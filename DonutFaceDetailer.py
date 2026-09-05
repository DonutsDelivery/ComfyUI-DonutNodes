"""
DonutFaceDetailer - FaceDetailer with max_faces limit and megapixel-based sizing.
Only processes the N largest detected faces, ignoring small background faces.
Uses total pixel count instead of max edge length for consistent VRAM usage.

Sampling canvases are aligned to 64 pixels to avoid diffusion/VAE quality loss on
awkward spatial grids. Multi-cycle refinement stays in latent space and performs
only one decode/downscale at the end.
"""

import inspect
import logging
import math

import numpy as np
import torch
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import nodes
from nodes import MAX_RESOLUTION

try:
    from .donut_detailer_core import offload_model_for_auxiliary_stage
except ImportError:
    from donut_detailer_core import offload_model_for_auxiliary_stage

try:
    from .krea2_edit_integration import (
        crop_image_padding,
        pad_image_to_multiple,
        prepare_krea2_edit,
    )
except ImportError:
    from krea2_edit_integration import (
        crop_image_padding,
        pad_image_to_multiple,
        prepare_krea2_edit,
    )

try:
    from .turbo_sampling import resolve_turbo_sampling
except ImportError:
    from turbo_sampling import resolve_turbo_sampling

try:
    import impact.core as core
    import impact.wildcards as impact_wildcards
    from impact import impact_sampling
    from impact import utils as impact_utils
    from comfy_extras import nodes_differential_diffusion
    IMPACT_AVAILABLE = True
except ImportError:
    IMPACT_AVAILABLE = False
    logging.warning("[DonutFaceDetailer] Impact Pack not found - node will be unavailable")


CANVAS_MULTIPLE = 64


def _snap_dimension(value, multiple=CANVAS_MULTIPLE):
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _ceil_dimension(value, multiple=CANVAS_MULTIPLE):
    return max(multiple, int(math.ceil(float(value) / multiple)) * multiple)


def _align_canvas(width, height, max_resolution=0, multiple=CANVAS_MULTIPLE):
    """Snap a canvas to a diffusion-friendly grid while preserving aspect closely."""
    width = float(width)
    height = float(height)
    if max_resolution > 0 and max(width, height) > max_resolution:
        scale = max_resolution / max(width, height)
        width *= scale
        height *= scale
    width = _snap_dimension(width, multiple)
    height = _snap_dimension(height, multiple)
    if max_resolution > 0 and max(width, height) > max_resolution:
        max_aligned = max(multiple, int(max_resolution // multiple) * multiple)
        scale = max_aligned / max(width, height)
        width = _snap_dimension(width * scale, multiple)
        height = _snap_dimension(height * scale, multiple)
        width = min(width, max_aligned)
        height = min(height, max_aligned)
    return int(width), int(height)


def _scale_to_target_pixels(width, height, target_pixels, max_resolution=0):
    pixels = width * height
    if pixels <= 0:
        return _align_canvas(width, height, max_resolution)
    scale = math.sqrt(target_pixels / pixels)
    return _align_canvas(width * scale, height * scale, max_resolution)


def _resize_mask(mask, height, width):
    """Normalize 2D/BHW/BHWC/BCHW masks to BHW at target size."""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    mask = mask.float()
    if mask.ndim == 2:
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask_4d = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask_4d = mask.permute(0, 3, 1, 2)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask_4d = mask
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
    resized = torch.nn.functional.interpolate(
        mask_4d, size=(height, width), mode="bilinear", align_corners=False,
    )
    return resized[:, 0]


def _combine_conditioning(primary, extra):
    if extra is None:
        return primary
    if hasattr(nodes, "ConditioningCombine"):
        return nodes.ConditioningCombine().combine(primary, extra)[0]
    return list(primary) + list(extra)


if IMPACT_AVAILABLE:
    class DonutFaceDetailer:
        """Face detailer with max-face filtering and megapixel target sizing."""

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                    "tooltip": "Equivalent square target size. Sampling canvases are snapped to 64-pixel multiples."}),
                "max_resolution": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Maximum edge length (0 = no limit)."}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"],),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                "bbox_detector": ("BBOX_DETECTOR",),
                "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "max_faces": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Maximum number of faces to process (largest by area)"}),
            }, "optional": {
                "sam_model_opt": ("SAM_MODEL",),
                "segm_detector_opt": ("SEGM_DETECTOR",),
                "detailer_hook": ("DETAILER_HOOK",),
                "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
                "edit_mode": ("BOOLEAN", {"default": False,
                    "tooltip": "Use Krea2 identity edit conditioning for each detected face."}),
                "edit_prompt": ("STRING", {"forceInput": True}),
                "edit_model": ("MODEL", {"tooltip": "Optional Krea2 model with the Identity Edit LoRA already applied. Falls back to model."}),
                "face_reference": ("IMAGE", {"tooltip": "Required in edit mode. The bbox detector extracts the identity face from this image."}),
                "edit_negative_prompt": ("STRING", {"forceInput": True}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64}),
                "vary_seed_per_face": ("BOOLEAN", {"default": False, "label_on": "per-face", "label_off": "shared",
                    "tooltip": "Use a unique seed offset for each detected face."}),
                "turbo_mode": ("BOOLEAN", {"default": False,
                    "tooltip": "Snap denoise to a valid Turbo scheduler point."}),
            }}

        RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
        RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
        OUTPUT_IS_LIST = (False, True, True, False, False, True)
        FUNCTION = "doit"
        CATEGORY = "ImpactPack/Simple"

        @staticmethod
        def enhance_detail_megapixel(
            image, model, clip, vae, resolution, max_resolution, guide_size_for_bbox,
            bbox, seed, steps, cfg, sampler_name, scheduler, positive, negative,
            denoise, noise_mask, force_inpaint, noise_mask_feather=0,
            inpaint_model=False, detailer_hook=None, scheduler_func=None,
            edit_mode=False, edit_prompt="Enhance facial details while preserving identity.",
            edit_negative_prompt="", grounding_px=768, edit_model=None,
            face_reference_crop=None, cycle=1, wildcard_opt=None,
            wildcard_concat_mode=None,
        ):
            h, w = image.shape[1:3]
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            if guide_size_for_bbox:
                bbox_pixels = bbox_w * bbox_h
                if bbox_pixels > resolution:
                    logging.info("[DonutFaceDetailer] Segment skip [bbox larger than target]")
                    return None, None
                if bbox_pixels > 0:
                    scale = math.sqrt(resolution / bbox_pixels)
                    new_w, new_h = _align_canvas(w * scale, h * scale, max_resolution)
                else:
                    new_w, new_h = _align_canvas(w, h, max_resolution)
            else:
                crop_pixels = w * h
                if crop_pixels > resolution:
                    logging.info("[DonutFaceDetailer] Segment skip [crop larger than target]")
                    return None, None
                new_w, new_h = _scale_to_target_pixels(w, h, resolution, max_resolution)

            upscale = new_w / max(w, 1)
            if upscale <= 1.0:
                if not force_inpaint:
                    logging.info("[DonutFaceDetailer] Segment skip [upscale=%.2f]", upscale)
                    return None, None
                new_w = _ceil_dimension(w)
                new_h = _ceil_dimension(h)
                upscale = new_w / max(w, 1)

            if detailer_hook is not None and hasattr(detailer_hook, "touch_scaled_size"):
                new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)
                new_w, new_h = _align_canvas(new_w, new_h, max_resolution)

            logging.info("[DonutFaceDetailer] Crop %dx%d -> %dx%d (%d pixels, target %d)",
                         w, h, new_w, new_h, new_w * new_h, resolution)
            scaled_image = impact_utils.tensor_resize(image, new_w, new_h)
            if noise_mask is not None:
                noise_mask = impact_utils.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
                noise_mask = _resize_mask(noise_mask, new_h, new_w)
            if detailer_hook is not None and hasattr(detailer_hook, "post_upscale"):
                scaled_image = detailer_hook.post_upscale(scaled_image, noise_mask)

            target_image = scaled_image
            target_padding = (0, 0, 0, 0)
            sampling_model = edit_model if edit_mode and edit_model is not None else model
            sampling_positive = positive
            sampling_negative = negative
            wildcard_positive = None
            if wildcard_opt:
                sampling_model, _, wildcard_positive = impact_wildcards.process_with_loras(wildcard_opt, sampling_model, clip)
                if not edit_mode:
                    if wildcard_concat_mode == "concat":
                        sampling_positive = nodes.ConditioningConcat().concat(sampling_positive, wildcard_positive)[0]
                    else:
                        sampling_positive = wildcard_positive

            if edit_mode:
                if face_reference_crop is None:
                    raise ValueError("DonutFaceDetailer edit_mode requires face_reference.")
                target_image, target_padding = pad_image_to_multiple(scaled_image)
                sampling_model, sampling_positive, sampling_negative, _, _ = prepare_krea2_edit(
                    sampling_model, clip, vae, face_reference_crop, edit_prompt,
                    edit_negative_prompt, grounding_px, target_image.shape[2], target_image.shape[1])
                if wildcard_positive is not None:
                    sampling_positive = _combine_conditioning(sampling_positive, wildcard_positive)
                if noise_mask is not None and any(target_padding):
                    left, top, right, bottom = target_padding
                    noise_mask = torch.nn.functional.pad(noise_mask, (left, right, top, bottom), value=0)

            if (noise_mask is not None and noise_mask_feather > 0
                    and "denoise_mask_function" not in sampling_model.model_options):
                sampling_model = nodes_differential_diffusion.DifferentialDiffusion().execute(sampling_model)[0]

            if inpaint_model and noise_mask is not None and not edit_mode:
                encode = nodes.InpaintModelConditioning().encode
                if "noise_mask" in inspect.signature(encode).parameters:
                    sampling_positive, sampling_negative, latent_image = encode(
                        sampling_positive, sampling_negative, target_image, vae,
                        mask=noise_mask, noise_mask=True)
                else:
                    sampling_positive, sampling_negative, latent_image = encode(
                        sampling_positive, sampling_negative, target_image, vae, noise_mask)
            else:
                latent_image = impact_utils.to_latent_image(target_image, vae)
                if noise_mask is not None:
                    latent_image["noise_mask"] = noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1]))

            if detailer_hook is not None and hasattr(detailer_hook, "post_encode"):
                latent_image = detailer_hook.post_encode(latent_image)
            skip_sampling = bool(detailer_hook is not None and hasattr(detailer_hook, "get_skip_sampling")
                                 and detailer_hook.get_skip_sampling())
            if skip_sampling:
                refined_image = target_image
            else:
                refined_latent = latent_image
                sampler_opt = detailer_hook.get_custom_sampler() if (
                    detailer_hook is not None and hasattr(detailer_hook, "get_custom_sampler")) else None
                for cycle_index in range(max(1, int(cycle))):
                    cycle_seed = (seed + cycle_index * 1000) & 0xffffffffffffffff
                    model2, seed2, steps2, cfg2 = sampling_model, cycle_seed, steps, cfg
                    sampler2, scheduler2 = sampler_name, scheduler
                    positive2, negative2 = sampling_positive, sampling_negative
                    latent2, denoise2, noise = refined_latent, denoise, None
                    if detailer_hook is not None:
                        if hasattr(detailer_hook, "set_steps"):
                            detailer_hook.set_steps((cycle_index, cycle))
                        if hasattr(detailer_hook, "cycle_latent"):
                            refined_latent = detailer_hook.cycle_latent(refined_latent)
                            latent2 = refined_latent
                        if hasattr(detailer_hook, "pre_ksample"):
                            (model2, seed2, steps2, cfg2, sampler2, scheduler2,
                             positive2, negative2, latent2, denoise2) = detailer_hook.pre_ksample(
                                sampling_model, cycle_seed, steps, cfg, sampler_name, scheduler,
                                sampling_positive, sampling_negative, refined_latent, denoise)
                        if hasattr(detailer_hook, "get_custom_noise"):
                            noise, touched = detailer_hook.get_custom_noise(
                                seed2, torch.zeros_like(latent2["samples"]), is_touched=False)
                            if not touched:
                                noise = None
                    refined_latent = impact_sampling.ksampler_wrapper(
                        model2, seed2, steps2, cfg2, sampler2, scheduler2,
                        positive2, negative2, latent2, denoise2, noise=noise,
                        scheduler_func=scheduler_func, sampler_opt=sampler_opt)
                if detailer_hook is not None and hasattr(detailer_hook, "pre_decode"):
                    refined_latent = detailer_hook.pre_decode(refined_latent)
                refined_image = vae.decode(refined_latent["samples"])
                if refined_image.ndim == 5:
                    refined_image = refined_image.reshape((-1,) + tuple(refined_image.shape[-3:]))
            if detailer_hook is not None and hasattr(detailer_hook, "post_decode"):
                refined_image = detailer_hook.post_decode(refined_image)
            if edit_mode and any(target_padding):
                refined_image = crop_image_padding(refined_image, target_padding)
            return refined_image, None

        @staticmethod
        def enhance_face(
            image, model, clip, vae, resolution, max_resolution, guide_size_for_bbox,
            seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
            feather, noise_mask_enabled, force_inpaint, bbox_threshold, bbox_dilation,
            bbox_crop_factor, sam_detection_hint, sam_dilation, sam_threshold,
            sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative,
            drop_size, bbox_detector, max_faces, segm_detector=None, sam_model_opt=None,
            wildcard_opt=None, detailer_hook=None, cycle=1, inpaint_model=False,
            noise_mask_feather=0, scheduler_func_opt=None, edit_mode=False,
            edit_prompt="Enhance facial details while preserving identity.",
            edit_negative_prompt="", grounding_px=768, edit_model=None,
            face_reference=None, vary_seed_per_face=False, turbo_mode=False,
        ):
            if turbo_mode:
                supported_steps = steps
                steps, denoise, matched_denoise = resolve_turbo_sampling(steps, denoise, scheduler)
                logging.info("[DonutFaceDetailer] Turbo: %d supported steps -> %d steps at scheduler denoise=%.3f (ComfyUI denoise=%.3f)",
                             supported_steps, steps, matched_denoise, denoise)
            offload_model_for_auxiliary_stage(model, comfy.model_management)
            if edit_mode and face_reference is None:
                raise ValueError("DonutFaceDetailer edit_mode requires face_reference.")

            bbox_detector.setAux("face")
            try:
                segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor,
                                            drop_size, detailer_hook=detailer_hook)
            finally:
                bbox_detector.setAux(None)
            if sam_model_opt is not None:
                sam_mask = core.make_sam_mask(sam_model_opt, segs, image, sam_detection_hint, sam_dilation,
                                              sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                                              sam_mask_hint_use_negative)
                segs = core.segs_bitwise_and_mask(segs, sam_mask)
            elif segm_detector is not None:
                segm_segs = segm_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
                if (hasattr(segm_detector, "override_bbox_by_segm") and segm_detector.override_bbox_by_segm
                        and not (detailer_hook is not None and not hasattr(detailer_hook, "override_bbox_by_segm"))):
                    segs = segm_segs
                else:
                    segs = core.segs_bitwise_and_mask(segs, core.segs_to_combined_mask(segm_segs))

            def segment_area(seg):
                return (seg.bbox[2] - seg.bbox[0]) * (seg.bbox[3] - seg.bbox[1])
            def has_nonzero_mask(seg):
                mask = seg.cropped_mask
                if mask is None:
                    return False
                return bool(torch.count_nonzero(mask)) if torch.is_tensor(mask) else bool(np.count_nonzero(mask))

            final_faces = [seg for seg in segs[1] if has_nonzero_mask(seg)]
            final_faces.sort(key=segment_area, reverse=True)
            segs = (segs[0], final_faces[:max_faces])
            reference_faces = []
            if edit_mode and segs[1]:
                bbox_detector.setAux("face")
                try:
                    reference_segs = bbox_detector.detect(face_reference, bbox_threshold, bbox_dilation,
                                                          bbox_crop_factor, drop_size, detailer_hook=detailer_hook)
                finally:
                    bbox_detector.setAux(None)
                reference_faces = sorted([seg for seg in reference_segs[1] if has_nonzero_mask(seg)],
                                         key=segment_area, reverse=True)
                if not reference_faces:
                    raise ValueError("DonutFaceDetailer found no face in face_reference.")

            wildcard_concat_mode = None
            wildcard_chooser = None
            wildcard_mode = None
            if wildcard_opt:
                wildcard_text = wildcard_opt
                if wildcard_text.startswith("[CONCAT]"):
                    wildcard_concat_mode, wildcard_text = "concat", wildcard_text[8:]
                wildcard_mode, wildcard_chooser = impact_wildcards.process_wildcard_for_segs(wildcard_text)

            if segs[1]:
                enhanced_img = image.clone()
                cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list = [], [], []
                for face_index, seg in enumerate(segs[1]):
                    crop_region, bbox, cropped_mask = seg.crop_region, seg.bbox, seg.cropped_mask
                    cropped_image = impact_utils.crop_image(enhanced_img, crop_region)
                    face_reference_crop = None
                    if edit_mode:
                        reference_seg = reference_faces[min(face_index, len(reference_faces) - 1)]
                        face_reference_crop = reference_seg.cropped_image
                        if face_reference_crop is None:
                            face_reference_crop = impact_utils.crop_image(face_reference, reference_seg.crop_region)
                    noise_mask = (_resize_mask(cropped_mask, cropped_image.shape[1], cropped_image.shape[2])
                                  if noise_mask_enabled and cropped_mask is not None else None)

                    wildcard_item, wildcard_seed = None, None
                    if wildcard_chooser is not None:
                        if wildcard_mode == "LAB":
                            wildcard_item = wildcard_chooser.get(seg)
                        else:
                            wildcard_seed, wildcard_item = wildcard_chooser.get(seg)
                        if wildcard_item and wildcard_item.strip() == "[SKIP]":
                            continue
                        if wildcard_item and wildcard_item.strip() == "[STOP]":
                            break
                    if wildcard_seed is not None:
                        face_seed = wildcard_seed
                    elif vary_seed_per_face:
                        face_seed = (seed + face_index) & 0xffffffffffffffff
                    else:
                        face_seed = seed

                    result, _ = DonutFaceDetailer.enhance_detail_megapixel(
                        cropped_image, model, clip, vae, resolution, max_resolution,
                        guide_size_for_bbox, bbox, face_seed, steps, cfg, sampler_name,
                        scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                        noise_mask_feather, inpaint_model, detailer_hook, scheduler_func_opt,
                        edit_mode, edit_prompt, edit_negative_prompt, grounding_px, edit_model,
                        face_reference_crop, cycle=cycle, wildcard_opt=wildcard_item,
                        wildcard_concat_mode=wildcard_concat_mode)
                    if result is None:
                        continue
                    enhanced_cropped = impact_utils.tensor_resize(result, cropped_image.shape[2], cropped_image.shape[1])
                    paste_mask = impact_utils.tensor_gaussian_blur_mask(impact_utils.to_tensor(seg.cropped_mask), feather)
                    enhanced_img, enhanced_cropped = enhanced_img.cpu(), enhanced_cropped.cpu()
                    impact_utils.tensor_paste(enhanced_img, enhanced_cropped,
                                              (crop_region[0], crop_region[1]), paste_mask)
                    if detailer_hook is not None and hasattr(detailer_hook, "post_paste"):
                        enhanced_img = detailer_hook.post_paste(enhanced_img)
                    cropped_enhanced.append(enhanced_cropped)
                    enhanced_alpha = impact_utils.tensor_convert_rgba(enhanced_cropped)
                    alpha_mask = paste_mask
                    if alpha_mask.shape[1:3] != enhanced_alpha.shape[1:3]:
                        alpha_mask = impact_utils.tensor_resize(alpha_mask, enhanced_alpha.shape[2], enhanced_alpha.shape[1])
                    impact_utils.tensor_putalpha(enhanced_alpha, alpha_mask)
                    cropped_enhanced_alpha.append(enhanced_alpha)
            else:
                enhanced_img, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list = image, [], [], []

            mask = core.segs_to_combined_mask(segs)
            if not cropped_enhanced:
                cropped_enhanced = [impact_utils.empty_pil_tensor()]
            if not cropped_enhanced_alpha:
                cropped_enhanced_alpha = [impact_utils.empty_pil_tensor()]
            if not cnet_pil_list:
                cnet_pil_list = [impact_utils.empty_pil_tensor()]
            return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list

        def doit(
            self, image, model, clip, vae, resolution, max_resolution, guide_size_for,
            seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
            feather, noise_mask, force_inpaint, bbox_threshold, bbox_dilation,
            bbox_crop_factor, sam_detection_hint, sam_dilation, sam_threshold,
            sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative,
            drop_size, bbox_detector, wildcard, cycle, max_faces, sam_model_opt=None,
            segm_detector_opt=None, detailer_hook=None, inpaint_model=False,
            noise_mask_feather=0, scheduler_func_opt=None, edit_mode=False,
            edit_prompt="Enhance facial details while preserving identity.",
            edit_negative_prompt="", grounding_px=768, edit_model=None,
            face_reference=None, vary_seed_per_face=False, turbo_mode=False,
        ):
            resolution *= resolution
            result_img = result_mask = None
            result_cropped_enhanced, result_cropped_enhanced_alpha, result_cnet_images = [], [], []
            if len(image) > 1:
                logging.warning("[DonutFaceDetailer] WARN: Not designed for video. Use Detailer For AnimateDiff.")
            for index, single_image in enumerate(image):
                single_face_reference = None
                if face_reference is not None:
                    reference_index = min(index, len(face_reference) - 1)
                    single_face_reference = face_reference[reference_index].unsqueeze(0)
                enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = DonutFaceDetailer.enhance_face(
                    single_image.unsqueeze(0), model, clip, vae, resolution, max_resolution,
                    guide_size_for, seed + index, steps, cfg, sampler_name, scheduler,
                    positive, negative, denoise, feather, noise_mask, force_inpaint,
                    bbox_threshold, bbox_dilation, bbox_crop_factor, sam_detection_hint,
                    sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                    sam_mask_hint_use_negative, drop_size, bbox_detector, max_faces,
                    segm_detector_opt, sam_model_opt, wildcard, detailer_hook, cycle=cycle,
                    inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
                    scheduler_func_opt=scheduler_func_opt, edit_mode=edit_mode,
                    edit_prompt=edit_prompt, edit_negative_prompt=edit_negative_prompt,
                    grounding_px=grounding_px, edit_model=edit_model,
                    face_reference=single_face_reference, vary_seed_per_face=vary_seed_per_face,
                    turbo_mode=turbo_mode)
                result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
                result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
                result_cropped_enhanced.extend(cropped_enhanced)
                result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
                result_cnet_images.extend(cnet_pil_list)
            pipe = (model, clip, vae, positive, negative, wildcard, bbox_detector,
                    segm_detector_opt, sam_model_opt, detailer_hook, None, None, None, None)
            return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images

    NODE_CLASS_MAPPINGS = {"DonutFaceDetailer": DonutFaceDetailer}
    NODE_DISPLAY_NAME_MAPPINGS = {"DonutFaceDetailer": "Face Detailer (Max Faces)"}
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

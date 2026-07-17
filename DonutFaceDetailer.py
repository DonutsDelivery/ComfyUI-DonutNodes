"""
DonutFaceDetailer - FaceDetailer with max_faces limit and megapixel-based sizing.
Only processes the N largest detected faces, ignoring small background faces.
Uses total pixel count instead of max edge length for consistent VRAM usage.
"""

import math
import numpy as np
import torch
import logging
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import nodes
from nodes import MAX_RESOLUTION

# Shared detailer helpers (extracted to remove duplication; behavior-identical)
try:
    from .donut_detailer_core import scale_to_megapixels, sample_and_decode
except ImportError:
    from donut_detailer_core import scale_to_megapixels, sample_and_decode

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

# Import from Impact Pack
try:
    import impact.core as core
    from impact import utils as impact_utils
    from impact import impact_sampling
    from comfy_extras import nodes_differential_diffusion
    IMPACT_AVAILABLE = True
except ImportError:
    IMPACT_AVAILABLE = False
    logging.warning("[DonutFaceDetailer] Impact Pack not found - node will be unavailable")


if IMPACT_AVAILABLE:
    class DonutFaceDetailer:
        """
        FaceDetailer with max_faces limit and megapixel-based sizing.
        Uses total pixel count (resolution) instead of max edge length for consistent VRAM usage.
        """

        @classmethod
        def INPUT_TYPES(s):
            return {"required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Target resolution as equivalent square size (e.g., 1024 = 1024x1024 pixels)"
                }),
                "max_resolution": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Maximum edge length (0 = no limit). Clamps if megapixel scaling would exceed this."
                }),
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

                "max_faces": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Maximum number of faces to process (largest by area)"
                }),
            },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL",),
                    "segm_detector_opt": ("SEGM_DETECTOR",),
                    "detailer_hook": ("DETAILER_HOOK",),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    "edit_mode": ("BOOLEAN", {
                        "default": False,
                        "tooltip": "Use natural 1-MiP Krea2 conditioning and pad the generated face target to a 32-pixel grid at the configured denoise.",
                    }),
                    "edit_prompt": ("STRING", {"forceInput": True}),
                    "edit_model": ("MODEL", {
                        "tooltip": "Optional Krea2 model with the Identity Edit LoRA already applied. Falls back to model.",
                    }),
                    "face_reference": ("IMAGE", {
                        "tooltip": "Required in edit mode. The connected bbox detector extracts the identity face independently of its image position.",
                    }),
                    "edit_negative_prompt": ("STRING", {"forceInput": True}),
                    "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64}),
                    "vary_seed_per_face": ("BOOLEAN", {
                        "default": False,
                        "label_on": "per-face",
                        "label_off": "shared",
                        "tooltip": "Use a unique seed offset for each detected face. Off preserves the same seed across faces.",
                    }),
                }}

        RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
        RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
        OUTPUT_IS_LIST = (False, True, True, False, False, True)
        FUNCTION = "doit"
        CATEGORY = "ImpactPack/Simple"

        @staticmethod
        def enhance_detail_megapixel(image, model, clip, vae, resolution, max_resolution, guide_size_for_bbox, bbox, seed, steps, cfg,
                                      sampler_name, scheduler, positive, negative, denoise,
                                      noise_mask, force_inpaint, noise_mask_feather=0,
                                      inpaint_model=False, detailer_hook=None, scheduler_func=None,
                                      edit_mode=False, edit_prompt="Enhance facial details while preserving identity.",
                                      edit_negative_prompt="", grounding_px=768, edit_model=None,
                                      face_reference_crop=None):
            """
            Enhanced detail function using megapixel-based sizing.
            Scales crop region to target total pixel count regardless of aspect ratio.
            """
            if noise_mask is not None:
                noise_mask = impact_utils.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
                noise_mask = noise_mask.squeeze(3)

            h = image.shape[1]
            w = image.shape[2]

            bbox_h = bbox[3] - bbox[1]
            bbox_w = bbox[2] - bbox[0]

            # Calculate target dimensions using megapixel approach
            if guide_size_for_bbox:
                # Scale based on bbox - smaller faces get more upscaling
                bbox_pixels = bbox_w * bbox_h
                if bbox_pixels > resolution:
                    logging.info("[DonutFaceDetailer] Segment skip [bbox larger than target]")
                    return None, None
                if bbox_pixels > 0:
                    scale = math.sqrt(resolution / bbox_pixels)
                    new_w = int(round(w * scale / 8) * 8)
                    new_h = int(round(h * scale / 8) * 8)
                    # Apply max_resolution clamp
                    if max_resolution > 0 and (new_w > max_resolution or new_h > max_resolution):
                        clamp_scale = max_resolution / max(new_w, new_h)
                        new_w = int(round(new_w * clamp_scale / 8) * 8)
                        new_h = int(round(new_h * clamp_scale / 8) * 8)
                    new_w = max(64, new_w)
                    new_h = max(64, new_h)
                else:
                    new_w, new_h = w, h
            else:
                # Scale based on crop region
                crop_pixels = w * h
                if crop_pixels > resolution:
                    logging.info("[DonutFaceDetailer] Segment skip [crop larger than target]")
                    return None, None
                new_w, new_h = scale_to_megapixels(w, h, resolution, max_resolution)

            # Calculate effective upscale factor
            upscale = new_w / w

            if not force_inpaint:
                if upscale <= 1.0:
                    logging.info(f"[DonutFaceDetailer] Segment skip [upscale={upscale:.2f}]")
                    return None, None

                if new_w == 0 or new_h == 0:
                    logging.info(f"[DonutFaceDetailer] Segment skip [zero size]")
                    return None, None
            else:
                if upscale <= 1.0 or new_w == 0 or new_h == 0:
                    logging.info("[DonutFaceDetailer] Force inpaint with original size")
                    upscale = 1.0
                    new_w = w
                    new_h = h

            if detailer_hook is not None:
                new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

            logging.info(f"[DonutFaceDetailer] Crop {w}x{h} -> {new_w}x{new_h} ({new_w*new_h:,} pixels, target {resolution:,})")

            # Scale image
            scaled_image = impact_utils.tensor_resize(image, new_w, new_h)
            target_image = scaled_image
            target_padding = (0, 0, 0, 0)

            sampling_model = model
            sampling_positive = positive
            sampling_negative = negative
            if edit_mode:
                if face_reference_crop is None:
                    raise ValueError("DonutFaceDetailer edit_mode requires face_reference.")
                target_image, target_padding = pad_image_to_multiple(scaled_image)
                sampling_model, sampling_positive, sampling_negative, _source_latent, _conditioning_image = prepare_krea2_edit(
                    edit_model if edit_model is not None else model,
                    clip, vae, face_reference_crop, edit_prompt,
                    edit_negative_prompt, grounding_px,
                    target_image.shape[2], target_image.shape[1],
                )
                latent_image = impact_utils.to_latent_image(target_image, vae)
            else:
                # Encode to latent for regular img2img sampling.
                latent_image = impact_utils.to_latent_image(scaled_image, vae)

            if (noise_mask is not None and noise_mask_feather > 0
                    and 'denoise_mask_function' not in sampling_model.model_options):
                sampling_model = nodes_differential_diffusion.DifferentialDiffusion().execute(
                    sampling_model,
                )[0]

            # Scale mask if present using torch interpolate
            if noise_mask is not None:
                # Ensure mask is 4D (N, C, H, W) for interpolate
                orig_shape = noise_mask.shape
                if len(orig_shape) == 2:
                    mask_4d = noise_mask.unsqueeze(0).unsqueeze(0)
                elif len(orig_shape) == 3:
                    mask_4d = noise_mask.unsqueeze(0)
                else:
                    mask_4d = noise_mask
                mask_4d = mask_4d.float()
                noise_mask = torch.nn.functional.interpolate(
                    mask_4d, size=(new_h, new_w), mode='bilinear', align_corners=False
                )
                # Restore original dimensions
                if len(orig_shape) == 2:
                    noise_mask = noise_mask.squeeze(0).squeeze(0)
                elif len(orig_shape) == 3:
                    noise_mask = noise_mask.squeeze(0)

                if edit_mode and any(target_padding):
                    left, top, right, bottom = target_padding
                    noise_mask = torch.nn.functional.pad(
                        noise_mask, (left, right, top, bottom), value=0,
                    )

            # Hook pre-processing
            if detailer_hook is not None:
                latent_image = detailer_hook.touch_latent(latent_image)

            # Prepare noise mask for latent
            if noise_mask is not None:
                latent_image['noise_mask'] = noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1]))

            # Sample -> VAE decode -> 5D->4D video-VAE reshape (shared helper)
            refined_image = sample_and_decode(sampling_model, seed, steps, cfg, sampler_name, scheduler,
                                              sampling_positive, sampling_negative, latent_image, denoise,
                                              vae, impact_sampling, scheduler_func=scheduler_func)

            if edit_mode and any(target_padding):
                refined_image = crop_image_padding(refined_image, target_padding)

            # Hook post-processing
            if detailer_hook is not None:
                refined_image = detailer_hook.post_detection(refined_image)

            return refined_image, None

        @staticmethod
        def enhance_face(image, model, clip, vae, resolution, max_resolution, guide_size_for_bbox, seed, steps, cfg,
                         sampler_name, scheduler, positive, negative, denoise, feather, noise_mask_enabled, force_inpaint,
                         bbox_threshold, bbox_dilation, bbox_crop_factor,
                         sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                         sam_mask_hint_use_negative, drop_size, bbox_detector, max_faces,
                         segm_detector=None, sam_model_opt=None, wildcard_opt=None, detailer_hook=None,
                         cycle=1, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None,
                         edit_mode=False, edit_prompt="Enhance facial details while preserving identity.",
                         edit_negative_prompt="", grounding_px=768, edit_model=None,
                         face_reference=None, vary_seed_per_face=False):

            # Unload diffusion model before detection to free VRAM for SAM/detector
            comfy.model_management.unload_all_models()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if edit_mode and face_reference is None:
                raise ValueError("DonutFaceDetailer edit_mode requires face_reference.")

            # Detect generated faces. Reference detection happens after final
            # SAM/segmentation refinement so pairing uses the actual targets.
            bbox_detector.setAux('face')
            try:
                segs = bbox_detector.detect(
                    image, bbox_threshold, bbox_dilation, bbox_crop_factor,
                    drop_size, detailer_hook=detailer_hook,
                )
            finally:
                bbox_detector.setAux(None)

            reference_faces = []

            # bbox + sam combination
            if sam_model_opt is not None:
                sam_mask = core.make_sam_mask(sam_model_opt, segs, image, sam_detection_hint, sam_dilation,
                                              sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                                              sam_mask_hint_use_negative)
                segs = core.segs_bitwise_and_mask(segs, sam_mask)

            elif segm_detector is not None:
                segm_segs = segm_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)

                if (hasattr(segm_detector, 'override_bbox_by_segm') and segm_detector.override_bbox_by_segm and
                        not (detailer_hook is not None and not hasattr(detailer_hook, 'override_bbox_by_segm'))):
                    segs = segm_segs
                else:
                    segm_mask = core.segs_to_combined_mask(segm_segs)
                    segs = core.segs_bitwise_and_mask(segs, segm_mask)

            def segment_area(seg):
                return (
                    (seg.bbox[2] - seg.bbox[0])
                    * (seg.bbox[3] - seg.bbox[1])
                )

            def has_nonzero_mask(seg):
                mask = seg.cropped_mask
                if mask is None:
                    return False
                if torch.is_tensor(mask):
                    return bool(torch.count_nonzero(mask))
                return bool(np.count_nonzero(mask))

            final_faces = [seg for seg in segs[1] if has_nonzero_mask(seg)]
            final_faces.sort(key=segment_area, reverse=True)
            segs = (segs[0], final_faces[:max_faces])

            if edit_mode and segs[1]:
                bbox_detector.setAux('face')
                try:
                    reference_segs = bbox_detector.detect(
                        face_reference, bbox_threshold, bbox_dilation,
                        bbox_crop_factor, drop_size,
                        detailer_hook=detailer_hook,
                    )
                finally:
                    bbox_detector.setAux(None)
                reference_faces = sorted(
                    reference_segs[1], key=segment_area, reverse=True,
                )
                if not reference_faces:
                    raise ValueError(
                        "DonutFaceDetailer found no face in face_reference."
                    )

            # Unload detection models before inpainting to free VRAM
            comfy.model_management.unload_all_models()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Process each segment with megapixel-based sizing
            if len(segs[1]) > 0:
                enhanced_img = image.clone()
                cropped_enhanced = []
                cropped_enhanced_alpha = []
                cnet_pil_list = []

                for face_index, seg in enumerate(segs[1]):
                    # Get crop region
                    cropped_image = seg.cropped_image
                    cropped_mask = seg.cropped_mask
                    crop_region = seg.crop_region
                    bbox = seg.bbox

                    face_reference_crop = None
                    if edit_mode:
                        reference_seg = reference_faces[
                            min(face_index, len(reference_faces) - 1)
                        ]
                        face_reference_crop = reference_seg.cropped_image
                        if face_reference_crop is None:
                            face_reference_crop = impact_utils.crop_image(
                                face_reference, reference_seg.crop_region,
                            )

                    if cropped_image is None:
                        cropped_image = impact_utils.crop_image(image, crop_region)

                    if noise_mask_enabled and cropped_mask is not None:
                        # Convert to tensor if numpy array
                        if isinstance(cropped_mask, np.ndarray):
                            cropped_mask = torch.from_numpy(cropped_mask)
                        # Resize mask using torch interpolate - ensure 4D (N, C, H, W)
                        orig_shape = cropped_mask.shape
                        if len(orig_shape) == 2:
                            mask_4d = cropped_mask.unsqueeze(0).unsqueeze(0)
                        elif len(orig_shape) == 3:
                            mask_4d = cropped_mask.unsqueeze(0)
                        else:
                            mask_4d = cropped_mask
                        mask_4d = mask_4d.float()
                        noise_mask = torch.nn.functional.interpolate(
                            mask_4d, size=(cropped_image.shape[1], cropped_image.shape[2]), mode='bilinear', align_corners=False
                        ).squeeze(0).squeeze(0)
                    else:
                        noise_mask = None

                    # Run detail enhancement cycles
                    enhanced_cropped = cropped_image
                    face_seed = (
                        seed + face_index if vary_seed_per_face else seed
                    ) & 0xffffffffffffffff
                    for c in range(cycle):
                        cycle_seed = (face_seed + c * 1000) & 0xffffffffffffffff
                        result, _ = DonutFaceDetailer.enhance_detail_megapixel(
                            enhanced_cropped, model, clip, vae, resolution, max_resolution, guide_size_for_bbox, bbox, cycle_seed, steps, cfg,
                            sampler_name, scheduler, positive, negative, denoise,
                            noise_mask if c == 0 else None, force_inpaint, noise_mask_feather,
                            inpaint_model, detailer_hook, scheduler_func_opt,
                            edit_mode, edit_prompt, edit_negative_prompt, grounding_px,
                            edit_model, face_reference_crop)

                        if result is not None:
                            # Resize back to original crop size
                            enhanced_cropped = impact_utils.tensor_resize(result, cropped_image.shape[2], cropped_image.shape[1])

                    # Paste back
                    if enhanced_cropped is not None:
                        # Prepare mask for pasting with feathering
                        paste_mask = impact_utils.to_tensor(seg.cropped_mask)
                        paste_mask = impact_utils.tensor_gaussian_blur_mask(paste_mask, feather)
                        enhanced_img = enhanced_img.cpu()
                        enhanced_cropped = enhanced_cropped.cpu()
                        impact_utils.tensor_paste(enhanced_img, enhanced_cropped, (crop_region[0], crop_region[1]), paste_mask)
                        cropped_enhanced.append(enhanced_cropped)

                        # Create alpha version (just use the enhanced image for now)
                        cropped_enhanced_alpha.append(enhanced_cropped)

            else:
                enhanced_img = image
                cropped_enhanced = []
                cropped_enhanced_alpha = []
                cnet_pil_list = []

            # Generate combined mask
            mask = core.segs_to_combined_mask(segs)

            if len(cropped_enhanced) == 0:
                cropped_enhanced = [impact_utils.empty_pil_tensor()]

            if len(cropped_enhanced_alpha) == 0:
                cropped_enhanced_alpha = [impact_utils.empty_pil_tensor()]

            if len(cnet_pil_list) == 0:
                cnet_pil_list = [impact_utils.empty_pil_tensor()]

            return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list

        def doit(self, image, model, clip, vae, resolution, max_resolution, guide_size_for, seed, steps, cfg, sampler_name,
                 scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                 bbox_threshold, bbox_dilation, bbox_crop_factor,
                 sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                 sam_mask_hint_use_negative, drop_size, bbox_detector, wildcard, cycle, max_faces,
                 sam_model_opt=None, segm_detector_opt=None, detailer_hook=None, inpaint_model=False,
                 noise_mask_feather=0, scheduler_func_opt=None, edit_mode=False,
                 edit_prompt="Enhance facial details while preserving identity.",
                 edit_negative_prompt="", grounding_px=768, edit_model=None,
                 face_reference=None, vary_seed_per_face=False):

            # Convert resolution from square side length to total pixels
            resolution = resolution * resolution

            result_img = None
            result_mask = None
            result_cropped_enhanced = []
            result_cropped_enhanced_alpha = []
            result_cnet_images = []

            if len(image) > 1:
                logging.warning("[DonutFaceDetailer] WARN: Not designed for video. Use Detailer For AnimateDiff.")

            for i, single_image in enumerate(image):
                single_face_reference = None
                if face_reference is not None:
                    reference_index = min(i, len(face_reference) - 1)
                    single_face_reference = face_reference[reference_index].unsqueeze(0)
                enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = \
                    DonutFaceDetailer.enhance_face(
                        single_image.unsqueeze(0), model, clip, vae, resolution, max_resolution, guide_size_for,
                        seed + i, steps, cfg, sampler_name, scheduler, positive, negative, denoise, feather,
                        noise_mask, force_inpaint, bbox_threshold, bbox_dilation, bbox_crop_factor,
                        sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
                        sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size, bbox_detector,
                        max_faces, segm_detector_opt, sam_model_opt, wildcard, detailer_hook,
                        cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
                        scheduler_func_opt=scheduler_func_opt, edit_mode=edit_mode,
                        edit_prompt=edit_prompt, edit_negative_prompt=edit_negative_prompt,
                        grounding_px=grounding_px, edit_model=edit_model,
                        face_reference=single_face_reference,
                        vary_seed_per_face=vary_seed_per_face)

                result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
                result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
                result_cropped_enhanced.extend(cropped_enhanced)
                result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
                result_cnet_images.extend(cnet_pil_list)

            pipe = (model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt,
                    detailer_hook, None, None, None, None)
            return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images

    NODE_CLASS_MAPPINGS = {
        "DonutFaceDetailer": DonutFaceDetailer,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "DonutFaceDetailer": "Face Detailer (Max Faces)",
    }

else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

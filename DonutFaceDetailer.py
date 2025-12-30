"""
DonutFaceDetailer - FaceDetailer with max_faces limit and megapixel-based sizing.
Only processes the N largest detected faces, ignoring small background faces.
Uses total pixel count instead of max edge length for consistent VRAM usage.
"""

import math
import torch
import logging
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import nodes
from nodes import MAX_RESOLUTION

# Import from Impact Pack
try:
    import impact.core as core
    from impact import utils as impact_utils
    from comfy_extras import nodes_differential_diffusion
    IMPACT_AVAILABLE = True
except ImportError:
    IMPACT_AVAILABLE = False
    logging.warning("[DonutFaceDetailer] Impact Pack not found - node will be unavailable")


def filter_segs_by_area(segs, max_count):
    """
    Filter SEGS to keep only the N largest by bounding box area.
    segs format: (shape, list_of_seg)
    """
    shape, seg_list = segs

    if not seg_list or max_count <= 0 or len(seg_list) <= max_count:
        return segs

    def get_area(seg):
        bbox = seg.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height

    sorted_segs = sorted(seg_list, key=get_area, reverse=True)
    return (shape, sorted_segs[:max_count])


def scale_to_megapixels(w, h, target_pixels):
    """
    Calculate new dimensions that maintain aspect ratio and hit target pixel count.
    Returns (new_w, new_h) rounded to nearest 8 pixels.
    """
    current_pixels = w * h
    if current_pixels <= 0:
        return w, h

    scale = math.sqrt(target_pixels / current_pixels)
    new_w = int(round(w * scale / 8) * 8)
    new_h = int(round(h * scale / 8) * 8)

    # Ensure minimum size
    new_w = max(64, new_w)
    new_h = max(64, new_h)

    return new_w, new_h


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
                }}

        RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
        RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
        OUTPUT_IS_LIST = (False, True, True, False, False, True)
        FUNCTION = "doit"
        CATEGORY = "ImpactPack/Simple"

        @staticmethod
        def enhance_detail_megapixel(image, model, clip, vae, resolution, bbox, seed, steps, cfg,
                                      sampler_name, scheduler, positive, negative, denoise,
                                      noise_mask, force_inpaint, noise_mask_feather=0,
                                      inpaint_model=False, detailer_hook=None, scheduler_func=None):
            """
            Enhanced detail function using megapixel-based sizing.
            Scales crop region to target total pixel count regardless of aspect ratio.
            """
            if noise_mask is not None:
                noise_mask = impact_utils.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
                noise_mask = noise_mask.squeeze(3)

                if noise_mask_feather > 0 and 'denoise_mask_function' not in model.model_options:
                    model = nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]

            h = image.shape[1]
            w = image.shape[2]

            bbox_h = bbox[3] - bbox[1]
            bbox_w = bbox[2] - bbox[0]

            # Calculate target dimensions using megapixel approach
            new_w, new_h = scale_to_megapixels(w, h, resolution)

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
            scaled_image = core.tensor_resize(image, new_w, new_h)

            # Scale mask if present
            if noise_mask is not None:
                noise_mask = core.tensor_resize(noise_mask, new_w, new_h)
                if inpaint_model:
                    latent_image = core.make_masked_latent_for_inpaint(scaled_image, noise_mask, vae)
                else:
                    latent_image = core.to_latent_image(scaled_image, vae)
            else:
                latent_image = core.to_latent_image(scaled_image, vae)

            # Hook pre-processing
            if detailer_hook is not None:
                latent_image = detailer_hook.touch_latent(latent_image)

            # Prepare noise mask for latent
            if noise_mask is not None:
                latent_image['noise_mask'] = noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1]))

            # Sample
            refined_latent = core.ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler,
                                                    positive, negative, latent_image, denoise,
                                                    scheduler_func=scheduler_func)

            # Decode
            refined_image = core.latent_to_image(refined_latent, vae)

            # Hook post-processing
            if detailer_hook is not None:
                refined_image = detailer_hook.post_detection(refined_image)

            return refined_image, None

        @staticmethod
        def enhance_face(image, model, clip, vae, resolution, seed, steps, cfg,
                         sampler_name, scheduler, positive, negative, denoise, feather, noise_mask_enabled, force_inpaint,
                         bbox_threshold, bbox_dilation, bbox_crop_factor,
                         sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                         sam_mask_hint_use_negative, drop_size, bbox_detector, max_faces,
                         segm_detector=None, sam_model_opt=None, wildcard_opt=None, detailer_hook=None,
                         cycle=1, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

            # Detect faces
            bbox_detector.setAux('face')
            segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size,
                                        detailer_hook=detailer_hook)
            bbox_detector.setAux(None)

            # Filter to keep only the N largest faces
            segs = filter_segs_by_area(segs, max_faces)

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

            # Process each segment with megapixel-based sizing
            if len(segs[1]) > 0:
                enhanced_img = image.clone()
                cropped_enhanced = []
                cropped_enhanced_alpha = []
                cnet_pil_list = []

                for seg in segs[1]:
                    # Get crop region
                    cropped_image = seg.cropped_image
                    cropped_mask = seg.cropped_mask
                    crop_region = seg.crop_region
                    bbox = seg.bbox

                    if cropped_image is None:
                        cropped_image = core.crop_image(image, crop_region)

                    if noise_mask_enabled and cropped_mask is not None:
                        noise_mask = core.tensor_resize(cropped_mask.unsqueeze(0).unsqueeze(0),
                                                        cropped_image.shape[2], cropped_image.shape[1])
                        noise_mask = noise_mask.squeeze(0).squeeze(0)
                    else:
                        noise_mask = None

                    # Run detail enhancement cycles
                    enhanced_cropped = cropped_image
                    for c in range(cycle):
                        cycle_seed = seed + c * 1000
                        result, _ = DonutFaceDetailer.enhance_detail_megapixel(
                            enhanced_cropped, model, clip, vae, resolution, bbox, cycle_seed, steps, cfg,
                            sampler_name, scheduler, positive, negative, denoise,
                            noise_mask if c == 0 else None, force_inpaint, noise_mask_feather,
                            inpaint_model, detailer_hook, scheduler_func_opt)

                        if result is not None:
                            # Resize back to original crop size
                            enhanced_cropped = core.tensor_resize(result, cropped_image.shape[2], cropped_image.shape[1])

                    # Paste back
                    if enhanced_cropped is not None:
                        enhanced_img = core.tensor_paste(enhanced_img, enhanced_cropped, crop_region, feather)
                        cropped_enhanced.append(enhanced_cropped)

                        # Create alpha version
                        if cropped_mask is not None:
                            alpha_cropped = core.tensor_add_alpha(enhanced_cropped, cropped_mask)
                            cropped_enhanced_alpha.append(alpha_cropped)
                        else:
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

        def doit(self, image, model, clip, vae, resolution, seed, steps, cfg, sampler_name,
                 scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                 bbox_threshold, bbox_dilation, bbox_crop_factor,
                 sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                 sam_mask_hint_use_negative, drop_size, bbox_detector, wildcard, cycle, max_faces,
                 sam_model_opt=None, segm_detector_opt=None, detailer_hook=None, inpaint_model=False,
                 noise_mask_feather=0, scheduler_func_opt=None):

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
                enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = \
                    DonutFaceDetailer.enhance_face(
                        single_image.unsqueeze(0), model, clip, vae, resolution,
                        seed + i, steps, cfg, sampler_name, scheduler, positive, negative, denoise, feather,
                        noise_mask, force_inpaint, bbox_threshold, bbox_dilation, bbox_crop_factor,
                        sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
                        sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size, bbox_detector,
                        max_faces, segm_detector_opt, sam_model_opt, wildcard, detailer_hook,
                        cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
                        scheduler_func_opt=scheduler_func_opt)

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

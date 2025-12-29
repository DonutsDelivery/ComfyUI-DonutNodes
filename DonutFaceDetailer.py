"""
DonutFaceDetailer - FaceDetailer with max_faces limit.
Only processes the N largest detected faces, ignoring small background faces.
"""

import torch
import logging
import comfy.samplers
import nodes
from nodes import MAX_RESOLUTION

# Import from Impact Pack
try:
    import impact.core as core
    from impact.impact_pack import DetailerForEach
    from impact import utils
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


if IMPACT_AVAILABLE:
    class DonutFaceDetailer:
        """
        FaceDetailer with max_faces limit.
        Only processes the N largest detected faces, useful for ignoring small background faces.
        """

        @classmethod
        def INPUT_TYPES(s):
            return {"required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
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
        def enhance_face(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg,
                         sampler_name, scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                         bbox_threshold, bbox_dilation, bbox_crop_factor,
                         sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                         sam_mask_hint_use_negative, drop_size, bbox_detector, max_faces,
                         segm_detector=None, sam_model_opt=None, wildcard_opt=None, detailer_hook=None,
                         refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                         refiner_negative=None, cycle=1, inpaint_model=False, noise_mask_feather=0,
                         scheduler_func_opt=None):

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

            if len(segs[1]) > 0:
                enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = \
                    DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size,
                                              seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                                              feather, noise_mask, force_inpaint, wildcard_opt, detailer_hook,
                                              refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                              refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                                              refiner_negative=refiner_negative, cycle=cycle, inpaint_model=inpaint_model,
                                              noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt)
            else:
                enhanced_img = image
                cropped_enhanced = []
                cropped_enhanced_alpha = []
                cnet_pil_list = []

            # Mask Generator
            mask = core.segs_to_combined_mask(segs)

            if len(cropped_enhanced) == 0:
                cropped_enhanced = [utils.empty_pil_tensor()]

            if len(cropped_enhanced_alpha) == 0:
                cropped_enhanced_alpha = [utils.empty_pil_tensor()]

            if len(cnet_pil_list) == 0:
                cnet_pil_list = [utils.empty_pil_tensor()]

            return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list

        def doit(self, image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                 scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                 bbox_threshold, bbox_dilation, bbox_crop_factor,
                 sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                 sam_mask_hint_use_negative, drop_size, bbox_detector, wildcard, cycle, max_faces,
                 sam_model_opt=None, segm_detector_opt=None, detailer_hook=None, inpaint_model=False,
                 noise_mask_feather=0, scheduler_func_opt=None):

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
                        single_image.unsqueeze(0), model, clip, vae, guide_size, guide_size_for, max_size,
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

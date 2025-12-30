"""
DonutUniversalDetailer - Universal object detailer using Florence-2 or Qwen2.5-VL.
Detects any object in image, generates prompts from descriptions, and enhances each.
"""

import math
import numpy as np
import torch
import logging
from collections import namedtuple
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
    from impact import impact_sampling
    from comfy_extras import nodes_differential_diffusion
    IMPACT_AVAILABLE = True
except ImportError:
    IMPACT_AVAILABLE = False
    logging.warning("[DonutUniversalDetailer] Impact Pack not found - node will be unavailable")

# Import Florence-2 from layerstyle
try:
    from custom_nodes.comfyui_layerstyle.py.florence2_ultra import process_image
    from custom_nodes.comfyui_layerstyle.py.imagefunc import tensor2pil, pil2tensor
    FLORENCE2_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        import sys
        import os
        layerstyle_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'comfyui_layerstyle', 'py')
        if layerstyle_path not in sys.path:
            sys.path.insert(0, layerstyle_path)
        from florence2_ultra import process_image
        from imagefunc import tensor2pil, pil2tensor
        FLORENCE2_AVAILABLE = True
    except ImportError:
        FLORENCE2_AVAILABLE = False
        logging.warning("[DonutUniversalDetailer] comfyui_layerstyle not found - Florence-2 unavailable")

# Define SEG if Impact Pack not available
if not IMPACT_AVAILABLE:
    SEG = namedtuple("SEG",
                     ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                     defaults=[None])
else:
    SEG = core.SEG


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    if FLORENCE2_AVAILABLE:
        return tensor2pil(tensor)
    # Fallback
    from PIL import Image
    if tensor.dim() == 4:
        tensor = tensor[0]
    np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def pil_to_tensor(pil_img):
    """Convert PIL Image to tensor."""
    if FLORENCE2_AVAILABLE:
        return pil2tensor(pil_img)
    # Fallback
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).unsqueeze(0)


def florence2_detect(florence2_model, image_pil, prompt, task='caption to phrase grounding'):
    """
    Run Florence-2 detection.

    Returns: list of (bbox, label) tuples
             bbox format: [x1, y1, x2, y2]
    """
    model = florence2_model['model']
    processor = florence2_model['processor']

    # Detection parameters
    max_new_tokens = 512
    num_beams = 3
    do_sample = False
    fill_mask = False

    results, _ = process_image(
        model, processor, image_pil, task,
        max_new_tokens, num_beams, do_sample,
        fill_mask, prompt
    )

    if not isinstance(results, dict) or 'bboxes' not in results:
        return [], []

    bboxes = []
    labels = []

    for i, bbox in enumerate(results['bboxes']):
        if len(bbox) == 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        elif len(bbox) == 8:
            # Polygon format - get bounding box
            x1 = int(min(bbox[0::2]))
            x2 = int(max(bbox[0::2]))
            y1 = int(min(bbox[1::2]))
            y2 = int(max(bbox[1::2]))
        else:
            continue

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        bboxes.append([x1, y1, x2, y2])
        label = results['labels'][i] if i < len(results['labels']) else prompt
        labels.append(label.removeprefix("</s>").strip())

    return bboxes, labels


def make_crop_region(w, h, bbox, crop_factor):
    """Create expanded crop region around bbox."""
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Expand by crop_factor
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_w = bbox_w * crop_factor
    new_h = bbox_h * crop_factor

    cx1 = int(max(0, center_x - new_w / 2))
    cy1 = int(max(0, center_y - new_h / 2))
    cx2 = int(min(w, center_x + new_w / 2))
    cy2 = int(min(h, center_y + new_h / 2))

    return [cx1, cy1, cx2, cy2]


def bboxes_to_segs(bboxes, labels, image_shape, crop_factor=3.0, dilation=10):
    """
    Convert Florence-2 bboxes to Impact Pack SEGS format.

    image_shape: (batch, height, width, channels) from tensor
    """
    h, w = image_shape[1], image_shape[2]
    result = []

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Skip invalid boxes
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            continue

        # Create crop_region with padding
        crop_region = make_crop_region(w, h, bbox, crop_factor)
        cx1, cy1, cx2, cy2 = crop_region

        # Create cropped mask (binary mask for the bbox within crop_region)
        cropped_mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.float32)

        # Translate bbox to crop_region coordinates
        rel_x1 = max(0, x1 - cx1)
        rel_y1 = max(0, y1 - cy1)
        rel_x2 = min(cx2 - cx1, x2 - cx1)
        rel_y2 = min(cy2 - cy1, y2 - cy1)
        cropped_mask[rel_y1:rel_y2, rel_x1:rel_x2] = 1.0

        # Apply dilation if requested
        if dilation > 0:
            import cv2
            kernel = np.ones((dilation, dilation), np.uint8)
            cropped_mask = cv2.dilate(cropped_mask, kernel, iterations=1)

        # Create SEG
        seg = SEG(
            cropped_image=None,
            cropped_mask=cropped_mask,
            confidence=1.0,
            crop_region=crop_region,
            bbox=[x1, y1, x2, y2],
            label=label,
            control_net_wrapper=None
        )
        result.append(seg)

    return ((h, w), result)


def filter_segs_by_area(segs, max_count):
    """Filter SEGS to keep only the N largest by bounding box area."""
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


def filter_segs_by_area_percent(segs, image_shape, min_percent):
    """Filter SEGS by minimum area percentage of total image."""
    if min_percent <= 0:
        return segs

    shape, seg_list = segs
    h, w = image_shape[1], image_shape[2]
    total_area = h * w
    min_area = total_area * (min_percent / 100.0)

    filtered = []
    for seg in seg_list:
        bbox = seg.bbox
        seg_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if seg_area >= min_area:
            filtered.append(seg)

    return (shape, filtered)


def scale_coords(segs, scale_factor):
    """Scale all coordinates in SEGS by a factor."""
    shape, seg_list = segs
    new_shape = (int(shape[0] * scale_factor), int(shape[1] * scale_factor))

    scaled_segs = []
    for seg in seg_list:
        # Scale bbox
        bbox = [int(v * scale_factor) for v in seg.bbox]

        # Scale crop_region
        crop_region = [int(v * scale_factor) for v in seg.crop_region]

        # Scale mask
        if seg.cropped_mask is not None:
            mask = seg.cropped_mask
            new_h = crop_region[3] - crop_region[1]
            new_w = crop_region[2] - crop_region[0]
            if isinstance(mask, np.ndarray):
                import cv2
                scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
        else:
            scaled_mask = None

        scaled_segs.append(SEG(
            cropped_image=None,
            cropped_mask=scaled_mask,
            confidence=seg.confidence,
            crop_region=crop_region,
            bbox=bbox,
            label=seg.label,
            control_net_wrapper=seg.control_net_wrapper
        ))

    return (new_shape, scaled_segs)


if IMPACT_AVAILABLE and FLORENCE2_AVAILABLE:
    class DonutUniversalDetailer:
        """
        Universal object detailer using Florence-2 (or future Qwen2.5-VL).
        Detects any objects, generates prompts from descriptions, enhances each.
        """

        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "image": ("IMAGE",),
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "vae": ("VAE",),
                    "florence2_model": ("FLORENCE2",),

                    "detection_prompt": ("STRING", {
                        "default": "face, hands, eyes",
                        "multiline": False,
                        "tooltip": "Objects to detect (comma-separated)"
                    }),

                    # Filtering
                    "min_area_percent": ("FLOAT", {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                        "tooltip": "Minimum object area as % of image (0 = no minimum)"
                    }),
                    "max_objects": ("INT", {
                        "default": 5,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Maximum objects to process (largest first)"
                    }),

                    # Resolution
                    "resolution": ("INT", {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Target resolution (1024 = 1024x1024 = ~1 megapixel)"
                    }),
                    "upscale_full_image": ("BOOLEAN", {
                        "default": True,
                        "label_on": "pre-scale",
                        "label_off": "per-crop",
                        "tooltip": "Pre-scale full image for quality vs scale each crop"
                    }),
                    "return_original_size": ("BOOLEAN", {
                        "default": False,
                        "label_on": "original",
                        "label_off": "upscaled",
                        "tooltip": "Return at original size or keep upscaled"
                    }),

                    # Sampling
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                    "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),

                    # Conditioning
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "prompt_mode": (["append_label", "replace_with_label", "use_parent"],),

                    # Mask/Inpaint
                    "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                    "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),

                    # Detection tuning
                    "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                    "bbox_dilation": ("INT", {"default": 10, "min": 0, "max": 512, "step": 1}),
                },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL",),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                }
            }

        RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
        RETURN_NAMES = ("image", "cropped_enhanced", "mask", "detected_labels")
        OUTPUT_IS_LIST = (False, True, False, False)
        FUNCTION = "doit"
        CATEGORY = "donut/Detailers"

        def get_object_conditioning(self, clip, positive, negative, label, prompt_mode):
            """Generate conditioning based on detected object label."""
            if prompt_mode == "use_parent":
                return positive, negative

            elif prompt_mode == "append_label":
                # Encode the label and concatenate
                tokens = clip.tokenize(f", {label}, detailed, high quality")
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                label_cond = [[cond, {"pooled_output": pooled}]]
                # Concatenate with parent
                combined = nodes.ConditioningConcat().concat(positive, label_cond)[0]
                return combined, negative

            elif prompt_mode == "replace_with_label":
                # Encode just the label
                tokens = clip.tokenize(f"{label}, detailed, high quality, sharp")
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                return [[cond, {"pooled_output": pooled}]], negative

            return positive, negative

        def doit(self, image, model, clip, vae, florence2_model, detection_prompt,
                 min_area_percent, max_objects, resolution, upscale_full_image, return_original_size,
                 seed, steps, cfg, sampler_name, scheduler, denoise,
                 positive, negative, prompt_mode, feather, noise_mask, force_inpaint,
                 bbox_crop_factor, bbox_dilation,
                 sam_model_opt=None, inpaint_model=False, noise_mask_feather=20):

            original_h, original_w = image.shape[1], image.shape[2]
            target_pixels = resolution * resolution

            # 1. Unload diffusion model before detection
            comfy.model_management.unload_all_models()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 2. Run Florence-2 detection on original image
            image_pil = tensor_to_pil(image)
            bboxes, labels = florence2_detect(florence2_model, image_pil, detection_prompt)

            if len(bboxes) == 0:
                logging.info("[DonutUniversalDetailer] No objects detected")
                # Return combined labels string
                return (image, [image], torch.zeros((1, original_h, original_w)), "")

            logging.info(f"[DonutUniversalDetailer] Detected {len(bboxes)} objects: {labels}")

            # 3. Convert to SEGS format
            segs = bboxes_to_segs(bboxes, labels, image.shape, bbox_crop_factor, bbox_dilation)

            # 4. Filter by area percentage
            segs = filter_segs_by_area_percent(segs, image.shape, min_area_percent)

            # 5. Filter by count (keep N largest)
            segs = filter_segs_by_area(segs, max_objects)

            if len(segs[1]) == 0:
                logging.info("[DonutUniversalDetailer] All objects filtered out")
                return (image, [image], torch.zeros((1, original_h, original_w)), "")

            # 6. Optional SAM refinement
            if sam_model_opt is not None:
                sam_mask = core.make_sam_mask(
                    sam_model_opt, segs, image, "center-1", 0,
                    0.93, 0, 0.7, "False"
                )
                segs = core.segs_bitwise_and_mask(segs, sam_mask)

            # 7. Unload detection model before sampling
            comfy.model_management.unload_all_models()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 8. Pre-scale optimization
            if upscale_full_image:
                current_pixels = original_h * original_w
                scale_factor = math.sqrt(target_pixels / current_pixels)

                if scale_factor > 1.0:
                    new_w = int(round(original_w * scale_factor / 8) * 8)
                    new_h = int(round(original_h * scale_factor / 8) * 8)
                    scale_factor = new_w / original_w  # Recalc after rounding

                    # Upscale full image
                    working_image = impact_utils.tensor_resize(image, new_w, new_h)

                    # Scale SEGS coordinates
                    segs = scale_coords(segs, scale_factor)

                    logging.info(f"[DonutUniversalDetailer] Pre-scaled {original_w}x{original_h} -> {new_w}x{new_h}")
                else:
                    working_image = image.clone()
                    scale_factor = 1.0
            else:
                working_image = image.clone()
                scale_factor = 1.0

            # 9. Process each segment
            enhanced_img = working_image.clone()
            cropped_enhanced = []
            detected_labels_list = []

            for i, seg in enumerate(segs[1]):
                crop_region = seg.crop_region
                bbox = seg.bbox
                label = seg.label
                detected_labels_list.append(label)

                # Get conditioning for this object
                seg_positive, seg_negative = self.get_object_conditioning(
                    clip, positive, negative, label, prompt_mode
                )

                # Crop from working image
                cropped_image = impact_utils.crop_image(working_image, crop_region)

                # Get mask
                if noise_mask and seg.cropped_mask is not None:
                    cropped_mask = seg.cropped_mask
                    if isinstance(cropped_mask, np.ndarray):
                        cropped_mask = torch.from_numpy(cropped_mask)

                    # Apply feathering
                    paste_mask = impact_utils.to_tensor(seg.cropped_mask)
                    paste_mask = impact_utils.tensor_gaussian_blur_mask(paste_mask, noise_mask_feather)
                    noise_mask_tensor = paste_mask.squeeze(3) if paste_mask.dim() == 4 else paste_mask
                else:
                    noise_mask_tensor = None
                    paste_mask = None

                # Encode to latent
                latent_image = impact_utils.to_latent_image(cropped_image, vae)

                # Add noise mask if present
                if noise_mask_tensor is not None:
                    latent_image['noise_mask'] = noise_mask_tensor.reshape(
                        (-1, 1, noise_mask_tensor.shape[-2], noise_mask_tensor.shape[-1])
                    )

                # Apply differential diffusion if feathering
                working_model = model
                if noise_mask_feather > 0 and noise_mask_tensor is not None:
                    if 'denoise_mask_function' not in model.model_options:
                        working_model = nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]

                # Sample
                refined_latent = impact_sampling.ksampler_wrapper(
                    working_model, seed + i, steps, cfg, sampler_name, scheduler,
                    seg_positive, seg_negative, latent_image, denoise
                )

                # Decode
                enhanced_cropped = vae.decode(refined_latent['samples'])

                # Paste back
                if paste_mask is None:
                    # Create simple rectangular mask
                    paste_mask = impact_utils.to_tensor(seg.cropped_mask)
                    paste_mask = impact_utils.tensor_gaussian_blur_mask(paste_mask, feather)

                enhanced_img = enhanced_img.cpu()
                enhanced_cropped = enhanced_cropped.cpu()
                impact_utils.tensor_paste(
                    enhanced_img, enhanced_cropped,
                    (crop_region[0], crop_region[1]), paste_mask
                )

                cropped_enhanced.append(enhanced_cropped)

            # 10. Generate combined mask
            mask = core.segs_to_combined_mask(segs)

            # 11. Optionally return to original size
            if return_original_size and scale_factor > 1.0:
                enhanced_img = impact_utils.tensor_resize(enhanced_img, original_w, original_h)
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(original_h, original_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

            # Ensure we have at least one crop for output
            if len(cropped_enhanced) == 0:
                cropped_enhanced = [impact_utils.empty_pil_tensor()]

            # Join detected labels
            detected_labels_str = ", ".join(detected_labels_list)

            return (enhanced_img, cropped_enhanced, mask, detected_labels_str)


    NODE_CLASS_MAPPINGS = {
        "DonutUniversalDetailer": DonutUniversalDetailer,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "DonutUniversalDetailer": "Universal Detailer (Florence-2)",
    }

else:
    # Dependencies not available
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

    if not IMPACT_AVAILABLE:
        logging.warning("[DonutUniversalDetailer] Impact Pack required but not found")
    if not FLORENCE2_AVAILABLE:
        logging.warning("[DonutUniversalDetailer] comfyui_layerstyle (Florence-2) required but not found")

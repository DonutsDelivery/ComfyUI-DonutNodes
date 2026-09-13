"""Edit Studio masks in reference-A coordinates and pixel-preserving compositing."""

import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch
import torch.nn.functional as F


def rasterize_mask(data, image_name, source_size, box, target_size, feather=8):
    """Replay normalized brush strokes, then apply the same crop as reference A."""
    try:
        document = json.loads(data)
        if not isinstance(document, dict):
            raise ValueError("The saved inpaint mask is invalid. Paint it again on image A.")
        if document.get("version") != 1 or document.get("image") != image_name:
            raise ValueError("Paint a mask for the current image A.")
        strokes = document["strokes"]
        if not isinstance(strokes, list) or len(strokes) > 10000:
            raise ValueError("Invalid inpaint mask strokes.")
        width, height = source_size
        mask = Image.new("L", source_size, 0)
        draw = ImageDraw.Draw(mask)
        for stroke in strokes:
            diameter = float(stroke["size"]) * min(source_size)
            if not math.isfinite(diameter) or not 0 < diameter <= min(source_size):
                raise ValueError("Invalid inpaint brush size.")
            points = stroke["points"]
            if not points or len(points) > 100000:
                raise ValueError("Invalid inpaint stroke points.")
            pixels = []
            for x, y in points:
                if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("Invalid inpaint mask coordinates.")
                pixels.append((x * (width - 1), y * (height - 1)))
            color = 0 if stroke.get("erase", False) else 255
            if stroke.get("shape") == "rectangle":
                if len(pixels) != 2:
                    raise ValueError("A rectangle selection requires two corners.")
                (x1, y1), (x2, y2) = pixels
                draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=color)
                continue
            brush = max(1, round(diameter))
            if len(pixels) > 1:
                draw.line(pixels, fill=color, width=brush, joint="curve")
            radius = brush / 2
            for x, y in pixels:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if document.get("inverted", False) is True:
            mask = ImageOps.invert(mask)
        mask = mask.crop(box).resize(target_size, Image.Resampling.BILINEAR)
        if mask.getbbox() is None:
            raise ValueError("Paint the area to edit on image A, or turn selected-area editing off.")
        # Feather inward: pixels outside the painted selection remain untouched.
        hard = np.asarray(mask).astype(np.float32) / 255
        if feather:
            soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(float(feather)))).astype(np.float32) / 255
            hard = np.minimum(hard, soft)
        return torch.from_numpy(hard.copy()).unsqueeze(0)
    except (KeyError, TypeError, json.JSONDecodeError, OverflowError) as error:
        raise ValueError("The saved inpaint mask is invalid. Paint it again on image A.") from error


def masked_edit_target(target, source_latent, inpaint):
    """Seed preserved regions with the encoded base; ComfyUI masks each noise step."""
    source = source_latent[0] if isinstance(source_latent, (list, tuple)) else source_latent
    samples = source["samples"]
    batch = target["samples"].shape[0]
    # Qwen Image/Wan VAEs return B,C,T,H,W even for one still image.
    # The Krea2 edit sampler uses B,C,H,W; never flatten actual video frames
    # into the target batch or mistake them for additional references.
    if samples.ndim == 5 and samples.shape[2] == 1:
        samples = samples.squeeze(2)
    if samples.ndim != 4 or samples.shape[0] != 1:
        raise ValueError(
            "Inpainting requires one encoded base image shaped B,C,H,W or "
            f"B,C,1,H,W; received {tuple(source['samples'].shape)}."
        )
    mask = inpaint["mask"]
    if mask.ndim != 3 or mask.shape[0] != 1 or not torch.isfinite(mask).all():
        raise ValueError("Inpainting requires a finite single-image mask.")
    if not torch.any(mask > 0):
        raise ValueError("Paint an area to edit before running inpainting.")
    result = target.copy()
    result["samples"] = samples.repeat(batch, 1, 1, 1)
    result["noise_mask"] = mask.clamp(0, 1).repeat(batch, 1, 1)
    return result


def composite_inpaint(image, inpaint):
    if inpaint is None:
        return image
    base = inpaint["image"].to(device=image.device, dtype=image.dtype)
    mask = inpaint["mask"].to(device=image.device, dtype=image.dtype)
    size = image.shape[1:3]
    if base.shape[1:3] != size:
        base = F.interpolate(base.movedim(-1, 1), size=size, mode="bicubic", align_corners=False).movedim(1, -1).clamp(0, 1)
    if mask.shape[1:] != size:
        mask = F.interpolate(mask.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)
    mask = mask.clamp(0, 1).unsqueeze(-1)
    return image * mask + base * (1 - mask)


class DonutInpaintComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "optional": {"inpaint": ("DONUT_INPAINT",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "donut/editing"
    DESCRIPTION = "Keep image A outside the Edit Studio selection, including after upscale and Face Detailer. Passes through when selected-area editing is off."

    def composite(self, image, inpaint=None):
        return (composite_inpaint(image, inpaint),)


NODE_CLASS_MAPPINGS = {"DonutInpaintComposite": DonutInpaintComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutInpaintComposite": "Donut Inpaint · Keep Surroundings"}

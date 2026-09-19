"""Source-coordinate crops and aspect-preserving fit; no ComfyUI dependency.

This is image preparation, not a new reference-token grid or encoder resolution.
All transforms are deterministic and use the same half-up rounding as the UI.
"""
from dataclasses import dataclass
import json
import math

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

LEGACY = "Legacy output-linked"
INDEPENDENT = "Independent crops"
FIT_KEY = "donut_reference_fit"
ASPECTS = {"1:1": (1, 1), "2:3": (2, 3), "3:2": (3, 2), "3:4": (3, 4),
           "4:3": (4, 3), "9:16": (9, 16), "16:9": (16, 9), "21:9": (21, 9)}


def half_up(value):
    return math.floor(value + 0.5)


def valid_size(size):
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in size)):
        raise ValueError("Image size must contain two positive integer dimensions.")
    return tuple(size)


def crop_pixels(data, image_name, source_size):
    """Empty means full image; stale/malformed saved crops fail explicitly."""
    width, height = valid_size(source_size)
    if not data:
        return (0, 0, width, height)
    try:
        doc = json.loads(data) if isinstance(data, str) else data
        if not isinstance(doc, dict) or isinstance(doc.get("version"), bool) or doc.get("version") != 1:
            raise ValueError("Invalid saved crop version.")
        if (doc.get("image") != image_name or doc.get("source_size") != [width, height]
                or any(isinstance(v, bool) for v in doc["source_size"])):
            raise ValueError("This crop belongs to a different image. Reset or select the crop again.")
        if doc.get("aspect", "Free") not in ("Original", "Free", *ASPECTS):
            raise ValueError("Unknown crop aspect ratio.")
        bounds = doc.get("bounds")
        if (not isinstance(bounds, list) or len(bounds) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or not 0 <= v <= 1 for v in bounds)):
            raise ValueError("Crop coordinates must be finite normalized numbers.")
        x1, y1, x2, y2 = bounds
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Crop must have positive width and height.")
        box = (half_up(x1 * width), half_up(y1 * height),
               half_up(x2 * width), half_up(y2 * height))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Crop is smaller than one source pixel.")
        return box
    except (TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("Invalid saved crop. Reset or select the crop again.") from error


@dataclass(frozen=True)
class Fit:
    canvas: tuple
    content: tuple
    offset: tuple


def fit_geometry(source_size, canvas_size):
    sw, sh = valid_size(source_size)
    cw, ch = valid_size(canvas_size)
    scale = min(cw / sw, ch / sh)
    width, height = min(cw, max(1, half_up(sw * scale))), min(ch, max(1, half_up(sh * scale)))
    return Fit((cw, ch), (width, height), ((cw - width) // 2, (ch - height) // 2))


def crop_fit_image(source, box, canvas_size, background=0.5):
    cropped = source.convert("RGB").crop(box)
    fit = fit_geometry(cropped.size, canvas_size)
    fill = half_up(255 * float(background))
    if not 0 <= fill <= 255:
        raise ValueError("Reference background must be between zero and one.")
    canvas = Image.new("RGB", fit.canvas, (fill,) * 3)
    if cropped.size != fit.content:
        cropped = cropped.resize(fit.content, Image.Resampling.LANCZOS)
    canvas.paste(cropped, fit.offset)
    return pil_tensor(canvas), fit


def pil_tensor(image):
    return torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).unsqueeze(0)


def fit_tensor(image, width, height, background=0.5):
    """Pixel-space fit before VAE encoding. Keeps the target latent contract."""
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("Reference fit requires a BHWC RGB image tensor.")
    fit = fit_geometry((int(image.shape[2]), int(image.shape[1])), (int(width), int(height)))
    if fit.canvas == fit.content == (image.shape[2], image.shape[1]):
        return image[..., :3]
    rgb = image[..., :3].movedim(-1, 1)
    resized = F.interpolate(rgb.float(), size=fit.content[::-1], mode="bilinear", align_corners=False)
    left, top = fit.offset
    right = fit.canvas[0] - left - fit.content[0]
    bottom = fit.canvas[1] - top - fit.content[1]
    return F.pad(resized, (left, right, top, bottom), value=float(background)).to(image.dtype).movedim(1, -1)


def fit_mask(mask, fit, feather=0):
    """Pad a cropped/resized A mask identically to A; feather in output pixels."""
    if (mask.ndim != 3 or mask.shape[0] != 1 or tuple(mask.shape[1:]) != fit.content[::-1]
            or not torch.isfinite(mask).all()):
        raise ValueError("Inpaint mask must match A's fitted content dimensions.")
    if isinstance(feather, bool) or not isinstance(feather, (int, float)) or not math.isfinite(feather) or not 0 <= feather <= 128:
        raise ValueError("Invalid inpaint edge softness.")
    width, height = fit.canvas
    left, top = fit.offset
    result = torch.zeros((1, height, width), dtype=mask.dtype, device=mask.device)
    result[:, top:top + fit.content[1], left:left + fit.content[0]] = mask.clamp(0, 1)
    if feather:
        hard = result[0].detach().float().cpu().numpy()
        matte = Image.fromarray((hard * 255).round().astype(np.uint8))
        soft = np.asarray(matte.filter(ImageFilter.GaussianBlur(float(feather)))).astype(np.float32) / 255
        result = torch.from_numpy(np.minimum(hard, soft).copy()).unsqueeze(0).to(device=mask.device, dtype=mask.dtype)
    return result


def output_dimensions(values, source_a_size=None, box_a=None, source_b_size=None, box_b=None):
    """Return None for legacy/off; caller then uses the exact existing recipe."""
    if values.get("geometry_mode", LEGACY) == LEGACY or not values.get("enabled", False):
        return None
    if values.get("geometry_mode") != INDEPENDENT:
        raise ValueError("Unknown reference geometry mode.")
    multiple = int(values.get("multiple", 64))
    if multiple not in (16, 32, 64):
        raise ValueError("Unsupported output pixel grid.")
    follow = values.get("output_canvas", "Follow A crop")
    if follow not in ("Follow A crop", "Independent output"):
        raise ValueError("Unknown output canvas mode.")
    if not source_a_size or not box_a:
        raise ValueError("Reference A is required for editing.")
    a = (box_a[2] - box_a[0], box_a[3] - box_a[1])
    b = (box_b[2] - box_b[0], box_b[3] - box_b[1]) if box_b else source_b_size
    if follow == "Follow A crop":
        if values.get("resolution_mode") == "Reference A · crop only":
            return tuple(max(multiple, n // multiple * multiple) for n in a)
        ratio = a
    elif values.get("resolution_mode") == "Custom":
        if any(not math.isfinite(float(values[name])) or not 16 <= float(values[name]) <= 16384 for name in ("width", "height")):
            raise ValueError("Invalid custom output dimensions.")
        return tuple(max(multiple, half_up(float(values[name]) / multiple) * multiple) for name in ("width", "height"))
    else:
        aspect = values.get("aspect_ratio", "4:3 Standard")
        if aspect == "Auto · Reference A":
            ratio = a
        elif aspect == "Auto · Reference B":
            ratio = b or (4, 3)
        else:
            key = str(aspect).split(" ", 1)[0]
            if key not in ASPECTS:
                raise ValueError("Unknown output aspect ratio.")
            ratio = ASPECTS[key]
    budget = float(values.get("megapixels", 1.0))
    if not math.isfinite(budget) or not 0.1 <= budget <= 16:
        raise ValueError("Output megapixels must be between 0.1 and 16.")
    scale = math.sqrt(budget * 1024 * 1024 / (ratio[0] * ratio[1]))
    result = tuple(max(multiple, half_up(edge * scale / multiple) * multiple) for edge in ratio)
    if max(result) > 16384:
        raise ValueError("The crop is too narrow for this pixel budget. Use a smaller budget or wider crop.")
    return result

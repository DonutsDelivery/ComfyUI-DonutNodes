"""Shared bridge to the optional ComfyUI-Krea2Edit node pack."""

import math

import nodes
import torch.nn.functional as F


_REQUIRED_NODE_IDS = ("Krea2EditModelPatch", "Krea2EditGroundedEncode")


def scale_image_to_megapixels(image, megapixels=1.0):
    """Scale NHWC images to a pixel budget without snapping either dimension."""
    height, width = int(image.shape[1]), int(image.shape[2])
    target_pixels = float(megapixels) * 1024 * 1024
    scale = math.sqrt(target_pixels / (width * height))
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    if (target_width, target_height) == (width, height):
        return image

    samples = image[..., :3].movedim(-1, 1)
    resized = F.interpolate(
        samples.float(), size=(target_height, target_width), mode="area",
    )
    return resized.to(image.dtype).movedim(1, -1)


def resize_center_crop(image, width, height):
    """Center-crop an NHWC image to target aspect, then resize exactly."""
    target_width, target_height = int(width), int(height)
    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height

    if source_ratio > target_ratio:
        crop_width = max(1, round(source_height * target_ratio))
        x1 = (source_width - crop_width) // 2
        cropped = image[:, :, x1:x1 + crop_width, :3]
    elif source_ratio < target_ratio:
        crop_height = max(1, round(source_width / target_ratio))
        y1 = (source_height - crop_height) // 2
        cropped = image[:, y1:y1 + crop_height, :, :3]
    else:
        cropped = image[..., :3]

    if tuple(cropped.shape[1:3]) == (target_height, target_width):
        return cropped
    resized = F.interpolate(
        cropped.movedim(-1, 1).float(),
        size=(target_height, target_width), mode="area",
    )
    return resized.to(image.dtype).movedim(1, -1)


def solve_multiple_target(width, height, megapixels=1.0, multiple=32,
                          area_tolerance=0.05):
    """Find the grid-aligned size with best aspect ratio near a pixel budget."""
    width, height, multiple = int(width), int(height), int(multiple)
    if width <= 0 or height <= 0 or multiple <= 0:
        raise ValueError("width, height, and multiple must be positive")

    target_pixels = float(megapixels) * 1024 * 1024
    source_ratio = width / height
    max_dimension = max(multiple, math.ceil(
        math.sqrt(target_pixels / min(source_ratio, 1 / source_ratio))
        * (1 + area_tolerance) / multiple,
    ) * multiple)

    candidates = []
    for target_width in range(multiple, max_dimension + multiple, multiple):
        ideal_height = target_pixels / target_width
        center = max(1, round(ideal_height / multiple))
        for height_units in range(max(1, center - 2), center + 3):
            target_height = height_units * multiple
            area_error = abs(target_width * target_height / target_pixels - 1)
            if area_error <= area_tolerance:
                aspect_error = abs(math.log(
                    (target_width / target_height) / source_ratio,
                ))
                candidates.append((aspect_error, area_error,
                                   target_width, target_height))

    if not candidates:
        raise ValueError("no grid-aligned target found within area tolerance")
    _, _, target_width, target_height = min(candidates)
    return target_width, target_height


def pad_image_to_multiple(image, multiple=32):
    """Symmetrically pad an NHWC image upward to a pixel-grid multiple."""
    height, width = int(image.shape[1]), int(image.shape[2])
    target_width = math.ceil(width / multiple) * multiple
    target_height = math.ceil(height / multiple) * multiple
    pad_x = target_width - width
    pad_y = target_height - height
    padding = (pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2)
    left, top, right, bottom = padding
    if not any(padding):
        return image, padding
    samples = F.pad(
        image.movedim(-1, 1), (left, right, top, bottom), mode="replicate",
    )
    return samples.movedim(1, -1), padding


def crop_image_padding(image, padding):
    """Remove left/top/right/bottom padding from an NHWC image."""
    left, top, right, bottom = padding
    height_end = image.shape[1] - bottom if bottom else image.shape[1]
    width_end = image.shape[2] - right if right else image.shape[2]
    return image[:, top:height_end, left:width_end, :]


def resize_reference_to_canvas(image, width, height):
    """Resize an NHWC reference batch to the exact generated-image canvas."""
    samples = image[..., :3].movedim(-1, 1)
    resized = F.interpolate(
        samples.float(), size=(int(height), int(width)),
        mode="bilinear", align_corners=False,
    )
    return resized.to(image.dtype).movedim(1, -1)


def crop_aligned_reference(image, canvas_width, canvas_height, region, batch_index=0):
    """Resize a source reference to the target canvas, then crop exact XYXY coordinates."""
    canvas = resize_reference_to_canvas(image, canvas_width, canvas_height)
    index = min(max(int(batch_index), 0), canvas.shape[0] - 1)
    x1, y1, x2, y2 = (int(round(value)) for value in region)
    x1 = min(max(x1, 0), int(canvas_width) - 1)
    y1 = min(max(y1, 0), int(canvas_height) - 1)
    x2 = min(max(x2, x1 + 1), int(canvas_width))
    y2 = min(max(y2, y1 + 1), int(canvas_height))
    return canvas[index:index + 1, y1:y2, x1:x2, :]


def patch_krea2_edit_model(model, source_latent):
    """Apply the installed Krea2 edit source-token patch to one model."""
    if "Krea2EditModelPatch" not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError(
            "Krea2 edit mode requires comfyui-krea2edit. Missing node: Krea2EditModelPatch"
        )
    return nodes.NODE_CLASS_MAPPINGS["Krea2EditModelPatch"]().patch(
        model, source_latent,
    )[0]


def make_krea2_edit_target(source_latent, width=None, height=None):
    """Create an empty target, optionally at independent pixel dimensions."""
    target = source_latent.copy()
    samples = source_latent["samples"]
    shape = samples.shape
    if width is not None and height is not None:
        shape = (shape[0], shape[1], int(height) // 8, int(width) // 8)
        target["downscale_ratio_spacial"] = 8
    target["samples"] = samples.new_zeros(shape)
    target.pop("noise_mask", None)
    return target


def prepare_krea2_edit(model, clip, vae, source_image, positive_prompt="",
                       negative_prompt="", grounding_px=768,
                       target_width=None, target_height=None):
    """Build target-matched VAE source tokens and natural 1-MiP grounding.

    The implementation is deliberately resolved through ComfyUI's node registry so
    Donut nodes use the installed ComfyUI-Krea2Edit implementation rather than
    carrying a fork that can drift from the Identity Edit LoRA's inference path.
    """
    if clip is None:
        raise ValueError("Krea2 edit mode requires a CLIP input")
    if vae is None:
        raise ValueError("Krea2 edit mode requires a VAE input")
    if source_image is None:
        raise ValueError("Krea2 edit mode requires a source image")

    missing = [node_id for node_id in _REQUIRED_NODE_IDS
               if node_id not in nodes.NODE_CLASS_MAPPINGS]
    if missing:
        raise RuntimeError(
            "Krea2 edit mode requires comfyui-krea2edit. Missing node(s): "
            + ", ".join(missing)
        )

    conditioning_image = scale_image_to_megapixels(source_image)
    patch_image = conditioning_image
    if target_width is not None and target_height is not None:
        patch_image = resize_center_crop(
            source_image, target_width, target_height,
        )
    source_latent = nodes.VAEEncode().encode(vae, patch_image)[0]

    grounded_node = nodes.NODE_CLASS_MAPPINGS["Krea2EditGroundedEncode"]()

    patched_model = patch_krea2_edit_model(model, source_latent)
    positive = grounded_node.encode(
        clip, positive_prompt, image=conditioning_image, grounding_px=grounding_px,
    )[0]
    negative = grounded_node.encode(
        clip, negative_prompt, image=conditioning_image, grounding_px=grounding_px,
    )[0]

    return patched_model, positive, negative, source_latent, conditioning_image

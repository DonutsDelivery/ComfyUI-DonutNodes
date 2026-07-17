"""Shared bridge to the optional ComfyUI-Krea2Edit node pack.

The appearance-token forward is adapted from ComfyUI-Krea2Edit revision
17af88332728c97ab5c7d26296b2cae59c935976 under Apache-2.0 and modified by
Donut for authoritative target latents, strict batching, composable wrappers,
and training-matched source modulation. See THIRD_PARTY_NOTICES.md.
"""

import math

import nodes
import torch
import torch.nn.functional as F

import comfy.ldm.common_dit
import comfy.patcher_extension
from comfy.ldm.flux.layers import timestep_embedding


_REQUIRED_NODE_IDS = ("Krea2EditGroundedEncode",)
_EDIT_WRAPPER_KEY = "donut_krea2_edit"
_LEGACY_EDIT_WRAPPER_KEY = "krea2_edit"


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


def _imgids(batch, frame, height, width, device):
    ids = torch.zeros(height, width, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, height * width, 3).repeat(batch, 1, 1)


def _to_4d(value):
    if value.ndim == 5:
        batch, channels, frames, height, width = value.shape
        return value.reshape(batch * frames, channels, height, width)
    return value


def _patchify(value, patch):
    batch, channels, height, width = value.shape
    patch_height, patch_width = height // patch, width // patch
    return value.reshape(
        batch, channels, patch_height, patch, patch_width, patch,
    ).permute(0, 2, 4, 1, 3, 5).reshape(
        batch, patch_height * patch_width, channels * patch * patch,
    )


def _match_source_batch(source, runtime_batch, target_batch):
    if target_batch < 1 or runtime_batch % target_batch != 0:
        raise ValueError(
            f"Krea2 edit runtime batch {runtime_batch} is not a multiple of target batch {target_batch}."
        )
    if source.shape[0] != 1:
        raise ValueError(
            f"Krea2 edit supports one source image broadcast across the target batch, got {source.shape[0]}."
        )
    return source.expand(runtime_batch, *source.shape[1:])


def krea2_edit_forward(model, x, timesteps, context, source_latent,
                       target_batch, transformer_options=None):
    """Krea2 edit forward with upstream training-matched source modulation."""
    transformer_options = transformer_options or {}
    patch = model.patch
    temporal = x.ndim == 5
    if temporal:
        batch_5d, _channels_5d, frames_5d, height_5d, width_5d = x.shape
    x = _to_4d(x)
    batch, _channels, original_height, original_width = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    height, width = x.shape[-2:]
    patch_height, patch_width = height // patch, width // patch

    source = _to_4d(source_latent).to(device=x.device, dtype=x.dtype)
    if source.shape[-2:] != (original_height, original_width):
        source = F.interpolate(
            source.float(), size=(original_height, original_width), mode="bilinear",
        ).to(x.dtype)
    source = comfy.ldm.common_dit.pad_to_patch_size(source, (patch, patch))

    context = model._unpack_context(context)
    if context.shape[0] != batch:
        raise ValueError(
            f"Krea2 edit conditioning batch must match runtime batch {batch}, got {context.shape[0]}."
        )
    target_image = model.first(_patchify(x, patch))
    source_image = model.first(_patchify(source, patch))
    source_image = _match_source_batch(source_image, batch, target_batch)

    active_t = model.tmlp(
        timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(target_image.dtype)
    )
    active_vec = model.tproj(active_t)

    context = model.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = model.txtmlp(context)
    text_length = context.shape[1]
    source_length = source_image.shape[1]
    target_length = target_image.shape[1]
    combined = torch.cat((context, source_image, target_image), dim=1)

    device = combined.device
    positions = torch.cat((
        torch.zeros(batch, text_length, 3, device=device, dtype=torch.float32),
        _imgids(batch, 1, patch_height, patch_width, device),
        _imgids(batch, 0, patch_height, patch_width, device),
    ), dim=1)
    freqs = model.pe_embedder(positions)

    for block in model.blocks:
        combined = block(
            combined, active_vec, freqs, None,
            transformer_options=transformer_options,
        )

    target_start = text_length + source_length
    target_tokens = combined[:, target_start:target_start + target_length]
    final = model.last(target_tokens, active_t)
    output = final.reshape(
        batch, patch_height, patch_width, model.channels, patch, patch,
    ).permute(0, 3, 1, 4, 2, 5).reshape(
        batch, model.channels, patch_height * patch, patch_width * patch,
    )
    output = output[:, :, :original_height, :original_width]
    if temporal:
        output = output.reshape(
            batch_5d, frames_5d, model.channels, height_5d, width_5d,
        ).movedim(1, 2)
    return output


def _is_krea2_model(model):
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    required = (
        "patch", "channels", "tdim", "first", "tmlp", "tproj",
        "txtfusion", "txtmlp", "blocks", "last", "pe_embedder",
        "_unpack_context",
    )
    return diffusion_model is not None and all(
        hasattr(diffusion_model, name) for name in required
    )


class _Krea2EditWrapper:
    def __init__(self, source_samples, target_batch):
        self.source_samples = source_samples
        self.target_batch = target_batch

    def to(self, device_or_dtype):
        return type(self)(self.source_samples.to(device_or_dtype), self.target_batch)

    def __call__(self, executor, x, timesteps, context, attention_mask=None,
                 transformer_options=None, **kwargs):
        diffusion_model = executor.class_obj

        def local_forward(x, timesteps, context, attention_mask=None,
                          transformer_options=None, **kwargs):
            return krea2_edit_forward(
                diffusion_model, x, timesteps, context,
                self.source_samples, self.target_batch, transformer_options,
            )

        # ComfyUI has no public replace-terminal API. Continue at the next wrapper
        # with Donut's edit forward as the terminal so wrappers on both sides run.
        remaining = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            local_forward,
            diffusion_model,
            executor.wrappers,
            idx=executor.idx + 1,
        )
        return remaining.execute(
            x, timesteps, context, attention_mask,
            transformer_options or {}, **kwargs,
        )


def _remove_edit_wrapper(model, key):
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    if hasattr(model, "remove_wrappers_with_key"):
        model.remove_wrappers_with_key(wrapper_type, key)
    transformer_options = model.model_options.setdefault("transformer_options", {})
    wrappers = transformer_options.get("wrappers", {})
    wrappers.get(wrapper_type, {}).pop(key, None)


def patch_krea2_edit_model(model, source_latent, target_batch=None):
    """Apply Donut's clean-reference Krea2 edit patch to one model clone."""
    if not _is_krea2_model(model):
        raise RuntimeError("Krea2 edit mode requires a Krea 2 diffusion model.")
    source_samples = model.model.process_latent_in(source_latent["samples"])
    source_batch = _to_4d(source_samples).shape[0]
    if target_batch is None:
        target_batch = 1
    target_batch = int(target_batch)
    if source_batch != 1:
        raise ValueError(
            f"Krea2 edit supports one source image broadcast across the target batch, got {source_batch}."
        )

    patched = model.clone()
    _remove_edit_wrapper(patched, _EDIT_WRAPPER_KEY)
    _remove_edit_wrapper(patched, _LEGACY_EDIT_WRAPPER_KEY)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        _EDIT_WRAPPER_KEY,
        _Krea2EditWrapper(source_samples, target_batch),
    )
    return patched


def make_krea2_edit_target(target_latent):
    """Return a zeroed edit target while preserving caller latent metadata."""
    target = target_latent.copy()
    target["samples"] = torch.zeros_like(target_latent["samples"])
    return target


def validate_krea2_edit_target(target_latent, source_image, mode, denoise,
                               add_noise, start_at_step):
    """Validate empty-target edit semantics and return samples plus pixel geometry."""
    if source_image is None:
        raise ValueError("DonutSampler edit_mode requires source_image.")
    if not isinstance(target_latent, dict) or "samples" not in target_latent:
        raise ValueError("DonutSampler edit_mode requires a valid latent_image target.")
    samples = target_latent["samples"]
    if not torch.is_tensor(samples) or samples.ndim != 4:
        raise ValueError("DonutSampler edit_mode requires 4D latent samples shaped B,C,H,W.")
    if samples.shape[0] < 1:
        raise ValueError("DonutSampler edit_mode target batch must be at least 1.")
    if (not torch.is_tensor(source_image) or source_image.ndim != 4
            or source_image.shape[0] != 1):
        raise ValueError(
            "DonutSampler edit_mode supports one source image broadcast across the target batch."
        )
    if not math.isclose(float(denoise), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("DonutSampler edit_mode requires denoise=1.0 for its empty target.")
    if mode in ("advanced", "multi_model"):
        if add_noise != "enable":
            raise ValueError("DonutSampler edit_mode requires add_noise=enable.")
        if start_at_step != 0:
            raise ValueError("DonutSampler edit_mode requires start_at_step=0.")

    noise_mask = target_latent.get("noise_mask")
    if noise_mask is not None:
        if not torch.is_tensor(noise_mask) or not bool(torch.all(noise_mask == 1)):
            raise ValueError(
                "DonutSampler edit_mode does not support partial noise_mask values; "
                "use an all-ones mask or remove it."
            )

    if target_latent.get("downscale_ratio_temporal") is not None:
        raise ValueError("DonutSampler edit_mode does not support temporal target latents.")
    downscale_ratio_value = target_latent.get("downscale_ratio_spacial")
    downscale_ratio = 8.0 if downscale_ratio_value is None else float(downscale_ratio_value)
    if not math.isfinite(downscale_ratio) or downscale_ratio <= 0:
        raise ValueError("DonutSampler edit_mode requires a positive finite latent downscale ratio.")
    target_width = round(int(samples.shape[-1]) * downscale_ratio)
    target_height = round(int(samples.shape[-2]) * downscale_ratio)
    if target_width * target_height > 2 * 1024 * 1024:
        raise ValueError(
            "DonutSampler edit_mode supports targets up to 2 MiP; upscale larger outputs afterward."
        )
    return samples, target_width, target_height


def prepare_krea2_edit(model, clip, vae, source_image, positive_prompt="",
                       negative_prompt="", grounding_px=768,
                       target_width=None, target_height=None, target_batch=None):
    """Build target-matched VAE source tokens and natural 1-MiP grounding.

    Grounded semantic conditioning remains delegated to the installed
    ComfyUI-Krea2Edit node. Donut owns the appearance-token model patch so its
    target, batch, and clean-reference semantics stay consistent.
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

    patched_model = patch_krea2_edit_model(
        model, source_latent, target_batch=target_batch,
    )
    positive = grounded_node.encode(
        clip, positive_prompt, image=conditioning_image, grounding_px=grounding_px,
    )[0]
    negative = grounded_node.encode(
        clip, negative_prompt, image=conditioning_image, grounding_px=grounding_px,
    )[0]

    return patched_model, positive, negative, source_latent, conditioning_image

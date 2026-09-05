"""Color-preservation extension for the existing DonutTiledUpscale node.

This module patches DonutTiledUpscale in-place. It intentionally registers no
additional ComfyUI node, so existing workflows keep using the same upscaler.
"""

from copy import deepcopy
import torch

from .DonutTiledUpscale import DonutTiledUpscale


def _match_color_stats(output, reference, strength=1.0, eps=1e-6):
    """AdaIN-style RGB mean/std transfer from reference to output."""
    strength = float(strength)
    if strength <= 0.0:
        return output

    ref = reference.to(device=output.device, dtype=output.dtype)
    matched = []
    for i in range(output.shape[0]):
        out_i = output[i]
        ref_i = ref[min(i, ref.shape[0] - 1)]

        out_mean = out_i.mean(dim=(0, 1), keepdim=True)
        out_std = out_i.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(eps)
        ref_mean = ref_i.mean(dim=(0, 1), keepdim=True)
        ref_std = ref_i.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(eps)

        transferred = (out_i - out_mean) / out_std * ref_std + ref_mean
        matched.append(out_i.lerp(transferred, strength).clamp(0.0, 1.0))

    return torch.stack(matched, dim=0)


def _patch_existing_node():
    if getattr(DonutTiledUpscale, "_color_preserve_patched", False):
        return

    original_input_types = DonutTiledUpscale.INPUT_TYPES.__func__
    original_upscale = DonutTiledUpscale.upscale

    @classmethod
    def input_types(cls):
        inputs = deepcopy(original_input_types(cls))
        optional = inputs.setdefault("optional", {})
        optional["color_reference"] = (
            "IMAGE",
            {"tooltip": "Optional color reference. Defaults to the input image."},
        )
        optional["color_preserve_strength"] = (
            "FLOAT",
            {
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "tooltip": "0 disables preservation; 1 fully matches output RGB mean/std to the reference after sampling and decode.",
            },
        )
        return inputs

    def upscale(self, image, upscale_model, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                rescale_factor, resampling_method, feather, tiled_vae,
                edit_mode=False, clip=None,
                edit_prompt="Enhance fine details while preserving the source image.",
                edit_negative_prompt="", grounding_px=768, edit_model=None,
                edit_source_image=None, turbo_mode=False, tiled_diffusion=True,
                color_reference=None, color_preserve_strength=0.0):
        output, debug = original_upscale(
            self, image, upscale_model, model, positive, negative, vae, seed,
            steps, cfg, sampler_name, scheduler, denoise,
            rescale_factor, resampling_method, feather, tiled_vae,
            edit_mode=edit_mode, clip=clip, edit_prompt=edit_prompt,
            edit_negative_prompt=edit_negative_prompt, grounding_px=grounding_px,
            edit_model=edit_model, edit_source_image=edit_source_image,
            turbo_mode=turbo_mode, tiled_diffusion=tiled_diffusion,
        )

        if color_preserve_strength > 0.0:
            reference = image if color_reference is None else color_reference
            output = _match_color_stats(output, reference, color_preserve_strength)

        return output, debug

    DonutTiledUpscale.INPUT_TYPES = input_types
    DonutTiledUpscale.upscale = upscale
    DonutTiledUpscale._color_preserve_patched = True


_patch_existing_node()

# Kept for the existing __init__.py import contract, but this module does not
# register a separate node.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

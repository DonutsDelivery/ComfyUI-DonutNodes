from copy import deepcopy
import torch
from .DonutTiledUpscale import DonutTiledUpscale


def match_color_stats(output, reference, strength=1.0, eps=1e-6):
    if strength <= 0.0:
        return output
    ref = reference.to(device=output.device, dtype=output.dtype)
    result = []
    for i in range(output.shape[0]):
        out_i = output[i]
        ref_i = ref[min(i, ref.shape[0] - 1)]
        out_mean = out_i.mean(dim=(0, 1), keepdim=True)
        out_std = out_i.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(eps)
        ref_mean = ref_i.mean(dim=(0, 1), keepdim=True)
        ref_std = ref_i.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(eps)
        transferred = (out_i - out_mean) / out_std * ref_std + ref_mean
        result.append(out_i.lerp(transferred, float(strength)).clamp(0.0, 1.0))
    return torch.stack(result, dim=0)


class DonutColorPreservingUpscale(DonutTiledUpscale):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(super().INPUT_TYPES())
        optional = inputs.setdefault("optional", {})
        optional["color_reference"] = ("IMAGE",)
        optional["color_preserve_strength"] = (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
        )
        return inputs

    FUNCTION = "upscale_color_preserve"
    CATEGORY = "donut/upscale"

    def upscale_color_preserve(self, image, upscale_model, model, positive, negative, vae, seed,
                               steps, cfg, sampler_name, scheduler, denoise,
                               rescale_factor, resampling_method, feather, tiled_vae,
                               edit_mode=False, clip=None,
                               edit_prompt="Enhance fine details while preserving the source image.",
                               edit_negative_prompt="", grounding_px=768, edit_model=None,
                               edit_source_image=None, turbo_mode=False, tiled_diffusion=True,
                               color_reference=None, color_preserve_strength=1.0):
        output, debug = super().upscale(
            image, upscale_model, model, positive, negative, vae, seed,
            steps, cfg, sampler_name, scheduler, denoise,
            rescale_factor, resampling_method, feather, tiled_vae,
            edit_mode=edit_mode, clip=clip, edit_prompt=edit_prompt,
            edit_negative_prompt=edit_negative_prompt, grounding_px=grounding_px,
            edit_model=edit_model, edit_source_image=edit_source_image,
            turbo_mode=turbo_mode, tiled_diffusion=tiled_diffusion,
        )
        reference = image if color_reference is None else color_reference
        return match_color_stats(output, reference, color_preserve_strength), debug


NODE_CLASS_MAPPINGS = {"DonutColorPreservingUpscale": DonutColorPreservingUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutColorPreservingUpscale": "Donut Tiled Upscale + Color Preserve"}

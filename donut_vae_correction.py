"""One-pass VAE damage subtraction shared by decoding and finishing stages."""
from copy import deepcopy

import torch
from torch.nn import functional as F
from nodes import VAEDecode
try:
    from . import donut_vae_upscale
except ImportError:
    import donut_vae_upscale


def vae_damage_input_types():
    return {
        "vae_damage_correction": ("BOOLEAN", {
            "default": False,
            "tooltip": "Subtract estimated VAE damage after decoding, using the selected VAE's encoder and decoder for one extra round trip per image or face crop. The 2x VAE filters back to the current image size before subtraction. No original reference is needed.",
        }),
        "vae_damage_strength": ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
            "tooltip": "0 skips correction; 1 is one-pass VAE damage subtraction. Values above 1 strengthen the same correction and can amplify artifacts. Does not add more iterations.",
        }),
    }


def subtract_vae_damage(image, vae, strength):
    """One RGB correction: clip(y + strength * (y - decode(encode(y))))."""
    if strength <= 0:
        return image
    vae = donut_vae_upscale.prepare_vae(vae)
    height, width = image.shape[1:3]
    grid = vae.spacial_compression_encode()
    pad_h, pad_w = (-height) % grid, (-width) % grid
    corrected = []
    # Video-capable VAEs can interpret an IMAGE batch as a sequence. Give each
    # still image its own round trip with the selected VAE.
    for sample in image.split(1):
        rgb = sample.float()
        pixels = rgb
        if pad_h or pad_w:
            mode = "reflect" if height > pad_h and width > pad_w else "replicate"
            pixels = F.pad(rgb.movedim(-1, 1), (0, pad_w, 0, pad_h), mode=mode).movedim(1, -1)
        decoded = vae.decode(vae.encode(pixels))
        if decoded.ndim == 5 and decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        if decoded.shape != pixels.shape:
            raise ValueError("VAE damage correction requires a VAE that reconstructs one RGB image at the same size.")
        decoded = decoded[:, :height, :width].to(device=rgb.device, dtype=torch.float32)
        corrected.append(rgb.add(rgb - decoded, alpha=strength).clamp(0.0, 1.0))
    return torch.cat(corrected, dim=0)


class DonutVAEDecode(VAEDecode):
    @classmethod
    def INPUT_TYPES(cls):
        result = deepcopy(super().INPUT_TYPES())
        result.setdefault("optional", {}).update(vae_damage_input_types())
        return result

    CATEGORY = "donut/image"
    DESCRIPTION = "Decode latents at the configured image size using the selected VAE, with optional VAE damage subtraction. The Wan2.1/Qwen 2x VAE filters its internal 2x output back to this size."

    def decode(self, vae, samples, vae_damage_correction=False, vae_damage_strength=1.0):
        vae = donut_vae_upscale.prepare_vae(vae)
        image, = super().decode(vae, samples)
        if vae_damage_correction and vae_damage_strength > 0:
            image = subtract_vae_damage(image, vae, float(vae_damage_strength))
        return (image,)


NODE_CLASS_MAPPINGS = {"DonutVAEDecode": DonutVAEDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutVAEDecode": "Donut VAE Decode"}

"""
DonutGammaCorrection - Gamma correction node using Gwenview's algorithm.
Gamma range is -100 to +100, where positive values brighten shadows
and negative values darken them.

Optional dithering to reduce banding artifacts in gradients.
"""

import torch
import math


class DonutGammaCorrection:
    """
    Apply gamma correction to images.
    Uses the same algorithm as Gwenview/localbooru:
        exponent = 3^(-gamma/100)
        output = input^exponent

    Optional dithering adds subtle noise to prevent banding/posterization
    artifacts that occur when gamma stretches limited 8-bit precision.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Gamma adjustment (-100 to +100). Positive brightens shadows, negative darkens them."
                }),
            },
            "optional": {
                "dither": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Dither amount to reduce banding (0 = off, 0.5 = subtle, 1.0 = 1 LSB worth of noise)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_gamma"
    CATEGORY = "donut/image"

    def apply_gamma(self, image, gamma, dither=0.0):
        if gamma == 0 and dither == 0:
            return (image,)

        # Calculate exponent using Gwenview's formula
        exponent = math.pow(3.0, -gamma / 100.0) if gamma != 0 else 1.0

        # Clamp to valid range before applying gamma
        img = torch.clamp(image, 0, 1)

        # Apply gamma correction
        if gamma != 0:
            img = torch.pow(img, exponent)

        # Apply dithering to reduce banding
        if dither > 0:
            # Triangular dither (TPDF) - better than uniform for audio/video
            # Two uniform randoms subtracted give triangular distribution
            noise1 = torch.rand_like(img)
            noise2 = torch.rand_like(img)
            # Scale to 1 LSB at 8-bit (1/255 ≈ 0.004)
            # dither=1.0 means ±0.5 LSB triangular noise
            noise_scale = dither * (1.0 / 255.0)
            noise = (noise1 - noise2) * noise_scale
            img = img + noise
            img = torch.clamp(img, 0, 1)

        return (img,)


NODE_CLASS_MAPPINGS = {
    "DonutGammaCorrection": DonutGammaCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutGammaCorrection": "Donut Gamma Correction",
}

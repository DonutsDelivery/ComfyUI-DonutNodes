"""
DonutHistogramStretch - Histogram stretching (contrast stretch / auto levels).

Remaps pixel values so that the darkest become 0 and brightest become 1,
maximizing the dynamic range of the image.

Two modes via percentile settings:
- Simple (0/100): Uses absolute min/max - sensitive to outliers
- Robust (e.g., 1/99): Uses percentiles - ignores outlier pixels

Three channel modes:
- global: Bounds from luminance, same stretch to all RGB (preserves color ratios)
- per_channel: Each RGB channel stretched independently (may shift colors)
- luminance: LAB L-channel only (perfectly preserves hue/saturation)

Pairs well with gamma correction:
1. Histogram stretch (maximize dynamic range)
2. Gamma correction (adjust midtone placement)
"""

import torch
import numpy as np
from PIL import Image


def tensor2pil(tensor):
    """Convert tensor [B, H, W, C] or [H, W, C] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0], mode='L')
    elif arr.shape[-1] == 3:
        return Image.fromarray(arr, mode='RGB')
    else:
        return Image.fromarray(arr, mode='RGBA')


def pil2tensor(pil_img):
    """Convert PIL Image to tensor [H, W, C]."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return torch.from_numpy(arr)


class DonutHistogramStretch:
    """
    Histogram stretching to maximize dynamic range.

    Maps the black_point percentile to 0 and white_point percentile to 1.
    - Set both to 0/100 for simple min/max stretching (matches LayerStyle)
    - Set to 1/99 or similar for robust stretching that ignores outliers
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "black_point": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 49.0,
                    "step": 0.5,
                    "tooltip": "Percentile for black point (0 = absolute min, 1+ = robust)"
                }),
                "white_point": ("FLOAT", {
                    "default": 100.0,
                    "min": 51.0,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": "Percentile for white point (100 = absolute max, 99- = robust)"
                }),
                "mode": (["global", "per_channel", "luminance"], {
                    "default": "luminance",
                    "tooltip": "global: bounds from luminance, same stretch to RGB (preserves ratios). per_channel: stretch RGB independently (may shift colors). luminance: LAB L-only (preserves hue/saturation, matches LayerStyle)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stretch_histogram"
    CATEGORY = "donut/image"

    def _get_bounds(self, values, black_point, white_point):
        """Get low/high bounds from values using percentiles."""
        if black_point == 0.0:
            low = values.min()
        else:
            low = torch.quantile(values, black_point / 100.0)

        if white_point == 100.0:
            high = values.max()
        else:
            high = torch.quantile(values, white_point / 100.0)

        return low, high

    def stretch_histogram(self, image, black_point, white_point, mode):
        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = image[b]  # [H, W, C]

            if mode == "per_channel":
                # Stretch each RGB channel independently
                img_out = img.clone()
                for c in range(min(img.shape[-1], 3)):  # Only RGB, not alpha
                    channel = img[..., c]
                    low, high = self._get_bounds(channel, black_point, white_point)

                    if high - low > 1e-8:
                        img_out[..., c] = (channel - low) / (high - low)

                img_out = torch.clamp(img_out, 0, 1)

            elif mode == "luminance":
                # LAB L-channel only stretch (matches LayerStyle behavior)
                # Convert to PIL for LAB conversion
                pil_img = tensor2pil(img)
                original_mode = pil_img.mode

                # Convert to LAB
                lab_img = pil_img.convert('LAB')
                l_channel, a_channel, b_channel = lab_img.split()

                # Get L channel as tensor for percentile calculation
                l_arr = np.array(l_channel).astype(np.float32) / 255.0
                l_tensor = torch.from_numpy(l_arr)

                low, high = self._get_bounds(l_tensor.flatten(), black_point, white_point)

                # Stretch L channel
                if high - low > 1e-8:
                    l_stretched = (l_tensor - low) / (high - low)
                    l_stretched = torch.clamp(l_stretched, 0, 1)
                    l_arr_new = (l_stretched.numpy() * 255).astype(np.uint8)
                    l_channel = Image.fromarray(l_arr_new, mode='L')

                # Merge back
                lab_stretched = Image.merge('LAB', (l_channel, a_channel, b_channel))
                rgb_result = lab_stretched.convert('RGB')

                # Handle alpha if present
                if original_mode == 'RGBA':
                    rgb_result = rgb_result.convert('RGBA')
                    rgb_result.putalpha(pil_img.split()[-1])

                img_out = pil2tensor(rgb_result)

                # Ensure same device as input
                img_out = img_out.to(img.device)

            else:  # global
                # Global stretch - compute bounds from luminance, apply same stretch to all RGB
                # This preserves color ratios while actually doing something useful
                rgb = img[..., :3]
                luminance = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
                low, high = self._get_bounds(luminance.flatten(), black_point, white_point)

                if high - low > 1e-8:
                    img_out = img.clone()
                    # Apply same linear transform to all RGB channels
                    img_out[..., :3] = (rgb - low) / (high - low)
                else:
                    img_out = img.clone()

                img_out = torch.clamp(img_out, 0, 1)

            results.append(img_out)

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "DonutHistogramStretch": DonutHistogramStretch,
}

# Hidden from the menu: folded into the unified "Donut Image Adjust" node
# (DonutImageAdjust). Kept in NODE_CLASS_MAPPINGS so saved workflows still load.
NODE_DISPLAY_NAME_MAPPINGS = {}

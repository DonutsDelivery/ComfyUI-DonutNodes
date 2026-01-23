"""
DonutAutoWhiteBalance - Automatic white balance / color cast removal.

Industry-standard algorithms for neutralizing color casts:
- gray_world: Assumes average color should be gray (fast, classic)
- white_patch: Assumes brightest pixels should be white
- combined: Gray World + White Patch together
- photoshop: Per-channel levels + neutral midtone snap (like Photoshop's Auto Color)

Apply BEFORE histogram stretch and gamma correction for best results.
"""

import torch


class DonutAutoWhiteBalance:
    """
    Automatic white balance using classic color constancy algorithms.

    Standard processing order:
    1. Auto White Balance (this node) - remove color casts
    2. Histogram Stretch - maximize dynamic range
    3. Gamma Correction - adjust midtone brightness
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["gray_world", "white_patch", "combined", "auto_levels", "photoshop"], {
                    "default": "gray_world",
                    "tooltip": "gray_world: shift means to gray. white_patch: scale by brightest. combined: both. auto_levels: per-channel stretch (Auto Tone). photoshop: per-channel + midtone snap (Auto Color)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Correction strength (1.0 = full correction)"
                }),
                "clip_percent": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Percent of pixels to clip from brightest/darkest (Photoshop default: 0.1%)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_white_balance"
    CATEGORY = "donut/image"

    def _gray_world(self, img, clip_percent):
        """
        Gray World algorithm: assumes average color should be neutral gray.
        Shifts each channel's mean to match the overall mean.
        """
        rgb = img[..., :3]

        # Compute channel means (optionally excluding extremes)
        if clip_percent > 0:
            # Flatten and sort each channel to exclude outliers
            means = []
            for c in range(3):
                channel = rgb[..., c].flatten()
                n = len(channel)
                clip_n = int(n * clip_percent / 100)
                if clip_n > 0:
                    sorted_vals, _ = torch.sort(channel)
                    trimmed = sorted_vals[clip_n:-clip_n] if clip_n * 2 < n else sorted_vals
                    means.append(trimmed.mean())
                else:
                    means.append(channel.mean())
            r_mean, g_mean, b_mean = means
        else:
            r_mean = rgb[..., 0].mean()
            g_mean = rgb[..., 1].mean()
            b_mean = rgb[..., 2].mean()

        # Target: overall mean (gray)
        overall_mean = (r_mean + g_mean + b_mean) / 3.0

        # Compute scale factors
        r_scale = overall_mean / (r_mean + 1e-8)
        g_scale = overall_mean / (g_mean + 1e-8)
        b_scale = overall_mean / (b_mean + 1e-8)

        # Apply correction
        img_out = img.clone()
        img_out[..., 0] = img[..., 0] * r_scale
        img_out[..., 1] = img[..., 1] * g_scale
        img_out[..., 2] = img[..., 2] * b_scale

        return img_out

    def _white_patch(self, img, clip_percent):
        """
        White Patch (Perfect Reflector) algorithm: assumes brightest pixels should be white.
        Scales each channel so its maximum becomes 1.0.
        """
        rgb = img[..., :3]

        # Find max values (optionally using percentile to exclude outliers)
        if clip_percent > 0:
            percentile = (100.0 - clip_percent) / 100.0
            r_max = torch.quantile(rgb[..., 0], percentile)
            g_max = torch.quantile(rgb[..., 1], percentile)
            b_max = torch.quantile(rgb[..., 2], percentile)
        else:
            r_max = rgb[..., 0].max()
            g_max = rgb[..., 1].max()
            b_max = rgb[..., 2].max()

        # Scale factors to make max = 1.0
        r_scale = 1.0 / (r_max + 1e-8)
        g_scale = 1.0 / (g_max + 1e-8)
        b_scale = 1.0 / (b_max + 1e-8)

        # Apply correction
        img_out = img.clone()
        img_out[..., 0] = img[..., 0] * r_scale
        img_out[..., 1] = img[..., 1] * g_scale
        img_out[..., 2] = img[..., 2] * b_scale

        return img_out

    def _combined(self, img, clip_percent):
        """
        Combined Gray World + White Patch.
        First applies gray world, then white patch.
        """
        img_out = self._gray_world(img, clip_percent)
        img_out = self._white_patch(img_out, clip_percent)
        return img_out

    def _auto_levels(self, img, clip_percent):
        """
        Auto Levels / Auto Tone (Photoshop's "Enhance Per Channel Contrast").

        Stretches each R, G, B channel independently to full range.
        This is the per-channel levels step WITHOUT neutral midtone snapping.
        May introduce color shifts but maximizes per-channel contrast.
        """
        rgb = img[..., :3]
        img_out = img.clone()

        clip_low = clip_percent / 100.0
        clip_high = (100.0 - clip_percent) / 100.0

        for c in range(3):
            channel = rgb[..., c]
            if clip_percent > 0:
                low = torch.quantile(channel, clip_low)
                high = torch.quantile(channel, clip_high)
            else:
                low = channel.min()
                high = channel.max()

            if high - low > 1e-8:
                img_out[..., c] = (channel - low) / (high - low)

        return img_out

    def _photoshop_style(self, img, clip_percent):
        """
        Photoshop Auto Color algorithm (Find Dark & Light Colors + Snap Neutral Midtones).

        Based on Adobe's documentation and reverse engineering:
        1. Per-channel black/white point clipping (default 0.1% each end)
        2. Stretch each R, G, B channel independently to full range
        3. Find near-neutral midtone pixels and adjust per-channel gamma
           to force them to RGB 128 (0.5 normalized) - true neutral gray

        References:
        - https://helpx.adobe.com/photoshop/using/making-quick-tonal-adjustments.html
        - https://geraldbakker.nl/psnumbers/auto-options.html
        """
        rgb = img[..., :3]
        img_out = img.clone()

        # Step 1: Per-channel levels (Find Dark & Light Colors)
        # Clip the specified percentage from each end of each channel's histogram
        clip_low = clip_percent / 100.0
        clip_high = (100.0 - clip_percent) / 100.0

        for c in range(3):
            channel = rgb[..., c]
            if clip_percent > 0:
                low = torch.quantile(channel, clip_low)
                high = torch.quantile(channel, clip_high)
            else:
                low = channel.min()
                high = channel.max()

            if high - low > 1e-8:
                img_out[..., c] = (channel - low) / (high - low)

        # Step 2: Snap Neutral Midtones
        # Find pixels that are nearly neutral and in midtone range
        rgb_balanced = img_out[..., :3]
        r, g, b = rgb_balanced[..., 0], rgb_balanced[..., 1], rgb_balanced[..., 2]

        # Calculate per-pixel "grayness" (how close to neutral)
        max_rgb = torch.maximum(torch.maximum(r, g), b)
        min_rgb = torch.minimum(torch.minimum(r, g), b)
        # Saturation-like measure: difference between max and min channel
        chroma = max_rgb - min_rgb

        # Mean brightness
        mean_rgb = (r + g + b) / 3.0

        # Find midtone neutrals:
        # - Midtone range: 0.15 to 0.85 (avoid shadows and highlights)
        # - Low chroma: < 0.1 (nearly neutral, R ≈ G ≈ B)
        midtone_mask = (mean_rgb > 0.15) & (mean_rgb < 0.85)
        neutral_mask = (chroma < 0.1) & midtone_mask

        if neutral_mask.any() and neutral_mask.sum() > 100:  # Need enough neutral pixels
            # Get average color of neutral pixels
            neutral_r = r[neutral_mask].mean()
            neutral_g = g[neutral_mask].mean()
            neutral_b = b[neutral_mask].mean()

            # Photoshop targets RGB 128 (0.5 in normalized space)
            # Adjust per-channel gamma so neutral pixels map to 0.5
            # Formula: new = old^gamma, we want neutral^gamma = 0.5
            # So: gamma = log(0.5) / log(neutral)
            target = 0.5

            if neutral_r > 0.02 and neutral_r < 0.98:
                gamma_r = torch.log(torch.tensor(target)) / torch.log(neutral_r + 1e-8)
                gamma_r = torch.clamp(gamma_r, 0.5, 2.0)
                img_out[..., 0] = torch.pow(img_out[..., 0].clamp(1e-8, 1), gamma_r)

            if neutral_g > 0.02 and neutral_g < 0.98:
                gamma_g = torch.log(torch.tensor(target)) / torch.log(neutral_g + 1e-8)
                gamma_g = torch.clamp(gamma_g, 0.5, 2.0)
                img_out[..., 1] = torch.pow(img_out[..., 1].clamp(1e-8, 1), gamma_g)

            if neutral_b > 0.02 and neutral_b < 0.98:
                gamma_b = torch.log(torch.tensor(target)) / torch.log(neutral_b + 1e-8)
                gamma_b = torch.clamp(gamma_b, 0.5, 2.0)
                img_out[..., 2] = torch.pow(img_out[..., 2].clamp(1e-8, 1), gamma_b)

        return img_out

    def apply_white_balance(self, image, method, strength, clip_percent):
        if strength == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = image[b]  # [H, W, C]

            # Apply selected method
            if method == "gray_world":
                img_corrected = self._gray_world(img, clip_percent)
            elif method == "white_patch":
                img_corrected = self._white_patch(img, clip_percent)
            elif method == "combined":
                img_corrected = self._combined(img, clip_percent)
            elif method == "auto_levels":
                img_corrected = self._auto_levels(img, clip_percent)
            else:  # photoshop
                img_corrected = self._photoshop_style(img, clip_percent)

            # Apply strength (blend with original)
            if strength < 1.0:
                img_out = img * (1 - strength) + img_corrected * strength
            elif strength > 1.0:
                # Overshoot
                diff = img_corrected - img
                img_out = img + diff * strength
            else:
                img_out = img_corrected

            # Clamp and preserve alpha if present
            img_out = torch.clamp(img_out, 0, 1)
            if img.shape[-1] == 4:
                img_out[..., 3] = img[..., 3]  # Preserve alpha

            results.append(img_out)

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "DonutAutoWhiteBalance": DonutAutoWhiteBalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutAutoWhiteBalance": "Donut Auto White Balance",
}

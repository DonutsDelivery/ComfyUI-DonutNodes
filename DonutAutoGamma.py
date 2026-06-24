"""
DonutAutoGamma - Automatic gamma correction nodes using various algorithms.

Main node: DonutAutoGamma - unified node with method selector
Legacy nodes kept for backwards compatibility.
"""

import torch
import math


class DonutAutoGamma:
    """
    Unified automatic gamma correction with multiple detection methods.

    All methods analyze the image to compute how much gamma correction is needed,
    then apply it using simple torch.pow() for clean color handling.

    Methods:
    - Simple Mean: Shifts mean brightness to target (default 0.5)
    - IAGCWD: Uses deviation from ideal mean (0.439) with quadratic scaling
    - GSLF: Histogram-based detection with trained polynomial (most sophisticated)
    - Percentile: Shifts a specific percentile to target value

    Based on:
    - Simple Mean: Babakhani & Zarei (2015)
    - IAGCWD: Cao et al. (2018)
    - GSLF: Rahman et al. (2021) - GLAGC paper
    """

    # IAGCWD constants
    IDEAL_MEAN = 0.439
    DIM_THRESHOLD = 0.307
    BRIGHT_THRESHOLD = 0.571

    # GSLF polynomial coefficients
    GSLF_COEFFS = (8.224, -5.534, 1.093)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["simple_mean", "iagcwd", "gslf", "percentile"], {
                    "default": "gslf",
                    "tooltip": "Detection method: simple_mean (fastest), iagcwd (deviation-based), gslf (histogram-based, most sophisticated), percentile (robust to outliers)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Correction strength (1.0 = full correction)"
                }),
            },
            "optional": {
                "target": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Target value for simple_mean and percentile methods"
                }),
                "percentile": ("FLOAT", {
                    "default": 50.0,
                    "min": 1.0,
                    "max": 99.0,
                    "step": 5.0,
                    "tooltip": "Which percentile to target (percentile method only)"
                }),
                "direction": (["both", "brighten_only", "darken_only"], {
                    "default": "both",
                    "tooltip": "both: auto brighten/darken, brighten_only: only brighten dark images, darken_only: only darken bright images"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "gamma")
    FUNCTION = "apply_auto_gamma"
    CATEGORY = "donut/image"

    def _compute_luminance(self, img):
        """Compute perceptual luminance from RGB."""
        if img.shape[-1] >= 3:
            return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
        return img[..., 0]

    def _method_simple_mean(self, luminance, target):
        """Compute gamma to shift mean to target."""
        mean_lum = luminance.mean().item()
        if mean_lum <= 0.01 or mean_lum >= 0.99:
            return 1.0
        return math.log(target) / math.log(mean_lum)

    def _method_iagcwd(self, luminance):
        """Compute gamma based on deviation from ideal mean with quadratic scaling."""
        mean_lum = luminance.mean().item()
        deviation = mean_lum - self.IDEAL_MEAN

        if deviation < 0:
            # Too dark - need gamma < 1 to brighten
            threshold_distance = self.IDEAL_MEAN - self.DIM_THRESHOLD
            normalized_dev = abs(deviation) / threshold_distance
            intensity = normalized_dev ** 2
            # Map to gamma: at full intensity, gamma ≈ 0.5 (significant brightening)
            gamma = 1.0 - intensity * 0.5
        else:
            # Too bright - need gamma > 1 to darken
            threshold_distance = self.BRIGHT_THRESHOLD - self.IDEAL_MEAN
            normalized_dev = abs(deviation) / threshold_distance
            intensity = normalized_dev ** 2
            # Map to gamma: at full intensity, gamma ≈ 2.0 (significant darkening)
            gamma = 1.0 + intensity * 1.0

        return max(0.2, min(3.0, gamma))

    def _method_gslf(self, luminance):
        """Compute gamma using histogram-based GSLF with polynomial."""
        values = luminance.flatten()
        values = torch.clamp(values, 0, 1)

        # Compute CDF
        hist = torch.histc(values, bins=256, min=0, max=1)
        cdf = torch.cumsum(hist, dim=0)
        cdf = cdf / (cdf[-1] + 1e-8)

        # Ideal CDF (uniform)
        ideal_cdf = torch.linspace(0, 1, 256, device=luminance.device)

        # GSLF = deviation from ideal
        cdf_safe = torch.clamp(cdf, min=1e-8)
        gslf = torch.sum(torch.abs(ideal_cdf - cdf) / cdf_safe) / 256
        gslf = torch.clamp(gslf, 0, 2).item()

        # Polynomial regression
        a, b, c = self.GSLF_COEFFS
        gamma = a * gslf**2 + b * gslf + c
        return max(0.2, min(3.0, gamma))

    def _method_percentile(self, luminance, percentile, target):
        """Compute gamma to shift a percentile to target."""
        percentile_val = torch.quantile(luminance, percentile / 100.0).item()
        if percentile_val <= 0.01 or percentile_val >= 0.99:
            return 1.0
        return math.log(target) / math.log(percentile_val)

    def apply_auto_gamma(self, image, method, strength, target=0.5, percentile=50.0, direction="both"):
        if strength == 0:
            return (image, 1.0)

        batch_size = image.shape[0]
        results = []
        gammas = []

        for b in range(batch_size):
            img = image[b]
            luminance = self._compute_luminance(img)

            # Compute gamma based on method
            if method == "simple_mean":
                gamma = self._method_simple_mean(luminance, target)
            elif method == "iagcwd":
                gamma = self._method_iagcwd(luminance)
            elif method == "gslf":
                gamma = self._method_gslf(luminance)
            elif method == "percentile":
                gamma = self._method_percentile(luminance, percentile, target)
            else:
                gamma = 1.0

            # Apply strength
            gamma = 1.0 + (gamma - 1.0) * strength

            # Apply direction constraint
            if direction == "brighten_only" and gamma > 1.0:
                gamma = 1.0  # Don't darken
            elif direction == "darken_only" and gamma < 1.0:
                gamma = 1.0  # Don't brighten

            gammas.append(gamma)

            if abs(gamma - 1.0) < 0.001:
                results.append(img)
                continue

            # Apply gamma
            img_out = torch.clamp(img, 0, 1)
            img_out = torch.pow(img_out, gamma)
            img_out = torch.clamp(img_out, 0, 1)
            results.append(img_out)

        avg_gamma = sum(gammas) / len(gammas)
        return (torch.stack(results), avg_gamma)


NODE_CLASS_MAPPINGS = {
    "DonutAutoGamma": DonutAutoGamma,
}

# Hidden from the menu: folded into the unified "Donut Image Adjust" node
# (DonutImageAdjust). Kept in NODE_CLASS_MAPPINGS so saved workflows still load.
NODE_DISPLAY_NAME_MAPPINGS = {}

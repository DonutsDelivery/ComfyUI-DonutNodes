"""
DonutSharpen - Industry standard sharpening methods.

Six methods implemented:
1. Unsharp Mask (USM) - Classic: original + amount * (original - blur)
2. High Pass - Isolates edges via frequency separation, blend overlay
3. Smart Sharpen - USM with edge detection to avoid sharpening noise
4. Deconvolution - Richardson-Lucy to actually recover detail
5. HiRaLoAm - High Radius Low Amount USM for local contrast
6. CAS - AMD Contrast Adaptive Sharpening

Apply LAST in the correction chain (after white balance, levels, gamma).
"""

import torch
import torch.nn.functional as F
import math


def gaussian_kernel_2d(radius, sigma=None, device='cpu'):
    """Create a 2D Gaussian kernel."""
    if sigma is None:
        sigma = radius / 3.0

    size = int(radius * 2) + 1
    x = torch.arange(size, device=device, dtype=torch.float32) - radius
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    kernel = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    return kernel


def apply_gaussian_blur(img, radius, device='cpu'):
    """Apply Gaussian blur to image tensor [H, W, C]."""
    if radius <= 0:
        return img

    kernel = gaussian_kernel_2d(radius, device=device)
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2

    # Reshape for conv2d: [H, W, C] -> [1, C, H, W]
    img_t = img.permute(2, 0, 1).unsqueeze(0)

    # Apply blur per channel
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

    blurred_channels = []
    for c in range(img_t.shape[1]):
        channel = img_t[:, c:c+1, :, :]
        blurred = F.conv2d(channel, kernel, padding=padding)
        blurred_channels.append(blurred)

    blurred = torch.cat(blurred_channels, dim=1)

    # Reshape back: [1, C, H, W] -> [H, W, C]
    return blurred.squeeze(0).permute(1, 2, 0)


class DonutUnsharpMask:
    """
    Unsharp Mask (USM) - The classic sharpening algorithm.

    Formula: sharpened = original + amount * (original - blur(original))

    The "unsharp" name comes from the darkroom technique of combining
    the original with a blurred (unsharp) negative to enhance edges.

    Parameters:
    - amount: Strength of sharpening (1.0 = 100%)
    - radius: Blur radius in pixels (controls edge width)
    - threshold: Minimum edge strength to sharpen (reduces noise amplification)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Sharpening strength (1.0 = 100%)"
                }),
                "radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Blur radius in pixels (controls edge width)"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Minimum edge strength to sharpen (0 = sharpen all)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_usm"
    CATEGORY = "donut/image/sharpen"

    def apply_usm(self, image, amount, radius, threshold):
        if amount == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = image[b]
            device = img.device

            # Create blurred version
            blurred = apply_gaussian_blur(img, radius, device)

            # Calculate edge mask (difference from blur)
            edge = img - blurred

            # Apply threshold if set
            if threshold > 0:
                edge_strength = edge.abs()
                mask = (edge_strength > threshold).float()
                # Smooth transition near threshold
                mask = torch.where(
                    edge_strength > threshold,
                    torch.ones_like(mask),
                    edge_strength / (threshold + 1e-8)
                )
                edge = edge * mask

            # Apply sharpening
            sharpened = img + amount * edge
            sharpened = torch.clamp(sharpened, 0, 1)

            results.append(sharpened)

        return (torch.stack(results),)


class DonutHighPassSharpen:
    """
    High Pass Sharpening - Frequency separation technique.

    Isolates high-frequency detail (edges) by subtracting a blurred
    version, then blends back using overlay mode for sharpening.

    Popular in portrait retouching for selective detail enhancement.
    Can also be used with soft light for gentler effect.

    Parameters:
    - radius: Blur radius (larger = broader edges, smaller = fine detail)
    - strength: Blend strength
    - blend_mode: overlay (punchy) or soft_light (subtle)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.5,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "High pass radius (larger = broader edges)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Blend strength"
                }),
                "blend_mode": (["overlay", "soft_light", "hard_light", "linear_light"], {
                    "default": "overlay",
                    "tooltip": "Blend mode: overlay (punchy), soft_light (subtle)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_high_pass"
    CATEGORY = "donut/image/sharpen"

    def _overlay_blend(self, base, blend):
        """Overlay blend mode."""
        return torch.where(
            base < 0.5,
            2 * base * blend,
            1 - 2 * (1 - base) * (1 - blend)
        )

    def _soft_light_blend(self, base, blend):
        """Soft light blend mode (Photoshop formula)."""
        return torch.where(
            blend < 0.5,
            base - (1 - 2 * blend) * base * (1 - base),
            base + (2 * blend - 1) * (torch.sqrt(base) - base)
        )

    def _hard_light_blend(self, base, blend):
        """Hard light blend mode (overlay with swapped inputs)."""
        return self._overlay_blend(blend, base)

    def _linear_light_blend(self, base, blend):
        """Linear light blend mode."""
        return base + 2 * blend - 1

    def apply_high_pass(self, image, radius, strength, blend_mode):
        if strength == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = image[b]
            device = img.device

            # Create high pass filter: original - blur, centered at 0.5
            blurred = apply_gaussian_blur(img, radius, device)
            high_pass = (img - blurred) + 0.5
            high_pass = torch.clamp(high_pass, 0, 1)

            # Blend using selected mode
            if blend_mode == "overlay":
                blended = self._overlay_blend(img, high_pass)
            elif blend_mode == "soft_light":
                blended = self._soft_light_blend(img, high_pass)
            elif blend_mode == "hard_light":
                blended = self._hard_light_blend(img, high_pass)
            else:  # linear_light
                blended = self._linear_light_blend(img, high_pass)

            # Apply strength
            result = img * (1 - strength) + blended * strength
            result = torch.clamp(result, 0, 1)

            results.append(result)

        return (torch.stack(results),)


class DonutSmartSharpen:
    """
    Smart Sharpen - USM with edge-aware masking.

    Improves on basic USM by detecting actual edges and avoiding
    sharpening of noise or flat areas. Uses Sobel edge detection
    to create a mask.

    Parameters:
    - amount: Sharpening strength
    - radius: Blur radius for USM
    - edge_threshold: How strong an edge must be to sharpen
    - reduce_noise: Strength of noise reduction in flat areas
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Sharpening strength"
                }),
                "radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Blur radius for USM"
                }),
                "edge_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Minimum edge strength to sharpen (higher = less sharpening)"
                }),
                "reduce_noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Blur flat areas to reduce noise (0 = off)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_smart_sharpen"
    CATEGORY = "donut/image/sharpen"

    def _detect_edges(self, img, device):
        """Detect edges using Sobel operator on luminance."""
        # Convert to luminance
        if img.shape[-1] >= 3:
            lum = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
        else:
            lum = img[..., 0]

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # Apply Sobel
        lum_t = lum.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        gx = F.conv2d(lum_t, sobel_x, padding=1)
        gy = F.conv2d(lum_t, sobel_y, padding=1)

        # Gradient magnitude
        edges = torch.sqrt(gx**2 + gy**2).squeeze()

        # Normalize to 0-1
        edges = edges / (edges.max() + 1e-8)

        return edges

    def apply_smart_sharpen(self, image, amount, radius, edge_threshold, reduce_noise):
        if amount == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = image[b]
            device = img.device

            # Detect edges
            edges = self._detect_edges(img, device)

            # Create edge mask with soft threshold
            edge_mask = torch.clamp((edges - edge_threshold * 0.5) / (edge_threshold + 1e-8), 0, 1)
            edge_mask = edge_mask.unsqueeze(-1)  # [H, W, 1]

            # USM sharpening
            blurred = apply_gaussian_blur(img, radius, device)
            edge_detail = img - blurred
            sharpened = img + amount * edge_detail * edge_mask

            # Optional noise reduction in flat areas
            if reduce_noise > 0:
                flat_mask = 1 - edge_mask
                noise_reduced = apply_gaussian_blur(sharpened, 0.5, device)
                sharpened = sharpened * (1 - flat_mask * reduce_noise) + noise_reduced * flat_mask * reduce_noise

            sharpened = torch.clamp(sharpened, 0, 1)
            results.append(sharpened)

        return (torch.stack(results),)


class DonutDeconvolutionSharpen:
    """
    Richardson-Lucy Deconvolution - Actually recovers lost detail.

    Unlike USM which just enhances edges, deconvolution attempts to
    reverse the blurring process and recover the original image.
    Used in astronomy, microscopy, and tools like Topaz Sharpen AI.

    This is iterative and slower but produces cleaner results with
    less haloing than USM for recovering genuinely blurred images.

    Parameters:
    - iterations: More = sharper but slower (10-30 typical)
    - psf_radius: Assumed blur radius to deconvolve
    - damping: Prevents noise amplification (higher = smoother)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "iterations": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Deconvolution iterations (more = sharper, slower)"
                }),
                "psf_radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Assumed blur radius (PSF size)"
                }),
                "damping": ("FLOAT", {
                    "default": 0.001,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Noise damping (higher = smoother)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_deconvolution"
    CATEGORY = "donut/image/sharpen"

    def apply_deconvolution(self, image, iterations, psf_radius, damping):
        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = image[b]
            device = img.device

            # Create Gaussian PSF (point spread function)
            psf = gaussian_kernel_2d(psf_radius, device=device)
            psf = psf / psf.sum()

            kernel_size = psf.shape[0]
            padding = kernel_size // 2

            # Prepare for conv2d
            psf_kernel = psf.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
            psf_flipped = torch.flip(psf_kernel, [2, 3])  # Flipped for correlation

            # Richardson-Lucy iteration per channel
            img_t = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            estimate = img_t.clone()

            for _ in range(iterations):
                # Convolve estimate with PSF
                convolved_channels = []
                for c in range(estimate.shape[1]):
                    channel = estimate[:, c:c+1, :, :]
                    conv = F.conv2d(channel, psf_kernel, padding=padding)
                    convolved_channels.append(conv)
                convolved = torch.cat(convolved_channels, dim=1)

                # Avoid division by zero
                convolved = torch.clamp(convolved, min=damping)

                # Ratio
                ratio = img_t / convolved

                # Correlate ratio with flipped PSF
                correlated_channels = []
                for c in range(ratio.shape[1]):
                    channel = ratio[:, c:c+1, :, :]
                    corr = F.conv2d(channel, psf_flipped, padding=padding)
                    correlated_channels.append(corr)
                correlated = torch.cat(correlated_channels, dim=1)

                # Update estimate
                estimate = estimate * correlated
                estimate = torch.clamp(estimate, 0, 1)

            # Convert back
            result = estimate.squeeze(0).permute(1, 2, 0)
            results.append(result)

        return (torch.stack(results),)


class DonutHiRaLoAm:
    """
    HiRaLoAm - High Radius, Low Amount sharpening.

    A USM technique using large radius (20-100px) with low amount (10-30%).
    Instead of enhancing fine detail, this enhances local contrast,
    giving images more "pop" and depth.

    Popular in landscape and portrait photography for "clarity" effect.
    Similar to Lightroom/Camera Raw Clarity slider.

    Parameters:
    - radius: Large radius for local contrast (20-100 typical)
    - amount: Keep low to avoid halos (0.1-0.3 typical)
    - threshold: Protect shadows/highlights from clipping
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {
                    "default": 50.0,
                    "min": 10.0,
                    "max": 200.0,
                    "step": 5.0,
                    "tooltip": "Large radius for local contrast enhancement"
                }),
                "amount": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Keep low to avoid halos (0.1-0.3 typical)"
                }),
                "protect_tones": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Protect shadows/highlights from clipping"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_hiraloam"
    CATEGORY = "donut/image/sharpen"

    def apply_hiraloam(self, image, radius, amount, protect_tones):
        if amount == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = image[b]
            device = img.device

            # Large radius blur
            blurred = apply_gaussian_blur(img, radius, device)

            # Local contrast enhancement (USM with large radius)
            detail = img - blurred
            enhanced = img + amount * detail

            # Protect extreme tones if enabled
            if protect_tones > 0:
                # Luminance
                if img.shape[-1] >= 3:
                    lum = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
                else:
                    lum = img[..., 0]

                # Create protection mask for shadows and highlights
                shadow_protect = torch.clamp(lum / 0.25, 0, 1)
                highlight_protect = torch.clamp((1 - lum) / 0.25, 0, 1)
                protect_mask = shadow_protect * highlight_protect
                protect_mask = protect_mask.unsqueeze(-1)

                # Blend based on protection
                protection_strength = protect_tones
                enhanced = img * (1 - protect_mask * (1 - protection_strength)) + \
                          enhanced * protect_mask * (1 - protection_strength) + \
                          enhanced * (1 - protect_mask)

            enhanced = torch.clamp(enhanced, 0, 1)
            results.append(enhanced)

        return (torch.stack(results),)


class DonutCAS:
    """
    CAS - AMD Contrast Adaptive Sharpening.

    Developed by AMD for recovering detail lost to temporal anti-aliasing (TAA)
    in games. Analyzes local contrast and adapts sharpening per-pixel:
    - Low contrast areas: Sharpen more
    - High contrast areas: Sharpen less (already sharp)

    Produces cleaner results than USM with less haloing on already-sharp edges.

    Parameters:
    - sharpness: Overall sharpening strength (0-1)
    - contrast: Local contrast adaptation strength
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpness": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Sharpening strength"
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Contrast adjustment (0 = neutral)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_cas"
    CATEGORY = "donut/image/sharpen"

    def apply_cas(self, image, sharpness, contrast):
        """
        AMD CAS algorithm adapted from FidelityFX.

        The core idea: sharpen inversely proportional to local contrast.
        This avoids over-sharpening already-sharp edges.
        """
        if sharpness == 0 and contrast == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        # CAS uses a cross-shaped 5-tap filter
        # Peak is calculated to achieve desired sharpening
        peak = -1.0 / (8.0 - 3.0 * sharpness) if sharpness < 1.0 else -0.125

        for b in range(batch_size):
            img = image[b]  # [H, W, C]
            H, W, C = img.shape
            device = img.device

            # Pad for neighborhood access
            padded = F.pad(img.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            padded = padded.squeeze(0).permute(1, 2, 0)  # [H+2, W+2, C]

            # Get neighbors (cross pattern)
            center = padded[1:-1, 1:-1, :]  # Center pixel
            north = padded[:-2, 1:-1, :]    # Top
            south = padded[2:, 1:-1, :]     # Bottom
            west = padded[1:-1, :-2, :]     # Left
            east = padded[1:-1, 2:, :]      # Right

            # Find local min/max for contrast detection
            local_min = torch.minimum(torch.minimum(torch.minimum(north, south), west), east)
            local_min = torch.minimum(local_min, center)
            local_max = torch.maximum(torch.maximum(torch.maximum(north, south), west), east)
            local_max = torch.maximum(local_max, center)

            # Adaptive weight based on local contrast
            # Low contrast = sharpen more, high contrast = sharpen less
            contrast_range = local_max - local_min

            # CAS weight calculation
            # w = saturate(min(local_min, 1-local_max) / contrast_range)
            amp = torch.minimum(local_min, 1.0 - local_max)
            amp = amp / (contrast_range + 1e-8)
            amp = torch.clamp(amp, 0, 1)

            # Calculate sharpening weight
            w = amp * peak

            # Apply 5-tap filter: center + w * (4*center - north - south - east - west)
            sum_neighbors = north + south + east + west
            sharpened = center + w * (4.0 * center - sum_neighbors)

            # Apply contrast adjustment if needed
            if contrast != 0:
                # Simple S-curve contrast
                mid = 0.5
                sharpened = mid + (sharpened - mid) * (1.0 + contrast)

            sharpened = torch.clamp(sharpened, 0, 1)
            results.append(sharpened)

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "DonutUnsharpMask": DonutUnsharpMask,
    "DonutHighPassSharpen": DonutHighPassSharpen,
    "DonutSmartSharpen": DonutSmartSharpen,
    "DonutDeconvolutionSharpen": DonutDeconvolutionSharpen,
    "DonutHiRaLoAm": DonutHiRaLoAm,
    "DonutCAS": DonutCAS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutUnsharpMask": "Donut Unsharp Mask (USM)",
    "DonutHighPassSharpen": "Donut High Pass Sharpen",
    "DonutSmartSharpen": "Donut Smart Sharpen",
    "DonutDeconvolutionSharpen": "Donut Deconvolution Sharpen",
    "DonutHiRaLoAm": "Donut HiRaLoAm (Local Contrast)",
    "DonutCAS": "Donut CAS (Contrast Adaptive)",
}

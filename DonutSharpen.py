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

    # Ensure kernel size is always odd to maintain output dimensions
    size = 2 * int(round(radius)) + 1
    center = size // 2
    x = torch.arange(size, device=device, dtype=torch.float32) - center
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


def apply_gaussian_blur_separable(img, radius, device='cpu'):
    """Apply Gaussian blur using separable 1D convolutions (faster for large radii)."""
    if radius <= 0:
        return img

    sigma = radius / 3.0
    size = 2 * int(round(radius)) + 1
    center = size // 2
    x = torch.arange(size, device=device, dtype=torch.float32) - center
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    padding = size // 2

    # Reshape: [H, W, C] -> [1, C, H, W]
    img_t = img.permute(2, 0, 1).unsqueeze(0)

    # Horizontal kernel [1, 1, 1, K]
    kernel_h = gauss_1d.view(1, 1, 1, -1)
    # Vertical kernel [1, 1, K, 1]
    kernel_v = gauss_1d.view(1, 1, -1, 1)

    blurred_channels = []
    for c in range(img_t.shape[1]):
        channel = img_t[:, c:c+1, :, :]
        # Horizontal pass
        blurred = F.conv2d(channel, kernel_h, padding=(0, padding))
        # Vertical pass
        blurred = F.conv2d(blurred, kernel_v, padding=(padding, 0))
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
                    "default": 2.0,
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
    Smart Sharpen - Texture sharpening with edge protection.

    Sharpens textures while protecting edges from halos. Uses Sobel
    edge detection to identify edges and applies USM only to
    non-edge areas (textures, fine detail).

    Similar goal to CAS but different approach.

    Parameters:
    - amount: Sharpening strength
    - radius: Blur radius for USM
    - edge_threshold: Edge detection sensitivity (higher = more protection)
    - reduce_noise: Blur edges slightly to reduce any artifacts
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
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Blur radius for USM"
                }),
                "edge_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Edge protection sensitivity (higher = protect more edges)"
                }),
                "reduce_noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Blur edges slightly to reduce artifacts (0 = off)"
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

            # Create texture mask (inverse of edges) - sharpen textures, protect edges
            edge_mask = torch.clamp((edges - edge_threshold * 0.5) / (edge_threshold + 1e-8), 0, 1)
            texture_mask = (1 - edge_mask).unsqueeze(-1)  # [H, W, 1]

            # USM sharpening on textures only
            blurred = apply_gaussian_blur(img, radius, device)
            detail = img - blurred
            sharpened = img + amount * detail * texture_mask

            # Optional noise reduction on edges (where we're not sharpening)
            if reduce_noise > 0:
                edge_mask_3d = edge_mask.unsqueeze(-1)
                noise_reduced = apply_gaussian_blur(sharpened, 0.5, device)
                sharpened = sharpened * (1 - edge_mask_3d * reduce_noise) + noise_reduced * edge_mask_3d * reduce_noise

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
                    "default": 20.0,
                    "min": 1.0,
                    "max": 200.0,
                    "step": 1.0,
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
                "fast_blur": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use separable blur (faster, identical quality)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_hiraloam"
    CATEGORY = "donut/image/sharpen"

    def apply_hiraloam(self, image, radius, amount, protect_tones, fast_blur):
        if amount == 0:
            return (image,)

        blur_fn = apply_gaussian_blur_separable if fast_blur else apply_gaussian_blur
        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = image[b]
            device = img.device

            # Large radius blur
            blurred = blur_fn(img, radius, device)

            # Local contrast enhancement (USM with large radius)
            detail = img - blurred

            # Protect extreme tones if enabled
            if protect_tones > 0:
                # Luminance
                if img.shape[-1] >= 3:
                    lum = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
                else:
                    lum = img[..., 0]

                # Midtone mask: 1 in midtones, 0 at shadows/highlights
                shadow_fade = torch.clamp(lum / 0.25, 0, 1)
                highlight_fade = torch.clamp((1 - lum) / 0.25, 0, 1)
                midtone_mask = shadow_fade * highlight_fade

                # Reduce effect at extremes based on protect_tones
                effect_mask = 1 - (1 - midtone_mask) * protect_tones
                enhanced = img + amount * detail * effect_mask.unsqueeze(-1)
            else:
                enhanced = img + amount * detail

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
        AMD CAS algorithm - faithful to FidelityFX implementation.

        Uses negative peak weight with normalized filter for artifact-free sharpening.
        Adapts per-pixel based on local contrast headroom.
        """
        if sharpness == 0 and contrast == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        # AMD CAS peak: negative value, more negative = more sharpening
        # lerp(8.0, 5.0, sharpness) = 8.0 - 3.0 * sharpness
        # At sharpness=0: peak = -1/8 = -0.125
        # At sharpness=1: peak = -1/5 = -0.2
        peak = -1.0 / (8.0 - 3.0 * sharpness + 1e-8)

        for b in range(batch_size):
            img = image[b]  # [H, W, C]
            device = img.device

            # Pad for neighborhood access
            padded = F.pad(img.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            padded = padded.squeeze(0).permute(1, 2, 0)  # [H+2, W+2, C]

            # Get neighbors (cross pattern - the 5-tap filter)
            center = padded[1:-1, 1:-1, :]
            north = padded[:-2, 1:-1, :]
            south = padded[2:, 1:-1, :]
            west = padded[1:-1, :-2, :]
            east = padded[1:-1, 2:, :]

            # Find local min/max (soft min/max from cross neighborhood)
            local_min = torch.minimum(torch.minimum(torch.minimum(north, south), west), east)
            local_min = torch.minimum(local_min, center)
            local_max = torch.maximum(torch.maximum(torch.maximum(north, south), west), east)
            local_max = torch.maximum(local_max, center)

            # AMD CAS adaptive weight: measures headroom before clipping
            # amp = saturate(min(mn, 1-mx) / mx)
            # This is high when there's room to sharpen, low near black/white
            amp = torch.minimum(local_min, 1.0 - local_max)
            amp = amp / (local_max + 1e-8)
            amp = torch.clamp(amp, 0, 1)

            # Per-pixel sharpening weight
            w = amp * peak

            # AMD CAS filter: normalized weighted average
            # output = (sum_neighbors * w + center) / (1 + 4*w)
            # With negative w, this emphasizes center over neighbors = sharpening
            sum_neighbors = north + south + east + west
            sharpened = (sum_neighbors * w + center) / (1.0 + 4.0 * w + 1e-8)

            # Apply contrast adjustment if needed
            if contrast != 0:
                mid = 0.5
                sharpened = mid + (sharpened - mid) * (1.0 + contrast)

            sharpened = torch.clamp(sharpened, 0, 1)
            results.append(sharpened)

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "DonutHiRaLoAm": DonutHiRaLoAm,
    "DonutCAS": DonutCAS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutHiRaLoAm": "Donut Local Contrast",
    "DonutCAS": "Donut CAS (Contrast Adaptive Sharpen)",
}

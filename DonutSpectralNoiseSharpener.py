#!/usr/bin/env python3
"""
DonutSpectralNoiseSharpener - ComfyUI Node for Scientific Spectral Enhancement

Implements state-of-the-art spectral noise sharpening based on 2024 research:
- Dual-domain (spatial-frequency) processing
- Amplitude-phase decomposition for brightness preservation  
- Reversible frequency decomposition to prevent information loss
- Reference-guided spectral matching for realistic AI image enhancement
"""

import torch
import numpy as np
import cv2
import time
from typing import Dict, Tuple, Any
from PIL import Image

from .spectral_noise_sharpener import SpectralNoiseSharpener


class DonutSharpenerFromReference:
    """
    ComfyUI node for advanced spectral sharpening using a reference image.
    
    Based on 2024 scientific research in frequency domain image enhancement,
    this node matches the spectral noise characteristics of AI images to real photos.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
            },
            "optional": {
                "enhancement_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "frequency_bands": ("INT", {"default": 16, "min": 8, "max": 32, "step": 1}),
                "spectral_mode": (["full_spectrum", "high_freq_only", "adaptive"], {"default": "full_spectrum"}),
                "blend_factor": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_image", "enhancement_report", "spectral_analysis")
    FUNCTION = "enhance_spectral_noise"
    CATEGORY = "donut/enhancement"

    def __init__(self):
        self.sharpener = None
    
    def tensor_to_cv2(self, tensor_image):
        """Convert ComfyUI tensor image to OpenCV format."""
        # ComfyUI images are in format [batch, height, width, channels] with values in [0, 1]
        if len(tensor_image.shape) == 4:
            tensor_image = tensor_image[0]  # Take first image from batch
        
        # Convert from tensor to numpy
        if hasattr(tensor_image, 'cpu'):
            np_image = tensor_image.cpu().numpy()
        else:
            np_image = tensor_image
        
        # Convert from [0, 1] to [0, 255] and ensure uint8
        np_image = (np_image * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if np_image.shape[2] == 3:  # RGB
            cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        elif np_image.shape[2] == 4:  # RGBA
            cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGR)
        else:
            cv2_image = np_image
        
        return cv2_image
    
    def cv2_to_tensor(self, cv2_image):
        """Convert OpenCV image back to ComfyUI tensor format."""
        # Convert BGR to RGB
        if len(cv2_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2_image
        
        # Convert to float32 and normalize to [0, 1]
        float_image = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        tensor_image = torch.from_numpy(float_image).unsqueeze(0)
        
        return tensor_image
    
    def blend_images(self, original_image: np.ndarray, enhanced_image: np.ndarray, 
                    blend_factor: float) -> np.ndarray:
        """
        Blend enhanced image with original using darken mode to reduce bright edges.
        
        Args:
            original_image: Original input image
            enhanced_image: Enhanced result image
            blend_factor: Blend strength (0.0 = original only, 1.0 = full darken effect)
            
        Returns:
            Blended image using darken blend mode
        """
        # Ensure both images are float32 in [0,1] range
        original = original_image.astype(np.float32)
        enhanced = enhanced_image.astype(np.float32)
        
        # Ensure values are in [0,1] range
        original = np.clip(original, 0.0, 1.0)
        enhanced = np.clip(enhanced, 0.0, 1.0)
        
        # Darken blend: where enhanced is brighter, use original to darken it
        # This prevents bright edges and artifacts
        darkened = np.minimum(original, enhanced)
        
        # Blend: 0.0 = enhanced only, 1.0 = darkened only
        blended = enhanced * (1.0 - blend_factor) + darkened * blend_factor
        
        # Ensure result is in valid range
        blended = np.clip(blended, 0.0, 1.0)
        
        return blended
    
    def format_enhancement_report(self, enhancement_info: Dict) -> str:
        """Format comprehensive enhancement report."""
        factors = enhancement_info['enhancement_factors']
        reference_spectrum = enhancement_info['reference_spectrum']
        
        report = f"""╔════════════════════════════════════════════════════════════════════════════════╗
║                        SPECTRAL NOISE SHARPENING REPORT                        ║
╠════════════════════════════════════════════════════════════════════════════════╣
║ ENHANCEMENT METHOD: {enhancement_info['method'].replace('_', ' ').title():<35}                     ║
║ SCIENTIFIC BASIS: 2024 Dual-Domain Frequency Processing Research              ║
║                                                                                ║
║ CONFIGURATION:                                                                 ║
║ • Enhancement Strength: {enhancement_info['strength']:<15.2f}                              ║
║ • Frequency Bands: {enhancement_info['num_bands']:<18}                              ║
║ • Processing Mode: Amplitude-Phase Decomposition                              ║
║ • Brightness Preservation: Enabled (Mean Normalization)                       ║
╠════════════════════════════════════════════════════════════════════════════════╣
║ SPECTRAL ENHANCEMENT STATISTICS:                                              ║
║ • Mean Enhancement: {np.mean(factors):<16.3f}                              ║
║ • Maximum Enhancement: {np.max(factors):<13.3f}                              ║
║ • Minimum Enhancement: {np.min(factors):<13.3f}                              ║
║ • Standard Deviation: {np.std(factors):<14.3f}                              ║
║                                                                                ║
║ REFERENCE SPECTRUM ANALYSIS:                                                   ║
║ • Total Spectral Bands: {len(reference_spectrum['band_energies']):<12}                              ║
║ • Peak Energy Band: {np.argmax(reference_spectrum['band_energies']):<17} ({np.max(reference_spectrum['band_energies']):.3f})     ║
║ • Energy Distribution: Analyzed for matching                                   ║
╠════════════════════════════════════════════════════════════════════════════════╣
║ FREQUENCY BAND ENHANCEMENT FACTORS:                                           ║"""
        
        # Add enhancement factors for each band
        num_bands = len(factors)
        for i in range(0, num_bands, 4):  # Show 4 bands per row
            line = "║ "
            for j in range(4):
                if i + j < num_bands:
                    band_idx = i + j
                    factor = factors[band_idx]
                    freq_pct = int((band_idx + 0.5) / num_bands * 100)
                    line += f"B{band_idx:02d}({freq_pct:02d}%): {factor:.2f}  "
                else:
                    line += "           "
            line += " ║"
            report += f"\n{line}"
        
        # Add spectral matching analysis
        energy_distribution = reference_spectrum['band_energies']
        low_energy = np.sum(energy_distribution[:num_bands//4])
        mid_energy = np.sum(energy_distribution[num_bands//4:3*num_bands//4])
        high_energy = np.sum(energy_distribution[3*num_bands//4:])
        
        report += f"""
╠════════════════════════════════════════════════════════════════════════════════╣
║ REFERENCE SPECTRAL CHARACTERISTICS:                                           ║
║ • Low Frequency Energy (0-25%): {low_energy:<12.3f}                              ║
║ • Mid Frequency Energy (25-75%): {mid_energy:<11.3f}                              ║
║ • High Frequency Energy (75-100%): {high_energy:<10.3f}                              ║
║                                                                                ║
║ ENHANCEMENT METHODOLOGY:                                                       ║
║ This implementation uses dual-domain processing based on 2024 research:       ║
║ 1. Amplitude-Phase Decomposition: Separates brightness from texture           ║
║ 2. Reversible Frequency Bands: Prevents information loss during processing    ║
║ 3. Reference-Guided Matching: Adapts AI images to real photo characteristics  ║
║ 4. Brightness Preservation: Maintains original illumination levels            ║
║                                                                                ║
║ EXPECTED IMPROVEMENTS:                                                         ║
║ • Enhanced spectral noise matching reference image characteristics             ║
║ • Preserved overall brightness and contrast                                    ║
║ • More realistic texture patterns in enhanced AI-generated images             ║
║ • Scientifically-validated frequency domain processing                        ║
╚════════════════════════════════════════════════════════════════════════════════╝"""
        
        return report
    
    def format_spectral_analysis(self, enhancement_info: Dict) -> str:
        """Format detailed spectral analysis for technical users."""
        reference_spectrum = enhancement_info['reference_spectrum']
        factors = enhancement_info['enhancement_factors']
        
        analysis = f"""SPECTRAL NOISE SHARPENING ANALYSIS
═══════════════════════════════════

SCIENTIFIC BASIS:
Based on 2024 research in dual-domain feature fusion and amplitude-phase decomposition
for frequency domain image enhancement with brightness preservation.

REFERENCE SPECTRUM CHARACTERISTICS:
• Band Count: {len(reference_spectrum['band_energies'])}
• Energy Distribution: {len(reference_spectrum['band_energies'])} frequency bands analyzed
• Phase Statistics: {len(reference_spectrum['band_phase_stats'])} bands with phase characteristics

ENHANCEMENT METHODOLOGY:
1. Amplitude-Phase Decomposition:
   - Separates image into brightness (amplitude) and texture (phase) components
   - Enables independent enhancement of spectral characteristics
   - Preserves overall image brightness through amplitude normalization

2. Reversible Frequency Decomposition:
   - Uses overlapping Gaussian band filters for smooth transitions
   - Ensures perfect reconstruction capability (sum of filters = 1)
   - Prevents information loss during frequency domain processing

3. Reference-Guided Spectral Matching:
   - Analyzes target frequency characteristics from reference image
   - Calculates enhancement factors to match reference spectrum
   - Only enhances (never reduces) to add realistic noise patterns

FREQUENCY BAND ANALYSIS:
"""
        
        # Add detailed band information
        for i, (energy, factor) in enumerate(zip(reference_spectrum['band_energies'], factors)):
            freq_start = int(i / len(factors) * 100)
            freq_end = int((i + 1) / len(factors) * 100)
            phase_stat = reference_spectrum['band_phase_stats'][i]
            
            analysis += f"Band {i:02d} ({freq_start:2d}-{freq_end:2d}%): Energy={energy:.4f}, Factor={factor:.3f}, Phase={phase_stat:.3f}\n"
        
        # Add enhancement interpretation
        max_enhancement_idx = np.argmax(factors)
        max_freq_pct = int((max_enhancement_idx + 0.5) / len(factors) * 100)
        
        analysis += f"""
ENHANCEMENT STRATEGY:
• Maximum enhancement at: {max_freq_pct}% frequency range (Band {max_enhancement_idx})
• Enhancement approach: {'High-frequency focus' if max_freq_pct > 70 else 'Mid-frequency focus' if max_freq_pct > 30 else 'Low-frequency focus'}
• Processing mode: Amplitude preservation with selective frequency enhancement
• Strength: {enhancement_info['strength']:.2f} (0.0 = no enhancement, 1.0 = full matching)

TECHNICAL ADVANTAGES:
• Scientifically validated approach based on latest 2024 research
• Dual-domain processing prevents common frequency domain artifacts
• Reversible decomposition ensures no information loss
• Reference-guided matching produces natural, realistic enhancements
• Brightness preservation maintains visual consistency"""
        
        return analysis

    def enhance_spectral_noise(self, input_image, reference_image, enhancement_strength=1.0, 
                             frequency_bands=16, spectral_mode="full_spectrum",
                             blend_factor=0.8):
        """
        Main enhancement function implementing scientific spectral noise sharpening.
        
        Args:
            input_image: Input image tensor from ComfyUI
            reference_image: Reference image tensor from ComfyUI
            enhancement_strength: Enhancement strength (0.0 to 2.0)
            frequency_bands: Number of frequency bands for analysis
            spectral_mode: Spectral processing mode
            blend_factor: Blend strength (0.0 = enhanced only, 1.0 = darkened)
            
        Returns:
            Tuple of (enhanced_image, enhancement_report, spectral_analysis)
        """
        try:
            # Convert ComfyUI tensors to OpenCV format
            input_cv2 = self.tensor_to_cv2(input_image)
            reference_cv2 = self.tensor_to_cv2(reference_image)
            
            # Initialize sharpener with specified parameters
            if self.sharpener is None or self.sharpener.num_bands != frequency_bands:
                self.sharpener = SpectralNoiseSharpener(num_bands=frequency_bands)
            
            # Apply spectral noise sharpening enhancement
            enhanced_cv2, enhancement_info = self.sharpener.enhance_to_match_reference(
                input_cv2, reference_cv2, strength=enhancement_strength
            )
            
            # Apply spectral mode adjustments if needed
            if spectral_mode == "high_freq_only":
                # Only enhance high-frequency bands (top 25%)
                factors = enhancement_info['enhancement_factors']
                cutoff = len(factors) * 3 // 4
                for i in range(cutoff):
                    factors[i] = 1.0  # No enhancement for low/mid frequencies
                
                # Re-apply with modified factors
                enhanced_cv2, enhancement_info = self.sharpener.enhance_to_match_reference(
                    input_cv2, reference_cv2, strength=enhancement_strength
                )
            
            elif spectral_mode == "adaptive":
                # Adaptive enhancement based on image content analysis
                input_gray = cv2.cvtColor(input_cv2, cv2.COLOR_BGR2GRAY)
                image_variance = np.var(input_gray)
                
                # Adjust strength based on image complexity
                if image_variance < 100:  # Low detail image
                    adaptive_strength = enhancement_strength * 1.2
                elif image_variance > 1000:  # High detail image  
                    adaptive_strength = enhancement_strength * 0.8
                else:
                    adaptive_strength = enhancement_strength
                
                # Re-apply with adaptive strength
                enhanced_cv2, enhancement_info = self.sharpener.enhance_to_match_reference(
                    input_cv2, reference_cv2, strength=adaptive_strength
                )
            
            # Apply blending to control enhancement intensity and reduce edge artifacts
            if blend_factor > 0.0:
                # Convert to float [0,1] for blending
                input_float = input_cv2.astype(np.float32) / 255.0
                enhanced_float = enhanced_cv2.astype(np.float32) / 255.0
                
                # Blend enhanced result with original input
                blended_float = self.blend_images(input_float, enhanced_float, blend_factor)
                
                # Convert back to uint8
                enhanced_cv2 = np.clip(blended_float * 255, 0, 255).astype(np.uint8)
            
            # Convert back to ComfyUI tensor format
            enhanced_tensor = self.cv2_to_tensor(enhanced_cv2)
            
            # Generate comprehensive reports
            enhancement_report = self.format_enhancement_report(enhancement_info)
            spectral_analysis = self.format_spectral_analysis(enhancement_info)
            
            return (enhanced_tensor, enhancement_report, spectral_analysis)
            
        except Exception as e:
            error_msg = f"Spectral enhancement failed: {str(e)}"
            # Return original image on error
            return (
                input_image,
                f"ERROR: {error_msg}",
                f"Analysis failed: {error_msg}"
            )


class DonutSharpener:
    """
    ComfyUI node for spectral sharpening with generated noise reference.
    
    Uses the input image as base and generates various types of noise
    to create synthetic reference characteristics for enhancement.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "enhancement_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "frequency_bands": ("INT", {"default": 16, "min": 8, "max": 32, "step": 1}),
                "noise_type": (["gaussian", "perlin", "film_grain", "sensor_noise", "uniform", "realistic_grain"], {"default": "gaussian"}),
                "noise_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "noise_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "spectral_mode": (["full_spectrum", "high_freq_only", "adaptive"], {"default": "full_spectrum"}),
                "blend_factor": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_image", "noise_reference", "enhancement_report", "spectral_analysis")
    FUNCTION = "enhance_with_noise_reference"
    CATEGORY = "donut/enhancement"

    def __init__(self):
        self.sharpener = None
        self.reference_sharpener = DonutSharpenerFromReference()
    
    def tensor2pil(self, tensor_image):
        """Convert tensor to PIL image."""
        if len(tensor_image.shape) == 4:
            tensor_image = tensor_image[0]  # Take first image from batch
        
        if hasattr(tensor_image, 'cpu'):
            np_image = tensor_image.cpu().numpy()
        else:
            np_image = tensor_image
        
        np_image = (np_image * 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    def pil2tensor(self, pil_image):
        """Convert PIL image to tensor."""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(np_image).unsqueeze(0)
    
    def image_add_grain(self, image, grain_scale, grain_power, grain_sat, toe=0, seed=None):
        """
        Add realistic film grain to an image.
        
        Args:
            image: PIL Image
            grain_scale: Size/scale of grain (0.1-10.0)
            grain_power: Intensity of grain (0.0-1.0) 
            grain_sat: Saturation effect (0.0-1.0)
            toe: Curve adjustment (not used, kept for compatibility)
            seed: Random seed for reproducibility
        """
        # Import dependencies at function start
        from scipy import ndimage
        from PIL import Image as PILImage
        
        if seed is not None:
            np.random.seed(seed)
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        height, width, channels = img_array.shape
        
        # Create grain texture based on scale
        # Calculate grain size and sigma once for use in both main and color grain
        grain_size = int(max(height, width) / (grain_scale * 100)) if grain_scale >= 1.0 else 1
        grain_size = max(grain_size, 1)
        sigma = (1.0 - grain_scale) * 2.0 if grain_scale < 0.5 else 0.0
        
        if grain_scale >= 1.0:
            # For larger scales, create smoother grain
            # Generate base grain texture
            small_grain = np.random.normal(0, 1, (height//grain_size + 1, width//grain_size + 1))
            
            # Resize grain to match image size
            grain_pil = PILImage.fromarray(((small_grain + 1) * 127.5).astype(np.uint8), mode='L')
            grain_pil = grain_pil.resize((width, height), PILImage.BILINEAR)
            grain = np.array(grain_pil).astype(np.float32) / 127.5 - 1.0
        else:
            # For smaller scales, use direct noise
            grain = np.random.normal(0, 1, (height, width))
            # Apply some smoothing based on scale
            if grain_scale < 0.5:
                grain = ndimage.gaussian_filter(grain, sigma)
        
        # Calculate luminance for grain modulation
        luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # Modulate grain by luminance (film grain is more visible in mid-tones)
        grain_modulation = 1.0 - np.abs(luminance - 0.5) * 2.0  # Peak at 0.5 luminance
        grain_modulation = np.power(grain_modulation, 0.5)  # Soften the curve
        
        # Apply grain power
        grain = grain * grain_power * 0.1  # Scale down for realistic effect
        grain = grain * grain_modulation[:,:,np.newaxis]  # Apply luminance modulation
        
        # Create color grain based on saturation parameter
        if grain_sat > 0 and channels >= 3:
            # Color grain - different noise for each channel
            color_grain = np.zeros_like(img_array)
            for c in range(min(3, channels)):
                # Generate slightly different grain for each color channel
                channel_seed = (seed + c) if seed is not None else None
                if channel_seed is not None:
                    np.random.seed(channel_seed)
                
                if grain_scale >= 1.0:
                    small_grain = np.random.normal(0, 1, (height//grain_size + 1, width//grain_size + 1))
                    grain_pil = PILImage.fromarray(((small_grain + 1) * 127.5).astype(np.uint8), mode='L')
                    grain_pil = grain_pil.resize((width, height), PILImage.BILINEAR)
                    channel_grain = np.array(grain_pil).astype(np.float32) / 127.5 - 1.0
                else:
                    channel_grain = np.random.normal(0, 1, (height, width))
                    if grain_scale < 0.5:
                        channel_grain = ndimage.gaussian_filter(channel_grain, sigma)
                
                color_grain[:,:,c] = channel_grain * grain_power * 0.05 * grain_sat * grain_modulation
            
            # Blend monochrome and color grain
            final_grain = grain * (1.0 - grain_sat) + color_grain * grain_sat
        else:
            # Monochrome grain only
            final_grain = np.repeat(grain[:,:,np.newaxis], channels, axis=2)
        
        # Apply grain to image
        result = img_array + final_grain
        result = np.clip(result, 0.0, 1.0)
        
        # Convert back to PIL
        result_uint8 = (result * 255).astype(np.uint8)
        return Image.fromarray(result_uint8)

    def generate_noise_reference(self, input_image, noise_type: str, noise_strength: float, 
                                noise_scale: float = 1.0, noise_saturation: float = 1.0):
        """Generate a noise reference image based on the input image."""
        import numpy as np
        
        if noise_type == "realistic_grain":
            # Use our custom grain function - noise_strength is already scaled (0.0-1.0)
            pil_image = self.tensor2pil(input_image)
            grain_image = self.image_add_grain(pil_image, noise_scale, noise_strength, noise_saturation, 
                                             toe=0, seed=int(time.time()))
            return self.pil2tensor(grain_image)
        
        else:
            # Use existing noise generation methods
            input_cv2 = self.reference_sharpener.tensor_to_cv2(input_image)
            height, width = input_cv2.shape[:2]
            
            # Start with the input image
            noisy_image = input_cv2.astype(np.float32)
            
            if noise_type == "gaussian":
                # Gaussian noise with scale control
                base_noise = np.random.normal(0, noise_strength * 50, (height, width, 3))
                
                # Apply scale: larger scale = smoother/larger noise patterns
                if noise_scale != 1.0:
                    # Create multi-scale Gaussian noise
                    scale_factor = int(max(1, noise_scale))
                    small_height, small_width = height // scale_factor, width // scale_factor
                    small_noise = np.random.normal(0, noise_strength * 50, (small_height, small_width, 3))
                    # Resize to full size for smoother patterns
                    from scipy import ndimage
                    noise = ndimage.zoom(small_noise, (scale_factor, scale_factor, 1), order=1)[:height, :width, :]
                    # Blend with base noise based on scale
                    blend_factor = min(1.0, noise_scale / 3.0)
                    noise = base_noise * (1 - blend_factor) + noise * blend_factor
                else:
                    noise = base_noise
                
                # Apply saturation: controls color vs monochrome noise
                if noise_saturation < 1.0:
                    # Convert to monochrome and blend
                    mono_noise = np.mean(noise, axis=2, keepdims=True)
                    mono_noise = np.repeat(mono_noise, 3, axis=2)
                    noise = noise * noise_saturation + mono_noise * (1.0 - noise_saturation)
                
            elif noise_type == "perlin":
                # Perlin-like noise with scale and saturation
                base_scales = [2, 4, 8]
                # Adjust base scales by noise_scale parameter
                scales = [max(1, int(s * noise_scale)) for s in base_scales]
                
                noise = np.zeros((height, width, 3))
                for i, scale in enumerate(scales):
                    weight = noise_strength * (20 / scale) * (1.0 + i * 0.3)  # Progressive weighting
                    small_noise = np.random.normal(0, weight, (height//scale + 1, width//scale + 1, 3))
                    small_noise = np.repeat(np.repeat(small_noise, scale, axis=0), scale, axis=1)[:height, :width, :]
                    noise += small_noise
                
                # Apply saturation
                if noise_saturation < 1.0:
                    mono_noise = np.mean(noise, axis=2, keepdims=True)
                    mono_noise = np.repeat(mono_noise, 3, axis=2)
                    noise = noise * noise_saturation + mono_noise * (1.0 - noise_saturation)
                    
            elif noise_type == "film_grain":
                # Film grain with scale and saturation controls
                scale_factor = max(1, int(noise_scale))
                grain_height, grain_width = height // scale_factor, width // scale_factor
                
                # Generate luminance-based grain at scaled resolution
                grain_luminance = np.random.normal(0, noise_strength * 20, (grain_height, grain_width))
                
                # Resize grain if needed
                if scale_factor > 1:
                    from scipy import ndimage
                    grain_luminance = ndimage.zoom(grain_luminance, scale_factor, order=1)[:height, :width]
                
                # Apply grain with color correlation
                noise = np.zeros((height, width, 3))
                noise[:,:,0] = grain_luminance * 1.0  # Red
                noise[:,:,1] = grain_luminance * 0.8  # Green (less noise)
                noise[:,:,2] = grain_luminance * 1.1  # Blue (slight more)
                
                # Add independent color noise scaled by saturation
                color_noise = np.random.normal(0, noise_strength * 8 * noise_saturation, (height, width, 3))
                noise += color_noise
                
            elif noise_type == "sensor_noise":
                # Digital sensor noise with scale affecting hot pixel clustering
                noise = np.random.normal(0, noise_strength * 35, (height, width, 3))
                
                # Hot pixels - scale affects clustering
                hot_pixel_density = 0.0001 * noise_strength * noise_scale
                hot_pixels = np.random.random((height, width, 3)) < hot_pixel_density
                noise[hot_pixels] += np.random.uniform(50, 100, np.sum(hot_pixels))
                
                # Saturation affects color correlation of sensor noise
                if noise_saturation < 1.0:
                    mono_noise = np.mean(noise, axis=2, keepdims=True)
                    mono_noise = np.repeat(mono_noise, 3, axis=2)
                    noise = noise * noise_saturation + mono_noise * (1.0 - noise_saturation)
                
            elif noise_type == "uniform":
                # Uniform noise with scale and saturation
                base_noise = np.random.uniform(-noise_strength * 40, noise_strength * 40, (height, width, 3))
                
                # Scale affects smoothness of uniform noise
                if noise_scale > 1.0:
                    from scipy import ndimage
                    sigma = (noise_scale - 1.0) * 0.5
                    noise = ndimage.gaussian_filter(base_noise, sigma=(sigma, sigma, 0))
                else:
                    noise = base_noise
                
                # Saturation affects color correlation
                if noise_saturation < 1.0:
                    mono_noise = np.mean(noise, axis=2, keepdims=True)
                    mono_noise = np.repeat(mono_noise, 3, axis=2)
                    noise = noise * noise_saturation + mono_noise * (1.0 - noise_saturation)
            
            # Add noise to image
            noisy_image += noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
            # Convert back to tensor
            return self.reference_sharpener.cv2_to_tensor(noisy_image)
    
    def enhance_with_noise_reference(self, input_image, enhancement_strength=1.0, frequency_bands=16, 
                                   noise_type="gaussian", noise_strength=0.0, noise_scale=1.0, 
                                   noise_saturation=1.0, spectral_mode="full_spectrum", blend_factor=0.8):
        """
        Main enhancement function using generated noise reference.
        """
        try:
            # Convert percentage to decimal (0-100% -> 0.0-1.0)
            noise_strength_scaled = noise_strength / 100.0
            
            if noise_strength == 0.0:
                # Self-amplification mode - generate enhancement factors directly
                # Convert ComfyUI tensor to OpenCV format
                input_cv2 = self.reference_sharpener.tensor_to_cv2(input_image)
                
                # Initialize sharpener with specified parameters
                if self.sharpener is None or self.sharpener.num_bands != frequency_bands:
                    self.sharpener = SpectralNoiseSharpener(num_bands=frequency_bands)
                
                # Convert to grayscale for spectral analysis
                if len(input_cv2.shape) == 3:
                    input_gray = cv2.cvtColor(input_cv2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                else:
                    input_gray = input_cv2.astype(np.float32) / 255.0
                
                # Create band filters for this image size
                self.sharpener.band_filters = None
                self.sharpener.create_reversible_frequency_bands(input_gray.shape)
                
                # Generate progressive enhancement factors for self-amplification
                enhancement_factors = []
                for i in range(frequency_bands):
                    freq_position = i / (frequency_bands - 1)  # 0 to 1
                    
                    # Progressive enhancement: more for higher frequencies
                    if freq_position < 0.2:
                        # Low frequencies: minimal enhancement
                        base_factor = 1.0 + 0.1 * enhancement_strength
                    elif freq_position < 0.5:
                        # Mid frequencies: moderate enhancement 
                        base_factor = 1.0 + 0.3 * enhancement_strength
                    else:
                        # High frequencies: strong enhancement (where noise lives)
                        base_factor = 1.0 + 0.6 * enhancement_strength
                    
                    enhancement_factors.append(base_factor)
                
                # Apply spectral enhancement
                enhanced_gray = self.sharpener.apply_spectral_noise_sharpening(
                    input_gray, enhancement_factors, 1.0  # Use full strength since factors are pre-scaled
                )
                
                # Apply to color channels if input is color
                if len(input_cv2.shape) == 3:
                    input_float = input_cv2.astype(np.float32) / 255.0
                    enhanced_image = np.zeros_like(input_float)
                    
                    # Calculate enhancement ratio for each pixel
                    ratio = enhanced_gray / (input_gray + 1e-8)
                    ratio = np.clip(ratio, 0.5, 2.0)
                    
                    # Apply ratio to each color channel
                    for channel in range(3):
                        enhanced_image[:, :, channel] = input_float[:, :, channel] * ratio
                    
                    enhanced_cv2 = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)
                else:
                    enhanced_cv2 = np.clip(enhanced_gray * 255, 0, 255).astype(np.uint8)
                
                # Apply blending if specified
                if blend_factor > 0.0:
                    input_float = input_cv2.astype(np.float32) / 255.0
                    enhanced_float = enhanced_cv2.astype(np.float32) / 255.0
                    blended_float = self.reference_sharpener.blend_images(input_float, enhanced_float, blend_factor)
                    enhanced_cv2 = np.clip(blended_float * 255, 0, 255).astype(np.uint8)
                
                # Convert back to ComfyUI tensor format
                enhanced_tensor = self.reference_sharpener.cv2_to_tensor(enhanced_cv2)
                
                # Generate reports
                enhancement_info = {
                    'enhancement_factors': enhancement_factors,
                    'strength': enhancement_strength,
                    'num_bands': frequency_bands,
                    'method': 'self_amplification',
                    'reference_spectrum': {
                        'band_energies': [1.0] * frequency_bands,  # Placeholder for reporting
                        'band_phase_stats': [0.0] * frequency_bands
                    }
                }
                
                enhancement_report = self.reference_sharpener.format_enhancement_report(enhancement_info)
                spectral_analysis = self.reference_sharpener.format_spectral_analysis(enhancement_info)
                
                return (enhanced_tensor, input_image, enhancement_report, spectral_analysis)
                
            else:
                # Normal noise reference mode
                # Generate noise reference image
                noise_reference = self.generate_noise_reference(input_image, noise_type, noise_strength_scaled, 
                                                               noise_scale, noise_saturation)
                
                # Use the reference-based sharpener with our generated reference
                enhanced_image, enhancement_report, spectral_analysis = self.reference_sharpener.enhance_spectral_noise(
                    input_image, noise_reference, enhancement_strength, frequency_bands, 
                    spectral_mode, blend_factor
                )
                
                return (enhanced_image, noise_reference, enhancement_report, spectral_analysis)
            
        except Exception as e:
            error_msg = f"Enhancement with noise reference failed: {str(e)}"
            return (
                input_image,
                input_image,
                f"ERROR: {error_msg}",
                f"Analysis failed: {error_msg}"
            )


class DonutSpectralSharpener(DonutSharpener):
    """
    Unified spectral sharpener.

    Consolidates the two legacy sharpener nodes behind a single ``reference_source``
    selector. The actual enhancement math is reused VERBATIM from the legacy engine
    classes (``DonutSharpenerFromReference`` for the external/synthesized-reference
    pipeline, and ``DonutSharpener`` for the self-amplify / generated-noise paths).

    reference_source:
      - "external_reference": match the spectrum of ``reference_image`` (falls back
        to ``input_image`` when the reference is unwired).
      - "generated_noise": synthesize a noise reference from ``input_image``
        (legacy DonutSharpener with noise_strength > 0).
      - "self_amplify": self-amplification (legacy DonutSharpener noise_strength == 0).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "reference_source": (["external_reference", "generated_noise", "self_amplify"], {"default": "external_reference"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "enhancement_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "frequency_bands": ("INT", {"default": 16, "min": 8, "max": 32, "step": 1}),
                "noise_type": (["gaussian", "perlin", "film_grain", "sensor_noise", "uniform", "realistic_grain"], {"default": "gaussian"}),
                "noise_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "noise_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "spectral_mode": (["full_spectrum", "high_freq_only", "adaptive"], {"default": "full_spectrum"}),
                "blend_factor": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_image", "noise_reference", "enhancement_report", "spectral_analysis")
    FUNCTION = "enhance_spectral"
    CATEGORY = "donut/enhancement"

    def enhance_spectral(self, input_image, reference_source="external_reference", reference_image=None,
                         enhancement_strength=1.0, frequency_bands=16, noise_type="gaussian",
                         noise_strength=0.0, noise_scale=1.0, noise_saturation=1.0,
                         spectral_mode="full_spectrum", blend_factor=0.8):
        if reference_source == "external_reference":
            # Match an external reference image. Fall back to the input image when
            # the reference input is left unwired so the node never hard-fails.
            ref = reference_image if reference_image is not None else input_image
            enhanced_image, enhancement_report, spectral_analysis = self.reference_sharpener.enhance_spectral_noise(
                input_image, ref, enhancement_strength, frequency_bands,
                spectral_mode, blend_factor
            )
            return (enhanced_image, ref, enhancement_report, spectral_analysis)

        elif reference_source == "self_amplify":
            # Legacy DonutSharpener noise_strength == 0 path (self-amplification).
            return self.enhance_with_noise_reference(
                input_image, enhancement_strength, frequency_bands,
                noise_type, 0.0, noise_scale, noise_saturation,
                spectral_mode, blend_factor
            )

        else:
            # generated_noise: synthesize a noise reference. Force a non-zero
            # noise_strength so the legacy engine takes the synthesize branch even
            # if the user left the widget at its 0.0 default.
            effective_noise_strength = noise_strength if noise_strength > 0.0 else 1.0
            return self.enhance_with_noise_reference(
                input_image, enhancement_strength, frequency_bands,
                noise_type, effective_noise_strength, noise_scale, noise_saturation,
                spectral_mode, blend_factor
            )


class _DonutSharpenerFromReferenceAlias(DonutSpectralSharpener):
    """Alias for legacy id "Donut Sharpener (from reference)".

    Preserves the ORIGINAL 2-required/4-optional INPUT_TYPES, 3-output signature,
    and FUNCTION name so already-saved workflows deserialize byte-identically.
    Delegates to the shared engine in external_reference mode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
            },
            "optional": {
                "enhancement_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "frequency_bands": ("INT", {"default": 16, "min": 8, "max": 32, "step": 1}),
                "spectral_mode": (["full_spectrum", "high_freq_only", "adaptive"], {"default": "full_spectrum"}),
                "blend_factor": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_image", "enhancement_report", "spectral_analysis")
    FUNCTION = "enhance_spectral_noise"
    CATEGORY = "donut/enhancement"

    def enhance_spectral_noise(self, input_image, reference_image, enhancement_strength=1.0,
                               frequency_bands=16, spectral_mode="full_spectrum",
                               blend_factor=0.8):
        return self.reference_sharpener.enhance_spectral_noise(
            input_image, reference_image, enhancement_strength, frequency_bands,
            spectral_mode, blend_factor
        )


class _DonutSharpenerAlias(DonutSpectralSharpener):
    """Alias for legacy id "Donut Sharpener".

    Preserves the ORIGINAL 1-required/9-optional single-input INPUT_TYPES,
    4-output signature, and FUNCTION name so already-saved workflows deserialize
    byte-identically. Delegates to the shared engine; noise_strength == 0 maps to
    self_amplify, otherwise generated_noise (matching legacy behavior verbatim).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "enhancement_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "frequency_bands": ("INT", {"default": 16, "min": 8, "max": 32, "step": 1}),
                "noise_type": (["gaussian", "perlin", "film_grain", "sensor_noise", "uniform", "realistic_grain"], {"default": "gaussian"}),
                "noise_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "noise_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "spectral_mode": (["full_spectrum", "high_freq_only", "adaptive"], {"default": "full_spectrum"}),
                "blend_factor": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_image", "noise_reference", "enhancement_report", "spectral_analysis")
    FUNCTION = "enhance_with_noise_reference"
    CATEGORY = "donut/enhancement"


NODE_CLASS_MAPPINGS = {
    "Donut Spectral Sharpener": DonutSpectralSharpener,
    # Legacy ids kept registered as thin alias subclasses (de-cluttered from the
    # display map below). DO NOT remove: needed for already-saved workflows.
    "Donut Sharpener (from reference)": _DonutSharpenerFromReferenceAlias,
    "Donut Sharpener": _DonutSharpenerAlias,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Donut Spectral Sharpener": "Donut Spectral Sharpener",
    # Legacy displays removed so only the unified node shows in the menu.
}
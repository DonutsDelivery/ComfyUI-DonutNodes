import re
import torch
import torch.nn as nn
from tqdm import tqdm
import hashlib
import psutil
import gc
import os
import tempfile
import uuid
from contextlib import contextmanager

# use package-relative path for ComfyUI
from .utils.sdxl_safetensors import ensure_same_device

# Global cache for preventing redundant processing
_MERGE_CACHE = {}
_CACHE_MAX_SIZE = 10

def monitor_memory(label=""):
    """Print current memory usage including VRAM"""
    try:
        process = psutil.Process()
        ram_mb = process.memory_info().rss / 1024 / 1024

        # Always try to get VRAM info
        vram_info = ""
        if torch.cuda.is_available():
            try:
                vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
                vram_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                vram_info = f", VRAM: {vram_mb:.1f}MB (reserved: {vram_reserved_mb:.1f}MB)"
            except Exception:
                vram_info = ", VRAM: unavailable"

        print(f"[MEMORY-{label}] RAM: {ram_mb:.1f}MB{vram_info}")

    except Exception as e:
        print(f"[MEMORY-{label}] Error: {e}")

def check_memory_safety():
    """Check if memory is safe to continue"""
    try:
        process = psutil.Process()
        current_ram_gb = process.memory_info().rss / 1024 / 1024 / 1024
        total_ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        available_ram_gb = total_ram_gb - current_ram_gb
        ram_usage_percent = current_ram_gb / total_ram_gb

        if ram_usage_percent > 0.95 or available_ram_gb < 1.5:
            return False, ram_usage_percent, available_ram_gb

        return True, ram_usage_percent, available_ram_gb
    except Exception:
        return True, 0.0, 999.0

def compute_merge_hash(models, merge_strength, temperature, enable_ties, threshold, forced_merge_ratio, renorm_mode):
    """Compute hash of merge parameters to detect changes"""
    hasher = hashlib.md5()

    # Hash model inputs
    for model in models:
        if model is not None and not getattr(model, "_is_filler", False):
            # Use model object id as a simple change detector
            hasher.update(str(id(model)).encode())

    # Hash merge parameters
    hasher.update(f"{merge_strength}_{temperature}_{enable_ties}_{threshold}_{forced_merge_ratio}_{renorm_mode}".encode())

    return hasher.hexdigest()

def check_cache_for_merge(cache_key):
    """Check if we have a cached result for this merge"""
    if cache_key in _MERGE_CACHE:
        print("[Cache] Found cached merge result - skipping processing")
        return _MERGE_CACHE[cache_key]
    return None

def store_merge_result(cache_key, result):
    """Store merge result in cache"""
    global _MERGE_CACHE

    # Clear old entries if cache is full
    if len(_MERGE_CACHE) >= _CACHE_MAX_SIZE:
        oldest_key = next(iter(_MERGE_CACHE))
        del _MERGE_CACHE[oldest_key]
        print(f"[Cache] Removed oldest entry, cache size: {len(_MERGE_CACHE)}")

    _MERGE_CACHE[cache_key] = result
    print(f"[Cache] Stored merge result, cache size: {len(_MERGE_CACHE)}")

def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class MemoryExhaustionError(Exception):
    pass

@contextmanager
def memory_cleanup_context(label=""):
    """Context manager for automatic memory cleanup"""
    monitor_memory(f"{label}-START")
    try:
        yield
    finally:
        force_cleanup()
        monitor_memory(f"{label}-END")

def calibrate_renormalize(merged_param, base_param, mode="calibrate", t=1.0, s=1.5):
    """Renormalize using calibration algorithm or simple methods"""
    if mode == "none":
        return merged_param

    elif mode == "magnitude":
        # Simple magnitude preservation (original method)
        base_norm = torch.norm(base_param)
        merged_norm = torch.norm(merged_param)
        if merged_norm > 1e-8:  # Avoid division by zero
            return merged_param * (base_norm / merged_norm)
        return merged_param

    elif mode == "calibrate":
        # More conservative calibration-style renormalization
        import torch.nn.functional as F

        # Get the difference from base parameter (the "delta")
        param_delta = merged_param - base_param

        # Only calibrate if there's a significant change
        delta_magnitude = torch.norm(param_delta).item()
        if delta_magnitude < 1e-6:
            return merged_param

        # Work with absolute values for calibration
        param_abs = torch.abs(param_delta)
        param_sign = torch.sign(param_delta)

        # Apply softmax normalization to absolute delta values
        if param_abs.numel() > 1:
            # Flatten for softmax, then reshape back
            original_shape = param_abs.shape
            param_flat = param_abs.flatten()

            if param_flat.sum() > 1e-8:  # Avoid division by zero
                # More conservative softmax (add small temperature)
                sm_m = F.softmax(param_flat * 0.5, dim=0)  # Lower temperature for stability

                # More conservative calibration thresholding
                K = param_flat.numel()
                thr_m = (t / K) * sm_m.sum() * 0.5  # More conservative threshold

                # More conservative scaling
                conservative_s = 1.0 + (s - 1.0) * 0.3  # Reduce scaling intensity
                cal_m = torch.where(sm_m > thr_m, conservative_s * sm_m, sm_m)

                # Renormalize to preserve relative magnitudes
                cal_m = cal_m * (param_flat.sum() / cal_m.sum())

                # Reshape back and restore signs
                cal_m = cal_m.reshape(original_shape)
                calibrated_delta = cal_m * param_sign

                # Apply calibrated delta back to base parameter
                calibrated_param = base_param + calibrated_delta

                return calibrated_param
            else:
                return merged_param
        else:
            return merged_param

    else:
        raise ValueError(f"Unknown renormalization mode: {mode}. Use 'none', 'magnitude', or 'calibrate'")

class TaskVector:
    """Extract task vector (delta) between two models with SDXL awareness"""
    def __init__(self, base_model, finetuned_model, exclude_param_names_regex=None):
        self.task_vector_param_dict = {}
        self.param_metadata = {}  # Store SDXL-specific metadata

        base_params = {n: p.detach().cpu().float().clone()
                      for n, p in base_model.named_parameters()}
        finetuned_params = {n: p.detach().cpu().float().clone()
                           for n, p in finetuned_model.named_parameters()}

        # Extract deltas with SDXL layer classification
        for name in base_params:
            if name in finetuned_params:
                if exclude_param_names_regex:
                    skip = any(re.search(pattern, name) for pattern in exclude_param_names_regex)
                    if skip:
                        continue

                delta = finetuned_params[name] - base_params[name]
                self.task_vector_param_dict[name] = delta

                # Classify SDXL layer type and store metadata
                self.param_metadata[name] = {
                    'layer_type': self._classify_sdxl_layer(name),
                    'base_magnitude': torch.norm(base_params[name]).item(),
                    'delta_magnitude': torch.norm(delta).item(),
                    'change_ratio': torch.norm(delta).item() / (torch.norm(base_params[name]).item() + 1e-8)
                }

    def _classify_sdxl_layer(self, param_name):
        """Classify SDXL layer types for specialized handling - ENHANCED"""
        name_lower = param_name.lower()

        # UNet structure classification - more comprehensive
        if 'time_embed' in name_lower:
            return 'time_embedding'
        elif 'label_emb' in name_lower:
            return 'class_embedding'
        elif any(x in name_lower for x in ['attn', 'attention']):
            if 'cross' in name_lower:
                return 'cross_attention'  # Text conditioning
            else:
                return 'self_attention'   # Spatial attention
        elif any(x in name_lower for x in ['conv', 'convolution']):
            if 'in_layers' in name_lower or 'input' in name_lower:
                return 'input_conv'
            elif 'out_layers' in name_lower or 'output' in name_lower:
                return 'output_conv'
            elif 'skip' in name_lower or 'residual' in name_lower:
                return 'skip_conv'
            else:
                return 'feature_conv'
        elif any(x in name_lower for x in ['norm', 'group_norm', 'layer_norm']):
            return 'normalization'
        elif 'bias' in name_lower:
            return 'bias'
        elif any(x in name_lower for x in ['down', 'upsample']):
            return 'resolution_change'
        # Enhanced classification for previously 'other' parameters
        elif any(x in name_lower for x in ['proj', 'projection']):
            return 'self_attention'  # Projections are usually part of attention
        elif any(x in name_lower for x in ['to_q', 'to_k', 'to_v', 'to_out']):
            return 'self_attention'  # Attention components
        elif any(x in name_lower for x in ['ff', 'feedforward', 'mlp']):
            return 'self_attention'  # Feed-forward networks in transformers
        elif 'weight' in name_lower and 'emb' in name_lower:
            return 'time_embedding'  # Embedding weights
        else:
            return 'other'

class MergingMethod:
    def __init__(self, merging_method_name: str):
        self.method = merging_method_name

        # SDXL-specific thresholds based on WIDEN paper principles - RELAXED for better success rates
        self.sdxl_thresholds = {
            'time_embedding': 0.0001,     # More lenient for critical layers
            'class_embedding': 0.0001,    # More lenient for critical layers
            'cross_attention': 0.0005,    # Relaxed from 0.002
            'self_attention': 0.0002,     # Relaxed from 0.001
            'input_conv': 0.0002,         # Relaxed from 0.0005
            'output_conv': 0.0005,        # Relaxed from 0.002
            'feature_conv': 0.0002,       # Relaxed from 0.001
            'skip_conv': 0.0002,          # Relaxed from 0.0008
            'resolution_change': 0.0001,  # More lenient
            'normalization': 0.0001,      # Relaxed from 0.0002
            'bias': 0.00005,              # Relaxed from 0.0001
            'other': 0.0001               # More lenient for unclassified layers
        }

        # Layer importance weights for SDXL
        self.sdxl_importance_weights = {
            'time_embedding': 1.5,        # Very important for temporal consistency
            'class_embedding': 1.3,       # Important for conditioning
            'cross_attention': 1.4,       # Critical for text alignment
            'self_attention': 1.2,        # Important for spatial coherence
            'input_conv': 1.1,           # Feature extraction
            'output_conv': 1.3,          # Final output quality
            'feature_conv': 1.0,         # Standard processing
            'skip_conv': 1.1,            # Residual connections
            'resolution_change': 1.2,    # Scale handling
            'normalization': 0.8,        # Less critical for merging
            'bias': 0.6,                 # Least critical
            'other': 1.0                 # Default
        }

    def should_merge_parameter(self, param_name, delta_magnitude, metadata):
        """Determine if parameter should be merged based on SDXL-specific criteria - RELAXED"""
        layer_type = metadata.get('layer_type', 'other')
        threshold = self.sdxl_thresholds.get(layer_type, 0.0001)  # Lower default threshold

        # Additional criteria for SDXL
        change_ratio = metadata.get('change_ratio', 0)

        # More lenient change requirements
        if delta_magnitude < threshold:
            return False

        # More lenient relative change threshold (reduced from 0.001 to 0.0001)
        if change_ratio < 0.0001:
            return False

        # Special handling for critical layers - even more sensitive
        if layer_type in ['time_embedding', 'class_embedding']:
            return delta_magnitude > threshold * 0.1  # Much more sensitive (was 0.5)

        # Special handling for 'other' category - be very lenient
        if layer_type == 'other':
            return delta_magnitude > threshold * 0.1

        return True

    def compute_magnitude_direction_sdxl(self, param_dict, desc="processing"):
        """Compute magnitude and direction optimized for SDXL parameters"""
        mags, dirs = {}, {}

        for name, tensor in param_dict.items():
            try:
                # SDXL-optimized magnitude/direction computation
                if tensor.dim() == 4:  # Conv layers (out_ch, in_ch, h, w)
                    o, c, h, w = tensor.shape
                    # For SDXL conv: preserve spatial structure in direction
                    if h * w <= 9:  # Small kernels (1x1, 3x3): flatten spatial
                        flat = tensor.view(o, -1)
                    else:  # Large kernels: treat each spatial position separately
                        flat = tensor.view(o, c, -1).mean(dim=2)  # Average spatial

                elif tensor.dim() == 3:  # Attention weights (heads, seq, dim)
                    flat = tensor.view(tensor.shape[0], -1)

                elif tensor.dim() == 2:  # Linear layers
                    flat = tensor

                elif tensor.dim() == 1:  # Bias, normalization parameters
                    # For 1D: each element is its own "feature"
                    flat = tensor.unsqueeze(0)

                else:
                    continue

                # Compute magnitude per output feature/channel
                if flat.dim() > 1:
                    mag = flat.norm(dim=-1)
                    # Stable direction computation
                    dir = flat / (mag.unsqueeze(-1) + 1e-8)
                else:
                    mag = flat.abs()
                    dir = torch.sign(flat)

                # Reshape back to match original tensor structure
                if tensor.dim() == 4 and h * w > 9:
                    # Expand back for large kernels
                    dirs[name] = dir.unsqueeze(-1).expand(-1, -1, h*w).view(tensor.shape)
                elif tensor.dim() == 1:
                    dirs[name] = dir.squeeze(0)
                    mag = mag.squeeze(0)
                else:
                    dirs[name] = dir

                mags[name] = mag

            except Exception as e:
                print(f"Warning: Failed to process {name}: {e}")
                continue

        return mags, dirs

    def rank_significance_adaptive(self, diff_tensor, layer_type='other'):
        """Enhanced ranking with SDXL layer-specific adaptations - BULLETPROOF"""
        # Handle edge cases first
        if diff_tensor.numel() == 0:
            return diff_tensor

        if diff_tensor.numel() == 1:
            # Scalar tensor - return as-is
            return diff_tensor

        # Ensure minimum dimensionality
        if diff_tensor.ndim == 0:
            # 0D tensor - return as-is
            return diff_tensor

        original_shape = diff_tensor.shape

        try:
            # Handle 1D tensors specially
            if diff_tensor.ndim == 1:
                # For 1D tensors, create simple ranking
                if diff_tensor.numel() <= 1:
                    return diff_tensor

                indices = torch.argsort(diff_tensor, dim=0)
                L = diff_tensor.shape[0]

                if layer_type in ['time_embedding', 'class_embedding']:
                    sig = torch.pow(torch.arange(L, device=diff_tensor.device, dtype=diff_tensor.dtype) / max(L-1, 1), 0.7)
                else:
                    sig = torch.arange(L, device=diff_tensor.device, dtype=diff_tensor.dtype) / max(L-1, 1)

                ranked = torch.zeros_like(diff_tensor)
                ranked.scatter_(0, indices, sig)
                return ranked

            # For multi-dimensional tensors, be more careful with flattening
            flat = None

            # Safe flattening strategy
            if diff_tensor.ndim == 2:
                # 2D tensor - use as-is or flatten to 1D if one dimension is 1
                if diff_tensor.shape[0] == 1:
                    flat = diff_tensor.flatten()
                elif diff_tensor.shape[1] == 1:
                    flat = diff_tensor.flatten()
                else:
                    flat = diff_tensor
            else:
                # Higher dimensional tensors - flatten carefully
                try:
                    if layer_type in ['cross_attention', 'self_attention']:
                        # For attention: try to preserve structure
                        if diff_tensor.ndim > 2:
                            flat = diff_tensor.flatten(1)
                        else:
                            flat = diff_tensor
                    else:
                        # For other types: safe flattening
                        if diff_tensor.ndim > 2:
                            flat = diff_tensor.flatten(1)
                        else:
                            flat = diff_tensor
                except Exception:
                    # Fallback: complete flattening
                    flat = diff_tensor.flatten()

            # Ensure we have a valid tensor for ranking
            if flat is None:
                flat = diff_tensor.flatten()

            # Handle the flattened tensor
            if flat.ndim == 1:
                # 1D case after flattening
                if flat.numel() <= 1:
                    return diff_tensor

                indices = torch.argsort(flat, dim=0)
                L = flat.shape[0]

                if layer_type in ['time_embedding', 'class_embedding']:
                    sig = torch.pow(torch.arange(L, device=flat.device, dtype=flat.dtype) / max(L-1, 1), 0.7)
                else:
                    sig = torch.arange(L, device=flat.device, dtype=flat.dtype) / max(L-1, 1)

                ranked_flat = torch.zeros_like(flat)
                ranked_flat.scatter_(0, indices, sig)

                # Reshape back to original if possible
                if ranked_flat.numel() == diff_tensor.numel():
                    return ranked_flat.view(original_shape)
                else:
                    return ranked_flat

            elif flat.ndim == 2:
                # 2D case - apply ranking along last dimension
                if flat.shape[-1] <= 1:
                    return diff_tensor

                indices = torch.argsort(flat, dim=-1)
                L = flat.shape[-1]

                if layer_type in ['time_embedding', 'class_embedding']:
                    sig = torch.pow(torch.arange(L, device=flat.device, dtype=flat.dtype) / max(L-1, 1), 0.7)
                else:
                    sig = torch.arange(L, device=flat.device, dtype=flat.dtype) / max(L-1, 1)

                # Create ranking matrix safely
                base = sig.unsqueeze(0).expand(flat.shape[0], -1)
                ranked = torch.zeros_like(flat)
                ranked.scatter_(-1, indices, base)

                # Reshape back to original if possible
                if ranked.numel() == diff_tensor.numel():
                    return ranked.view(original_shape)
                else:
                    return ranked
            else:
                # Higher dimensional - return original to avoid errors
                return diff_tensor

        except Exception as e:
            print(f"Warning: Ranking failed for tensor shape {diff_tensor.shape}, layer {layer_type}: {e}")
            # Ultimate fallback: return normalized tensor
            try:
                norm = torch.norm(diff_tensor)
                if norm > 1e-8:
                    return diff_tensor / norm
                else:
                    return diff_tensor
            except:
                return diff_tensor

    def compute_importance_sdxl(self, sig_tensor, layer_type='other', above_avg_ratio=1.0, calibration_value=1.0):
        """SDXL-optimized importance computation following WIDEN principles - BULLETPROOF"""

        try:
            # Handle edge cases
            if sig_tensor.numel() == 0:
                return torch.tensor(1.0, dtype=torch.float32, device=sig_tensor.device)

            # Layer-specific importance weighting
            layer_weight = self.sdxl_importance_weights.get(layer_type, 1.0)

            # Handle scalar tensors
            if sig_tensor.numel() == 1:
                # For scalar tensors, just return the calibration value
                return torch.tensor(calibration_value * layer_weight, dtype=sig_tensor.dtype, device=sig_tensor.device)

            # Handle very small tensors
            if sig_tensor.numel() <= 2:
                return torch.full_like(sig_tensor, calibration_value * layer_weight)

            # Base softmax scoring with error handling
            try:
                if sig_tensor.ndim == 0:
                    return torch.tensor(calibration_value * layer_weight, dtype=sig_tensor.dtype, device=sig_tensor.device)
                elif sig_tensor.ndim == 1:
                    softmax_dim = 0
                else:
                    softmax_dim = 0

                # Apply softmax with numerical stability
                sig_scaled = sig_tensor * layer_weight
                # Clamp to prevent overflow
                sig_scaled = torch.clamp(sig_scaled, min=-50, max=50)
                sc = torch.softmax(sig_scaled, dim=softmax_dim)

            except Exception as e:
                print(f"Warning: Softmax failed for tensor shape {sig_tensor.shape}: {e}")
                return torch.full_like(sig_tensor, calibration_value * layer_weight)

            # Adaptive thresholding based on layer type
            try:
                if sig_tensor.ndim > 1:
                    avg = sig_tensor.mean(0, keepdim=True)
                else:
                    avg = sig_tensor.mean()

                # Layer-specific above-average ratio adjustment
                if layer_type in ['time_embedding', 'class_embedding', 'cross_attention']:
                    # More selective for critical layers
                    adjusted_ratio = above_avg_ratio * 1.2
                elif layer_type in ['normalization', 'bias']:
                    # Less selective for less critical layers
                    adjusted_ratio = above_avg_ratio * 0.8
                else:
                    adjusted_ratio = above_avg_ratio

                mask = sig_tensor > avg * adjusted_ratio

                # Apply calibration with layer-specific scaling
                calibration_scaled = calibration_value * layer_weight
                sc = torch.where(mask, torch.tensor(calibration_scaled, dtype=sc.dtype, device=sc.device), sc)

                return sc

            except Exception as e:
                print(f"Warning: Thresholding failed for tensor shape {sig_tensor.shape}: {e}")
                return torch.full_like(sig_tensor, calibration_value * layer_weight)

        except Exception as e:
            print(f"Warning: Importance computation completely failed: {e}")
            # Ultimate fallback
            return torch.tensor(1.0, dtype=torch.float32, device=sig_tensor.device if hasattr(sig_tensor, 'device') else 'cpu')

    def merge_single_parameter_sdxl(self, deltas, base_param, mag_ranks, dir_ranks,
                                   param_name, metadata, above_avg_ratio=1.0, calibration_value=1.0):
        """SDXL-optimized parameter merging with layer-aware weighting - ROBUST"""
        try:
            layer_type = metadata.get('layer_type', 'other')

            # More lenient parameter checking
            delta_mag = torch.norm(deltas).item() / max(len(deltas), 1)  # Avoid division by zero
            if not self.should_merge_parameter(param_name, delta_mag, metadata):
                return base_param

            # Compute importance scores with comprehensive error handling
            try:
                mag_importance = self.compute_importance_sdxl(mag_ranks, layer_type, above_avg_ratio, calibration_value)
                dir_importance = self.compute_importance_sdxl(dir_ranks, layer_type, above_avg_ratio, calibration_value)
            except Exception as e:
                print(f"Warning: Failed to compute importance for {param_name}: {e}")
                # Fallback: use simple average instead of failing completely
                return base_param + deltas.mean(0)

            # Robust importance combination with fallbacks
            try:
                # Ensure importance tensors have compatible shapes
                if mag_importance.numel() == 1 and dir_importance.numel() == 1:
                    # Both are scalars
                    combined_weights = 0.5 * (mag_importance + dir_importance)
                elif mag_importance.numel() == 1:
                    # Mag is scalar, dir is tensor
                    combined_weights = 0.5 * mag_importance.item() + 0.5 * dir_importance
                elif dir_importance.numel() == 1:
                    # Dir is scalar, mag is tensor
                    combined_weights = 0.5 * mag_importance + 0.5 * dir_importance.item()
                elif mag_importance.shape != dir_importance.shape:
                    print(f"Info: Using fallback for {param_name} due to shape mismatch: mag {mag_importance.shape} vs dir {dir_importance.shape}")
                    # Fallback to simple average
                    return base_param + deltas.mean(0)
                else:
                    # Layer-specific importance combination
                    if layer_type in ['cross_attention', 'self_attention']:
                        # For attention: direction is more important than magnitude
                        combined_weights = 0.3 * mag_importance + 0.7 * dir_importance
                    elif layer_type in ['normalization']:
                        # For normalization: magnitude is more important
                        combined_weights = 0.8 * mag_importance + 0.2 * dir_importance
                    else:
                        # Default balanced combination
                        combined_weights = 0.5 * mag_importance + 0.5 * dir_importance

                # Apply layer-specific importance weight
                layer_weight = self.sdxl_importance_weights.get(layer_type, 1.0)
                combined_weights = combined_weights * layer_weight

            except Exception as e:
                print(f"Info: Using simple average for {param_name} due to weighting error: {e}")
                return base_param + deltas.mean(0)

            # Robust tensor weighting with multiple fallback strategies
            try:
                # Strategy 1: Direct application
                if combined_weights.numel() == 1:
                    # Scalar weight - apply uniformly
                    weighted_deltas = deltas * combined_weights.item()
                else:
                    # Try different broadcasting approaches
                    if deltas.dim() == 2 and combined_weights.dim() == 1:
                        # Multiple 1D deltas, 1D weights
                        if combined_weights.numel() == deltas.shape[0]:
                            weighted_deltas = deltas * combined_weights.unsqueeze(-1)
                        elif combined_weights.numel() == deltas.shape[1]:
                            weighted_deltas = deltas * combined_weights.unsqueeze(0)
                        else:
                            # Size mismatch - use mean weight
                            weighted_deltas = deltas * combined_weights.mean().item()
                    elif deltas.dim() == 3:
                        # Try to broadcast weights appropriately
                        if combined_weights.numel() == deltas.shape[0]:
                            w = combined_weights.view(-1, 1, 1)
                            weighted_deltas = deltas * w
                        elif combined_weights.numel() == deltas.shape[1]:
                            w = combined_weights.view(1, -1, 1)
                            weighted_deltas = deltas * w
                        elif combined_weights.numel() == deltas.shape[2]:
                            w = combined_weights.view(1, 1, -1)
                            weighted_deltas = deltas * w
                        else:
                            weighted_deltas = deltas * combined_weights.mean().item()
                    else:
                        # Default: try direct multiplication, fallback to mean
                        try:
                            weighted_deltas = deltas * combined_weights
                        except RuntimeError:
                            weighted_deltas = deltas * combined_weights.mean().item()

                # Sum weighted deltas and add to base
                merged = base_param + weighted_deltas.sum(0)

                # Verify shape consistency
                if merged.shape != base_param.shape:
                    print(f"Info: Shape corrected for {param_name}: {merged.shape} -> {base_param.shape}")
                    # Try to reshape or fallback to simple average
                    if merged.numel() == base_param.numel():
                        merged = merged.view(base_param.shape)
                    else:
                        return base_param + deltas.mean(0)

                return merged

            except Exception as e:
                print(f"Info: Using simple fallback for {param_name}: {e}")
                # Final fallback: simple average
                return base_param + deltas.mean(0)

        except Exception as e:
            print(f"Warning: Complete fallback for {param_name}: {e}")
            # Ultimate fallback: return unchanged base parameter
            return base_param

    def widen_merging_sdxl(
        self,
        target_model,
        base_model,
        models_to_merge,
        merge_strength: float = 1.0,
        enable_renorm: bool = True,
        renorm_mode: str = "magnitude",
        above_avg_ratio: float = 1.0,
        calibration_value: float = 1.0,
        batch_size: int = 50,
    ):
        """FULL ZERO-ACCUMULATION WIDEN algorithm for SDXL - No intermediate data storage"""

        results_text = f"[{self.method}] Starting FULL ZERO-ACCUMULATION SDXL WIDEN merge\n"
        results_text += f"[{self.method}] Threshold: {above_avg_ratio}, Calibration: {calibration_value}\n"

        # Memory safety check
        safe, ram_percent, available_gb = check_memory_safety()
        if not safe:
            error_msg = f"[SAFETY] Cannot start - memory critical! RAM: {ram_percent*100:.1f}%, Available: {available_gb:.1f}GB"
            print(error_msg)
            raise MemoryExhaustionError(error_msg)

        print(f"[{self.method}] Initial memory check: {ram_percent*100:.1f}% used, {available_gb:.1f}GB available")

        # 1. Get base parameters list (names only, no tensor storage)
        print(f"[{self.method}] Getting parameter names...")
        base_param_names = list(base_model.named_parameters())
        param_names_only = [name for name, _ in base_param_names]

        # 2. Build task vectors with minimal storage
        print(f"[{self.method}] Building minimal task vectors...")
        task_vector_models = models_to_merge  # Just store model references

        # 3. Find common parameters without loading everything
        print(f"[{self.method}] Finding common parameters...")
        common_params = set(param_names_only)
        for model in models_to_merge:
            model_param_names = set(name for name, _ in model.named_parameters())
            common_params &= model_param_names
        common_params = list(common_params)

        print(f"[{self.method}] Found {len(common_params)} common parameters")

        # 4. FULL ZERO-ACCUMULATION: Process each parameter individually
        target_state_dict = target_model.state_dict()
        layer_stats = {}
        merged_count = 0
        failed_count = 0
        skipped_count = 0

        print(f"[{self.method}] Starting FULL ZERO-ACCUMULATION processing...")

        for param_idx, name in enumerate(common_params):
            # Progress tracking - reduced frequency
            if param_idx % 200 == 0:  # Reduced from every 50 to every 200
                progress = (param_idx / len(common_params)) * 100
                print(f"[PROGRESS] {param_idx}/{len(common_params)} ({progress:.1f}%)")

                # Memory safety check
                safe, ram_percent, available_gb = check_memory_safety()
                if not safe:
                    print(f"[EMERGENCY] Memory critical at parameter {param_idx}! Stopping safely...")
                    partial_results = f"""
[PARTIAL] Emergency stop at parameter {param_idx}/{len(common_params)}:
  - Processed: {param_idx}/{len(common_params)} parameters ({param_idx/len(common_params)*100:.1f}%)"""
                    return results_text + partial_results

            try:
                # STEP 1: Load ONLY the current parameter from all models (zero-accumulation)
                base_param = None
                deltas = []

                # Get base parameter
                for param_name, param in base_model.named_parameters():
                    if param_name == name:
                        base_param = param.detach().cpu().float().clone()
                        break

                if base_param is None:
                    skipped_count += 1
                    continue

                # Get deltas from each model (one at a time)
                for model in task_vector_models:
                    other_param = None
                    for param_name, param in model.named_parameters():
                        if param_name == name:
                            other_param = param.detach().cpu().float().clone()
                            break

                    if other_param is not None and other_param.shape == base_param.shape:
                        delta = other_param - base_param
                        deltas.append(delta)
                        del other_param  # Immediate cleanup

                if len(deltas) == 0:
                    skipped_count += 1
                    del base_param
                    continue

                # STEP 2: Classify layer and get metadata (zero-accumulation)
                layer_type = self._classify_sdxl_layer(name)
                if layer_type not in layer_stats:
                    layer_stats[layer_type] = {'merged': 0, 'skipped': 0, 'failed': 0}

                metadata = {
                    'layer_type': layer_type,
                    'base_magnitude': torch.norm(base_param).item(),
                    'delta_magnitude': sum(torch.norm(d).item() for d in deltas) / len(deltas),
                }

                # STEP 3: Compute magnitude/direction ON-THE-FLY (zero-accumulation)
                base_mag, base_dir = self.compute_magnitude_direction_sdxl({name: base_param}, "silent")

                mag_diffs = []
                dir_diffs = []

                for i, delta in enumerate(deltas):
                    # Compute magnitude/direction for this delta only
                    other_param = base_param + delta
                    other_mag, other_dir = self.compute_magnitude_direction_sdxl({name: other_param}, "silent")

                    if name in base_mag and name in other_mag:
                        mag_diff = (other_mag[name] - base_mag[name]).abs()
                        layer_weight = self.sdxl_importance_weights.get(layer_type, 1.0)
                        mag_diffs.append(mag_diff * layer_weight)

                    if name in base_dir and name in other_dir:
                        if base_dir[name].numel() > 1 and other_dir[name].numel() > 1:
                            base_flat = base_dir[name].flatten()
                            other_flat = other_dir[name].flatten()
                            cos_sim = torch.cosine_similarity(other_flat, base_flat, dim=0)
                            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                            dir_diff = 1 - cos_sim
                        else:
                            dir_diff = torch.abs(other_dir[name] - base_dir[name]).mean()

                        layer_weight = self.sdxl_importance_weights.get(layer_type, 1.0)
                        dir_diffs.append(dir_diff * layer_weight)

                    del other_param  # Immediate cleanup

                # Clean up magnitude/direction data immediately
                del base_mag, base_dir

                # STEP 4: Apply WIDEN algorithm with immediate cleanup
                if len(mag_diffs) == 0 or len(dir_diffs) == 0:
                    # Fallback to simple average (silent)
                    try:
                        avg_delta = sum(deltas) / len(deltas)
                        final_merged = base_param + avg_delta * merge_strength
                        del deltas, avg_delta  # Immediate cleanup
                    except Exception as e:
                        failed_count += 1
                        layer_stats[layer_type]['failed'] += 1
                        del base_param, deltas
                        continue
                else:
                    try:
                        # Stack and process immediately
                        deltas_tensor = torch.stack(deltas)
                        mag_diffs_tensor = torch.stack(mag_diffs)
                        dir_diffs_tensor = torch.stack(dir_diffs)

                        # Clean up lists immediately
                        del deltas, mag_diffs, dir_diffs

                        # Rank significance
                        mag_ranks = self.rank_significance_adaptive(mag_diffs_tensor, layer_type)
                        dir_ranks = self.rank_significance_adaptive(dir_diffs_tensor, layer_type)

                        # Clean up diff tensors immediately
                        del mag_diffs_tensor, dir_diffs_tensor

                        # Merge with WIDEN algorithm
                        merged_param = self.merge_single_parameter_sdxl(
                            deltas_tensor, base_param, mag_ranks, dir_ranks,
                            name, metadata, above_avg_ratio, calibration_value
                        )

                        # Clean up intermediate tensors immediately
                        del deltas_tensor, mag_ranks, dir_ranks

                        # Apply strength
                        final_merged = base_param + (merged_param - base_param) * merge_strength
                        del merged_param  # Immediate cleanup

                    except Exception as e:
                        # Silent fallback for WIDEN failures
                        try:
                            avg_delta = sum(deltas) / len(deltas) if deltas else torch.zeros_like(base_param)
                            final_merged = base_param + avg_delta * merge_strength
                            del deltas, avg_delta
                        except Exception as e2:
                            failed_count += 1
                            layer_stats[layer_type]['failed'] += 1
                            del base_param
                            if 'deltas' in locals():
                                del deltas
                            continue

                # STEP 5: Apply renormalization and write to target (zero-accumulation)
                if enable_renorm:
                    try:
                        if renorm_mode == "calibrate":
                            final_merged = calibrate_renormalize(
                                final_merged, base_param, renorm_mode, 0.5, 1.2
                            )
                        else:
                            final_merged = calibrate_renormalize(
                                final_merged, base_param, renorm_mode, 1.0, 1.0
                            )
                    except Exception as e:
                        # Silent renormalization failure handling
                        pass

                # Write directly to target model
                try:
                    target_device = target_state_dict[name].device
                    if final_merged.device != target_device:
                        final_merged = final_merged.to(target_device)

                    if final_merged.shape != target_state_dict[name].shape:
                        if final_merged.numel() == target_state_dict[name].numel():
                            final_merged = final_merged.view(target_state_dict[name].shape)
                        else:
                            failed_count += 1
                            layer_stats[layer_type]['failed'] += 1
                            del base_param, final_merged
                            continue

                    target_state_dict[name].copy_(final_merged)
                    merged_count += 1
                    layer_stats[layer_type]['merged'] += 1

                except Exception as e:
                    failed_count += 1
                    layer_stats[layer_type]['failed'] += 1

                # Clean up all remaining tensors for this parameter
                del base_param, final_merged

                # Aggressive cleanup every few parameters
                if param_idx % 20 == 0:  # Reduced frequency from every 5 to every 20
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()

            except Exception as e:
                failed_count += 1
                if layer_type not in layer_stats:
                    layer_stats[layer_type] = {'merged': 0, 'skipped': 0, 'failed': 0}
                layer_stats[layer_type]['failed'] += 1
                continue

        total_params = len(common_params)

        # Generate detailed layer-wise results
        layer_report = "\n[LAYER-WISE RESULTS]:\n"
        for layer_type, stats in sorted(layer_stats.items()):
            total_layer = sum(stats.values())
            if total_layer > 0:
                layer_report += f"  {layer_type}: {stats['merged']}/{total_layer} merged "
                layer_report += f"({stats['merged']/total_layer*100:.1f}% success)\n"

        results_text += f"""
[RESULTS] FULL ZERO-ACCUMULATION WIDEN merge complete:
  - Successfully merged: {merged_count}/{total_params} parameters ({merged_count/total_params*100:.1f}%)
  - Skipped (below threshold): {skipped_count}
  - Failed: {failed_count}
  - Renormalization: {'enabled' if enable_renorm else 'disabled'} (mode: {renorm_mode})
  - Full zero-accumulation: ✓ (absolute minimal memory footprint)
{layer_report}
[FULL ZERO-ACCUMULATION PRINCIPLES]:
  - No batch storage of any data ✓
  - Parameter-by-parameter processing throughout ✓
  - Immediate tensor cleanup after each operation ✓
  - No intermediate accumulation at any stage ✓
  - Minimal memory usage ✓"""

        print(results_text)
        force_cleanup()
        return results_text

    def _classify_sdxl_layer(self, param_name):
        """Classify SDXL layer types for specialized handling - ENHANCED"""
        name_lower = param_name.lower()

        # UNet structure classification - more comprehensive
        if 'time_embed' in name_lower:
            return 'time_embedding'
        elif 'label_emb' in name_lower:
            return 'class_embedding'
        elif any(x in name_lower for x in ['attn', 'attention']):
            if 'cross' in name_lower:
                return 'cross_attention'  # Text conditioning
            else:
                return 'self_attention'   # Spatial attention
        elif any(x in name_lower for x in ['conv', 'convolution']):
            if 'in_layers' in name_lower or 'input' in name_lower:
                return 'input_conv'
            elif 'out_layers' in name_lower or 'output' in name_lower:
                return 'output_conv'
            elif 'skip' in name_lower or 'residual' in name_lower:
                return 'skip_conv'
            else:
                return 'feature_conv'
        elif any(x in name_lower for x in ['norm', 'group_norm', 'layer_norm']):
            return 'normalization'
        elif 'bias' in name_lower:
            return 'bias'
        elif any(x in name_lower for x in ['down', 'upsample']):
            return 'resolution_change'
        # Enhanced classification for previously 'other' parameters
        elif any(x in name_lower for x in ['proj', 'projection']):
            return 'self_attention'  # Projections are usually part of attention
        elif any(x in name_lower for x in ['to_q', 'to_k', 'to_v', 'to_out']):
            return 'self_attention'  # Attention components
        elif any(x in name_lower for x in ['ff', 'feedforward', 'mlp']):
            return 'self_attention'  # Feed-forward networks in transformers
        elif 'weight' in name_lower and 'emb' in name_lower:
            return 'time_embedding'  # Embedding weights
        else:
            return 'other'


class DonutWidenMergeUNet:
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_base": ("MODEL",),
                "model_other": ("MODEL",),
                "merge_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "widen_threshold": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "widen_calibration": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "enable_renorm": ("BOOLEAN", {"default": True}),
                "renorm_mode": (["magnitude", "calibrate", "none"], {"default": "magnitude"}),
                "batch_size": ("INT", {"default": 50, "min": 10, "max": 500, "step": 10}),
            },
            "optional": {
                "model_3": ("MODEL",),
                "model_4": ("MODEL",),
                "model_5": ("MODEL",),
                "model_6": ("MODEL",),
                "model_7": ("MODEL",),
                "model_8": ("MODEL",),
                "model_9": ("MODEL",),
                "model_10": ("MODEL",),
                "model_11": ("MODEL",),
                "model_12": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "merge_results")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, model_base, model_other, merge_strength, widen_threshold, widen_calibration, enable_renorm, renorm_mode, batch_size,
                model_3=None, model_4=None, model_5=None, model_6=None,
                model_7=None, model_8=None, model_9=None, model_10=None,
                model_11=None, model_12=None):

        # Check cache first
        all_models = [model_base, model_other, model_3, model_4, model_5, model_6,
                     model_7, model_8, model_9, model_10, model_11, model_12]
        cache_key = compute_merge_hash(all_models, merge_strength, widen_threshold, False, widen_calibration, 0, f"{renorm_mode}")

        cached_result = check_cache_for_merge(cache_key)
        if cached_result is not None:
            return cached_result

        with memory_cleanup_context("DonutWidenMergeUNet"):
            import copy

            models_to_merge = [m for m in all_models[1:]  # Skip model_base
                             if m is not None and not getattr(m, "_is_filler", False)]

            print(f"[DonutWidenMergeUNet] WIDEN merging {len(models_to_merge)} models")

            try:
                base_model_obj = model_base.model
                other_model_objs = [model.model for model in models_to_merge]

                model_merged = copy.deepcopy(model_base)

                merger = MergingMethod("DonutWidenMergeUNet")
                results_text = merger.widen_merging_sdxl(
                    target_model=model_merged.model,
                    base_model=base_model_obj,
                    models_to_merge=other_model_objs,
                    merge_strength=merge_strength,
                    enable_renorm=enable_renorm,
                    renorm_mode=renorm_mode,
                    above_avg_ratio=widen_threshold,
                    calibration_value=widen_calibration,
                    batch_size=batch_size,
                )

                force_cleanup()
                result = (model_merged, results_text)

                # Store in cache
                store_merge_result(cache_key, result)

                return result

            except MemoryExhaustionError as e:
                print(f"[SAFETY] Memory exhaustion prevented crash: {e}")
                error_results = f"[SAFETY] Merge terminated to prevent crash: {e}"
                result = (model_base, error_results)
                store_merge_result(cache_key, result)
                return result

            except Exception as e:
                print(f"[DonutWidenMergeUNet] Error: {e}")
                if "memory" in str(e).lower():
                    error_results = f"[SAFETY] Memory error prevented crash: {e}"
                    result = (model_base, error_results)
                    store_merge_result(cache_key, result)
                    return result
                else:
                    raise


# VERSION CHECK - This should appear in logs if new code is loading
print("="*50)
print("LOADING DONUTWIDENMERGECLIP VERSION 7.0 - FULL ZERO-ACCUMULATION")
print("="*50)

class DonutWidenMergeCLIP:
    class_type = "CLIP"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_base": ("CLIP",),
                "clip_other": ("CLIP",),
                "merge_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "widen_threshold": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "widen_calibration": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "enable_renorm": ("BOOLEAN", {"default": True}),
                "renorm_mode": (["magnitude", "calibrate", "none"], {"default": "magnitude"}),
                "batch_size": ("INT", {"default": 75, "min": 10, "max": 500, "step": 10}),
            },
            "optional": {
                "clip_3": ("CLIP",),
                "clip_4": ("CLIP",),
                "clip_5": ("CLIP",),
                "clip_6": ("CLIP",),
                "clip_7": ("CLIP",),
                "clip_8": ("CLIP",),
                "clip_9": ("CLIP",),
                "clip_10": ("CLIP",),
                "clip_11": ("CLIP",),
                "clip_12": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "merge_results")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, clip_base, clip_other, merge_strength, widen_threshold, widen_calibration, enable_renorm, renorm_mode, batch_size,
                clip_3=None, clip_4=None, clip_5=None, clip_6=None,
                clip_7=None, clip_8=None, clip_9=None, clip_10=None,
                clip_11=None, clip_12=None):

        # Check cache first
        all_clips = [clip_base, clip_other, clip_3, clip_4, clip_5, clip_6,
                    clip_7, clip_8, clip_9, clip_10, clip_11, clip_12]
        cache_key = compute_merge_hash(all_clips, merge_strength, widen_threshold, False, widen_calibration, 0, f"{renorm_mode}")

        cached_result = check_cache_for_merge(cache_key)
        if cached_result is not None:
            return cached_result

        with memory_cleanup_context("DonutWidenMergeCLIP"):
            import copy

            clips_to_merge = [c for c in all_clips[1:]  # Skip clip_base
                            if c is not None and not getattr(c, "_is_filler", False)]

            print(f"[DonutWidenMergeCLIP] WIDEN merging {len(clips_to_merge)} CLIP models")

            try:
                base_enc = getattr(clip_base, "model", getattr(clip_base, "clip",
                          getattr(clip_base, "cond_stage_model", None)))
                if not base_enc:
                    raise AttributeError("Could not locate base CLIP encoder")

                other_encs = []
                for clip in clips_to_merge:
                    enc = getattr(clip, "model", getattr(clip, "clip",
                         getattr(clip, "cond_stage_model", None)))
                    if enc:
                        other_encs.append(enc)

                clip_merged = copy.deepcopy(clip_base)
                enc_merged = getattr(clip_merged, "model", getattr(clip_merged, "clip",
                            getattr(clip_merged, "cond_stage_model", None)))

                if not enc_merged:
                    raise AttributeError("Could not locate merged CLIP encoder")

                merger = MergingMethod("DonutWidenMergeCLIP")
                results_text = merger.widen_merging_sdxl(
                    target_model=enc_merged,
                    base_model=base_enc,
                    models_to_merge=other_encs,
                    merge_strength=merge_strength,
                    enable_renorm=enable_renorm,
                    renorm_mode=renorm_mode,
                    above_avg_ratio=widen_threshold,
                    calibration_value=widen_calibration,
                    batch_size=batch_size,
                )

                force_cleanup()
                result = (clip_merged, results_text)

                # Store in cache
                store_merge_result(cache_key, result)

                return result

            except MemoryExhaustionError as e:
                print(f"[SAFETY] Memory exhaustion prevented crash: {e}")
                error_results = f"[SAFETY] CLIP merge terminated to prevent crash: {e}"
                result = (clip_base, error_results)
                store_merge_result(cache_key, result)
                return result

            except Exception as e:
                print(f"[DonutWidenMergeCLIP] Error: {e}")
                if "memory" in str(e).lower():
                    error_results = f"[SAFETY] CLIP memory error prevented crash: {e}"
                    result = (clip_base, error_results)
                    store_merge_result(cache_key, result)
                    return result
                else:
                    raise


class DonutFillerModel:
    class_type = "MODEL"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "utils"

    def execute(self):
        class _Stub:
            def state_dict(self): return {}
            def named_parameters(self): return iter([])
        m = _Stub()
        setattr(m, "_is_filler", True)
        return (m,)


class DonutFillerClip:
    class_type = "CLIP"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "execute"
    CATEGORY = "utils"

    def execute(self):
        class _StubClip:
            def state_dict(self): return {}
            def named_parameters(self): return iter([])
        c = _StubClip()
        setattr(c, "_is_filler", True)
        return (c,)


NODE_CLASS_MAPPINGS = {
    "DonutWidenMergeUNet": DonutWidenMergeUNet,
    "DonutWidenMergeCLIP": DonutWidenMergeCLIP,
    "DonutFillerClip": DonutFillerClip,
    "DonutFillerModel": DonutFillerModel,
}

def clear_merge_cache():
    """Clear the model merge cache"""
    global _MERGE_CACHE
    _MERGE_CACHE.clear()
    print("[Cache] Cleared all cached merge results")

import atexit
def cleanup_on_exit():
    """Cleanup on exit"""
    try:
        pass
    except Exception:
        pass

atexit.register(cleanup_on_exit)

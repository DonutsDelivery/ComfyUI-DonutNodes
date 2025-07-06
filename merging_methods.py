import re
from collections import defaultdict
from tqdm import tqdm
import traceback
import torch
import torch.nn as nn

from .task_vector import TaskVector
from .mask_weights_utils import mask_model_weights
from .utils.utils import get_param_names_to_merge


class MergingMethod:
    def __init__(self, merging_method_name: str, vram_limit_bytes: int = None):
        """
        merging_method_name: name used in logs
        vram_limit_bytes: if set, will only merge on GPU when free VRAM ≥ this
        """
        self.method      = merging_method_name
        self.vram_limit  = vram_limit_bytes
        
        # SDXL layer thresholds for parameter merging
        self.sdxl_thresholds = {
            'time_embedding': 0.1,        # MUCH higher threshold for critical layers
            'class_embedding': 0.1,       # MUCH higher threshold for critical layers
            'cross_attention': 0.05,      # Higher threshold for attention
            'self_attention': 0.05,       # Higher threshold for attention
            'input_conv': 0.05,           # Higher threshold for convolutions
            'output_conv': 0.1,           # High threshold for output layers
            'feature_conv': 0.05,         # Higher threshold for convolutions
            'skip_conv': 0.03,            # Moderate threshold for skip connections
            'resolution_change': 0.05,    # Higher threshold
            'normalization': 0.02,        # Moderate threshold for normalization
            'bias': 0.01,                 # Higher threshold for bias
            'other': 0.05                 # Higher threshold for unclassified layers
        }
        
        # SDXL importance weights for layer-specific processing
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

    def _choose_device(self):
        if torch.cuda.is_available() and self.vram_limit is not None:
            dev = torch.cuda.current_device()
            free, _ = torch.cuda.mem_get_info(dev)
            if free >= self.vram_limit:
                return torch.device("cuda")
        return torch.device("cpu")

    def widen_merging(
        self,
        merged_model: nn.Module,
        models_to_merge: list,
        exclude_param_names_regex: list,
        above_average_value_ratio: float = 1.0,
        score_calibration_value: float = 1.0,
    ):
        """
        Widen merging with automatic CPU/GPU offload and per-param fallbacks.
        Returns merged parameter dict.
        """
        device = self._choose_device()
        print(f"[{self.method}] merging on {device}")

        # 1) snapshot original parameters on CPU/float32
        pre_params = {n: p.detach().cpu().float().clone()
                      for n, p in merged_model.named_parameters()}
        finetuned_dicts = [
            {n: p.detach().cpu().float().clone() for n, p in m.named_parameters()}
            for m in models_to_merge
        ]

        # 2) transpose token embeddings if present
        def transpose_tok(d):
            if "model.embed_tokens.weight" in d:
                d["model.embed_tokens.weight"] = d["model.embed_tokens.weight"].T
        transpose_tok(pre_params)
        for d in finetuned_dicts:
            transpose_tok(d)

        # 3) build TaskVectors to extract deltas
        task_vectors = [
            TaskVector(merged_model, m, exclude_param_names_regex)
            for m in models_to_merge
        ]

        # 4) compute magnitude & direction for each param
        def compute_mag_dir(param_dict, desc):
            mags, dirs = {}, {}
            for name, tensor in tqdm(param_dict.items(), desc=desc):
                try:
                    if tensor.dim() not in (2,4):  # only 2D or conv4D
                        continue
                    if tensor.dim() == 4:
                        o,c,h,w = tensor.shape
                        flat = tensor.view(o, -1)
                    else:
                        flat = tensor
                    mag = flat.norm(dim=0)
                    dir = flat / (mag.unsqueeze(0) + 1e-8)
                    dirs[name] = dir.view(tensor.shape) if tensor.dim()==4 else dir
                    mags[name] = mag
                except Exception:
                    continue
            return mags, dirs

        pre_mag, pre_dir = compute_mag_dir(pre_params,   "[mag/dir] pretrained")
        diff_list = []
        for fin in finetuned_dicts:
            fin_mag, fin_dir = compute_mag_dir(fin, "[mag/dir] finetuned")
            mag_diff = {k:(fin_mag[k] - pre_mag[k]).abs() for k in pre_mag if k in fin_mag}
            dir_diff = {k:1 - torch.cosine_similarity(fin_dir[k], pre_dir[k], dim=0)
                        for k in pre_dir if k in fin_dir}
            diff_list.append((mag_diff, dir_diff))

        # 5) helper to rank & score
        def rank_sig(diff: torch.Tensor):
            if diff.ndim == 1:
                # Handle single model case - convert to 2D
                diff = diff.unsqueeze(0)
            elif diff.ndim != 2:
                raise IndexError(f"rank_sig expects 1D or 2D tensor, got {diff.ndim}D")
            
            n,dim = diff.shape
            flat  = diff.reshape(n,-1)
            idx   = torch.argsort(flat, dim=1)
            L     = flat.shape[1]
            sig   = torch.arange(L, device=flat.device)/L
            base  = sig.unsqueeze(0).repeat(n,1)
            return base.scatter(1, idx, sig)

        def importance(sig: torch.Tensor):
            sc = torch.softmax(sig, dim=0)
            avg= sig.mean(1, keepdim=True)
            mask = sig > avg * above_average_value_ratio
            sc[mask] = score_calibration_value
            return sc

        def merge_param(delta, base, mag_rank, dir_rank):
            try:
                ms  = importance(mag_rank)
                ds  = importance(dir_rank)
                w   = 0.5 * (ms + ds)
                if delta.dim() == 3:
                    w = w.unsqueeze(1)
                elif delta.dim() == 5:
                    n,dim = w.shape
                    w = w.view(n,1,dim,1,1)
                else:
                    shape = [1]*(delta.dim()-2)+[w.shape[1]]
                    w = w.view(delta.shape[0], *shape)
                merged = base + (delta * w).sum(0)
                return merged if merged.shape == base.shape else base
            except Exception:
                return base

        merged_params = {}
        fell_back = 0
        common = set(pre_mag.keys())
        for tv in task_vectors:
            common &= set(tv.task_vector_param_dict.keys())

        for name in tqdm(common, desc=f"[{self.method}] merging"):
            try:
                delta = torch.stack([tv.task_vector_param_dict[name] for tv in task_vectors])
                magd  = torch.stack([d[0][name] for d in diff_list])
                dird  = torch.stack([d[1][name] for d in diff_list])
                rankm = rank_sig(magd)
                rankd = rank_sig(dird)
            except Exception:
                merged_params[name] = pre_params[name]
                fell_back += 1
                continue
            merged = merge_param(delta, pre_params[name], rankm, rankd)
            merged_params[name] = merged
            if torch.allclose(merged, pre_params[name]):
                fell_back += 1

        total = len(common)
        print(f"[{self.method}] merged {total - fell_back} / {total} parameters")
        
        return merged_params

    def should_merge_parameter(self, param_name, delta_magnitude, metadata, widen_threshold=0.5):
        """Determine if parameter should be merged based on SDXL-specific criteria - FIXED: More aggressive threshold logic"""
        layer_type = metadata.get('layer_type', 'other')
        base_threshold = self.sdxl_thresholds.get(layer_type, 0.0001)

        # FIXED: Much more aggressive exponential scaling
        # 0.0 -> 0.001x (extremely permissive)
        # 0.5 -> 1.0x (standard)
        # 1.0 -> 1000x (extremely selective)
        if widen_threshold <= 0.5:
            # Permissive range: 0.0-0.5 maps to 0.001x-1.0x
            threshold_multiplier = 0.001 + (widen_threshold * 2) ** 3 * 0.999
        else:
            # Selective range: 0.5-1.0 maps to 1.0x-1000x
            threshold_multiplier = 1.0 + ((widen_threshold - 0.5) * 2) ** 4 * 999

        threshold = base_threshold * threshold_multiplier

        # Additional criteria for SDXL
        change_ratio = metadata.get('change_ratio', 0)
        scaled_change_threshold = 0.0001 * threshold_multiplier

        # FIXED: More strict checking - both conditions must pass
        magnitude_check = delta_magnitude >= threshold
        ratio_check = change_ratio >= scaled_change_threshold

        if not magnitude_check or not ratio_check:
            return False

        # Special handling for critical layers - even more selective at high thresholds
        if layer_type in ['time_embedding', 'class_embedding']:
            critical_threshold = threshold * (0.5 + widen_threshold * 0.5)  # 0.5-1.0x multiplier
            return delta_magnitude > critical_threshold

        # Special handling for 'other' category - less lenient at high thresholds
        if layer_type == 'other':
            other_threshold = threshold * (0.3 + widen_threshold * 0.7)  # 0.3-1.0x multiplier
            return delta_magnitude > other_threshold

        return True

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
        """Enhanced ranking with SDXL layer-specific adaptations - BULLETPROOF - FIXED: Infinite loop prevention"""
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
            # FIXED: Handle 1D tensors specially to prevent infinite recursion
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

            # FIXED: Safe flattening strategy to prevent infinite recursion
            if diff_tensor.ndim == 2:
                # 2D tensor - use as-is or flatten to 1D if one dimension is 1
                if diff_tensor.shape[0] == 1:
                    flat = diff_tensor.view(-1)  # Use view instead of flatten to prevent recursion
                elif diff_tensor.shape[1] == 1:
                    flat = diff_tensor.view(-1)  # Use view instead of flatten to prevent recursion
                else:
                    flat = diff_tensor
            else:
                # Higher dimensional tensors - flatten carefully
                try:
                    if layer_type in ['cross_attention', 'self_attention']:
                        # For attention: try to preserve structure
                        if diff_tensor.ndim > 2:
                            flat = diff_tensor.view(diff_tensor.shape[0], -1)  # Use view instead of flatten
                        else:
                            flat = diff_tensor
                    else:
                        # For other types: safe flattening
                        if diff_tensor.ndim > 2:
                            flat = diff_tensor.view(diff_tensor.shape[0], -1)  # Use view instead of flatten
                        else:
                            flat = diff_tensor
                except Exception:
                    # Fallback: complete flattening using view
                    flat = diff_tensor.view(-1)

            # Ensure we have a valid tensor for ranking
            if flat is None:
                flat = diff_tensor.view(-1)  # Use view instead of flatten

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

    def compute_importance_sdxl(self, sig_tensor, layer_type='other', widen_threshold=0.5, calibration_value=0.0):
        """SDXL-optimized importance computation following WIDEN principles - FIXED: 0-1 calibration range"""

        # Debug: Log tensor properties for first few calls
        debug_scores = False
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:  # Log first 3 calls
            self._debug_count += 1
            debug_scores = True
        
        if debug_scores:
            print(f"[SDXL SCORE DEBUG] Input tensor shape: {sig_tensor.shape}, layer: {layer_type}")
            print(f"[SDXL SCORE DEBUG] Input tensor min/max: {sig_tensor.min().item():.6f}/{sig_tensor.max().item():.6f}")
            print(f"[SDXL SCORE DEBUG] widen_threshold: {widen_threshold}, calibration_value: {calibration_value}")

        try:
            # Handle edge cases
            if sig_tensor.numel() == 0:
                return torch.tensor(1.0, dtype=torch.float32, device=sig_tensor.device)

            # Layer-specific importance weighting
            layer_weight = self.sdxl_importance_weights.get(layer_type, 1.0)

            # Handle scalar tensors
            if sig_tensor.numel() == 1:
                # FIXED: Map 0-1 calibration to meaningful range (0.1-2.0)
                calibration_mapped = 0.1 + calibration_value * 1.9
                return torch.tensor(calibration_mapped * layer_weight, dtype=sig_tensor.dtype, device=sig_tensor.device)

            # Handle very small tensors
            if sig_tensor.numel() <= 2:
                calibration_mapped = 0.1 + calibration_value * 1.9
                return torch.full_like(sig_tensor, calibration_mapped * layer_weight)

            # Base softmax scoring with error handling
            try:
                if sig_tensor.ndim == 0:
                    calibration_mapped = 0.1 + calibration_value * 1.9
                    return torch.tensor(calibration_mapped * layer_weight, dtype=sig_tensor.dtype, device=sig_tensor.device)
                
                # CRITICAL FIX: Same softmax dimension logic as CLIP fix
                # For single model (shape [1, features]), softmax across features (dim=1)
                # For multiple models (shape [models, features]), softmax across models (dim=0)
                if sig_tensor.shape[0] == 1:
                    softmax_dim = 1  # Single model: softmax across features
                    if debug_scores:
                        print(f"[SDXL SOFTMAX FIX] Single model detected, using dim=1")
                else:
                    softmax_dim = 0  # Multiple models: softmax across models  
                    if debug_scores:
                        print(f"[SDXL SOFTMAX FIX] Multiple models detected, using dim=0")

                # Apply softmax with numerical stability
                sig_scaled = sig_tensor * layer_weight
                # Clamp to prevent overflow
                sig_scaled = torch.clamp(sig_scaled, min=-50, max=50)
                sc = torch.softmax(sig_scaled, dim=softmax_dim)
                
                if debug_scores:
                    print(f"[SDXL SOFTMAX FIX] Input shape: {sig_tensor.shape}, softmax_dim: {softmax_dim}")
                    print(f"[SDXL SOFTMAX FIX] Softmax output sample: {sc.flatten()[:10]}")

            except Exception as e:
                print(f"Warning: Softmax failed for tensor shape {sig_tensor.shape}: {e}")
                calibration_mapped = 0.1 + calibration_value * 1.9
                return torch.full_like(sig_tensor, calibration_mapped * layer_weight)

            # FIXED: Much more aggressive adaptive thresholding
            try:
                if sig_tensor.ndim > 1:
                    avg = sig_tensor.mean(0, keepdim=True)
                else:
                    avg = sig_tensor.mean()

                # FIXED: Use same aggressive scaling as should_merge_parameter
                if widen_threshold <= 0.5:
                    # Permissive range: 0.0-0.5 maps to 0.1x-1.0x
                    threshold_multiplier = 0.1 + (widen_threshold * 2) ** 2 * 0.9
                else:
                    # Selective range: 0.5-1.0 maps to 1.0x-100x
                    threshold_multiplier = 1.0 + ((widen_threshold - 0.5) * 2) ** 3 * 99

                # Layer-specific threshold adjustment
                if layer_type in ['time_embedding', 'class_embedding', 'cross_attention']:
                    # More selective for critical layers
                    adjusted_multiplier = threshold_multiplier * 1.2
                elif layer_type in ['normalization', 'bias']:
                    # Less selective for less critical layers
                    adjusted_multiplier = threshold_multiplier * 0.8
                else:
                    adjusted_multiplier = threshold_multiplier

                mask = sig_tensor > avg * adjusted_multiplier
                mask_ratio = mask.float().mean().item()

                # FIXED: Apply calibration with 0-1 mapping to 0.1-2.0 range
                # 0.0 = minimal importance weighting (0.1x)
                # 0.5 = standard importance weighting (1.0x)
                # 1.0 = maximum importance weighting (2.0x)
                calibration_mapped = 0.1 + calibration_value * 1.9
                calibration_scaled = calibration_mapped * layer_weight
                
                # CRITICAL FIX: Apply calibration as multiplicative factor to preserve variation
                if mask_ratio < 0.95:  # Less than 95% of values above threshold
                    # Apply calibration as boost factor while preserving relative differences
                    sc = torch.where(mask, sc * calibration_scaled, sc)
                    mask_applied = True
                else:
                    mask_applied = False
                    # keep softmax scores to preserve variation

                if debug_scores:
                    print(f"[SDXL SCORE DEBUG] Mask ratio: {mask_ratio:.3f}, applied: {mask_applied}")
                    print(f"[SDXL SCORE DEBUG] Final scores min/max: {sc.min().item():.6f}/{sc.max().item():.6f}")

                return sc

            except Exception as e:
                print(f"Warning: Thresholding failed for tensor shape {sig_tensor.shape}: {e}")
                calibration_mapped = 0.1 + calibration_value * 1.9
                return torch.full_like(sig_tensor, calibration_mapped * layer_weight)

        except Exception as e:
            print(f"Warning: Importance computation completely failed: {e}")
            # Ultimate fallback
            return torch.tensor(1.0, dtype=torch.float32, device=sig_tensor.device if hasattr(sig_tensor, 'device') else 'cpu')

    def merge_single_parameter_sdxl(self, deltas, base_param, mag_ranks, dir_ranks,
                                   param_name, metadata, widen_threshold=0.5, calibration_value=0.0):
        """SDXL-optimized parameter merging with layer-aware weighting"""
        try:
            layer_type = metadata.get('layer_type', 'other')

            # More robust delta magnitude calculation
            if len(deltas) == 0:
                return base_param

            # Calculate average magnitude safely
            total_norm = sum(torch.norm(delta).item() for delta in deltas if delta.numel() > 0)
            delta_mag = total_norm / max(len(deltas), 1)

            if not self.should_merge_parameter(param_name, delta_mag, metadata, widen_threshold):
                return base_param

            # Compute importance scores with comprehensive error handling
            try:
                mag_importance = self.compute_importance_sdxl(mag_ranks, layer_type, widen_threshold, calibration_value)
                dir_importance = self.compute_importance_sdxl(dir_ranks, layer_type, widen_threshold, calibration_value)
            except Exception as e:
                print(f"Warning: Failed to compute importance for {param_name}: {e}")
                # Fallback: use simple average instead of failing completely
                if hasattr(deltas, 'mean'):
                    return base_param + deltas.mean(0)
                else:
                    avg_delta = sum(deltas) / len(deltas)
                    return base_param + avg_delta

            # Robust importance combination with proper shape validation
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
                    # Better fallback - use scalar weights instead of tensor operations
                    mag_scalar = mag_importance.mean().item()
                    dir_scalar = dir_importance.mean().item()
                    combined_weights = 0.5 * (mag_scalar + dir_scalar)
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

                # Ensure combined_weights is always scalar when multiplying with layer_weight
                if hasattr(combined_weights, 'numel') and combined_weights.numel() > 1:
                    combined_weights = combined_weights.mean() * layer_weight
                else:
                    if hasattr(combined_weights, 'item'):
                        combined_weights = combined_weights.item() * layer_weight
                    else:
                        combined_weights = combined_weights * layer_weight

            except Exception as e:
                print(f"Info: Using simple average for {param_name} due to weighting error: {e}")
                if hasattr(deltas, 'mean'):
                    return base_param + deltas.mean(0)
                else:
                    avg_delta = sum(deltas) / len(deltas)
                    return base_param + avg_delta

            # Simplified tensor weighting - always use scalar weights to avoid shape issues
            try:
                # Ensure we always have a scalar weight
                if hasattr(combined_weights, 'item'):
                    weight_scalar = combined_weights.item()
                elif hasattr(combined_weights, '__len__') and len(combined_weights) > 1:
                    weight_scalar = float(torch.tensor(combined_weights).mean().item())
                else:
                    weight_scalar = float(combined_weights)

                # Apply scalar weight to deltas - much simpler and more reliable
                if hasattr(deltas, 'shape'):  # It's a tensor
                    weighted_deltas = deltas * weight_scalar
                else:  # It's a list
                    weighted_deltas = torch.stack([delta * weight_scalar for delta in deltas])

                # Sum weighted deltas and add to base
                merged = base_param + weighted_deltas.sum(0)

                # Verify shape consistency more robustly
                if merged.shape != base_param.shape:
                    print(f"Info: Shape corrected for {param_name}: {merged.shape} -> {base_param.shape}")
                    # Try to reshape or fallback to simple average
                    if merged.numel() == base_param.numel():
                        merged = merged.view(base_param.shape)
                    else:
                        # Clean up failed merged tensor before fallback
                        del merged, weighted_deltas
                        if hasattr(deltas, 'mean'):
                            return base_param + deltas.mean(0)
                        else:
                            avg_delta = sum(deltas) / len(deltas)
                            return base_param + avg_delta

                # Clean up weighted_deltas immediately after use
                del weighted_deltas
                return merged

            except Exception as e:
                print(f"Info: Using simple fallback for {param_name}: {e}")
                # Simple cleanup without complex variable checking
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # Final fallback: simple average
                if hasattr(deltas, 'mean'):
                    return base_param + deltas.mean(0)
                else:
                    avg_delta = sum(deltas) / len(deltas)
                    return base_param + avg_delta

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
        renorm_mode: str = "magnitude",
        widen_threshold: float = 0.5,
        calibration_value: float = 0.0,
        batch_size: int = 50,
    ):
        """Simplified WIDEN merging for SDXL - fallback implementation"""
        
        print(f"[{self.method}] Starting fallback WIDEN merge")
        print(f"[{self.method}] Threshold: {widen_threshold}, Calibration: {calibration_value}")
        
        # Get parameter names
        base_params = dict(base_model.named_parameters())
        target_state_dict = target_model.state_dict()
        
        # Find common parameters
        common_params = set(base_params.keys())
        for model in models_to_merge:
            model_param_names = set(name for name, _ in model.named_parameters())
            common_params &= model_param_names
        common_params = list(common_params)
        
        print(f"[{self.method}] Processing {len(common_params)} common parameters")
        
        merged_count = 0
        skipped_count = 0
        failed_count = 0
        
        for param_name in common_params:
            try:
                # Get base parameter
                base_param = base_params[param_name].detach().cpu().float()
                
                # Get deltas from other models
                deltas = []
                for model in models_to_merge:
                    model_params = dict(model.named_parameters())
                    if param_name in model_params:
                        other_param = model_params[param_name].detach().cpu().float()
                        if other_param.shape == base_param.shape:
                            delta = other_param - base_param
                            deltas.append(delta)
                
                if not deltas:
                    skipped_count += 1
                    continue
                
                # Simple merging with thresholding
                layer_type = self._classify_sdxl_layer(param_name)
                avg_delta_magnitude = sum(torch.norm(d).item() for d in deltas) / len(deltas)
                
                metadata = {
                    'layer_type': layer_type,
                    'delta_magnitude': avg_delta_magnitude,
                    'change_ratio': avg_delta_magnitude / (torch.norm(base_param).item() + 1e-8)
                }
                
                # Apply thresholding
                if not self.should_merge_parameter(param_name, avg_delta_magnitude, metadata, widen_threshold):
                    skipped_count += 1
                    continue
                
                # Simple average merge with strength
                avg_delta = sum(deltas) / len(deltas)
                merged_param = base_param + avg_delta * merge_strength
                
                # Write to target
                target_device = target_state_dict[param_name].device
                if merged_param.device != target_device:
                    merged_param = merged_param.to(target_device)
                    
                if merged_param.shape == target_state_dict[param_name].shape:
                    target_state_dict[param_name].copy_(merged_param)
                    merged_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                continue
        
        total_params = len(common_params)
        results_text = f"""
[{self.method}] Fallback WIDEN merge complete:
  - Successfully merged: {merged_count}/{total_params} parameters ({merged_count/total_params*100:.1f}%)
  - Skipped (below threshold): {skipped_count} ({skipped_count/total_params*100:.1f}%)
  - Failed: {failed_count}
  - Threshold: {widen_threshold}"""
        
        print(results_text)
        return results_text

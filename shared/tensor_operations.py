"""
Tensor Operations Module for DonutNodes

This module contains extracted tensor manipulation functions from DonutWidenMerge
for safe tensor multiplication, alignment, vectorized batch processing, and 
importance score computation. These functions handle complex tensor shape matching
and broadcasting scenarios while maintaining numerical stability and performance.

The functions in this module are designed to be JIT-compilable and optimized for
batch processing with torch.vmap where applicable.
"""

import torch
import torch.nn.functional as F
import numpy as np

# Import shared components
from .logging_config import diagnostic_logger
from .alignment import align_linear_layer, transpose_embeddings_if_needed

# Check for scipy availability
try:
    import scipy.optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Advanced tensor alignment functions for structure-preserving operations (from bd89e6c)
def align_and_stack(t1, t2, spatial_dim=None):
    """
    Align two tensors using adaptive pooling instead of flattening.
    Preserves per-feature structure while handling dimension mismatches.
    """
    if t1.shape == t2.shape:
        return torch.stack([t1, t2], dim=0)
    
    # Handle different tensor types with appropriate pooling
    if t1.ndim == 4 and t2.ndim == 4:  # Conv weights (N, C, H, W)
        # Pool spatial dimensions to the smaller size
        target_h = min(t1.size(2), t2.size(2))
        target_w = min(t1.size(3), t2.size(3))
        t1_pooled = F.adaptive_avg_pool2d(t1, (target_h, target_w))
        t2_pooled = F.adaptive_avg_pool2d(t2, (target_h, target_w))
        return torch.stack([t1_pooled, t2_pooled], dim=0)
    
    elif t1.ndim == 2 and t2.ndim == 2:  # Linear weights (in, out)
        # Pool to the smaller dimension in each axis
        target_shape = (min(t1.size(0), t2.size(0)), min(t1.size(1), t2.size(1)))
        t1_pooled = F.adaptive_avg_pool1d(t1.unsqueeze(0), target_shape[1]).squeeze(0)
        t2_pooled = F.adaptive_avg_pool1d(t2.unsqueeze(0), target_shape[1]).squeeze(0)
        
        # Handle first dimension
        if t1_pooled.size(0) > target_shape[0]:
            t1_pooled = t1_pooled[:target_shape[0]]
        if t2_pooled.size(0) > target_shape[0]:
            t2_pooled = t2_pooled[:target_shape[0]]
            
        return torch.stack([t1_pooled, t2_pooled], dim=0)
    
    # Fallback: use broadcasting if possible
    try:
        bcast = torch.broadcast_tensors(t1, t2)
        return torch.stack(bcast, dim=0)
    except RuntimeError:
        # Last resort: pool to scalars but preserve semantic meaning
        return torch.stack([t1.mean(), t2.mean()], dim=0)


def safe_stack(tensors, dim=0):
    """
    Stack tensors safely using broadcast-aware operations.
    """
    if not tensors:
        return None
    
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    
    # Check if all tensors have the same shape
    shapes = [t.shape for t in tensors]
    if all(s == shapes[0] for s in shapes):
        return torch.stack(tensors, dim)
    
    # Try broadcasting first
    try:
        bcast = torch.broadcast_tensors(*tensors)
        return torch.stack(bcast, dim)
    except RuntimeError:
        # Use adaptive alignment
        return align_tensors(tensors, dim)


def align_tensors(tensors, stack_dim=0):
    """
    Align tensors with different shapes using optimized einsum-style patterns.
    Uses efficient tensor operations and vectorized pooling for better performance.
    """
    if not tensors:
        return None
    
    # Find target shape by taking minimum in each dimension (conservative pooling)
    shapes = [t.shape for t in tensors]
    ndims = [len(s) for s in shapes]
    
    # Handle tensors with different number of dimensions
    if not all(nd == ndims[0] for nd in ndims):
        # Pad smaller tensors with singleton dimensions
        max_ndim = max(ndims)
        padded_tensors = []
        for t in tensors:
            while t.ndim < max_ndim:
                t = t.unsqueeze(-1)
            padded_tensors.append(t)
        tensors = padded_tensors
        shapes = [t.shape for t in tensors]
    
    # Compute target shape (minimum size in each dimension)
    target_shape = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))
    
    aligned = []
    for tensor in tensors:
        aligned_tensor = tensor
        
        # Handle 4D tensors (conv weights) with adaptive pooling
        if tensor.ndim == 4 and (tensor.shape[2] > target_shape[2] or tensor.shape[3] > target_shape[3]):
            aligned_tensor = F.adaptive_avg_pool2d(aligned_tensor, target_shape[2:4])
        
        # Handle 2D tensors (linear weights) with optimized einsum-style pooling
        elif tensor.ndim == 2:
            h, w = aligned_tensor.shape
            target_h, target_w = target_shape[0], target_shape[1]
            
            # Optimized pooling using einsum patterns for better memory efficiency
            if h > target_h:
                # Pool along first dimension using vectorized operations
                pool_factor = h // target_h
                remainder = h % target_h
                
                if remainder == 0:
                    # Perfect division - use einsum for efficient reshape + mean
                    # einsum 'hpw->hw' where h=target_h, p=pool_factor, w=width
                    aligned_tensor = torch.einsum('hpw->hw', 
                        aligned_tensor[:target_h * pool_factor].view(target_h, pool_factor, w))
                else:
                    # Handle remainder with efficient concatenation
                    main_part = torch.einsum('hpw->hw',
                        aligned_tensor[:target_h * pool_factor].view(target_h, pool_factor, w))
                    remainder_part = aligned_tensor[target_h * pool_factor:].mean(dim=0, keepdim=True)
                    aligned_tensor = torch.cat([main_part, remainder_part], dim=0)[:target_h]
            
            if aligned_tensor.size(1) > target_w:
                # Pool along second dimension using vectorized operations
                h_curr, w_curr = aligned_tensor.shape
                pool_factor = w_curr // target_w
                remainder = w_curr % target_w
                
                if remainder == 0:
                    # Perfect division - use einsum for efficient reshape + mean
                    # einsum 'hwp->hw' where h=height, w=target_w, p=pool_factor
                    aligned_tensor = torch.einsum('hwp->hw',
                        aligned_tensor[:, :target_w * pool_factor].view(h_curr, target_w, pool_factor))
                else:
                    # Handle remainder with efficient concatenation
                    main_part = torch.einsum('hwp->hw',
                        aligned_tensor[:, :target_w * pool_factor].view(h_curr, target_w, pool_factor))
                    remainder_part = aligned_tensor[:, target_w * pool_factor:].mean(dim=1, keepdim=True)
                    aligned_tensor = torch.cat([main_part, remainder_part], dim=1)[:, :target_w]
        
        # Handle other dimensions with optimized slicing and padding
        else:
            # Efficient slicing using advanced indexing
            if aligned_tensor.shape != target_shape:
                # Create slice indices once and reuse
                slices = tuple(slice(0, min(aligned_tensor.size(i), target_shape[i])) 
                             for i in range(aligned_tensor.ndim))
                aligned_tensor = aligned_tensor[slices]
                
                # Vectorized padding calculation
                current_shape = aligned_tensor.shape
                pad_needed = any(current_shape[i] < target_shape[i] for i in range(len(current_shape)))
                
                if pad_needed:
                    # Compute padding in one pass using vectorized calculation
                    pad_widths = []
                    for i in range(aligned_tensor.ndim - 1, -1, -1):
                        pad_widths.extend([0, max(0, target_shape[i] - current_shape[i])])
                    
                    aligned_tensor = F.pad(aligned_tensor, pad_widths, mode='constant', value=0)
        
        aligned.append(aligned_tensor)
    
    # Use efficient stacking
    return torch.stack(aligned, dim=stack_dim)


# Core tensor multiplication with robust axis-matching (from bd89e6c)
def _safe_tensor_multiply_core(models_to_merge_delta_param: torch.Tensor, weight_scores: torch.Tensor) -> torch.Tensor:
    """
    Multiply delta_param by weight_scores using sophisticated tensor alignment for robust dimension handling.
    This implements the robust axis-matching logic from bd89e6c that made it better than the original.
    """
    # Use sophisticated tensor alignment for robust dimension handling
    if models_to_merge_delta_param.dim() == weight_scores.dim():
        # Same dimensions - direct multiplication preserves structure
        return models_to_merge_delta_param * weight_scores
    elif models_to_merge_delta_param.dim() > weight_scores.dim():
        # Delta tensor has more dimensions - use PyTorch broadcasting
        # This preserves the parameter's structural information
        try:
            # Try PyTorch's native broadcasting first
            weighted_deltas = models_to_merge_delta_param * weight_scores.view(
                *weight_scores.shape, *([1] * (models_to_merge_delta_param.dim() - weight_scores.dim()))
            )
            return weighted_deltas
        except RuntimeError:
            # If broadcasting fails, use adaptive alignment
            # Robust axis-matching approach: find where weight_scores shape matches delta tensor
            S = tuple(models_to_merge_delta_param.shape)
            R = tuple(weight_scores.shape)
            
            # Find the axis where R matches a slice of S
            for axis in range(len(S) - len(R) + 1):
                if S[axis:axis+len(R)] == R:
                    # Insert singleton dimensions to align properly
                    aligned_weights = weight_scores.view(
                        *([1] * axis), *R, *([1] * (len(S) - axis - len(R)))
                    )
                    return models_to_merge_delta_param * aligned_weights
            
            # Final fallback: broadcast weight_scores along the last dims  
            extra = models_to_merge_delta_param.dim() - weight_scores.dim()
            aligned_weights = weight_scores.view(*([1] * extra), *weight_scores.shape)
            return models_to_merge_delta_param * aligned_weights
    else:
        # weight_scores has more dimensions than delta - unusual case
        # Try standard broadcasting
        try:
            return models_to_merge_delta_param * weight_scores
        except RuntimeError:
            # Fallback: reduce weight_scores to scalar
            return models_to_merge_delta_param * weight_scores.mean()


def _safe_tensor_multiply(models_to_merge_delta_param, weight_scores, param_name, failure_tracker=None):
    """
    Safely multiply delta tensors with weight scores, handling dimension mismatches.
    Returns weighted_deltas or None if operation fails.
    """
    try:
        # Apply neuron alignment for linear layers if appropriate
        if SCIPY_AVAILABLE and models_to_merge_delta_param.ndim == 2 and "weight" in param_name.lower():
            # Check if this looks like a linear layer (2D weight matrix)
            if models_to_merge_delta_param.shape[0] > 1 and models_to_merge_delta_param.shape[1] > 1:
                # For multi-model case, align each model to the first one
                if models_to_merge_delta_param.shape[0] > 1:  # Multiple models
                    aligned_deltas = models_to_merge_delta_param.clone()
                    base_delta = aligned_deltas[0]  # Use first model as reference
                    
                    for i in range(1, aligned_deltas.shape[0]):
                        aligned_deltas[i] = align_linear_layer(base_delta, aligned_deltas[i])
                    
                    models_to_merge_delta_param = aligned_deltas
        
        # Apply automatic embedding transpose if needed
        if "embed" in param_name.lower() and models_to_merge_delta_param.ndim >= 2:
            # Get hint for hidden size from tensor shape
            hidden_size_hint = models_to_merge_delta_param.shape[-1]
            models_to_merge_delta_param = transpose_embeddings_if_needed(
                param_name, models_to_merge_delta_param, hidden_size_hint
            )
        
        # First try the JIT-compiled core for maximum performance
        return _safe_tensor_multiply_core(models_to_merge_delta_param, weight_scores)
            
    except Exception as e:
        # Fallback: try to find matching dimensions for complex cases
        try:
            delta = models_to_merge_delta_param.to(torch.float32)
            weights = weight_scores.to(torch.float32)
            
            # Handle edge cases that JIT core might miss
            if weights.dim() == 1 and delta.dim() >= 2:
                # Look for a dimension that matches weights.size(0)
                for axis in range(delta.dim()):
                    if delta.size(axis) == weights.size(0):
                        # Create shape that broadcasts weights along this axis
                        shape = [1] * delta.dim()
                        shape[axis] = weights.size(0)
                        weights_shaped = weights.reshape(shape)
                        return delta * weights_shaped
                        
            # If no specific match found, try general broadcasting
            return delta * weights
                        
        except Exception as e2:
            # Final fallback to complex alignment
            try:
                return _align_and_multiply_tensors(models_to_merge_delta_param, weight_scores, param_name)
            except Exception as e3:
                # Record the failure if tracker is provided
                if failure_tracker:
                    failure_tracker.record_failure(param_name, e3, fallback_used=False)
                diagnostic_logger.warning(f"All tensor multiplication methods failed for {param_name}: {e}, {e2}, {e3}")
                return None


def _align_and_multiply_tensors(models_to_merge_delta_param, weight_scores, param_name):
    """Helper function for complex tensor alignment cases using structure-preserving operations"""
    
    # First, try to preserve structure by using mean to collapse extra dimensions
    delta = models_to_merge_delta_param
    weights = weight_scores
    
    try:
        # If delta has more than 2 dims, preserve first 2 dims and average the rest
        if delta.dim() > 2:
            # Preserve [out_features, in_features] structure, average spatial/extra dims
            condensed_delta = delta.mean(dim=tuple(range(2, delta.ndim)))
        else:
            condensed_delta = delta
        
        # Similarly condense weights if needed
        if weights.dim() > 2:
            condensed_weights = weights.mean(dim=tuple(range(2, weights.ndim)))
        else:
            condensed_weights = weights
        
        # Now try multiplication on the condensed tensors
        if condensed_delta.dim() == 2 and condensed_weights.dim() == 1:
            # Standard case: 2D delta with 1D weights
            if condensed_weights.size(0) == condensed_delta.size(0):
                # Per-output-channel weighting
                result = condensed_delta * condensed_weights.unsqueeze(1)
            elif condensed_weights.size(0) == condensed_delta.size(1):
                # Per-input-channel weighting
                result = condensed_delta * condensed_weights.unsqueeze(0)
            else:
                # Fallback: broadcast or use mean
                result = condensed_delta * condensed_weights.mean()
        else:
            # Try broadcasting the condensed tensors
            result = condensed_delta * condensed_weights
            
        # If original delta had spatial dims, expand result back with proper broadcasting
        if delta.dim() > 2 and result.dim() == 2:
            # Create the proper broadcast shape
            target_shape = list(delta.shape)
            # Result should broadcast to delta's shape
            result = result.view(target_shape[0], target_shape[1], *([1] * (delta.dim() - 2)))
            result = result.expand(delta.shape)
                
        return result
        
    except Exception as e:
        diagnostic_logger.debug(f"Structure-preserving alignment failed for {param_name}: {e}, using fallback")
        
        # Fallback to original logic for compatibility
        if models_to_merge_delta_param.dim() == 4:  # Conv weights
            target_shape = models_to_merge_delta_param.shape[2:]
            aligned_weights = weight_scores.unsqueeze(-1).unsqueeze(-1)
            aligned_weights = F.adaptive_avg_pool2d(aligned_weights, target_shape)
            return models_to_merge_delta_param * aligned_weights
        else:
            # Robust axis-matching approach: find where weight_scores shape matches delta tensor
            S = tuple(models_to_merge_delta_param.shape)
            R = tuple(weight_scores.shape)
            
            # Find the axis where R matches a slice of S
            for axis in range(len(S) - len(R) + 1):
                if S[axis:axis+len(R)] == R:
                    # Insert singleton dimensions to align properly
                    aligned_weights = weight_scores.view(
                        *([1] * axis), *R, *([1] * (len(S) - axis - len(R)))
                    )
                    return models_to_merge_delta_param * aligned_weights
            
            # Final fallback: broadcast weight_scores along the last dims
            extra = models_to_merge_delta_param.dim() - weight_scores.dim()
            aligned_weights = weight_scores.view(*([1] * extra), *weight_scores.shape)
            return models_to_merge_delta_param * aligned_weights


def _vectorized_parameter_batch_merge(param_tensors_batch: list, task_vector_deltas_batch: list, 
                                     weight_scores_batch: list, device: torch.device) -> list:
    """
    Vectorized batch processing of multiple parameters using torch.vmap for elimination of sequential overhead
    
    Args:
        param_tensors_batch: List of base parameter tensors  
        task_vector_deltas_batch: List of corresponding task vector deltas
        weight_scores_batch: List of corresponding weight scores
        device: Target computation device
        
    Returns:
        List of merged parameter tensors
    """
    if not param_tensors_batch:
        return []
    
    try:
        # Group parameters by shape for efficient batch processing
        shape_groups = {}
        for i, param_tensor in enumerate(param_tensors_batch):
            shape_key = tuple(param_tensor.shape)
            if shape_key not in shape_groups:
                shape_groups[shape_key] = []
            shape_groups[shape_key].append(i)
        
        merged_results = [None] * len(param_tensors_batch)
        
        # Process each shape group with vectorized operations
        for shape_key, indices in shape_groups.items():
            if len(indices) == 1:
                # Single parameter - no vectorization benefit
                idx = indices[0]
                base_param = param_tensors_batch[idx].to(device, non_blocking=True)
                delta = task_vector_deltas_batch[idx].to(device, non_blocking=True) 
                weights = weight_scores_batch[idx].to(device, non_blocking=True)
                
                merged_results[idx] = base_param + _safe_tensor_multiply_core(delta, weights)
            else:
                # Multiple parameters with same shape - use vectorized batch processing
                batch_base_params = torch.stack([param_tensors_batch[i].to(device, non_blocking=True) for i in indices])
                batch_deltas = torch.stack([task_vector_deltas_batch[i].to(device, non_blocking=True) for i in indices])
                batch_weights = torch.stack([weight_scores_batch[i].to(device, non_blocking=True) for i in indices])
                
                # Vectorized multiplication using vmap for parallel processing
                def single_param_merge(base_param, delta, weights):
                    return base_param + _safe_tensor_multiply_core(delta, weights)
                
                # Use vmap to process entire batch in parallel
                # Force all inputs to float32 to avoid autocast issues  
                batch_base_float32 = batch_base_params.to(torch.float32)
                batch_deltas_float32 = batch_deltas.to(torch.float32)
                batch_weights_float32 = batch_weights.to(torch.float32)
                
                # Explicitly disable autocast around vmap to prevent TorchScript issues
                with torch.cuda.amp.autocast(enabled=False):
                    batch_merged = torch.vmap(single_param_merge)(batch_base_float32, batch_deltas_float32, batch_weights_float32)
                
                # Store results back in original order
                for i, idx in enumerate(indices):
                    merged_results[idx] = batch_merged[i]
        
        return merged_results
        
    except Exception as e:
        print(f"[WARNING] Vectorized batch merge failed: {e}")
        # Fallback to sequential processing
        merged_results = []
        for i in range(len(param_tensors_batch)):
            base_param = param_tensors_batch[i].to(device, non_blocking=True)
            delta = task_vector_deltas_batch[i].to(device, non_blocking=True)
            weights = weight_scores_batch[i].to(device, non_blocking=True)
            merged_results.append(base_param + _safe_tensor_multiply_core(delta, weights))
        return merged_results


def _batch_importance_score_computation(magnitude_diffs_batch: list, direction_diffs_batch: list,
                                       above_average_value_ratio: float, score_calibration_value: float,
                                       device: torch.device) -> tuple:
    """
    Vectorized batch computation of importance scores for multiple parameters
    
    Returns:
        (batch_importance_scores, batch_variances) - Lists of importance scores and variance info
    """
    if not magnitude_diffs_batch:
        return [], []
    
    try:
        # Process parameters in vectorized batches
        batch_importance_scores = []
        batch_variances = []
        
        # Group by compatible tensor shapes for batch processing
        compatible_groups = []
        for i, (mag_diff, dir_diff) in enumerate(zip(magnitude_diffs_batch, direction_diffs_batch)):
            found_group = False
            for group in compatible_groups:
                sample_mag, sample_dir = magnitude_diffs_batch[group[0]], direction_diffs_batch[group[0]]
                if (mag_diff.shape == sample_mag.shape and dir_diff.shape == sample_dir.shape):
                    group.append(i)
                    found_group = True
                    break
            if not found_group:
                compatible_groups.append([i])
        
        # Process each compatible group with vectorization
        results = [None] * len(magnitude_diffs_batch)
        for group_indices in compatible_groups:
            if len(group_indices) == 1:
                # Single parameter
                idx = group_indices[0]
                mag_tensor = magnitude_diffs_batch[idx].to(device, non_blocking=True)
                dir_tensor = direction_diffs_batch[idx].to(device, non_blocking=True)
                
                mag_scores, mag_uniform = _compute_importance_scores_core(mag_tensor, above_average_value_ratio, score_calibration_value)
                dir_scores, dir_uniform = _compute_importance_scores_core(dir_tensor, above_average_value_ratio, score_calibration_value) 
                
                results[idx] = ((mag_scores, dir_scores), {'magnitude_uniform': mag_uniform.item() if hasattr(mag_uniform, 'item') else mag_uniform, 'direction_uniform': dir_uniform.item() if hasattr(dir_uniform, 'item') else dir_uniform})
            else:
                # Batch process compatible parameters
                batch_mag_tensors = torch.stack([magnitude_diffs_batch[i].to(device, non_blocking=True) for i in group_indices])
                batch_dir_tensors = torch.stack([direction_diffs_batch[i].to(device, non_blocking=True) for i in group_indices])
                
                # Vectorized importance score computation
                def compute_single_importance(mag_tensor, dir_tensor):
                    mag_scores, mag_uniform = _compute_importance_scores_core(mag_tensor, above_average_value_ratio, score_calibration_value)
                    dir_scores, dir_uniform = _compute_importance_scores_core(dir_tensor, above_average_value_ratio, score_calibration_value)
                    return mag_scores, dir_scores, mag_uniform, dir_uniform
                
                # Use vmap for parallel batch processing 
                # Force all inputs to float32 to avoid autocast issues
                batch_mag_float32 = batch_mag_tensors.to(torch.float32)
                batch_dir_float32 = batch_dir_tensors.to(torch.float32)
                
                # Explicitly disable autocast around vmap to prevent TorchScript issues
                with torch.cuda.amp.autocast(enabled=False):
                    batch_mag_scores, batch_dir_scores, batch_mag_uniform, batch_dir_uniform = torch.vmap(compute_single_importance)(batch_mag_float32, batch_dir_float32)
                
                # Store results
                for i, idx in enumerate(group_indices):
                    results[idx] = ((batch_mag_scores[i], batch_dir_scores[i]), 
                                   {'magnitude_uniform': batch_mag_uniform[i].item() if hasattr(batch_mag_uniform[i], 'item') else batch_mag_uniform[i], 'direction_uniform': batch_dir_uniform[i].item() if hasattr(batch_dir_uniform[i], 'item') else batch_dir_uniform[i]})
        
        # Extract final results
        for result in results:
            batch_importance_scores.append(result[0])
            batch_variances.append(result[1])
            
        return batch_importance_scores, batch_variances
        
    except Exception as e:
        print(f"[WARNING] Batch importance computation failed: {e}")
        # Fallback to sequential processing
        batch_importance_scores = []
        batch_variances = []
        
        for mag_diff, dir_diff in zip(magnitude_diffs_batch, direction_diffs_batch):
            mag_tensor = mag_diff.to(device, non_blocking=True)
            dir_tensor = dir_diff.to(device, non_blocking=True)
            
            mag_scores, mag_uniform = _compute_importance_scores_core(mag_tensor, above_average_value_ratio, score_calibration_value)
            dir_scores, dir_uniform = _compute_importance_scores_core(dir_tensor, above_average_value_ratio, score_calibration_value)
            
            batch_importance_scores.append((mag_scores, dir_scores))
            batch_variances.append({'magnitude_uniform': mag_uniform.item() if hasattr(mag_uniform, 'item') else mag_uniform, 'direction_uniform': dir_uniform.item() if hasattr(dir_uniform, 'item') else dir_uniform})
        
        return batch_importance_scores, batch_variances


def _compute_importance_scores_core(input_significance_tensor: torch.Tensor, 
                                       above_average_value_ratio: float = 1.0, 
                                       score_calibration_value: float = 1.0):
    """Core importance scoring logic - pure tensor operations for vmap compatibility"""
    # Handle scalar or 1D tensors (e.g. clip.logit_scale) as uniform scores
    if input_significance_tensor.ndim <= 1:
        return torch.ones_like(input_significance_tensor), torch.tensor(True, device=input_significance_tensor.device)
    
    # Check if input is uniform (all values nearly identical) - use mask instead of if/else
    tensor_variance = torch.var(input_significance_tensor)
    is_uniform = tensor_variance < 1e-8
    
    # Uniform scores fallback (always computed, selected by mask)
    uniform_scores = torch.full_like(input_significance_tensor, 1.0 / input_significance_tensor.numel())
    
    # Compute only the softmax axis actually needed: single-model tensors
    # (2D with batch 1) softmax over the feature axis, multi-model tensors
    # softmax over the model axis. The branch depends only on shape, so it is
    # safe under vmap and avoids allocating the unused softmax entirely.
    is_single_model = (input_significance_tensor.dim() == 2) and (input_significance_tensor.size(0) == 1)
    if is_single_model:
        importance_scores = torch.softmax(input_significance_tensor, dim=1)
    else:
        importance_scores = torch.softmax(input_significance_tensor, dim=0)
    
    # Apply above-average boost (pure tensor operations)
    if above_average_value_ratio != 1.0 and above_average_value_ratio > 0.0:
        mean_val = torch.mean(input_significance_tensor)
        above_avg_mask = input_significance_tensor > mean_val
        importance_scores = torch.where(above_avg_mask, importance_scores * above_average_value_ratio, importance_scores)
    
    # Apply score calibration (pure tensor operation)
    if score_calibration_value != 1.0:
        importance_scores = importance_scores * score_calibration_value
    
    # Select final scores based on uniformity mask (pure tensor operation)
    final_scores = torch.where(is_uniform, uniform_scores, importance_scores)
    
    return final_scores, is_uniform
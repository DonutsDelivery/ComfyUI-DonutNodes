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
import time

# use package-relative path for ComfyUI
from .utils.sdxl_safetensors import ensure_same_device

# Import merging methods and utilities
from .merging_methods import MergingMethod
from .task_vector import TaskVector
from .utils.utils import get_param_names_to_merge
import numpy as np

# SDXL block grouping function for preserving WIDEN cross-parameter validation
def _group_parameters_by_blocks(param_names):
    """Group SDXL parameters by individual architectural blocks to preserve cross-parameter context"""
    import re
    from collections import defaultdict
    
    blocks = defaultdict(list)
    
    for param_name in param_names:
        name_lower = param_name.lower()
        
        # Time and class embeddings (critical for temporal/conditional consistency)
        if 'time_embed' in name_lower:
            blocks['time_embedding'].append(param_name)
        elif 'label_emb' in name_lower or 'class_emb' in name_lower:
            blocks['class_embedding'].append(param_name)
        
        # UNet block structure - Extract individual block numbers
        elif 'input_blocks' in name_lower or 'down_blocks' in name_lower:
            # Extract block number (e.g., input_blocks.0, input_blocks.1, etc.)
            block_match = re.search(r'(input_blocks|down_blocks)\.(\d+)', param_name)
            if block_match:
                block_num = block_match.group(2)
                blocks[f'input_block_{block_num}'].append(param_name)
            else:
                blocks['input_blocks_other'].append(param_name)
                
        elif 'middle_block' in name_lower:
            # Extract middle block sub-components if numbered
            block_match = re.search(r'middle_block\.(\d+)', param_name)
            if block_match:
                block_num = block_match.group(1)
                blocks[f'middle_block_{block_num}'].append(param_name)
            else:
                blocks['middle_block'].append(param_name)
                
        elif 'output_blocks' in name_lower or 'up_blocks' in name_lower:
            # Extract block number (e.g., output_blocks.0, output_blocks.1, etc.)
            block_match = re.search(r'(output_blocks|up_blocks)\.(\d+)', param_name)
            if block_match:
                block_num = block_match.group(2)
                blocks[f'output_block_{block_num}'].append(param_name)
            else:
                blocks['output_blocks_other'].append(param_name)
        
        # Attention layers (preserve cross-parameter relationships)
        elif 'cross' in name_lower and ('attn' in name_lower or 'attention' in name_lower):
            blocks['cross_attention'].append(param_name)
        elif 'attn' in name_lower or 'attention' in name_lower or any(x in name_lower for x in ['to_q', 'to_k', 'to_v', 'to_out']):
            blocks['self_attention'].append(param_name)
        
        # Convolution layers
        elif 'conv' in name_lower or 'convolution' in name_lower:
            blocks['convolutions'].append(param_name)
        
        # Normalization layers
        elif any(x in name_lower for x in ['norm', 'group_norm', 'layer_norm']):
            blocks['normalization'].append(param_name)
        
        # Everything else
        else:
            blocks['other'].append(param_name)
    
    # Convert to list of tuples and sort input/output blocks numerically
    result = []
    for block_name, params in blocks.items():
        if params:  # Only include non-empty blocks
            result.append((block_name, params))
    
    # Sort to ensure input_block_0, input_block_1, etc. are in order
    def sort_key(item):
        block_name = item[0]
        if 'input_block_' in block_name:
            try:
                num = int(block_name.split('_')[-1])
                return (0, num)  # Input blocks first, then by number
            except:
                return (0, 999)
        elif 'middle_block' in block_name:
            if block_name == 'middle_block':
                return (1, 0)
            else:
                try:
                    num = int(block_name.split('_')[-1])
                    return (1, num)
                except:
                    return (1, 999)
        elif 'output_block_' in block_name:
            try:
                num = int(block_name.split('_')[-1])
                return (2, num)  # Output blocks after middle, then by number
            except:
                return (2, 999)
        else:
            return (3, block_name)  # Other blocks at end, alphabetically
    
    result.sort(key=sort_key)
    return result

# Enhanced WIDEN merging with dynamic compatibility-based strength
def enhanced_widen_merging_with_dynamic_strength(
    merger,
    merged_model,
    models_to_merge,
    exclude_param_names_regex,
    importance_threshold,
    importance_boost,
    base_merge_strength,
    rank_sensitivity,
    skip_threshold,
    normalization_mode
):
    """Enhanced WIDEN merging with dynamic compatibility-based merge strength"""
    
    # Determine devices for hybrid processing
    target_device = next(merged_model.parameters()).device
    
    # Check available VRAM and RAM for smart processing strategy
    vram_available_mb = 0
    ram_available_gb = 0
    
    if torch.cuda.is_available():
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            vram_available_mb = free_memory / (1024 * 1024)
            allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"[ENHANCED WIDEN] GPU memory - Available: {vram_available_mb:.1f}MB, Allocated: {allocated_mb:.1f}MB")
        except:
            vram_available_mb = 1000  # Conservative fallback
    
    # Check system RAM
    try:
        import psutil
        ram_info = psutil.virtual_memory()
        ram_available_gb = ram_info.available / (1024 * 1024 * 1024)
        ram_total_gb = ram_info.total / (1024 * 1024 * 1024)
        ram_used_percent = ram_info.percent
        print(f"[ENHANCED WIDEN] System RAM - Available: {ram_available_gb:.1f}GB/{ram_total_gb:.1f}GB ({100-ram_used_percent:.1f}% free)")
        
        if ram_available_gb < 2.0:
            print(f"[WARNING] Low RAM detected ({ram_available_gb:.1f}GB available). Using conservative processing.")
    except:
        ram_available_gb = 4.0  # Conservative fallback
    
    # Simplified processing strategy: Always use CPU to avoid redundant GPU/CPU computation
    # Since we're doing all ranking on CPU anyway, using GPU for intermediate computations is wasteful
    computation_device = torch.device("cpu")
    storage_device = torch.device("cpu")
    ranking_device = torch.device("cpu")
    
    if ram_available_gb < 2.0:
        print(f"[ENHANCED WIDEN] Low RAM mode ({ram_available_gb:.1f}GB): Conservative CPU processing")
    elif ram_available_gb > 4.0:
        print(f"[ENHANCED WIDEN] High RAM mode ({ram_available_gb:.1f}GB): Optimized CPU processing")
    else:
        print(f"[ENHANCED WIDEN] Standard mode ({ram_available_gb:.1f}GB): CPU processing")
    
    print(f"[ENHANCED WIDEN] All processing on CPU for maximum efficiency and WIDEN validation")
    
    # Create task vectors efficiently (these contain the deltas we need)
    print("[ENHANCED WIDEN] Creating TaskVectors for delta computation...")
    models_to_merge_task_vectors = [
        TaskVector(merged_model, model_to_merge, exclude_param_names_regex) 
        for model_to_merge in models_to_merge
    ]
    
    # Create minimal pretrained parameter reference (only what we need for final merging)
    print("[ENHANCED WIDEN] Creating minimal parameter reference...")
    pretrained_param_dict = {}
    for param_name, param_value in merged_model.named_parameters():
        pretrained_param_dict[param_name] = param_value.detach().to(storage_device).float()
    
    # Transpose token embeddings in TaskVectors only (we don't need separate copies)
    for task_vector in models_to_merge_task_vectors:
        _transpose_token_embeddings(task_vector.task_vector_param_dict)
    _transpose_token_embeddings(pretrained_param_dict)  # Only transpose our minimal reference
    
    with torch.no_grad():
        print("[ENHANCED WIDEN] Computing magnitude and direction differences from TaskVector deltas...")
        
        # Step 1: Use TaskVector deltas directly instead of recomputing everything
        models_to_merge_param_magnitude_direction_diff_tuples = []
        
        for task_vector in models_to_merge_task_vectors:
            # Compute magnitude and direction differences directly from deltas
            magnitude_diffs = {}
            direction_diffs = {}
            
            for param_name, delta in task_vector.task_vector_param_dict.items():
                # Compute magnitude and direction differences from the delta directly
                if delta.dim() == 1:
                    # For 1D: magnitude is absolute delta, direction is sign of delta
                    magnitude_diffs[param_name] = torch.abs(delta)
                    direction_diffs[param_name] = torch.sign(delta)
                elif delta.dim() == 2:
                    # For 2D: compute along features (dim=0) - both should be 1D for ranking
                    magnitude_diffs[param_name] = torch.norm(delta, p=2, dim=0)  # Shape: [features]
                    # For direction: compute cosine similarity with a reference direction (mean direction)
                    delta_normalized = delta / (torch.norm(delta, p=2, dim=0, keepdim=True) + 1e-8)
                    mean_direction = delta_normalized.mean(dim=0)  # Shape: [features]
                    direction_diffs[param_name] = 1.0 - torch.cosine_similarity(
                        delta_normalized, mean_direction.unsqueeze(0), dim=0
                    )  # Shape: [features] - how much each feature differs from mean direction
                elif delta.dim() > 2:
                    # For >2D: flatten to 2D first
                    original_shape = delta.shape
                    delta_2d = delta.view(original_shape[0], -1)
                    magnitude_diffs[param_name] = torch.norm(delta_2d, p=2, dim=0)  # Shape: [flattened_features]
                    # For direction: same approach as 2D
                    delta_normalized = delta_2d / (torch.norm(delta_2d, p=2, dim=0, keepdim=True) + 1e-8)
                    mean_direction = delta_normalized.mean(dim=0)  # Shape: [flattened_features]
                    direction_diffs[param_name] = 1.0 - torch.cosine_similarity(
                        delta_normalized, mean_direction.unsqueeze(0), dim=0
                    )  # Shape: [flattened_features]
            
            models_to_merge_param_magnitude_direction_diff_tuples.append((magnitude_diffs, direction_diffs))
        
        print(f"[ENHANCED WIDEN] Computed differences for {len(models_to_merge_task_vectors)} models efficiently")
        
        # Step 2: Enhanced parameter merging with dynamic compatibility-based strength
        merged_params = _merge_param_magnitude_direction_with_dynamic_strength(
            models_to_merge_param_magnitude_direction_diff_tuples,
            pretrained_param_dict,
            models_to_merge_task_vectors,
            exclude_param_names_regex,
            importance_threshold,
            importance_boost,
            base_merge_strength,
            rank_sensitivity,
            skip_threshold,
            normalization_mode,
            computation_device,
            target_device,
            storage_device,
            ranking_device
        )
        
        # Transpose back
        _transpose_token_embeddings(merged_params)
    
    return merged_params

# Helper functions for enhanced WIDEN merging
def _transpose_token_embeddings(param_dict):
    """Transpose token embeddings"""
    for param_name in param_dict:
        if param_name == "model.embed_tokens.weight":
            param_dict[param_name] = param_dict[param_name].transpose(dim0=0, dim1=1)

def _compute_param_magnitude_direction(param_dict, module_dict):
    """Compute magnitude vector and direction matrix for parameters"""
    param_magnitude_dict, param_direction_dict = {}, {}
    
    for param_name in param_dict:
        param_last_name = param_name.split(".")[-1]
        module_name = param_name[:-len(f".{param_last_name}")]
        
        if param_dict[param_name].dim() == 1:
            # Handle 1D parameters (bias, norm) - treat each element as a feature
            param_tensor = param_dict[param_name]
            magnitude_vector = torch.abs(param_tensor)  # For 1D, magnitude is just absolute value
            direction_matrix = torch.sign(param_tensor)  # For 1D, direction is just the sign
            
            param_magnitude_dict[param_name] = magnitude_vector
            param_direction_dict[param_name] = direction_matrix
            
        elif param_dict[param_name].dim() == 2:
            # Compute magnitude and direction for 2D parameters
            magnitude_vector = torch.norm(param_dict[param_name], p=2, dim=0)
            direction_matrix = param_dict[param_name] / (magnitude_vector + 1e-8)
            
            param_magnitude_dict[param_name] = magnitude_vector
            param_direction_dict[param_name] = direction_matrix
            
        elif param_dict[param_name].dim() > 2:
            # Handle higher dimensional parameters (conv layers, etc.)
            # Flatten to 2D for magnitude/direction computation
            original_shape = param_dict[param_name].shape
            param_2d = param_dict[param_name].view(original_shape[0], -1)
            
            magnitude_vector = torch.norm(param_2d, p=2, dim=0)
            direction_matrix = param_2d / (magnitude_vector + 1e-8)
            
            param_magnitude_dict[param_name] = magnitude_vector
            param_direction_dict[param_name] = direction_matrix
    
    return param_magnitude_dict, param_direction_dict

def _compute_param_magnitude_direction_differences(pretrained_param_magnitude_dict, pretrained_param_direction_dict,
                                                 finetuned_param_magnitude_dict, finetuned_param_direction_dict, module_dict):
    """Compute magnitude and direction differences"""
    param_magnitude_diff_dict, param_direction_diff_dict = {}, {}
    
    for param_name in pretrained_param_magnitude_dict:
        # Ensure tensors are on the same device
        pretrained_mag = pretrained_param_magnitude_dict[param_name]
        finetuned_mag = finetuned_param_magnitude_dict[param_name]
        pretrained_dir = pretrained_param_direction_dict[param_name]
        finetuned_dir = finetuned_param_direction_dict[param_name]
        
        # Move to same device (prefer CUDA if available)
        target_device = pretrained_mag.device
        if finetuned_mag.device != target_device:
            finetuned_mag = finetuned_mag.to(target_device)
        if finetuned_dir.device != target_device:
            finetuned_dir = finetuned_dir.to(target_device)
        if pretrained_dir.device != target_device:
            pretrained_dir = pretrained_dir.to(target_device)
        
        # Compute magnitude difference
        param_magnitude_diff = torch.abs(finetuned_mag - pretrained_mag)
        
        # Compute direction difference
        param_direction_diff = 1.0 - torch.cosine_similarity(
            finetuned_dir,
            pretrained_dir,
            dim=0
        )
        
        param_magnitude_diff_dict[param_name] = param_magnitude_diff
        param_direction_diff_dict[param_name] = param_direction_diff
    
    return param_magnitude_diff_dict, param_direction_diff_dict

def _rank_per_param_magnitude_or_direction_within_model(models_to_merge_param_diff):
    """Rank the magnitude or direction within model"""
    # Ensure all tensors are on the same device
    device = models_to_merge_param_diff.device
    
    sort_indices = torch.argsort(models_to_merge_param_diff, dim=1, descending=False, stable=True)
    within_model_significance = (torch.arange(models_to_merge_param_diff.shape[1], device=device) / models_to_merge_param_diff.shape[1]).repeat(
        models_to_merge_param_diff.shape[0]
    ).reshape(models_to_merge_param_diff.shape)
    
    models_to_merge_param_within_model_significance = torch.zeros(within_model_significance.shape, device=device)
    models_to_merge_param_within_model_significance = torch.scatter(
        input=models_to_merge_param_within_model_significance,
        dim=1,
        index=sort_indices,
        src=within_model_significance
    )
    
    return models_to_merge_param_within_model_significance

def _compute_importance_scores(input_significance_tensor, above_average_value_ratio=1.0, score_calibration_value=1.0):
    """Compute importance scores (fixed WIDEN algorithm)"""
    # Debug: Log tensor properties - ALWAYS DEBUG FOR NOW
    debug_scores = True
    if hasattr(_compute_importance_scores, 'debug_count'):
        _compute_importance_scores.debug_count += 1
    else:
        _compute_importance_scores.debug_count = 1
    
    if debug_scores:
        print(f"[SCORE DEBUG] Input tensor shape: {input_significance_tensor.shape}")
        print(f"[SCORE DEBUG] Input tensor min/max: {input_significance_tensor.min().item():.6f}/{input_significance_tensor.max().item():.6f}")
        print(f"[SCORE DEBUG] Input tensor variance: {input_significance_tensor.var().item():.6f}")
        print(f"[SCORE DEBUG] above_average_value_ratio: {above_average_value_ratio}, score_calibration_value: {score_calibration_value}")
    
    # Check if input tensor has no variation (all values identical)
    tensor_variance = input_significance_tensor.var().item()
    if tensor_variance < 1e-8:
        # If all values are identical, use uniform scores to avoid bias
        uniform_score = 1.0 / input_significance_tensor.shape[0]  # Equal probability for each model
        importance_scores = torch.full_like(input_significance_tensor, uniform_score)
        if debug_scores:
            print(f"[SCORE DEBUG] Uniform input detected (variance={tensor_variance:.2e}), using uniform scores: {uniform_score:.6f}")
        return importance_scores
    
    # Compute softmax scores for varied inputs
    # For single model (shape [1, features]), softmax across features (dim=1)
    # For multiple models (shape [models, features]), softmax across models (dim=0)
    if input_significance_tensor.shape[0] == 1:
        # Debug: Test what happens with softmax
        print(f"[SOFTMAX TEST] Input shape: {input_significance_tensor.shape}")
        print(f"[SOFTMAX TEST] Sample values: {input_significance_tensor[0][:10]}")
        importance_scores = torch.softmax(input_significance_tensor, dim=1)
        print(f"[SOFTMAX TEST] Output sample: {importance_scores[0][:10]}")
        print(f"[SOFTMAX TEST] Output sum: {importance_scores.sum().item()}")
    else:
        importance_scores = torch.softmax(input_significance_tensor, dim=0)
    
    if debug_scores:
        print(f"[SCORE DEBUG] Softmax scores min/max: {importance_scores.min().item():.6f}/{importance_scores.max().item():.6f}")
    
    # Apply mask only if it won't make everything uniform
    # CRITICAL FIX: Base mask on importance_scores (after softmax), not input tensor
    if importance_scores.dim() >= 2:
        # For 2D+ tensors, compute mean along dim=1 (within each model)
        avg_importance_scores = torch.mean(importance_scores, dim=1, keepdim=True)
    else:
        # For 1D tensors, compute overall mean
        avg_importance_scores = torch.mean(importance_scores, dim=0, keepdim=True)
    
    mask = importance_scores > (avg_importance_scores * above_average_value_ratio)
    mask_ratio = mask.float().mean().item()
    
    # FIXED: Apply calibration as multiplicative factor to preserve variation, not replacement
    if mask_ratio < 0.95:  # Less than 95% of values above threshold
        # Apply calibration as a boost factor while preserving relative differences
        importance_scores[mask] = importance_scores[mask] * score_calibration_value
        mask_applied = True
        if debug_scores:
            print(f"[SCORE DEBUG] Applied calibration factor {score_calibration_value} to {mask.sum().item()} values")
    else:
        mask_applied = False
        if debug_scores:
            print(f"[SCORE DEBUG] Mask covers {mask_ratio*100:.1f}% of values - skipping to preserve score variation")
    
    if debug_scores:
        mask_count = mask.sum().item()
        total_count = mask.numel()
        print(f"[SCORE DEBUG] Average threshold: {(avg_importance_scores * above_average_value_ratio).item():.6f}")
        print(f"[SCORE DEBUG] Mask would apply to {mask_count}/{total_count} elements ({100*mask_count/total_count:.1f}%)")
        print(f"[SCORE DEBUG] Mask actually applied: {mask_applied}")
        print(f"[SCORE DEBUG] Final scores min/max: {importance_scores.min().item():.6f}/{importance_scores.max().item():.6f}")
    
    return importance_scores

def _compatibility_to_merge_strength(compatibility_score, base_merge_strength, sensitivity):
    """Convert compatibility score to merge strength"""
    # If sensitivity is 0, disable dynamic strength (use static base strength)
    if sensitivity == 0.0:
        return float(base_merge_strength)
    
    # Normalize compatibility score (typical range is 0.0 to 1.0, but can go higher)
    normalized_score = min(compatibility_score, 1.0)
    
    # Apply sigmoid transformation
    sigmoid_input = (normalized_score - 0.5) * sensitivity
    sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
    
    # Dynamic range: 10% to 100% of base merge strength
    min_strength = 0.1 * base_merge_strength
    max_strength = base_merge_strength
    
    # Map to merge strength range
    merge_strength = min_strength + (max_strength - min_strength) * sigmoid_output
    
    return float(merge_strength)

def _merge_param_magnitude_direction_with_dynamic_strength(
    models_to_merge_param_magnitude_direction_diff_tuples,
    pretrained_param_dict,
    models_to_merge_task_vectors,
    exclude_param_names_regex,
    importance_threshold,
    importance_boost,
    base_merge_strength,
    rank_sensitivity,
    skip_threshold,
    normalization_mode,
    computation_device,
    target_device,
    storage_device,
    ranking_device
):
    """Enhanced parameter merging with dynamic strength based on compatibility"""
    
    # Map new parameter names to original WIDEN algorithm variable names
    above_average_value_ratio = importance_threshold
    score_calibration_value = importance_boost
    
    # Check available RAM for memory management
    ram_available_gb = 4.0  # Conservative fallback
    try:
        import psutil
        ram_info = psutil.virtual_memory()
        ram_available_gb = ram_info.available / (1024 * 1024 * 1024)
    except:
        pass
    
    # Get parameters to merge
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()),
        exclude_param_names_regex=exclude_param_names_regex
    )
    
    # Initialize WIDEN diagnostics collection
    widen_diagnostics = {
        'compatibility_scores': [],
        'importance_score_variances': [],
        'magnitude_score_ranges': [],
        'direction_score_ranges': [],
        'parameters_with_rankings': 0,
        'parameters_skipped_threshold': 0,
        'parameters_skipped_no_rankings': 0,
        'uniform_score_count': 0,
        'varied_score_count': 0
    }
    
    # Unpack magnitude and direction differences
    print(f"[DEBUG] Input magnitude/direction tuples: {len(models_to_merge_param_magnitude_direction_diff_tuples)}")
    
    if not models_to_merge_param_magnitude_direction_diff_tuples:
        print(f"[CRITICAL] No magnitude/direction differences computed!")
        # Fallback: create empty rankings and skip to Phase 2
        magnitude_rankings = {}
        direction_rankings = {}
        param_names_merged_by_magnitude_direction = []
    else:
        # Extract magnitude and direction differences from the new format
        models_to_merge_param_magnitude_diff_tuple = [model_diffs[0] for model_diffs in models_to_merge_param_magnitude_direction_diff_tuples]
        models_to_merge_param_direction_diff_tuple = [model_diffs[1] for model_diffs in models_to_merge_param_magnitude_direction_diff_tuples]
        
        param_names_merged_by_magnitude_direction = list(models_to_merge_param_magnitude_diff_tuple[0].keys())
        print(f"[DEBUG] Parameters eligible for magnitude/direction ranking: {len(param_names_merged_by_magnitude_direction)}")
        if param_names_merged_by_magnitude_direction:
            print(f"[DEBUG] Sample eligible parameters: {param_names_merged_by_magnitude_direction[:3]}")
        else:
            print(f"[CRITICAL] No parameters eligible for ranking - all filtered out!")
    
    # PHASE 1: Compute ALL rankings on CPU to preserve WIDEN cross-parameter validation
    if param_names_merged_by_magnitude_direction:
        print(f"[ENHANCED WIDEN] Phase 1: Computing rankings for {len(param_names_merged_by_magnitude_direction)} eligible parameters...")
        if 'magnitude_rankings' not in locals():
            magnitude_rankings = {}
        if 'direction_rankings' not in locals():  
            direction_rankings = {}
            
        # Organize parameters by SDXL blocks to preserve WIDEN cross-parameter context
        block_groups = _group_parameters_by_blocks(param_names_merged_by_magnitude_direction)
        total_blocks = len(block_groups)
        total_params = len(param_names_merged_by_magnitude_direction)
        
        print(f"[ENHANCED WIDEN] Block-wise processing: {total_blocks} blocks containing {total_params} parameters")
        print(f"[ENHANCED WIDEN] Blocks: {[(name, len(params)) for name, params in block_groups]}")
        
        import time
        start_time = time.time()
        
        # Process parameters block by block to preserve WIDEN cross-parameter validation
        for block_idx, (block_name, block_params) in enumerate(block_groups):
            block_start_time = time.time()
            print(f"[ENHANCED WIDEN] Processing block {block_idx+1}/{total_blocks}: {block_name} ({len(block_params)} parameters)")
            
            # Batch process all parameters in this block at once for efficiency
            block_magnitude_diffs = {}
            block_direction_diffs = {}
            valid_params_in_block = []
            
            # Collect all magnitude/direction diffs for this block
            for param_name in block_params:
                try:
                    magnitude_diffs = []
                    direction_diffs = []
                    
                    for model_idx in range(len(models_to_merge_param_magnitude_diff_tuple)):
                        if param_name in models_to_merge_param_magnitude_diff_tuple[model_idx]:
                            mag_diff = models_to_merge_param_magnitude_diff_tuple[model_idx][param_name]
                            dir_diff = models_to_merge_param_direction_diff_tuple[model_idx][param_name]
                            
                            # Skip extremely large tensors to prevent memory issues
                            if mag_diff.numel() > 50000000:  # 50M elements
                                print(f"[ENHANCED WIDEN] Skipping large tensor {param_name} ({mag_diff.numel()} elements)")
                                break
                                
                            magnitude_diffs.append(mag_diff.to(ranking_device))
                            direction_diffs.append(dir_diff.to(ranking_device))
                    
                    if magnitude_diffs and direction_diffs:
                        block_magnitude_diffs[param_name] = magnitude_diffs
                        block_direction_diffs[param_name] = direction_diffs
                        valid_params_in_block.append(param_name)
                        
                except Exception as e:
                    print(f"[WARNING] Failed to collect data for {param_name}: {e}")
            
            # Now compute rankings for all valid parameters in this block
            print(f"[ENHANCED WIDEN]   Computing rankings for {len(valid_params_in_block)} valid parameters in block {block_name}")
            
            for param_name in valid_params_in_block:
                try:
                    mag_tensor = torch.stack(block_magnitude_diffs[param_name], dim=0)
                    dir_tensor = torch.stack(block_direction_diffs[param_name], dim=0)
                    
                    # Compute rankings with FULL parameter visibility (critical for WIDEN)
                    magnitude_rankings[param_name] = _rank_per_param_magnitude_or_direction_within_model(mag_tensor)
                    direction_rankings[param_name] = _rank_per_param_magnitude_or_direction_within_model(dir_tensor)
                    
                    # Verify ranking shapes for WIDEN correctness (first parameter only)
                    if block_idx == 0 and param_name == valid_params_in_block[0]:
                        print(f"[WIDEN VERIFICATION] Parameter '{param_name}': magnitude shape {mag_tensor.shape} -> ranking shape {magnitude_rankings[param_name].shape}")
                        print(f"[WIDEN VERIFICATION] This ranks {mag_tensor.shape[1]} features across {mag_tensor.shape[0]} models - CORRECT WIDEN behavior")
                    
                    # Free memory immediately
                    del mag_tensor, dir_tensor
                    
                except Exception as e:
                    print(f"[WARNING] Failed to compute rankings for {param_name}: {e}")
            
            # Clean up block data
            del block_magnitude_diffs, block_direction_diffs
            
            # Block timing and cleanup
            block_time = time.time() - block_start_time
            print(f"[ENHANCED WIDEN] Block {block_name} completed in {block_time:.1f}s")
            
            # Aggressive memory cleanup after each block
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"[ENHANCED WIDEN] Phase 1 complete: Rankings computed for {len(magnitude_rankings)} parameters")
    
    # CRITICAL DIAGNOSTIC: Check if Phase 1 failed completely
    if len(magnitude_rankings) == 0:
        print(f"[CRITICAL] Phase 1 FAILED: No rankings computed!")
        print(f"[CRITICAL] Input parameters available: {len(param_names_merged_by_magnitude_direction)}")
        print(f"[CRITICAL] Task vectors: {len(models_to_merge_task_vectors)}")
        print(f"[CRITICAL] Magnitude diff tuples: {len(models_to_merge_param_magnitude_diff_tuple) if models_to_merge_param_magnitude_diff_tuple else 'NONE'}")
        if param_names_merged_by_magnitude_direction:
            print(f"[CRITICAL] Sample parameters: {param_names_merged_by_magnitude_direction[:3]}")
    else:
        print(f"[ENHANCED WIDEN] Phase 1 successful: {len(magnitude_rankings)} parameters have rankings")
        print(f"[ENHANCED WIDEN] Sample ranking parameters: {list(magnitude_rankings.keys())[:3]}")
    
    # PHASE 2: Process individual parameters using precomputed rankings
    print("[ENHANCED WIDEN] Phase 2: Processing individual parameters...")
    print("[ENHANCED WIDEN] Note: All parameters (1D, 2D, >2D) will use WIDEN rankings for consistent merging")
    merged_params = {}
    skipped_count = 0
    failed_count = 0
    no_rankings_count = 0
    widen_merged_count = 0  # For parameters using WIDEN rankings
    
    # Memory monitoring function (for logging only - never stop processing)
    def check_memory_status():
        try:
            import psutil
            ram_info = psutil.virtual_memory()
            ram_available_gb = ram_info.available / (1024 * 1024 * 1024)
            if ram_available_gb < 1.0:
                print(f"[WARNING] Critical RAM: {ram_available_gb:.1f}GB available - continuing but may be slow")
                # Force aggressive cleanup but don't stop
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return ram_available_gb
        except:
            return 4.0  # Assume sufficient if can't check
    
    # Process parameters block-wise for Phase 2 as well for consistency
    phase2_block_groups = _group_parameters_by_blocks(param_names_to_merge)
    total_phase2_blocks = len(phase2_block_groups)
    processed_block_count = 0
    
    for block_idx, (block_name, block_params) in enumerate(phase2_block_groups):
        processed_block_count += 1
        ram_gb = check_memory_status()
        print(f"[ENHANCED WIDEN] Phase 2 - Processing block {processed_block_count}/{total_phase2_blocks}: {block_name} ({len(block_params)} parameters) [RAM: {ram_gb:.1f}GB]")
        
        for param_name in block_params:
            try:
                # Initialize ranking variables to None
                models_to_merge_param_magnitude_rank = None
                models_to_merge_param_direction_rank = None
                models_to_merge_delta_param = None
                
                # Parameters that can be merged by magnitudes and directions using precomputed rankings
                if param_name in param_names_merged_by_magnitude_direction and param_name in magnitude_rankings:
                    # Get delta tensors for computation
                    delta_tensors = []
                    for models_to_merge_task_vector in models_to_merge_task_vectors:
                        if param_name in models_to_merge_task_vector.task_vector_param_dict:
                            delta = models_to_merge_task_vector.task_vector_param_dict[param_name]
                            # Move to computation device for processing
                            delta_tensors.append(delta.to(computation_device))
                    
                    if not delta_tensors:
                        print(f"[WARNING] No delta tensors found for {param_name}, skipping")
                        merged_params[param_name] = pretrained_param_dict[param_name].to(target_device)
                        continue
                        
                    models_to_merge_delta_param = torch.stack(delta_tensors, dim=0)
                    
                    # Use precomputed rankings (move to computation device if needed)
                    models_to_merge_param_magnitude_rank = magnitude_rankings[param_name].to(computation_device)
                    models_to_merge_param_direction_rank = direction_rankings[param_name].to(computation_device)
                
                # Only compute importance scores if we have valid ranking tensors
                if models_to_merge_param_magnitude_rank is not None and models_to_merge_param_direction_rank is not None:
                    # Debug input tensors before computing scores
                    print(f"[INPUT DEBUG] {param_name}: magnitude_rank shape={models_to_merge_param_magnitude_rank.shape}, values={models_to_merge_param_magnitude_rank}")
                    print(f"[INPUT DEBUG] {param_name}: direction_rank shape={models_to_merge_param_direction_rank.shape}, values={models_to_merge_param_direction_rank}")
                    
                    # Compute importance scores using original WIDEN algorithm
                    magnitude_scores = _compute_importance_scores(
                        models_to_merge_param_magnitude_rank, above_average_value_ratio, score_calibration_value
                    )
                    direction_scores = _compute_importance_scores(
                        models_to_merge_param_direction_rank, above_average_value_ratio, score_calibration_value
                    )
                    
                    # Collect WIDEN diagnostics for enhanced reporting
                    widen_diagnostics['parameters_with_rankings'] += 1
                    
                    # Check score variance (detect uniform vs varied scores)
                    mag_variance = magnitude_scores.var().item()
                    dir_variance = direction_scores.var().item()
                    widen_diagnostics['importance_score_variances'].extend([mag_variance, dir_variance])
                    
                    # Track score ranges
                    widen_diagnostics['magnitude_score_ranges'].append((magnitude_scores.min().item(), magnitude_scores.max().item()))
                    widen_diagnostics['direction_score_ranges'].append((direction_scores.min().item(), direction_scores.max().item()))
                    
                    # Count uniform vs varied scores (variance threshold: 1e-6)
                    if mag_variance < 1e-6 and dir_variance < 1e-6:
                        widen_diagnostics['uniform_score_count'] += 1
                    else:
                        widen_diagnostics['varied_score_count'] += 1
                else:
                    # Skip parameters that don't have rankings (shouldn't happen if magnitude/direction computation worked)
                    merged_params[param_name] = pretrained_param_dict[param_name].to(target_device)
                    no_rankings_count += 1
                    widen_diagnostics['parameters_skipped_no_rankings'] += 1
                    continue
                
                # Combine scores (original WIDEN approach)
                combined_scores = 0.5 * (magnitude_scores + direction_scores)
                
                # Debug combined scores
                print(f"[COMBINE DEBUG] {param_name}: mag_scores={magnitude_scores.min().item():.6f}-{magnitude_scores.max().item():.6f}, dir_scores={direction_scores.min().item():.6f}-{direction_scores.max().item():.6f}")
                print(f"[COMBINE DEBUG] {param_name}: combined_scores={combined_scores.min().item():.6f}-{combined_scores.max().item():.6f}")
                
                # Compute compatibility score - higher means more compatible/important
                compatibility_score = torch.mean(combined_scores).item()
                
                # Collect compatibility score for diagnostics
                widen_diagnostics['compatibility_scores'].append(compatibility_score)
                
                # Debug: Log compatibility scores for debugging
                param_count = len(merged_params) + no_rankings_count + skipped_count
                if param_count < 10 or skip_threshold > 0.0:
                    print(f"[SKIP DEBUG] {param_name}: compatibility_score={compatibility_score:.4f}, skip_threshold={skip_threshold}")
                
                # Skip parameters with very low compatibility (if threshold > 0)
                if skip_threshold > 0.0 and compatibility_score <= skip_threshold:
                    print(f"[SKIP] Skipping {param_name} (score={compatibility_score:.4f} < threshold={skip_threshold})")
                    merged_params[param_name] = pretrained_param_dict[param_name].to(target_device)
                    skipped_count += 1
                    widen_diagnostics['parameters_skipped_threshold'] += 1
                    continue
                
                # Compute dynamic merge strength
                merge_strength = _compatibility_to_merge_strength(
                    compatibility_score, base_merge_strength, rank_sensitivity
                )
                
                # Merge parameter with dynamic strength
                weight_scores = combined_scores
                
                # Handle different tensor dimensions for weight application
                try:
                    # Reshape weight_scores to broadcast with models_to_merge_delta_param
                    if models_to_merge_delta_param.dim() == weight_scores.dim():
                        # Same dimensions, can multiply directly
                        weighted_deltas = models_to_merge_delta_param * weight_scores
                    elif models_to_merge_delta_param.dim() > weight_scores.dim():
                        # Need to add dimensions to weight_scores
                        weight_shape = [weight_scores.shape[0]] + [1] * (models_to_merge_delta_param.dim() - 1)
                        weight_scores_reshaped = weight_scores.view(weight_shape)
                        weighted_deltas = models_to_merge_delta_param * weight_scores_reshaped
                    else:
                        # Fallback: use simple averaging
                        weighted_deltas = models_to_merge_delta_param
                    
                    merged_delta_param = weighted_deltas.sum(dim=0)
                except Exception as e:
                    # Fallback to simple averaging if weighting fails
                    merged_delta_param = models_to_merge_delta_param.mean(dim=0)
                
                merged_param = pretrained_param_dict[param_name].to(computation_device) + merged_delta_param * merge_strength
                
                # Apply renormalization if enabled
                if hasattr(merged_param, 'shape') and normalization_mode != "none":
                    try:
                        base_param_for_renorm = pretrained_param_dict[param_name].to(computation_device)
                        if normalization_mode == "calibrate":
                            # Use conservative calibrate parameters
                            merged_param = calibrate_renormalize(merged_param, base_param_for_renorm, normalization_mode, 0.3, 1.1)
                        else:  # magnitude
                            merged_param = calibrate_renormalize(merged_param, base_param_for_renorm, normalization_mode, 1.0, 1.0)
                    except Exception as e:
                        # Silent renormalization failure - continue with non-renormalized parameter
                        pass
                
                # Move final result to target device and immediately free processing memory
                merged_params[param_name] = merged_param.to(target_device)
                widen_merged_count += 1  # Count successful WIDEN merges
                
                # Clean up computation tensors to free VRAM and RAM
                try:
                    del models_to_merge_delta_param, merged_delta_param, merged_param
                    if models_to_merge_param_magnitude_rank is not None:
                        del models_to_merge_param_magnitude_rank
                    if models_to_merge_param_direction_rank is not None:
                        del models_to_merge_param_direction_rank
                    del magnitude_scores, direction_scores, combined_scores, weight_scores
                except:
                    pass
                    
            except Exception as e:
                print(f"[ERROR] Failed to merge parameter {param_name}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: use original parameter
                merged_params[param_name] = pretrained_param_dict[param_name].to(target_device)
                failed_count += 1
    
    # Calculate actual merge statistics
    total_merged_count = len(merged_params)
    
    # Generate enhanced WIDEN diagnostics
    compatibility_scores = widen_diagnostics['compatibility_scores']
    score_variances = widen_diagnostics['importance_score_variances']
    
    print(f"[ENHANCED WIDEN] MERGE RESULTS:")
    print(f"  Total parameters: {len(param_names_to_merge)}")
    print(f"  WIDEN merged (all dimensions): {widen_merged_count}")
    print(f"  Total successfully merged: {total_merged_count}")
    print(f"  Skipped (no rankings): {no_rankings_count}")
    print(f"  Skipped (low compatibility): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Overall merge success rate: {total_merged_count/len(param_names_to_merge)*100:.1f}%")
    print(f"  WIDEN success rate: {widen_merged_count/len(param_names_to_merge)*100:.1f}%")
    
    # ENHANCED: WIDEN Algorithm Health Diagnostics
    print(f"\n[WIDEN ALGORITHM HEALTH]:")
    if compatibility_scores:
        compat_min, compat_max = min(compatibility_scores), max(compatibility_scores)
        compat_mean = sum(compatibility_scores) / len(compatibility_scores)
        compat_variance = sum((x - compat_mean)**2 for x in compatibility_scores) / len(compatibility_scores)
        print(f"  Compatibility Scores: {compat_min:.4f} - {compat_max:.4f} (mean: {compat_mean:.4f}, var: {compat_variance:.6f})")
        print(f"  Score Distribution: {'✓ VARIED' if compat_variance > 1e-6 else '✗ UNIFORM (BUG!)'}")
    else:
        print(f"  Compatibility Scores: NONE COMPUTED")
    
    if score_variances:
        avg_variance = sum(score_variances) / len(score_variances)
        print(f"  Importance Score Variance: {avg_variance:.6f} ({'✓ VARIED' if avg_variance > 1e-6 else '✗ UNIFORM (BUG!)'})")
    
    varied_count = widen_diagnostics['varied_score_count']
    uniform_count = widen_diagnostics['uniform_score_count']
    total_scored = varied_count + uniform_count
    if total_scored > 0:
        print(f"  Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored:.1f}%)")
        print(f"  Ranking Algorithm: {'✓ HEALTHY' if varied_count > uniform_count else '✗ FAILING (UNIFORM SCORES!)'}")
    
    if skip_threshold > 0.0:
        skip_effectiveness = widen_diagnostics['parameters_skipped_threshold']
        print(f"  Skip Threshold: {skip_threshold} -> {skip_effectiveness} parameters skipped")
        print(f"  Threshold Status: {'✓ ACTIVE' if skip_effectiveness > 0 else 'ⓘ NO EFFECT'}")
    
    # Critical diagnostic: If no parameters merged, investigate Phase 1
    if total_merged_count == 0:
        print(f"[CRITICAL] Zero parameters merged! Phase 1 rankings: {len(magnitude_rankings)} magnitude, {len(direction_rankings)} direction")
        print(f"[CRITICAL] Parameter list sample: {param_names_to_merge[:5] if param_names_to_merge else 'EMPTY'}")
        print(f"[CRITICAL] Magnitude rankings sample: {list(magnitude_rankings.keys())[:5] if magnitude_rankings else 'EMPTY'}")
    
    # Verify we processed all parameters (critical for WIDEN integrity)
    processed_count = len(merged_params)
    if processed_count != len(param_names_to_merge):
        print(f"[WARNING] Parameter count mismatch: processed {processed_count}, expected {len(param_names_to_merge)}")
    else:
        print(f"[ENHANCED WIDEN] ✓ All parameters processed - WIDEN merge integrity maintained")
    
    # Clean up intermediate variables to free memory
    del models_to_merge_param_magnitude_direction_diff_tuples
    del pretrained_param_dict
    del models_to_merge_task_vectors
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return merged_params, widen_diagnostics

# LoRA Delta-Only Processing Classes
class LoRADelta:
    """Memory-efficient LoRA delta storage for WIDEN merging"""
    def __init__(self, base_model, lora_model, lora_name="unknown"):
        self.base_model = base_model  # Reference to base model (no copy)
        self.lora_name = lora_name
        self.deltas = {}  # Only store the differences
        self.param_metadata = {}
        
        print(f"[LoRADelta] Computing deltas for {lora_name}...")
        self._compute_deltas(lora_model)
        
    def _compute_deltas(self, lora_model):
        """Compute only the parameter differences between base and LoRA-enhanced model"""
        try:
            base_params = dict(self.base_model.named_parameters())
            lora_params = dict(lora_model.named_parameters())
            
            delta_count = 0
            total_delta_size = 0
            
            for name, lora_param in lora_params.items():
                if name in base_params:
                    base_param = base_params[name]
                    if base_param.shape == lora_param.shape:
                        # Compute delta
                        delta = lora_param.detach().cpu().float() - base_param.detach().cpu().float()
                        
                        # Only store if there's a meaningful difference
                        delta_magnitude = torch.norm(delta).item()
                        if delta_magnitude > 1e-8:
                            self.deltas[name] = delta
                            self.param_metadata[name] = {
                                'delta_magnitude': delta_magnitude,
                                'base_magnitude': torch.norm(base_param).item(),
                                'change_ratio': delta_magnitude / (torch.norm(base_param).item() + 1e-8)
                            }
                            delta_count += 1
                            total_delta_size += delta.numel() * 4  # 4 bytes per float32
            
            # Memory usage summary
            size_mb = total_delta_size / (1024 * 1024)
            print(f"[LoRADelta] {self.lora_name}: {delta_count} changed parameters, {size_mb:.1f}MB delta storage")
            
        except Exception as e:
            print(f"[LoRADelta] Error computing deltas for {self.lora_name}: {e}")
            self.deltas = {}
    
    def get_parameter(self, name):
        """Get parameter value (base + delta if exists)"""
        if name in self.deltas:
            base_param = dict(self.base_model.named_parameters())[name]
            return base_param + self.deltas[name].to(base_param.device)
        else:
            return dict(self.base_model.named_parameters())[name]
    
    def named_parameters(self):
        """Generator that yields (name, parameter) tuples with deltas applied"""
        base_params = dict(self.base_model.named_parameters())
        for name, base_param in base_params.items():
            if name in self.deltas:
                # Apply delta
                enhanced_param = base_param + self.deltas[name].to(base_param.device)
                yield name, enhanced_param
            else:
                # Use base parameter unchanged
                yield name, base_param
    
    def get_delta_info(self):
        """Get information about stored deltas"""
        return {
            'lora_name': self.lora_name,
            'delta_count': len(self.deltas),
            'changed_parameters': list(self.deltas.keys()),
            'metadata': self.param_metadata
        }

class LoRAStackProcessor:
    """Process LoRA stacks efficiently for WIDEN merging"""
    def __init__(self, base_model, base_clip=None):
        self.base_model = base_model
        self.base_clip = base_clip
        self.lora_deltas = []
        
    def add_lora_from_stack(self, lora_stack):
        """Add LoRAs from a LoRA stack, creating deltas for each"""
        if lora_stack is None:
            return
            
        try:
            # Handle LoRA stack format from DonutLoRAStack: list of (name, model_weight, clip_weight, block_vector)
            if hasattr(lora_stack, '__iter__'):
                for idx, lora_item in enumerate(lora_stack):
                    lora_name = f"LoRA_{idx+1}"
                    if isinstance(lora_item, tuple) and len(lora_item) >= 4:
                        name, model_weight, clip_weight, block_vector = lora_item
                        lora_name = f"{name}_{idx+1}"
                    self._process_single_lora(lora_item, lora_name)
            else:
                self._process_single_lora(lora_stack, "LoRA_1")
                
        except Exception as e:
            print(f"[LoRAStackProcessor] Error processing LoRA stack: {e}")
    
    def _process_single_lora_for_unet(self, lora_item, lora_name):
        """Process UNet part of LoRA (Step 1 of DonutApplyLoRAStack)"""
        try:
            # Extract LoRA details from DonutLoRAStack format: (name, model_weight, clip_weight, block_vector)
            if isinstance(lora_item, tuple) and len(lora_item) >= 4:
                name, model_weight, clip_weight, block_vector = lora_item
            else:
                print(f"[LoRAStackProcessor] Invalid LoRA format for {lora_name}: {lora_item}")
                return
                
            print(f"[LoRADelta] Processing UNet LoRA {lora_name}: {name} (model_weight: {model_weight})")
            
            try:
                # Use ComfyUI's LoRA loading system (same as DonutApplyLoRAStack Step 1)
                import comfy.utils
                import folder_paths
                from .lora_block_weight import LoraLoaderBlockWeight
                
                # Get the full path to the LoRA file
                path = folder_paths.get_full_path("loras", name)
                if path is None:
                    raise FileNotFoundError(f"LoRA file not found: {name}")
                
                print(f"[LoRADelta] Loading UNet LoRA from: {path}")
                
                # Load the LoRA file
                lora = comfy.utils.load_torch_file(path, safe_load=True)
                
                # Create a temporary copy of the base model to apply LoRA
                if hasattr(self.base_model, 'clone'):
                    temp_model = self.base_model.clone()
                else:
                    import copy
                    temp_model = copy.copy(self.base_model)
                    if hasattr(self.base_model, 'model'):
                        temp_model.model = copy.deepcopy(self.base_model.model)
                
                # Apply UNet LoRA using block weights (DonutApplyLoRAStack Step 1)
                loader = LoraLoaderBlockWeight()
                vector = block_vector if block_vector else ",".join(["1"] * 12)
                
                # Step 1: block-weighted UNet merge (clip_strength=0)
                enhanced_model, _, _ = loader.load_lora_for_models(
                    temp_model, None, lora,  # UNet only, no CLIP
                    strength_model=model_weight,
                    strength_clip=0.0,  # No CLIP changes in UNet step
                    inverse=False,
                    seed=0,
                    A=1.0,
                    B=1.0,
                    block_vector=vector
                )
                
                # Create delta object comparing base vs LoRA-enhanced UNet
                lora_delta = LoRADelta(self.base_model, enhanced_model, lora_name)
                if len(lora_delta.deltas) > 0:
                    self.lora_deltas.append(lora_delta)
                    print(f"[LoRADelta] Successfully created UNet delta for {lora_name} with {len(lora_delta.deltas)} changed parameters")
                else:
                    print(f"[LoRADelta] No UNet deltas found for {lora_name}, skipping")
                
                # Clean up temporary model
                del temp_model, enhanced_model
                gc.collect()
                
            except Exception as e:
                print(f"[LoRADelta] Error processing UNet LoRA {name}: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"[LoRAStackProcessor] Error processing UNet LoRA {lora_name}: {e}")

    def _process_single_lora_for_clip(self, lora_item, lora_name):
        """Process CLIP part of LoRA (Step 2 of DonutApplyLoRAStack)"""
        try:
            # Extract LoRA details from DonutLoRAStack format: (name, model_weight, clip_weight, block_vector)
            if isinstance(lora_item, tuple) and len(lora_item) >= 4:
                name, model_weight, clip_weight, block_vector = lora_item
            else:
                print(f"[LoRAStackProcessor] Invalid LoRA format for {lora_name}: {lora_item}")
                return
                
            print(f"[LoRADelta] Processing CLIP LoRA {lora_name}: {name} (clip_weight: {clip_weight})")
            
            try:
                # Use ComfyUI's LoRA loading system (same as DonutApplyLoRAStack Step 2)
                import comfy.utils
                import comfy.sd
                import folder_paths
                
                # Get the full path to the LoRA file
                path = folder_paths.get_full_path("loras", name)
                if path is None:
                    raise FileNotFoundError(f"LoRA file not found: {name}")
                
                print(f"[LoRADelta] Loading CLIP LoRA from: {path}")
                
                # Load the LoRA file
                lora = comfy.utils.load_torch_file(path, safe_load=True)
                
                # Create a temporary copy of the base CLIP to apply LoRA
                if hasattr(self.base_model, 'clone'):
                    temp_clip = self.base_model.clone()
                else:
                    import copy
                    temp_clip = copy.copy(self.base_model)
                    # For CLIP, copy the actual encoder
                    if hasattr(self.base_model, 'cond_stage_model'):
                        temp_clip.cond_stage_model = copy.deepcopy(self.base_model.cond_stage_model)
                    elif hasattr(self.base_model, 'clip'):
                        temp_clip.clip = copy.deepcopy(self.base_model.clip)
                
                # Step 2: uniform CLIP merge (no block control)
                _, enhanced_clip = comfy.sd.load_lora_for_models(
                    None, temp_clip, lora,
                    0.0,         # No UNet change in CLIP step
                    clip_weight  # CLIP strength
                )
                
                # Create delta object comparing base vs LoRA-enhanced CLIP
                lora_delta = LoRADelta(self.base_model, enhanced_clip, lora_name)
                if len(lora_delta.deltas) > 0:
                    self.lora_deltas.append(lora_delta)
                    print(f"[LoRADelta] Successfully created CLIP delta for {lora_name} with {len(lora_delta.deltas)} changed parameters")
                else:
                    print(f"[LoRADelta] No CLIP deltas found for {lora_name}, skipping")
                
                # Clean up temporary model
                del temp_clip, enhanced_clip
                gc.collect()
                
            except Exception as e:
                print(f"[LoRADelta] Error processing CLIP LoRA {name}: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"[LoRAStackProcessor] Error processing CLIP LoRA {lora_name}: {e}")

    def _process_single_lora(self, lora_item, lora_name):
        """Process LoRA for the specific model type (UNet or CLIP)"""
        # Determine if we're processing UNet or CLIP and call the appropriate method
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'diffusion_model'):
            # This is a UNet MODEL - process UNet LoRA
            self._process_single_lora_for_unet(lora_item, lora_name)
        elif hasattr(self.base_model, 'cond_stage_model') or hasattr(self.base_model, 'clip'):
            # This is a CLIP object - process CLIP LoRA  
            self._process_single_lora_for_clip(lora_item, lora_name)
        else:
            print(f"[LoRADelta] Unknown model type for {lora_name}, skipping")
    
    
    def get_virtual_models(self):
        """Return list of virtual models (base + each delta)"""
        models = [self.base_model]  # Include base model
        models.extend(self.lora_deltas)  # Add delta models
        return models
    
    def get_summary(self):
        """Get summary of processed LoRAs"""
        total_deltas = sum(len(delta.deltas) for delta in self.lora_deltas)
        return {
            'base_model': 'included',
            'lora_count': len(self.lora_deltas),
            'total_delta_parameters': total_deltas,
            'lora_names': [delta.lora_name for delta in self.lora_deltas]
        }

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

def compute_merge_hash(models, merge_strength, importance_threshold, importance_boost, rank_sensitivity, skip_threshold, normalization_mode):
    """Compute hash of merge parameters to detect changes - FIXED: More robust hashing"""
    hasher = hashlib.sha256()  # Changed from md5 to sha256 for better collision resistance

    # Hash model inputs - FIXED: Use model state checksum instead of object ID
    for model in models:
        if model is not None and not getattr(model, "_is_filler", False):
            try:
                # Create a more stable hash based on model parameters
                model_params = list(model.named_parameters()) if hasattr(model, 'named_parameters') else []
                if model_params:
                    # Use first and last parameter shapes and a few sample values for hash
                    first_param = model_params[0][1] if model_params else None
                    last_param = model_params[-1][1] if len(model_params) > 1 else first_param

                    if first_param is not None:
                        hasher.update(str(first_param.shape).encode())
                        hasher.update(str(first_param.flatten()[:10].tolist()).encode())
                    if last_param is not None and last_param is not first_param:
                        hasher.update(str(last_param.shape).encode())
                        hasher.update(str(last_param.flatten()[:10].tolist()).encode())
                else:
                    # Fallback to object id with timestamp for uniqueness
                    hasher.update(f"{id(model)}_{time.time()}".encode())
            except Exception:
                # Ultimate fallback
                hasher.update(f"{id(model)}_{time.time()}".encode())

    # Hash merge parameters
    hasher.update(f"{merge_strength}_{importance_threshold}_{importance_boost}_{rank_sensitivity}_{skip_threshold}_{normalization_mode}".encode())

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
    """Renormalize using calibration algorithm or simple methods - ENHANCED with conservative options"""
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
        # More conservative calibration-style renormalization with adjustable parameters
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
                # ENHANCED: Adjustable softmax temperature for more/less smoothing
                # Lower t = more conservative (sharper), higher t = more smoothing
                temperature = max(0.1, min(2.0, t))  # Clamp between 0.1-2.0
                sm_m = F.softmax(param_flat * temperature, dim=0)

                # ENHANCED: Adjustable calibration thresholding
                # Lower t = higher threshold (more selective), higher t = lower threshold
                K = param_flat.numel()
                threshold_factor = max(0.1, min(1.0, t))  # Clamp between 0.1-1.0
                thr_m = (threshold_factor / K) * sm_m.sum() * 0.5

                # ENHANCED: Adjustable scaling intensity
                # Lower s = less aggressive scaling, higher s = more aggressive
                scaling_factor = max(1.0, min(3.0, s))  # Clamp between 1.0-3.0
                conservative_intensity = max(0.1, min(1.0, (scaling_factor - 1.0) * 0.5))  # More conservative
                cal_m = torch.where(sm_m > thr_m, 1.0 + conservative_intensity * sm_m, sm_m)

                # Renormalize to preserve relative magnitudes
                if cal_m.sum() > 1e-8:
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

        # FIXED: Proper memory management with explicit cleanup
        try:
            base_params = {n: p.detach().cpu().float()  # FIXED: Removed redundant .clone()
                          for n, p in base_model.named_parameters()}
            finetuned_params = {n: p.detach().cpu().float()  # FIXED: Removed redundant .clone()
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
                    base_magnitude = torch.norm(base_params[name]).item()
                    delta_magnitude = torch.norm(delta).item()
                    self.param_metadata[name] = {
                        'layer_type': self._classify_sdxl_layer(name),
                        'base_magnitude': base_magnitude,
                        'delta_magnitude': delta_magnitude,
                        'change_ratio': delta_magnitude / (base_magnitude + 1e-8)  # FIXED: Avoid division by zero
                    }

        finally:
            # FIXED: Explicit cleanup to prevent memory leaks
            if 'base_params' in locals():
                del base_params
            if 'finetuned_params' in locals():
                del finetuned_params
            gc.collect()

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

        # SDXL-specific thresholds based on WIDEN paper principles - FIXED: Much higher base thresholds
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

        # Debug logging for high thresholds
        if widen_threshold > 0.8:
            print(f"[DEBUG] {param_name[:30]}: mag={delta_magnitude:.6f} vs thresh={threshold:.6f}, "
                  f"ratio={change_ratio:.6f} vs ratio_thresh={scaled_change_threshold:.6f}, "
                  f"passes={magnitude_check and ratio_check}")

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
                elif sig_tensor.ndim == 1:
                    softmax_dim = 0
                else: # ndim > 1
                    softmax_dim = -1 # FIXED: Apply softmax along the last dimension

                # Apply softmax with numerical stability
                sig_scaled = sig_tensor * layer_weight
                # Clamp to prevent overflow
                sig_scaled = torch.clamp(sig_scaled, min=-50, max=50)
                sc = torch.softmax(sig_scaled, dim=softmax_dim)

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

                # FIXED: Apply calibration with 0-1 mapping to 0.1-2.0 range
                # 0.0 = minimal importance weighting (0.1x)
                # 0.5 = standard importance weighting (1.0x)
                # 1.0 = maximum importance weighting (2.0x)
                calibration_mapped = 0.1 + calibration_value * 1.9
                calibration_scaled = calibration_mapped * layer_weight
                sc = torch.where(mask, torch.tensor(calibration_scaled, dtype=sc.dtype, device=sc.device), sc)

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
        """SDXL-optimized parameter merging with layer-aware weighting - FIXED: Updated parameter name"""
        try:
            layer_type = metadata.get('layer_type', 'other')

            # FIXED: More robust delta magnitude calculation
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

            # FIXED: Robust importance combination with proper shape validation
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
                    # FIXED: Better fallback - use scalar weights instead of tensor operations
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

                # FIXED: Ensure combined_weights is always scalar when multiplying with layer_weight
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

            # FIXED: Simplified tensor weighting - always use scalar weights to avoid shape issues
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

                # FIXED: Verify shape consistency more robustly
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
        """FULL ZERO-ACCUMULATION WIDEN algorithm for SDXL - No intermediate data storage - FIXED: Better memory management"""

        results_text = f"[{self.method}] Starting FULL ZERO-ACCUMULATION SDXL WIDEN merge\n"
        results_text += f"[{self.method}] Threshold: {widen_threshold}, Calibration: {calibration_value}\n"

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

        # FIXED: Clean up the parameter list immediately
        del base_param_names
        gc.collect()

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

        # FIXED: Clean up parameter names immediately
        del param_names_only
        gc.collect()

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
                        base_param = param.detach().cpu().float()  # FIXED: Removed redundant .clone()
                        break

                if base_param is None:
                    skipped_count += 1
                    continue

                # Get deltas from each model (one at a time)
                for model in task_vector_models:
                    other_param = None
                    for param_name, param in model.named_parameters():
                        if param_name == name:
                            other_param = param.detach().cpu().float()  # FIXED: Removed redundant .clone()
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

                # FIXED: More robust metadata calculation
                base_magnitude = torch.norm(base_param).item()
                delta_magnitudes = [torch.norm(d).item() for d in deltas if d.numel() > 0]
                avg_delta_magnitude = sum(delta_magnitudes) / max(len(delta_magnitudes), 1)

                metadata = {
                    'layer_type': layer_type,
                    'base_magnitude': base_magnitude,
                    'delta_magnitude': avg_delta_magnitude,
                    'change_ratio': avg_delta_magnitude / (base_magnitude + 1e-8)  # FIXED: Avoid division by zero
                }

                # DIAGNOSTIC: Log threshold analysis for first few parameters
                if param_idx < 10:
                    base_threshold = self.sdxl_thresholds.get(layer_type, 0.0001)
                    if widen_threshold <= 0.5:
                        threshold_multiplier = 0.001 + (widen_threshold * 2) ** 3 * 0.999
                    else:
                        threshold_multiplier = 1.0 + ((widen_threshold - 0.5) * 2) ** 4 * 999
                    final_threshold = base_threshold * threshold_multiplier

                    print(f"[DIAGNOSTIC] {name[:30]}: layer={layer_type}, "
                          f"delta_mag={avg_delta_magnitude:.6f}, base_thresh={base_threshold:.6f}, "
                          f"multiplier={threshold_multiplier:.3f}, final_thresh={final_threshold:.6f}, "
                          f"passes={avg_delta_magnitude >= final_threshold}")

                # Early threshold check for efficiency
                if not self.should_merge_parameter(name, avg_delta_magnitude, metadata, widen_threshold):
                    skipped_count += 1
                    layer_stats[layer_type]['skipped'] += 1
                    del base_param, deltas
                    continue

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
                        try:
                            b_dir_p = base_dir[name]
                            o_dir_p = other_dir[name]
                            
                            # Case 1: Per-feature magnitude case (Linear layers)
                            is_per_feature_magnitude_case = (
                                name in base_mag and
                                base_mag[name].ndim == 1 and
                                b_dir_p.ndim == 2 and o_dir_p.ndim == 2 and
                                b_dir_p.shape == o_dir_p.shape and
                                b_dir_p.shape[0] == base_mag[name].shape[0]
                            )

                            if is_per_feature_magnitude_case:
                                # Compute cosine similarity along feature embedding dimension
                                cos_sim = torch.cosine_similarity(o_dir_p, b_dir_p, dim=-1)
                                cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                                layer_weight_val = self.sdxl_importance_weights.get(layer_type, 1.0)
                                current_dir_diff_val = (1.0 - cos_sim) * layer_weight_val
                                dir_diffs.append(current_dir_diff_val)
                            else:
                                # Case 2: Scalar magnitude case (bias, etc.)
                                if b_dir_p.numel() == 1 and o_dir_p.numel() == 1:
                                    dir_diff = torch.abs(o_dir_p - b_dir_p)
                                    layer_weight_val = self.sdxl_importance_weights.get(layer_type, 1.0)
                                    dir_diffs.append(dir_diff * layer_weight_val)
                                else:
                                    # Case 3: General tensor case - flatten and compute cosine similarity
                                    try:
                                        base_flat = b_dir_p.view(-1)
                                        other_flat = o_dir_p.view(-1)
                                        cos_sim = torch.cosine_similarity(other_flat, base_flat, dim=0)
                                        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                                        dir_diff = 1 - cos_sim
                                        layer_weight_val = self.sdxl_importance_weights.get(layer_type, 1.0)
                                        dir_diffs.append(dir_diff * layer_weight_val)
                                    except Exception as fallback_e:
                                        # Fallback to mean absolute difference
                                        dir_diff = torch.abs(o_dir_p - b_dir_p).mean()
                                        layer_weight_val = self.sdxl_importance_weights.get(layer_type, 1.0)
                                        dir_diffs.append(dir_diff * layer_weight_val)
                        except Exception as e:
                            # FIXED: Better error handling for direction computation
                            print(f"Warning: Direction computation failed for {name}: {e}")
                            dir_diff = torch.tensor(0.1, dtype=torch.float32)  # Small default value
                            layer_weight = self.sdxl_importance_weights.get(layer_type, 1.0)
                            dir_diffs.append(dir_diff * layer_weight)

                    del other_param  # Immediate cleanup

                # Clean up magnitude/direction data immediately
                del base_mag, base_dir

                # STEP 4: Apply WIDEN algorithm with immediate cleanup
                if len(mag_diffs) == 0 or len(dir_diffs) == 0:
                    # WIDEN failed - check if we should still merge with simple average
                    avg_delta_mag = metadata['delta_magnitude']
                    if not self.should_merge_parameter(name, avg_delta_mag, metadata, widen_threshold):
                        skipped_count += 1
                        layer_stats[layer_type]['skipped'] += 1
                        del base_param, deltas
                        continue

                    # Fallback to simple average (silent)
                    try:
                        if len(deltas) > 0:
                            avg_delta = sum(deltas) / len(deltas)
                            final_merged = base_param + avg_delta * merge_strength
                            del deltas, avg_delta  # Immediate cleanup
                        else:
                            final_merged = base_param
                            del deltas
                    except Exception as e:
                        failed_count += 1
                        layer_stats[layer_type]['failed'] += 1
                        del base_param, deltas
                        continue
                else:
                    try:
                        # FIXED: Better tensor stacking with validation
                        if not deltas:
                            final_merged = base_param
                        else:
                            deltas_tensor = torch.stack(deltas)

                            # FIXED: Enhanced tensor stacking validation for mag_diffs and dir_diffs
                            # Ensure all mag_diffs have the same shape before stacking
                            if mag_diffs and all(isinstance(d, torch.Tensor) for d in mag_diffs):
                                mag_shapes = [d.shape for d in mag_diffs]
                                if all(shape == mag_shapes[0] for shape in mag_shapes):
                                    mag_diffs_tensor = torch.stack(mag_diffs)
                                else:
                                    # Convert to scalars if shapes are inconsistent
                                    mag_scalars = [d.mean().item() if d.numel() > 1 else d.item() for d in mag_diffs]
                                    mag_diffs_tensor = torch.tensor(mag_scalars, dtype=torch.float32)
                            else:
                                # Fallback for empty or invalid mag_diffs
                                mag_diffs_tensor = torch.ones(len(deltas), dtype=torch.float32)

                            if dir_diffs and all(isinstance(d, torch.Tensor) for d in dir_diffs):
                                dir_shapes = [d.shape for d in dir_diffs]
                                if all(shape == dir_shapes[0] for shape in dir_shapes):
                                    dir_diffs_tensor = torch.stack(dir_diffs)
                                else:
                                    # Convert to scalars if shapes are inconsistent
                                    dir_scalars = [d.mean().item() if d.numel() > 1 else d.item() for d in dir_diffs]
                                    dir_diffs_tensor = torch.tensor(dir_scalars, dtype=torch.float32)
                            else:
                                # Fallback for empty or invalid dir_diffs
                                dir_diffs_tensor = torch.ones(len(deltas), dtype=torch.float32)

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
                                name, metadata, widen_threshold, calibration_value
                            )

                            # Clean up intermediate tensors immediately
                            del deltas_tensor, mag_ranks, dir_ranks

                            # Apply strength
                            final_merged = base_param + (merged_param - base_param) * merge_strength
                            del merged_param  # Immediate cleanup

                    except Exception as e:
                        # WIDEN failed - check threshold before fallback
                        avg_delta_mag = metadata['delta_magnitude']
                        if not self.should_merge_parameter(name, avg_delta_mag, metadata, widen_threshold):
                            skipped_count += 1
                            layer_stats[layer_type]['skipped'] += 1
                            del base_param
                            if 'deltas' in locals():
                                del deltas
                            continue

                        # Silent fallback for WIDEN failures
                        try:
                            if 'deltas' in locals() and deltas:
                                avg_delta = sum(deltas) / len(deltas)
                                final_merged = base_param + avg_delta * merge_strength
                                del deltas, avg_delta
                            else:
                                final_merged = base_param
                                if 'deltas' in locals():
                                    del deltas
                        except Exception as e2:
                            failed_count += 1
                            layer_stats[layer_type]['failed'] += 1
                            del base_param
                            if 'deltas' in locals():
                                del deltas
                            continue

                # STEP 5: Apply renormalization and write to target (zero-accumulation)
                if renorm_mode != "none":
                    try:
                        if renorm_mode == "calibrate":
                            # FIXED: More conservative calibrate parameters
                            # t=0.3 (lower = more selective), s=1.1 (lower = less aggressive scaling)
                            final_merged = calibrate_renormalize(
                                final_merged, base_param, renorm_mode, 0.3, 1.1
                            )
                        else:  # magnitude
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

                # FIXED: Less aggressive cleanup - only every 100 parameters instead of 50
                if param_idx % 100 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()

            except Exception as e:
                failed_count += 1
                if layer_type not in layer_stats:
                    layer_stats[layer_type] = {'merged': 0, 'skipped': 0, 'failed': 0}
                layer_stats[layer_type]['failed'] += 1

                # Simple cleanup on exception
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                continue

        total_params = len(common_params)

        # Generate detailed layer-wise results
        layer_report = "\n[LAYER-WISE RESULTS]:\n"
        for layer_type, stats in sorted(layer_stats.items()):
            total_layer = sum(stats.values())
            if total_layer > 0:
                layer_report += f"  {layer_type}: {stats['merged']}/{total_layer} merged "
                layer_report += f"({stats['merged']/total_layer*100:.1f}% success), "
                layer_report += f"{stats['skipped']} skipped, {stats['failed']} failed\n"

        # FIXED: Better summary stats
        total_processed = merged_count + skipped_count + failed_count
        fallback_count = merged_count - sum(1 for layer_stats_dict in layer_stats.values()
                                          for count in [layer_stats_dict.get('merged', 0)])

        results_text += f"""
[RESULTS] FULL ZERO-ACCUMULATION WIDEN merge complete:
  - Total parameters processed: {total_processed}
  - Successfully merged with WIDEN: {merged_count}/{total_params} parameters ({merged_count/total_params*100:.1f}%)
  - Skipped (below threshold): {skipped_count} ({skipped_count/total_params*100:.1f}%)
  - Failed: {failed_count}
  - Threshold effectiveness: {(total_params - skipped_count)/total_params*100:.1f}% of parameters met threshold
  - Renormalization: {'enabled' if renorm_mode != 'none' else 'disabled'} (mode: {renorm_mode})
  - Full zero-accumulation: ✓ (absolute minimal memory footprint)
{layer_report}
[THRESHOLD ANALYSIS]:
  - widen_threshold: {widen_threshold} (0.0=permissive, 1.0=selective)
  - Parameters above threshold: {merged_count + failed_count}/{total_params}
  - Selectivity working: {'YES' if skipped_count > 0 else 'NO - All parameters passed threshold'}"""

        print(results_text)

        # FIXED: Extra aggressive cleanup at end of merge
        print("[CLEANUP] Post-merge aggressive cleanup...")
        del target_state_dict, layer_stats
        force_cleanup()
        force_cleanup()  # Double cleanup for stubborn references

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
                "normalization_mode": (["magnitude", "calibrate", "none"], {"default": "magnitude"}),  # (renorm_mode)
                # Enhanced WIDEN parameters
                "importance_threshold": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 500.0, "step": 0.1}),  # (above_average_value_ratio)
                "importance_boost": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 3.0, "step": 0.1}),  # (score_calibration_value)
                # Dynamic compatibility settings  
                "rank_sensitivity": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),  # (compatibility_sensitivity)
                "skip_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),  # (compatibility_threshold)
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
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

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "merge_results", "parameter_info")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, model_base, model_other, merge_strength, normalization_mode,
                importance_threshold, importance_boost,
                rank_sensitivity, skip_threshold,
                lora_stack=None, model_3=None, model_4=None, model_5=None, model_6=None,
                model_7=None, model_8=None, model_9=None, model_10=None,
                model_11=None, model_12=None):

        # FIXED: Aggressive memory cleanup before starting
        print("[MEMORY] Pre-merge cleanup...")
        force_cleanup()

        # Check cache first
        all_models = [model_base, model_other, model_3, model_4, model_5, model_6,
                     model_7, model_8, model_9, model_10, model_11, model_12]
        cache_key = compute_merge_hash(all_models, merge_strength, importance_threshold, importance_boost, rank_sensitivity, skip_threshold, f"{normalization_mode}_enhanced_widen")

        cached_result = check_cache_for_merge(cache_key)
        if cached_result is not None:
            return cached_result

        with memory_cleanup_context("DonutWidenMergeUNet"):
            import copy
            import gc

            # Process LoRA stack if provided
            lora_processor = None
            if lora_stack is not None:
                print("[DonutWidenMergeUNet] Processing LoRA stack for delta-based merging...")
                # For UNet LoRA processing, we need a CLIP object for proper key mapping
                # We'll pass None for now and warn about potential key mapping issues
                lora_processor = LoRAStackProcessor(model_base, base_clip=None)
                lora_processor.add_lora_from_stack(lora_stack)
                
                # Get summary of LoRA processing
                summary = lora_processor.get_summary()
                print(f"[LoRADelta] Processed {summary['lora_count']} LoRAs with {summary['total_delta_parameters']} delta parameters")
                print(f"[LoRADelta] LoRA names: {summary['lora_names']}")

            # FIXED: Filter out None models and filler models more safely
            models_to_merge = []
            for m in all_models[1:]:  # Skip model_base
                if m is not None and not getattr(m, "_is_filler", False):
                    models_to_merge.append(m)

            # Add LoRA-enhanced virtual models if available
            if lora_processor is not None:
                virtual_models = lora_processor.get_virtual_models()
                # Skip the base model (first item) since we already have it
                lora_virtual_models = virtual_models[1:]  # Only LoRA deltas
                models_to_merge.extend(lora_virtual_models)
                print(f"[LoRADelta] Added {len(lora_virtual_models)} LoRA-enhanced virtual models")

            print(f"[DonutWidenMergeUNet] WIDEN merging {len(models_to_merge)} models ({len([m for m in models_to_merge if hasattr(m, 'lora_name')])} from LoRA stack)")

            try:
                base_model_obj = model_base.model
                # Handle both regular models and LoRADelta objects
                other_model_objs = []
                for model in models_to_merge:
                    if hasattr(model, 'lora_name'):  # LoRADelta object
                        other_model_objs.append(model)
                    else:  # Regular model
                        other_model_objs.append(model.model)

                # FIXED: Use deepcopy only for the wrapper, not the heavy model tensors
                model_merged = copy.copy(model_base)  # Shallow copy of wrapper
                model_merged.model = copy.deepcopy(base_model_obj)  # Deep copy only the model

                # Use enhanced WIDEN merger with dynamic compatibility
                print("[DonutWidenMergeUNet] Using WIDEN merge with dynamic compatibility-based strength...")
                
                merger = MergingMethod("DonutWidenMergeUNet")
                
                if rank_sensitivity > 0.0:
                    # Use WIDEN with dynamic strength based on compatibility
                    merged_params, widen_diagnostics = enhanced_widen_merging_with_dynamic_strength(
                        merger=merger,
                        merged_model=model_merged.model,
                        models_to_merge=other_model_objs,
                        exclude_param_names_regex=[],
                        importance_threshold=importance_threshold,
                        importance_boost=importance_boost,
                        base_merge_strength=merge_strength,
                        rank_sensitivity=rank_sensitivity,
                        skip_threshold=skip_threshold,
                        normalization_mode=normalization_mode
                    )
                    
                    # Apply merged parameters with shape validation
                    applied_count = 0
                    shape_mismatch_count = 0
                    for param_name, param_value in merged_params.items():
                        for name, param in model_merged.model.named_parameters():
                            if name == param_name:
                                if param_value.shape == param.shape:
                                    param.data.copy_(param_value)
                                    applied_count += 1
                                else:
                                    print(f"[WARNING] Shape mismatch for {param_name}: expected {param.shape}, got {param_value.shape}")
                                    shape_mismatch_count += 1
                                break
                    
                    print(f"[ENHANCED WIDEN] Applied {applied_count} parameters, {shape_mismatch_count} shape mismatches")
                    
                    # Create detailed merge results with enhanced WIDEN diagnostics
                    total_models = len([m for m in [model_other, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10, model_11, model_12] if m is not None]) + 1
                    
                    # Generate enhanced diagnostics from widen_diagnostics
                    compatibility_scores = widen_diagnostics['compatibility_scores']
                    varied_count = widen_diagnostics['varied_score_count']
                    uniform_count = widen_diagnostics['uniform_score_count']
                    skipped_threshold = widen_diagnostics['parameters_skipped_threshold']
                    
                    if compatibility_scores:
                        compat_min, compat_max = min(compatibility_scores), max(compatibility_scores)
                        compat_mean = sum(compatibility_scores) / len(compatibility_scores)
                        compat_variance = sum((x - compat_mean)**2 for x in compatibility_scores) / len(compatibility_scores)
                        score_health = "✓ VARIED" if compat_variance > 1e-6 else "✗ UNIFORM (BUG!)"
                    else:
                        compat_min = compat_max = compat_mean = 0.0
                        score_health = "NO SCORES"
                    
                    ranking_health = "✓ HEALTHY" if varied_count > uniform_count else "✗ FAILING"
                    total_scored = varied_count + uniform_count
                    
                    results_text = f"""╔═ WIDEN MERGE RESULTS (Dynamic Compatibility) ═╗
║ Models: {total_models} | Strength: {merge_strength} | Mode: {normalization_mode}
║ Threshold: {importance_threshold} | Boost: {importance_boost} | Sensitivity: {rank_sensitivity}
╠═══════════════════════════════════════════════════╣
║ Parameters Merged: {len(merged_params)} | Applied: {applied_count}
║ Shape Mismatches: {shape_mismatch_count} | Success: {(applied_count/(applied_count+shape_mismatch_count)*100):.1f}%
║ Status: ✓ Enhanced WIDEN with Dynamic Compatibility
╠═══════════════════════════════════════════════════╣
║ 🔍 WIDEN ALGORITHM HEALTH DIAGNOSTICS:
║ Compatibility Range: {compat_min:.4f} - {compat_max:.4f} (avg: {compat_mean:.4f})
║ Score Distribution: {score_health}
║ Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored if total_scored > 0 else 0:.1f}%) - {ranking_health}
║ Skip Threshold: {skip_threshold} → {skipped_threshold} parameters skipped
╚═══════════════════════════════════════════════════╝"""
                    # Create detailed parameter information
                    parameter_info = f"""╔═ WIDEN PARAMETER DETAILS ═╗
║ Merge Strength: Controls blend intensity between models
║ Importance Threshold: Multiplier for classifying important parameters
║ Importance Boost: Score amplification for important parameters  
║ Rank Sensitivity: Dynamic compatibility adjustment strength
║ Skip Threshold: Excludes low-compatibility parameters from merge
║ Normalization: {normalization_mode} - post-merge parameter scaling
╠═══════════════════════════════════════════════════╣
║ Dynamic Mode: Adapts strength based on compatibility
║ Total Processed: {len(merged_params)} parameters analyzed
║ Applied Successfully: {applied_count} parameters merged
╚═══════════════════════════════════════════════════╝"""
                    
                else:
                    # Use enhanced WIDEN without dynamic strength (fallback mode)
                    merged_params, widen_diagnostics = enhanced_widen_merging_with_dynamic_strength(
                        merger=merger,
                        merged_model=model_merged.model,
                        models_to_merge=other_model_objs,
                        exclude_param_names_regex=[],
                        importance_threshold=importance_threshold,
                        importance_boost=importance_boost,
                        base_merge_strength=merge_strength,
                        rank_sensitivity=0.0,  # Disable dynamic strength
                        skip_threshold=skip_threshold,
                        normalization_mode=normalization_mode
                    )
                    
                    # Apply merged parameters with shape validation
                    applied_count = 0
                    shape_mismatch_count = 0
                    for param_name, param_value in merged_params.items():
                        for name, param in model_merged.model.named_parameters():
                            if name == param_name:
                                if param_value.shape == param.shape:
                                    param.data.copy_(param_value)
                                    applied_count += 1
                                else:
                                    print(f"[WARNING] Shape mismatch for {param_name}: expected {param.shape}, got {param_value.shape}")
                                    shape_mismatch_count += 1
                                break
                    
                    print(f"[ENHANCED WIDEN] Applied {applied_count} parameters, {shape_mismatch_count} shape mismatches")
                    
                    # Create detailed merge results with enhanced WIDEN diagnostics
                    total_models = len([m for m in [model_other, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10, model_11, model_12] if m is not None]) + 1
                    
                    # Generate enhanced diagnostics from widen_diagnostics
                    compatibility_scores = widen_diagnostics['compatibility_scores']
                    varied_count = widen_diagnostics['varied_score_count']
                    uniform_count = widen_diagnostics['uniform_score_count']
                    skipped_threshold = widen_diagnostics['parameters_skipped_threshold']
                    
                    if compatibility_scores:
                        compat_min, compat_max = min(compatibility_scores), max(compatibility_scores)
                        compat_mean = sum(compatibility_scores) / len(compatibility_scores)
                        compat_variance = sum((x - compat_mean)**2 for x in compatibility_scores) / len(compatibility_scores)
                        score_health = "✓ VARIED" if compat_variance > 1e-6 else "✗ UNIFORM (BUG!)"
                    else:
                        compat_min = compat_max = compat_mean = 0.0
                        score_health = "NO SCORES"
                    
                    ranking_health = "✓ HEALTHY" if varied_count > uniform_count else "✗ FAILING"
                    total_scored = varied_count + uniform_count
                    
                    results_text = f"""╔═ WIDEN MERGE RESULTS (Static Strength) ═╗
║ Models: {total_models} | Strength: {merge_strength} | Mode: {normalization_mode}
║ Threshold: {importance_threshold} | Boost: {importance_boost} | Sensitivity: {rank_sensitivity} (off)
╠═══════════════════════════════════════════════════╣
║ Parameters Merged: {len(merged_params)} | Applied: {applied_count}
║ Shape Mismatches: {shape_mismatch_count} | Success: {(applied_count/(applied_count+shape_mismatch_count)*100):.1f}%
║ Status: ✓ Enhanced WIDEN with Static Strength
╠═══════════════════════════════════════════════════╣
║ 🔍 WIDEN ALGORITHM HEALTH DIAGNOSTICS:
║ Compatibility Range: {compat_min:.4f} - {compat_max:.4f} (avg: {compat_mean:.4f})
║ Score Distribution: {score_health}
║ Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored if total_scored > 0 else 0:.1f}%) - {ranking_health}
║ Skip Threshold: {skip_threshold} → {skipped_threshold} parameters skipped
╚═══════════════════════════════════════════════════╝"""
                    # Create detailed parameter information  
                    parameter_info = f"""╔═ WIDEN PARAMETER DETAILS ═╗
║ Merge Strength: Controls blend intensity between models
║ Importance Threshold: Multiplier for classifying important parameters
║ Importance Boost: Score amplification for important parameters
║ Rank Sensitivity: {rank_sensitivity} (disabled) - no dynamic adjustment
║ Skip Threshold: Excludes low-compatibility parameters from merge
║ Normalization: {normalization_mode} - post-merge parameter scaling
╠═══════════════════════════════════════════════════╣
║ Static Mode: Uses fixed strength for all parameters
║ Total Processed: {len(merged_params)} parameters analyzed
║ Applied Successfully: {applied_count} parameters merged
╚═══════════════════════════════════════════════════╝"""

                # FIXED: Aggressive cleanup before returning
                del base_model_obj, other_model_objs, models_to_merge
                if rank_sensitivity <= 0.0:
                    del merger
                force_cleanup()

                result = (model_merged, results_text, parameter_info)

                # Store in cache
                store_merge_result(cache_key, result)

                return result

            except MemoryExhaustionError as e:
                print(f"[SAFETY] Memory exhaustion prevented crash: {e}")
                # FIXED: Cleanup on error
                force_cleanup()
                error_results = f"""╔═ WIDEN MERGE RESULTS (MEMORY ERROR) ═╗
║ ERROR: Memory exhaustion prevented crash
║ DETAILS: {str(e)[:40]}...
║ STATUS: ✗ Failed - Memory limit exceeded
║ FIX: Reduce batch size or model count
╚═══════════════════════════════════════════════════╝"""
                error_param_info = """╔═ ERROR PARAMETER INFO ═╗
║ Merge was terminated due to memory limits
║ No parameter analysis available
║ Try reducing model count or batch size
╚═══════════════════════════════════════════════════╝"""
                result = (model_base, error_results, error_param_info)
                store_merge_result(cache_key, result)
                return result

            except Exception as e:
                print(f"[DonutWidenMergeUNet] Error: {e}")
                # FIXED: Cleanup on error
                force_cleanup()
                if "memory" in str(e).lower():
                    error_results = f"""╔═ WIDEN MERGE RESULTS (MEMORY ERROR) ═╗
║ ERROR: Memory error prevented crash
║ DETAILS: {str(e)[:40]}...
║ STATUS: ✗ Failed - Memory limit exceeded
║ FIX: Reduce batch size or model count
╚═══════════════════════════════════════════════════╝"""
                    error_param_info = """╔═ ERROR PARAMETER INFO ═╗
║ Merge was terminated due to memory error
║ No parameter analysis available  
║ Try reducing model count or batch size
╚═══════════════════════════════════════════════════╝"""
                    result = (model_base, error_results, error_param_info)
                    store_merge_result(cache_key, result)
                    return result
                else:
                    raise


# VERSION CHECK - This should appear in logs if new code is loading
print("="*50)
print("LOADING DONUTWIDENMERGECLIP VERSION 7.0 - FULL ZERO-ACCUMULATION - BUGFIXED")
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
                "normalization_mode": (["magnitude", "calibrate", "none"], {"default": "magnitude"}),  # (renorm_mode)
                # Enhanced WIDEN parameters
                "importance_threshold": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 500.0, "step": 0.1}),  # (above_average_value_ratio)
                "importance_boost": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 3.0, "step": 0.1}),  # (score_calibration_value)
                # Dynamic compatibility settings  
                "rank_sensitivity": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),  # (compatibility_sensitivity)
                "skip_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),  # (compatibility_threshold)
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
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

    RETURN_TYPES = ("CLIP", "STRING", "STRING")
    RETURN_NAMES = ("clip", "merge_results", "parameter_info")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, clip_base, clip_other, merge_strength, normalization_mode,
                importance_threshold, importance_boost,
                rank_sensitivity, skip_threshold,
                lora_stack=None, clip_3=None, clip_4=None, clip_5=None, clip_6=None,
                clip_7=None, clip_8=None, clip_9=None, clip_10=None,
                clip_11=None, clip_12=None):

        # FIXED: Aggressive memory cleanup before starting
        print("[MEMORY] Pre-merge cleanup...")
        force_cleanup()

        # Check cache first
        all_clips = [clip_base, clip_other, clip_3, clip_4, clip_5, clip_6,
                    clip_7, clip_8, clip_9, clip_10, clip_11, clip_12]
        cache_key = compute_merge_hash(all_clips, merge_strength, importance_threshold, importance_boost, rank_sensitivity, skip_threshold, f"{normalization_mode}_enhanced_widen")

        cached_result = check_cache_for_merge(cache_key)
        if cached_result is not None:
            return cached_result

        with memory_cleanup_context("DonutWidenMergeCLIP"):
            import copy
            import gc

            # Get base encoder first for LoRA processing
            base_enc = getattr(clip_base, "model", getattr(clip_base, "clip",
                      getattr(clip_base, "cond_stage_model", None)))
            if not base_enc:
                raise AttributeError("Could not locate base CLIP encoder")

            # Process LoRA stack if provided
            lora_processor = None
            if lora_stack is not None:
                print("[DonutWidenMergeCLIP] Processing LoRA stack for delta-based merging...")
                lora_processor = LoRAStackProcessor(clip_base)  # Pass the wrapper, not base_enc
                lora_processor.add_lora_from_stack(lora_stack)
                
                # Get summary of LoRA processing
                summary = lora_processor.get_summary()
                print(f"[LoRADelta] Processed {summary['lora_count']} LoRAs with {summary['total_delta_parameters']} delta parameters")
                print(f"[LoRADelta] LoRA names: {summary['lora_names']}")

            # FIXED: Filter out None clips and filler clips more safely
            clips_to_merge = []
            for c in all_clips[1:]:  # Skip clip_base
                if c is not None and not getattr(c, "_is_filler", False):
                    clips_to_merge.append(c)

            # Add LoRA-enhanced virtual models if available
            if lora_processor is not None:
                virtual_models = lora_processor.get_virtual_models()
                # Skip the base model (first item) since we already have it
                lora_virtual_models = virtual_models[1:]  # Only LoRA deltas
                clips_to_merge.extend(lora_virtual_models)
                print(f"[LoRADelta] Added {len(lora_virtual_models)} LoRA-enhanced virtual CLIP models")

            print(f"[DonutWidenMergeCLIP] WIDEN merging {len(clips_to_merge)} CLIP models ({len([m for m in clips_to_merge if hasattr(m, 'lora_name')])} from LoRA stack)")

            try:
                # Handle both regular clips and LoRADelta objects
                other_encs = []
                for clip in clips_to_merge:
                    if hasattr(clip, 'lora_name'):  # LoRADelta object
                        other_encs.append(clip)
                    else:  # Regular clip
                        enc = getattr(clip, "model", getattr(clip, "clip",
                             getattr(clip, "cond_stage_model", None)))
                        if enc:
                            other_encs.append(enc)

                # FIXED: Use deepcopy only for the wrapper, not the heavy model tensors
                clip_merged = copy.copy(clip_base)  # Shallow copy of wrapper
                enc_merged = copy.deepcopy(base_enc)  # Deep copy only the encoder

                # Set the copied encoder back to the merged clip
                if hasattr(clip_merged, "model"):
                    clip_merged.model = enc_merged
                elif hasattr(clip_merged, "clip"):
                    clip_merged.clip = enc_merged
                elif hasattr(clip_merged, "cond_stage_model"):
                    clip_merged.cond_stage_model = enc_merged

                # Use enhanced WIDEN merger with dynamic compatibility
                print("[DonutWidenMergeCLIP] Using WIDEN merge with dynamic compatibility-based strength...")
                
                merger = MergingMethod("DonutWidenMergeCLIP")
                
                if rank_sensitivity > 0.0:
                    # Use WIDEN with dynamic strength based on compatibility
                    merged_params, widen_diagnostics = enhanced_widen_merging_with_dynamic_strength(
                        merger=merger,
                        merged_model=enc_merged,
                        models_to_merge=other_encs,
                        exclude_param_names_regex=[],
                        importance_threshold=importance_threshold,
                        importance_boost=importance_boost,
                        base_merge_strength=merge_strength,
                        rank_sensitivity=rank_sensitivity,
                        skip_threshold=skip_threshold,
                        normalization_mode=normalization_mode
                    )
                    
                    # Apply merged parameters with shape validation
                    applied_count = 0
                    shape_mismatch_count = 0
                    for param_name, param_value in merged_params.items():
                        for name, param in enc_merged.named_parameters():
                            if name == param_name:
                                if param_value.shape == param.shape:
                                    param.data.copy_(param_value)
                                    applied_count += 1
                                else:
                                    print(f"[WARNING] Shape mismatch for {param_name}: expected {param.shape}, got {param_value.shape}")
                                    shape_mismatch_count += 1
                                break
                    
                    print(f"[ENHANCED WIDEN] Applied {applied_count} parameters, {shape_mismatch_count} shape mismatches")
                    
                    # Create detailed merge results with enhanced WIDEN diagnostics
                    total_clips = len([c for c in [clip_other, clip_3, clip_4, clip_5, clip_6, clip_7, clip_8, clip_9, clip_10, clip_11, clip_12] if c is not None]) + 1
                    
                    # Generate enhanced diagnostics from widen_diagnostics
                    compatibility_scores = widen_diagnostics['compatibility_scores']
                    varied_count = widen_diagnostics['varied_score_count']
                    uniform_count = widen_diagnostics['uniform_score_count']
                    skipped_threshold = widen_diagnostics['parameters_skipped_threshold']
                    
                    if compatibility_scores:
                        compat_min, compat_max = min(compatibility_scores), max(compatibility_scores)
                        compat_mean = sum(compatibility_scores) / len(compatibility_scores)
                        compat_variance = sum((x - compat_mean)**2 for x in compatibility_scores) / len(compatibility_scores)
                        score_health = "✓ VARIED" if compat_variance > 1e-6 else "✗ UNIFORM (BUG!)"
                    else:
                        compat_min = compat_max = compat_mean = 0.0
                        score_health = "NO SCORES"
                    
                    ranking_health = "✓ HEALTHY" if varied_count > uniform_count else "✗ FAILING"
                    total_scored = varied_count + uniform_count
                    
                    results_text = f"""╔═ WIDEN CLIP MERGE RESULTS (Dynamic Compatibility) ═╗
║ CLIP Models: {total_clips} | Strength: {merge_strength} | Mode: {normalization_mode}
║ Threshold: {importance_threshold} | Boost: {importance_boost} | Sensitivity: {rank_sensitivity}
╠═══════════════════════════════════════════════════╣
║ Parameters Merged: {len(merged_params)} | Applied: {applied_count}
║ Shape Mismatches: {shape_mismatch_count} | Success: {(applied_count/(applied_count+shape_mismatch_count)*100):.1f}%
║ Status: ✓ Enhanced WIDEN with Dynamic Compatibility
╠═══════════════════════════════════════════════════╣
║ 🔍 WIDEN ALGORITHM HEALTH DIAGNOSTICS:
║ Compatibility Range: {compat_min:.4f} - {compat_max:.4f} (avg: {compat_mean:.4f})
║ Score Distribution: {score_health}
║ Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored if total_scored > 0 else 0:.1f}%) - {ranking_health}
║ Skip Threshold: {skip_threshold} → {skipped_threshold} parameters skipped
╚═══════════════════════════════════════════════════╝"""
                    # Create detailed parameter information
                    parameter_info = f"""╔═ WIDEN CLIP PARAMETER DETAILS ═╗
║ Merge Strength: Controls blend intensity between CLIP models
║ Importance Threshold: Multiplier for classifying important parameters
║ Importance Boost: Score amplification for important parameters
║ Rank Sensitivity: Dynamic compatibility adjustment strength  
║ Skip Threshold: Excludes low-compatibility parameters from merge
║ Normalization: {normalization_mode} - post-merge parameter scaling
╠═══════════════════════════════════════════════════╣
║ Dynamic Mode: Adapts strength based on compatibility
║ Total Processed: {len(merged_params)} parameters analyzed
║ Applied Successfully: {applied_count} parameters merged
╚═══════════════════════════════════════════════════╝"""
                    
                else:
                    # Use enhanced WIDEN without dynamic strength (fallback mode)
                    merged_params, widen_diagnostics = enhanced_widen_merging_with_dynamic_strength(
                        merger=merger,
                        merged_model=enc_merged,
                        models_to_merge=other_encs,
                        exclude_param_names_regex=[],
                        importance_threshold=importance_threshold,
                        importance_boost=importance_boost,
                        base_merge_strength=merge_strength,
                        rank_sensitivity=0.0,  # Disable dynamic strength
                        skip_threshold=skip_threshold,
                        normalization_mode=normalization_mode
                    )
                    
                    # Apply merged parameters with shape validation
                    applied_count = 0
                    shape_mismatch_count = 0
                    for param_name, param_value in merged_params.items():
                        for name, param in enc_merged.named_parameters():
                            if name == param_name:
                                if param_value.shape == param.shape:
                                    param.data.copy_(param_value)
                                    applied_count += 1
                                else:
                                    print(f"[WARNING] Shape mismatch for {param_name}: expected {param.shape}, got {param_value.shape}")
                                    shape_mismatch_count += 1
                                break
                    
                    print(f"[ENHANCED WIDEN] Applied {applied_count} parameters, {shape_mismatch_count} shape mismatches")
                    
                    # Create detailed merge results with enhanced WIDEN diagnostics
                    total_clips = len([c for c in [clip_other, clip_3, clip_4, clip_5, clip_6, clip_7, clip_8, clip_9, clip_10, clip_11, clip_12] if c is not None]) + 1
                    
                    # Generate enhanced diagnostics from widen_diagnostics
                    compatibility_scores = widen_diagnostics['compatibility_scores']
                    varied_count = widen_diagnostics['varied_score_count']
                    uniform_count = widen_diagnostics['uniform_score_count']
                    skipped_threshold = widen_diagnostics['parameters_skipped_threshold']
                    
                    if compatibility_scores:
                        compat_min, compat_max = min(compatibility_scores), max(compatibility_scores)
                        compat_mean = sum(compatibility_scores) / len(compatibility_scores)
                        compat_variance = sum((x - compat_mean)**2 for x in compatibility_scores) / len(compatibility_scores)
                        score_health = "✓ VARIED" if compat_variance > 1e-6 else "✗ UNIFORM (BUG!)"
                    else:
                        compat_min = compat_max = compat_mean = 0.0
                        score_health = "NO SCORES"
                    
                    ranking_health = "✓ HEALTHY" if varied_count > uniform_count else "✗ FAILING"
                    total_scored = varied_count + uniform_count
                    
                    results_text = f"""╔═ WIDEN CLIP MERGE RESULTS (Static Strength) ═╗
║ CLIP Models: {total_clips} | Strength: {merge_strength} | Mode: {normalization_mode}
║ Threshold: {importance_threshold} | Boost: {importance_boost} | Sensitivity: {rank_sensitivity} (off)
╠═══════════════════════════════════════════════════╣
║ Parameters Merged: {len(merged_params)} | Applied: {applied_count}
║ Shape Mismatches: {shape_mismatch_count} | Success: {(applied_count/(applied_count+shape_mismatch_count)*100):.1f}%
║ Status: ✓ Enhanced WIDEN with Static Strength
╠═══════════════════════════════════════════════════╣
║ 🔍 WIDEN ALGORITHM HEALTH DIAGNOSTICS:
║ Compatibility Range: {compat_min:.4f} - {compat_max:.4f} (avg: {compat_mean:.4f})
║ Score Distribution: {score_health}
║ Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored if total_scored > 0 else 0:.1f}%) - {ranking_health}
║ Skip Threshold: {skip_threshold} → {skipped_threshold} parameters skipped
║ Status: ✓ Enhanced WIDEN with Static Strength
╚═══════════════════════════════════════════════════╝"""
                    # Create detailed parameter information
                    parameter_info = f"""╔═ WIDEN CLIP PARAMETER DETAILS ═╗
║ Merge Strength: Controls blend intensity between CLIP models
║ Importance Threshold: Multiplier for classifying important parameters
║ Importance Boost: Score amplification for important parameters
║ Rank Sensitivity: {rank_sensitivity} (disabled) - no dynamic adjustment
║ Skip Threshold: Excludes low-compatibility parameters from merge  
║ Normalization: {normalization_mode} - post-merge parameter scaling
╠═══════════════════════════════════════════════════╣
║ Static Mode: Uses fixed strength for all parameters
║ Total Processed: {len(merged_params)} parameters analyzed
║ Applied Successfully: {applied_count} parameters merged
╚═══════════════════════════════════════════════════╝"""

                # FIXED: Aggressive cleanup before returning
                del base_enc, other_encs, clips_to_merge, enc_merged
                if rank_sensitivity <= 0.0:
                    del merger
                force_cleanup()

                result = (clip_merged, results_text, parameter_info)

                # Store in cache
                store_merge_result(cache_key, result)

                return result

            except MemoryExhaustionError as e:
                print(f"[SAFETY] Memory exhaustion prevented crash: {e}")
                # FIXED: Cleanup on error
                force_cleanup()
                error_results = f"""╔═ WIDEN CLIP MERGE RESULTS (MEMORY ERROR) ═╗
║ ERROR: Memory exhaustion prevented crash
║ DETAILS: {str(e)[:40]}...
║ STATUS: ✗ Failed - Memory limit exceeded
║ FIX: Reduce batch size or CLIP model count
╚═══════════════════════════════════════════════════╝"""
                error_param_info = """╔═ ERROR PARAMETER INFO ═╗
║ CLIP merge was terminated due to memory limits
║ No parameter analysis available
║ Try reducing CLIP model count or batch size
╚═══════════════════════════════════════════════════╝"""
                result = (clip_base, error_results, error_param_info)
                store_merge_result(cache_key, result)
                return result

            except Exception as e:
                print(f"[DonutWidenMergeCLIP] Error: {e}")
                # FIXED: Cleanup on error
                force_cleanup()
                if "memory" in str(e).lower():
                    error_results = f"""╔═ WIDEN CLIP MERGE RESULTS (MEMORY ERROR) ═╗
║ ERROR: Memory error prevented crash
║ DETAILS: {str(e)[:40]}...
║ STATUS: ✗ Failed - Memory limit exceeded
║ FIX: Reduce batch size or CLIP model count
╚═══════════════════════════════════════════════════╝"""
                    error_param_info = """╔═ ERROR PARAMETER INFO ═╗
║ CLIP merge was terminated due to memory error
║ No parameter analysis available
║ Try reducing CLIP model count or batch size  
╚═══════════════════════════════════════════════════╝"""
                    result = (clip_base, error_results, error_param_info)
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

# FIXED: Add manual cleanup function for ComfyUI
def manual_cleanup():
    """Manual cleanup function that users can call"""
    print("="*50)
    print("MANUAL MEMORY CLEANUP INITIATED")
    print("="*50)
    clear_merge_cache()
    force_cleanup()
    print("="*50)
    print("MANUAL CLEANUP COMPLETE")
    print("="*50)

# Export the manual cleanup function
NODE_CLASS_MAPPINGS["DonutManualCleanup"] = type("DonutManualCleanup", (), {
    "class_type": "FUNCTION",
    "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
    "RETURN_TYPES": ("STRING",),
    "RETURN_NAMES": ("status",),
    "FUNCTION": "execute",
    "CATEGORY": "donut/utils",
    "execute": lambda self: (f"Memory cleanup completed at {time.time()}",)
})

def clear_merge_cache():
    """Clear the model merge cache"""
    global _MERGE_CACHE
    _MERGE_CACHE.clear()
    print("[Cache] Cleared all cached merge results")

import atexit
def cleanup_on_exit():
    """Cleanup on exit"""
    try:
        clear_merge_cache()  # FIXED: Actually call the cache clearing function
        force_cleanup()      # FIXED: Call force cleanup on exit
    except Exception:
        pass

atexit.register(cleanup_on_exit)

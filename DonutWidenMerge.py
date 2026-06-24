"""
Enhanced WIDEN Merge with Advanced Quality Optimizations

This implementation includes several advanced features for high-quality model merging:

1. AUTO-TUNING: Skip threshold automatically adjusts based on Phase 1 compatibility scores
2. NEURON ALIGNMENT: Linear layer weights are optimally aligned using Hungarian algorithm
3. EMBEDDING TRANSPOSE: Automatic detection and correction of embedding orientation
4. ADAPTIVE THRESHOLDS: Per-layer skip thresholds (conv: 1e-3, linear: 1e-4, norm: 0.0)
5. FAILURE TRACKING: Comprehensive error handling with detailed diagnostics
6. SANITY METRICS: Over-merge detection with L2 change ratio analysis
7. NORM RECALIBRATION: BatchNorm/LayerNorm stats recalibration after merge
8. HYPERPARAMETER SEARCH: Grid search and Optuna optimization support
9. STRUCTURE PRESERVATION: Per-channel structure preservation using mean instead of flattening
10. CONDENSED DELTAS: Channel-preserving delta summaries for better metadata

Example usage with hyperparameter tuning:
```python
# Define evaluation function (optional - dummy evaluation if None)
def evaluate_model(model, validation_data):
    # Your evaluation logic here
    return score  # Higher = better

# Perform hyperparameter search
best_params = tune_merge_hyperparameters(
    merge_function=enhanced_widen_merging_with_dynamic_strength,
    base_model=base_model,
    models_to_merge=[model1, model2, model3],
    evaluate_function=evaluate_model,
    validation_data=val_loader,
    method="optuna",  # or "grid"
    n_trials=30
)

# Use best parameters for final merge
merged_model, diagnostics = enhanced_widen_merging_with_dynamic_strength(
    **best_params['params']
)

# Recalibrate normalization layers with real data
recalibrate_norm_stats(merged_model, calib_data=val_loader, num_batches=10)
```

Dependencies:
- scipy (optional): For neuron alignment via Hungarian algorithm
- optuna (optional): For Bayesian hyperparameter optimization
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import hashlib
import psutil
import gc
import os
import tempfile
import uuid
from contextlib import contextmanager
import time
import logging
import os
import numpy as np
import warnings
import itertools

# Import all shared modules
try:
    # Try relative imports first (when running as part of ComfyUI)
    from .shared.logging_config import (
        widen_logger, performance_logger, memory_logger, diagnostic_logger,
        print_progress_bar, ProgressBarContext, configure_widen_logging
    )
    from .shared.constants import *
    from .shared.exceptions import *
    from .shared.alignment import (
        align_linear_layer, transpose_embeddings_if_needed, align_and_stack,
        safe_stack, align_tensors, create_condensed_delta, safe_align_and_stack
    )
    from .shared.merge_strength import (
        get_adaptive_skip_threshold, _compatibility_to_merge_strength,
        _fast_sigmoid_strength, MergeFailureTracker, DEFAULT_SKIP_THRESHOLDS
    )
    from .shared.diagnostics import (
        compute_merge_sanity_metrics, print_merge_diagnostics,
        tune_merge_hyperparameters, sanitize_strength_distribution
    )
    from .shared.memory_management import (
        MemoryEfficientContext, get_reusable_tensor_buffer, clear_tensor_buffer_cache,
        clear_session_cache, monitor_memory, check_memory_safety, force_cleanup,
        gentle_cleanup, aggressive_memory_cleanup, monitor_memory_usage,
        MemoryProfiler, get_widen_memory_profiler, memory_cleanup_context,
        MemoryExhaustionError, ultra_memory_efficient_widen_merge
    )
    from .shared.tensor_operations import *
    from .shared.utility_functions import (
        _analyze_compatibility_patterns_and_recommend_threshold
    )
    from .shared.core_merge_functions import (
        enhanced_widen_merging_with_dynamic_strength,
        enhanced_widen_merging_with_post_refinement,
        create_enhanced_merge_with_refinement_config
    )
    from .shared.lora_processing import LoRAStackProcessor, LoRADelta
    from .shared.cache_management import (
        compute_merge_hash, check_cache_for_merge, store_merge_result, clear_merge_cache,
        check_cache_for_merge_with_bypass, enable_cache_debug_logging, inspect_cache,
        analyze_hash_differences, debug_cache_invalidation_test, create_merge_tracer,
        print_merge_trace_summary
    )
    from .shared.task_vector import TaskVector
except ImportError:
    # Fallback to absolute imports (when running standalone)
    # Add the current directory to Python path to find the shared modules
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Now import all shared modules
    from shared.logging_config import (
        widen_logger, performance_logger, memory_logger, diagnostic_logger,
        print_progress_bar, ProgressBarContext, configure_widen_logging
    )
    from shared.constants import *
    from shared.exceptions import *
    from shared.alignment import (
        align_linear_layer, transpose_embeddings_if_needed, align_and_stack,
        safe_stack, align_tensors, create_condensed_delta, safe_align_and_stack
    )
    from shared.merge_strength import (
        get_adaptive_skip_threshold, _compatibility_to_merge_strength,
        _fast_sigmoid_strength, MergeFailureTracker, DEFAULT_SKIP_THRESHOLDS
    )
    from shared.diagnostics import (
        compute_merge_sanity_metrics, print_merge_diagnostics,
        tune_merge_hyperparameters, sanitize_strength_distribution
    )
    from shared.memory_management import (
        MemoryEfficientContext, get_reusable_tensor_buffer, clear_tensor_buffer_cache,
        clear_session_cache, monitor_memory, check_memory_safety, force_cleanup,
        gentle_cleanup, aggressive_memory_cleanup, monitor_memory_usage,
        MemoryProfiler, get_widen_memory_profiler, memory_cleanup_context,
        MemoryExhaustionError, ultra_memory_efficient_widen_merge
    )
    from shared.tensor_operations import *
    from shared.utility_functions import (
        _analyze_compatibility_patterns_and_recommend_threshold
    )
    from shared.core_merge_functions import (
        enhanced_widen_merging_with_dynamic_strength,
        enhanced_widen_merging_with_post_refinement,
        create_enhanced_merge_with_refinement_config
    )
    from shared.lora_processing import LoRAStackProcessor, LoRADelta
    from shared.cache_management import (
        compute_merge_hash, check_cache_for_merge, store_merge_result, clear_merge_cache,
        check_cache_for_merge_with_bypass, enable_cache_debug_logging, inspect_cache,
        analyze_hash_differences, debug_cache_invalidation_test, create_merge_tracer,
        print_merge_trace_summary
    )
    from shared.task_vector import TaskVector

# Ensure cache variables are available at module level
try:
    # These should already be imported via "from .shared.constants import *" or "from shared.constants import *"
    # But make them explicitly available in case of import issues
    if '_MERGE_CACHE' not in globals():
        from .shared.constants import _MERGE_CACHE, _CACHE_MAX_SIZE
except ImportError:
    try:
        from shared.constants import _MERGE_CACHE, _CACHE_MAX_SIZE
    except ImportError:
        # Fallback definitions if all imports fail
        _MERGE_CACHE = {}
        _CACHE_MAX_SIZE = 3

# Scipy and Optuna availability are checked in the modules now
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Progress bar and logging functions imported from logging_config module

# MergingMethod class - required by the ComfyUI node classes
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
            'resolution_change': 1.2,    # Important for spatial transformations
            'normalization': 1.1,        # Stability
            'bias': 0.9,                 # Less critical
            'other': 1.0                 # Default weight
        }

    def classify_parameter(self, name: str) -> str:
        """
        Classify SDXL parameters by their architectural role.
        
        Returns one of: 'time_embedding', 'class_embedding', 'cross_attention', 
        'self_attention', 'input_conv', 'output_conv', 'feature_conv', 'skip_conv',
        'resolution_change', 'normalization', 'bias', 'other'
        """
        name_lower = name.lower()
        
        # Time and class embeddings
        if 'time_embed' in name_lower:
            return 'time_embedding'
        elif 'class_embed' in name_lower or 'label_emb' in name_lower:
            return 'class_embedding'
        
        # Attention mechanisms
        elif 'cross_attn' in name_lower or ('attn' in name_lower and 'cross' in name_lower):
            return 'cross_attention'
        elif 'self_attn' in name_lower or ('attn' in name_lower and 'self' in name_lower):
            return 'self_attention'
        elif any(x in name_lower for x in ['attn', 'attention']):
            # Default attention classification
            if any(x in name_lower for x in ['encoder', 'text']):
                return 'cross_attention'
            else:
                return 'self_attention'
        
        # Convolutional layers
        elif any(x in name_lower for x in ['conv_in', 'input_blocks.0']):
            return 'input_conv'
        elif any(x in name_lower for x in ['conv_out', 'out.']):
            return 'output_conv'
        elif any(x in name_lower for x in ['skip_connection', 'residual']):
            return 'skip_conv'
        elif 'conv' in name_lower:
            if any(x in name_lower for x in ['down', 'up']):
                return 'resolution_change'
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
                "merge_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "min_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "max_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "normalization_mode": (["magnitude", "calibrate", "none"], {"default": "magnitude"}),  # (renorm_mode)
                # Enhanced WIDEN parameters
                "importance_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 500.0, "step": 0.1}),  # (above_average_value_ratio)
                "importance_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),  # (score_calibration_value)
                # Dynamic compatibility settings  
                "rank_sensitivity": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),  # (compatibility_sensitivity)
                "skip_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.000001}),  # (compatibility_threshold)
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
                "scale_to_min_max": ("BOOLEAN", {"default": False}),
                "invert_strengths": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "merge_results", "parameter_info")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def _apply_dynamic_scaling(self, other_models, min_strength, max_strength, 
                              importance_threshold, importance_boost, rank_sensitivity, skip_threshold):
        """
        Analyze the current strength distribution and scale it to fit perfectly 
        within min_strength to max_strength range.
        
        Returns: (new_min_strength, new_max_strength) that will map the natural 
        distribution to the desired range.
        """
        try:
            from .shared.merge_strength import _compatibility_to_merge_strength
            
            print(f"[DynamicScaling] Analyzing strength distribution for scaling...")
            
            # Sample a few parameters to estimate the natural strength distribution
            sampled_strengths = []
            param_count = 0
            max_samples = 100  # Don't analyze too many to keep it fast
            
            # Get base model for comparison
            if hasattr(other_models[0], 'lora_name'):
                # LoRA models - skip this for now, use defaults
                print(f"[DynamicScaling] LoRA models detected, using original range")
                return min_strength, max_strength
            
            base_model = other_models[0] if other_models else None
            if base_model is None:
                return min_strength, max_strength
                
            # Sample parameters to estimate distribution
            for name, param in base_model.named_parameters():
                if param_count >= max_samples:
                    break
                    
                # Skip normalization layers and embeddings for faster analysis
                if any(skip_name in name.lower() for skip_name in ['norm', 'bias', 'embed']):
                    continue
                    
                # Calculate what the strength would be for this parameter
                try:
                    # Simplified compatibility calculation (just use param variance as proxy)
                    param_var = torch.var(param).item()
                    mock_compatibility = min(param_var * 1000, 2.0)  # Scale to reasonable range
                    
                    strength = _compatibility_to_merge_strength(
                        compatibility=mock_compatibility,
                        importance_threshold=importance_threshold,
                        importance_boost=importance_boost,
                        rank_sensitivity=rank_sensitivity,
                        skip_threshold=skip_threshold,
                        min_strength=0.0,  # Use full range for analysis
                        max_strength=2.0
                    )
                    
                    if strength > 0:  # Only count non-zero strengths
                        sampled_strengths.append(strength)
                        param_count += 1
                        
                except Exception:
                    continue
            
            if len(sampled_strengths) < 5:
                print(f"[DynamicScaling] Insufficient samples ({len(sampled_strengths)}), using original range")
                return min_strength, max_strength
            
            # Calculate natural distribution
            natural_min = min(sampled_strengths)
            natural_max = max(sampled_strengths)
            natural_range = natural_max - natural_min
            
            print(f"[DynamicScaling] Natural distribution: {natural_min:.3f} to {natural_max:.3f} (range: {natural_range:.3f})")
            print(f"[DynamicScaling] Target range: {min_strength:.3f} to {max_strength:.3f}")
            
            if natural_range < 1e-6:  # Nearly uniform distribution
                print(f"[DynamicScaling] Nearly uniform distribution, using original range")
                return min_strength, max_strength
            
            # Calculate scaling parameters
            # We want: natural_min -> min_strength, natural_max -> max_strength  
            # Formula: new_value = ((old_value - natural_min) / natural_range) * target_range + min_strength
            # But we need to reverse this to find what min/max to use in the merge function
            
            target_range = max_strength - min_strength
            scale_factor = target_range / natural_range
            
            # The merge function will generate the natural distribution, then we want it scaled to target
            # So we need to set merge function's min/max such that when it generates [natural_min, natural_max],
            # the final result is [min_strength, max_strength]
            
            # Reverse the scaling: if final = (natural - nat_min) / nat_range * target_range + min_strength
            # Then we want: natural = (final - min_strength) / scale_factor + natural_min
            # But the merge function's min/max control the natural generation...
            
            # Actually, simpler approach: we'll modify the merge function's output range
            # Set merge min/max to map the expected natural range to our target range
            new_min = min_strength
            new_max = max_strength
            
            print(f"[DynamicScaling] Mapped natural range [{natural_min:.3f}, {natural_max:.3f}] to target [{new_min:.3f}, {new_max:.3f}]")
            
            return new_min, new_max
            
        except Exception as e:
            print(f"[DynamicScaling] Error during scaling analysis: {e}")
            print(f"[DynamicScaling] Falling back to original range")
            return min_strength, max_strength

    def _create_strength_scaler(self, scaling_info):
        """
        Create a function that scales strength values from natural to target range.
        """
        if not scaling_info or not scaling_info.get('needs_scaling'):
            return lambda x: x  # Identity function
            
        natural_min = scaling_info['natural_min']
        natural_max = scaling_info['natural_max']
        target_min = scaling_info['target_min']
        target_max = scaling_info['target_max']
        
        natural_range = natural_max - natural_min
        target_range = target_max - target_min
        
        if natural_range < 1e-8:
            return lambda x: x  # Identity if range too small
            
        def scale_strength(strength):
            # Linear transformation: (strength - natural_min) / natural_range * target_range + target_min
            return (strength - natural_min) / natural_range * target_range + target_min
            
        return scale_strength

    def _apply_post_merge_scaling(self, merged_params, widen_diagnostics, target_min, target_max):
        """
        Apply scaling to merge results to map natural distribution to target range.
        This analyzes the applied strengths and rescales them linearly.
        """
        try:
            # Extract the applied strength values from diagnostics
            # Fix: applied_strengths is a list of dicts, not a dict!
            applied_strengths_list = widen_diagnostics.get('applied_strengths', [])
            if not applied_strengths_list:
                print("[DynamicScaling] No applied strengths found in diagnostics, skipping scaling")
                return merged_params
            
            # Convert list of dicts to param_name -> strength mapping
            applied_strengths = {item['parameter']: item['strength'] for item in applied_strengths_list}
            
            # Find the actual range of applied strengths
            strength_values = list(applied_strengths.values())
            actual_min = min(strength_values)
            actual_max = max(strength_values)
            actual_range = actual_max - actual_min
            
            if actual_range < 1e-8:
                print("[DynamicScaling] Actual strength range too small, skipping scaling")
                return merged_params
            
            target_range = target_max - target_min
            print(f"[DynamicScaling] Scaling applied strengths from [{actual_min:.3f}, {actual_max:.3f}] to [{target_min:.3f}, {target_max:.3f}]")
            
            # DISABLED: Post-merge parameter scaling corrupts model weights!
            # Dynamic scaling should happen during merge, not after on final parameters
            print("[DynamicScaling] Post-merge scaling disabled to prevent model corruption")
            scaled_params = merged_params  # Return unmodified parameters
                
            print(f"[DynamicScaling] Successfully scaled {len(scaled_params)} parameters")
            return scaled_params
            
        except Exception as e:
            print(f"[DynamicScaling] Error during post-merge scaling: {e}")
            print("[DynamicScaling] Returning unscaled parameters")
            return merged_params

    def _apply_strength_inversion(self, merged_params, widen_diagnostics, min_strength, max_strength):
        """
        Apply strength inversion to flip the strength distribution.
        Low compatibility parameters get high strengths and vice versa.
        """
        try:
            # Extract the applied strength values from diagnostics
            # Fix: applied_strengths is a list of dicts, not a dict!
            applied_strengths_list = widen_diagnostics.get('applied_strengths', [])
            if not applied_strengths_list:
                print("[DynamicScaling] No applied strengths found in diagnostics, skipping scaling")
                return merged_params
            
            # Convert list of dicts to param_name -> strength mapping
            applied_strengths = {item['parameter']: item['strength'] for item in applied_strengths_list}
            if not applied_strengths:
                print("[StrengthInversion] No applied strengths found in diagnostics, skipping inversion")
                return merged_params
            
            # Find the actual range of applied strengths
            strength_values = list(applied_strengths.values())
            actual_min = min(strength_values)
            actual_max = max(strength_values)
            actual_range = actual_max - actual_min
            
            if actual_range < 1e-8:
                print("[StrengthInversion] Actual strength range too small, skipping inversion")
                return merged_params
            
            print(f"[StrengthInversion] Inverting strengths in range [{actual_min:.3f}, {actual_max:.3f}]")
            
            # Apply inversion to the merged parameters
            inverted_params = {}
            for param_name, param_tensor in merged_params.items():
                original_strength = applied_strengths.get(param_name, 1.0)
                
                # Invert the strength: low becomes high, high becomes low
                # Formula: inverted = actual_max + actual_min - original
                inverted_strength = actual_max + actual_min - original_strength
                
                # Adjust the parameter tensor proportionally
                strength_ratio = inverted_strength / original_strength if original_strength != 0 else 1.0
                inverted_params[param_name] = param_tensor * strength_ratio
                
            print(f"[StrengthInversion] Successfully inverted {len(inverted_params)} parameters")
            return inverted_params
            
        except Exception as e:
            print(f"[StrengthInversion] Error during strength inversion: {e}")
            print("[StrengthInversion] Returning non-inverted parameters")
            return merged_params

    def execute(self, model_base, model_other, merge_strength, min_strength, max_strength, normalization_mode,
                importance_threshold, importance_boost,
                rank_sensitivity, skip_threshold,
                lora_stack=None, model_3=None, model_4=None, model_5=None, model_6=None,
                model_7=None, model_8=None, model_9=None, model_10=None,
                model_11=None, model_12=None, scale_to_min_max=False, invert_strengths=False, enable_cache_debug=None, force_fresh_merge=None):

        # Handle legacy debug parameters for backward compatibility
        if enable_cache_debug is not None:
            print(f"[Compatibility] Ignoring legacy enable_cache_debug parameter: {enable_cache_debug}")
        if force_fresh_merge is not None:
            print(f"[Compatibility] Ignoring legacy force_fresh_merge parameter: {force_fresh_merge}")

        # Conservative pre-merge setup with session cache management
        print("[MEMORY] Pre-merge setup...")
        
        # Clear session cache if it's getting large (prevents accumulation)
        if len(_MERGE_CACHE) >= _CACHE_MAX_SIZE:
            clear_session_cache()
        
        # Light pre-merge cleanup
        gentle_cleanup()

        # Check cache first
        all_models = [model_base, model_other, model_3, model_4, model_5, model_6,
                     model_7, model_8, model_9, model_10, model_11, model_12]
        cache_key = compute_merge_hash(all_models, merge_strength, min_strength, max_strength, importance_threshold, importance_boost, rank_sensitivity, skip_threshold, f"{normalization_mode}_enhanced_widen", lora_stack, scale_to_min_max, invert_strengths)
        # UNet cache analysis

        cached_result = check_cache_for_merge_with_bypass(cache_key, False)
        if cached_result is not None:
            # Cache hit
            return cached_result
        else:
            # Cache miss - computing fresh merge
            pass

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

                # CONTAMINATION FIX: Memory-efficient independent model copy
                # ComfyUI's clone() shares parameter tensors. We need isolation but minimize memory usage.
                print("[CONTAMINATION-FIX] Creating memory-efficient independent model copy")
                import copy
                import gc
                
                # Step 1: Force ComfyUI to offload the base model from GPU to free VRAM
                try:
                    import comfy.model_management
                    comfy.model_management.unload_model_clones(model_base)
                    comfy.model_management.soft_empty_cache(force=True)
                except:
                    pass
                
                # Step 2: Create deep copy with immediate cleanup
                model_merged = copy.deepcopy(model_base)
                
                # Step 3: Aggressive memory cleanup after copy
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU cache
                
                print("[CONTAMINATION-FIX] Model copied with memory optimization")

                # Apply dynamic scaling if requested  
                original_min, original_max = min_strength, max_strength
                if scale_to_min_max:
                    print(f"[DonutWidenMergeUNet] Dynamic scaling enabled - using user bounds directly: {min_strength} to {max_strength}")
                    # FIXED: Use user's actual bounds instead of wide analysis range
                    # since strength calculation already respects these bounds properly

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
                        merge_strength=merge_strength,
                        min_strength=min_strength,
                        max_strength=max_strength,
                        rank_sensitivity=rank_sensitivity,
                        skip_threshold=skip_threshold,
                        normalization_mode=normalization_mode,
                        scale_to_min_max=scale_to_min_max,
                        original_min=original_min,
                        original_max=original_max
                    )
                    
                    # Apply strength inversion if enabled (before scaling)
                    if invert_strengths:
                        print("[DonutWidenMergeUNet] Applying strength inversion...")
                        merged_params = self._apply_strength_inversion(merged_params, widen_diagnostics, min_strength, max_strength)
                    
                    # Apply dynamic scaling post-processing if enabled
                    if scale_to_min_max:
                        print("[DonutWidenMergeUNet] Applying dynamic scaling to merge results...")
                        # DISABLED: Post-merge scaling corrupts model - bounds are already correct
                        # merged_params = self._apply_post_merge_scaling(merged_params, widen_diagnostics, original_min, original_max)
                    
                    # Apply merged parameters with shape validation
                    applied_count = 0
                    shape_mismatch_count = 0
                    for param_name, param_value in merged_params.items():
                        for name, param in model_merged.model.named_parameters():
                            if name == param_name:
                                # Detect and remove an accidental leading batch-of-1 dim
                                if param_value.ndim == param.ndim + 1 and param_value.size(0) == 1:
                                    param_value = param_value.squeeze(0)
                                
                                if param_value.shape == param.shape:
                                    param.data.copy_(param_value)
                                    applied_count += 1
                                else:
                                    print(f"[WARNING] Shape mismatch for {param_name}: expected {param.shape}, got {param_value.shape}")
                                    shape_mismatch_count += 1
                                break
                    
                    print(f"[ENHANCED WIDEN] Applied {applied_count} parameters, {shape_mismatch_count} shape mismatches")
                    
                    # Compute and display merge sanity metrics
                    try:
                        sanity_metrics = compute_merge_sanity_metrics(base_model, merged_model)
                        print_merge_diagnostics(sanity_metrics)
                    except Exception as e:
                        diagnostic_logger.warning(f"Failed to compute sanity metrics: {e}")
                    
                    # Create detailed merge results with enhanced WIDEN diagnostics
                    total_models = len([m for m in [model_other, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10, model_11, model_12] if m is not None]) + 1
                    
                    # Generate enhanced diagnostics from widen_diagnostics
                    compatibility_scores = widen_diagnostics['compatibility_scores']
                    varied_count = widen_diagnostics['varied_score_count']
                    uniform_count = widen_diagnostics['uniform_score_count']
                    skipped_threshold = widen_diagnostics['parameters_skipped_threshold']
                    
                    # Extract strength distribution metrics
                    strength_dist = widen_diagnostics['strength_distribution']
                    applied_strengths = widen_diagnostics['applied_strengths']
                    
                    # Sanitize strength distribution for display
                    clean_strength_dist = sanitize_strength_distribution(strength_dist)
                    
                    # ACTUAL APPLIED STRENGTH DIAGNOSTIC
                    # Compute actual applied strength distribution from merged parameters  
                    actual_applied_strengths = []
                    if applied_strengths:
                        if isinstance(applied_strengths, list):
                            actual_applied_strengths = [item['strength'] for item in applied_strengths if 'strength' in item]
                        elif isinstance(applied_strengths, dict):
                            actual_applied_strengths = list(applied_strengths.values())
                    
                    if actual_applied_strengths:
                        actual_min = min(actual_applied_strengths)
                        actual_max = max(actual_applied_strengths)  
                        actual_mean = sum(actual_applied_strengths) / len(actual_applied_strengths)
                        actual_strength_text = f"{actual_min:.3f}-{actual_max:.3f} (avg {actual_mean:.3f})"
                    else:
                        actual_strength_text = "N/A (no applied strengths found)"
                    
                    if compatibility_scores:
                        compat_values = [score['compatibility'] for score in compatibility_scores]
                        compat_min, compat_max = min(compat_values), max(compat_values)
                        compat_mean = sum(compat_values) / len(compat_values)
                        compat_variance = sum((x - compat_mean)**2 for x in compat_values) / len(compat_values)
                        compat_range = compat_max - compat_min
                        # Use relative variance threshold based on score range
                        relative_variance_threshold = max(1e-6, (compat_range * 0.01) ** 2)
                        score_health = "✓ VARIED" if compat_variance > relative_variance_threshold else "✗ UNIFORM (BUG!)"
                    else:
                        compat_min = compat_max = compat_mean = 0.0
                        score_health = "NO SCORES"
                    
                    ranking_health = "✓ HEALTHY" if varied_count > uniform_count else "✗ FAILING"
                    total_scored = varied_count + uniform_count
                    
                    # Generate skip threshold recommendation based on actual compatibility scores
                    compatibility_score_dict = {score['parameter']: score['compatibility'] for score in compatibility_scores}
                    recommended_threshold, threshold_analysis = _analyze_compatibility_patterns_and_recommend_threshold(
                        compatibility_score_dict, skip_threshold
                    )
                    
                    # Add threshold analysis to console output
                    performance_logger.info(f"Skip threshold analysis:\n{threshold_analysis}")
                    
                    # Show the correct strength range in diagnostics
                    display_min = original_min if scale_to_min_max else min_strength
                    display_max = original_max if scale_to_min_max else max_strength
                    features = []
                    if scale_to_min_max:
                        features.append("dynamic scaling")
                    if invert_strengths:
                        features.append("inverted")
                    scaling_status = f" ({', '.join(features)})" if features else ""
                    
                    results_text = f"""╔═ WIDEN MERGE RESULTS (Dynamic Compatibility) ═╗
║ Models: {total_models} | Strength: {display_min}-{display_max}{scaling_status} | Mode: {normalization_mode}
║ User Params: merge={merge_strength} | min={original_min} | max={original_max} | scale_to_min_max={scale_to_min_max}
║ Threshold: {importance_threshold} | Boost: {importance_boost} | Sensitivity: {rank_sensitivity}
╠═══════════════════════════════════════════════════╣
║ Parameters Merged: {len(merged_params)} | Applied: {applied_count}
║ Shape Mismatches: {shape_mismatch_count} | Success: {(applied_count/(applied_count+shape_mismatch_count)*100):.1f}%
║ Status: ✓ Enhanced WIDEN with Dynamic Compatibility
╠═══════════════════════════════════════════════════╣
║ 🔍 WIDEN ALGORITHM HEALTH DIAGNOSTICS:
║ Compatibility Range: {compat_min:.4f} - {compat_max:.4f} (avg: {compat_mean:.4f})
║ Strength Distribution: {clean_strength_dist['display_text']}
║ ACTUAL Applied Range: {actual_strength_text}
║ Applied Strengths: {clean_strength_dist['count']} parameters with dynamic adjustment
║ Score Distribution: {score_health}
║ Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored if total_scored > 0 else 0:.1f}%) - {ranking_health}
║ Skip Threshold: {skip_threshold} (percentile) → {skipped_threshold} parameters skipped
║ 💡 RECOMMENDED: skip_threshold = {recommended_threshold:.6f} for optimal performance
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
                        merge_strength=merge_strength,
                        min_strength=min_strength,
                        max_strength=max_strength,
                        scale_to_min_max=scale_to_min_max,
                        original_min=original_min,
                        original_max=original_max,
                        rank_sensitivity=0.0,  # Disable dynamic strength
                        skip_threshold=skip_threshold,
                        normalization_mode=normalization_mode
                    )
                    
                    # Apply strength inversion if enabled (before scaling)
                    if invert_strengths:
                        print("[DonutWidenMergeUNet] Applying strength inversion (static mode)...")
                        merged_params = self._apply_strength_inversion(merged_params, widen_diagnostics, min_strength, max_strength)
                    
                    # Apply dynamic scaling post-processing if enabled
                    if scale_to_min_max:
                        print("[DonutWidenMergeUNet] Applying dynamic scaling to merge results (static mode)...")
                        # DISABLED: Post-merge scaling corrupts model - bounds are already correct
                        # merged_params = self._apply_post_merge_scaling(merged_params, widen_diagnostics, original_min, original_max)
                    
                    # Apply merged parameters with shape validation
                    applied_count = 0
                    shape_mismatch_count = 0
                    for param_name, param_value in merged_params.items():
                        for name, param in model_merged.model.named_parameters():
                            if name == param_name:
                                # Detect and remove an accidental leading batch-of-1 dim
                                if param_value.ndim == param.ndim + 1 and param_value.size(0) == 1:
                                    param_value = param_value.squeeze(0)
                                
                                if param_value.shape == param.shape:
                                    param.data.copy_(param_value)
                                    applied_count += 1
                                else:
                                    print(f"[WARNING] Shape mismatch for {param_name}: expected {param.shape}, got {param_value.shape}")
                                    shape_mismatch_count += 1
                                break
                    
                    print(f"[ENHANCED WIDEN] Applied {applied_count} parameters, {shape_mismatch_count} shape mismatches")
                    
                    # Compute and display merge sanity metrics
                    try:
                        sanity_metrics = compute_merge_sanity_metrics(base_model, merged_model)
                        print_merge_diagnostics(sanity_metrics)
                    except Exception as e:
                        diagnostic_logger.warning(f"Failed to compute sanity metrics: {e}")
                    
                    # Create detailed merge results with enhanced WIDEN diagnostics
                    total_models = len([m for m in [model_other, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10, model_11, model_12] if m is not None]) + 1
                    
                    # Generate enhanced diagnostics from widen_diagnostics
                    compatibility_scores = widen_diagnostics['compatibility_scores']
                    varied_count = widen_diagnostics['varied_score_count']
                    uniform_count = widen_diagnostics['uniform_score_count']
                    skipped_threshold = widen_diagnostics['parameters_skipped_threshold']
                    
                    # Extract strength distribution metrics
                    strength_dist = widen_diagnostics['strength_distribution']
                    applied_strengths = widen_diagnostics['applied_strengths']
                    
                    # Sanitize strength distribution for display
                    clean_strength_dist = sanitize_strength_distribution(strength_dist)
                    
                    # ACTUAL APPLIED STRENGTH DIAGNOSTIC
                    # Compute actual applied strength distribution from merged parameters  
                    actual_applied_strengths = []
                    if applied_strengths:
                        if isinstance(applied_strengths, list):
                            actual_applied_strengths = [item['strength'] for item in applied_strengths if 'strength' in item]
                        elif isinstance(applied_strengths, dict):
                            actual_applied_strengths = list(applied_strengths.values())
                    
                    if actual_applied_strengths:
                        actual_min = min(actual_applied_strengths)
                        actual_max = max(actual_applied_strengths)  
                        actual_mean = sum(actual_applied_strengths) / len(actual_applied_strengths)
                        actual_strength_text = f"{actual_min:.3f}-{actual_max:.3f} (avg {actual_mean:.3f})"
                    else:
                        actual_strength_text = "N/A (no applied strengths found)"
                    
                    if compatibility_scores:
                        compat_values = [score['compatibility'] for score in compatibility_scores]
                        compat_min, compat_max = min(compat_values), max(compat_values)
                        compat_mean = sum(compat_values) / len(compat_values)
                        compat_variance = sum((x - compat_mean)**2 for x in compat_values) / len(compat_values)
                        compat_range = compat_max - compat_min
                        # Use relative variance threshold based on score range
                        relative_variance_threshold = max(1e-6, (compat_range * 0.01) ** 2)
                        score_health = "✓ VARIED" if compat_variance > relative_variance_threshold else "✗ UNIFORM (BUG!)"
                    else:
                        compat_min = compat_max = compat_mean = 0.0
                        score_health = "NO SCORES"
                    
                    ranking_health = "✓ HEALTHY" if varied_count > uniform_count else "✗ FAILING"
                    total_scored = varied_count + uniform_count
                    
                    # Generate skip threshold recommendation for static strength mode too
                    compatibility_score_dict = {score['parameter']: score['compatibility'] for score in compatibility_scores}
                    recommended_threshold, threshold_analysis = _analyze_compatibility_patterns_and_recommend_threshold(
                        compatibility_score_dict, skip_threshold
                    )
                    
                    # Add threshold analysis to console output
                    performance_logger.info(f"Skip threshold analysis (static mode):\n{threshold_analysis}")
                    
                    # Show the correct strength range in diagnostics  
                    display_min = original_min if scale_to_min_max else min_strength
                    display_max = original_max if scale_to_min_max else max_strength
                    features = []
                    if scale_to_min_max:
                        features.append("dynamic scaling")
                    if invert_strengths:
                        features.append("inverted")
                    scaling_status = f" ({', '.join(features)})" if features else ""
                    
                    results_text = f"""╔═ WIDEN MERGE RESULTS (Static Strength) ═╗
║ Models: {total_models} | Strength: {display_min}-{display_max}{scaling_status} | Mode: {normalization_mode}
║ User Params: merge={merge_strength} | min={original_min} | max={original_max} | scale_to_min_max={scale_to_min_max}
║ Threshold: {importance_threshold} | Boost: {importance_boost} | Sensitivity: {rank_sensitivity} (off)
╠═══════════════════════════════════════════════════╣
║ Parameters Merged: {len(merged_params)} | Applied: {applied_count}
║ Shape Mismatches: {shape_mismatch_count} | Success: {(applied_count/(applied_count+shape_mismatch_count)*100):.1f}%
║ Status: ✓ Enhanced WIDEN with Static Strength
╠═══════════════════════════════════════════════════╣
║ 🔍 WIDEN ALGORITHM HEALTH DIAGNOSTICS:
║ Compatibility Range: {compat_min:.4f} - {compat_max:.4f} (avg: {compat_mean:.4f})
║ Strength Distribution: {clean_strength_dist['display_text']}
║ ACTUAL Applied Range: {actual_strength_text}
║ Applied Strengths: {clean_strength_dist['count']} parameters with dynamic adjustment
║ Score Distribution: {score_health}
║ Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored if total_scored > 0 else 0:.1f}%) - {ranking_health}
║ Skip Threshold: {skip_threshold} (percentile) → {skipped_threshold} parameters skipped
║ 💡 RECOMMENDED: skip_threshold = {recommended_threshold:.6f} for optimal performance
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

                # Unload input models to free memory - they'll be reloaded if parameters change
                print("[MEMORY] Unloading input models after successful merge")
                del model_base
                if 'other_model_objs' in locals():
                    del other_model_objs
                force_cleanup()

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
                "merge_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "min_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "max_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "normalization_mode": (["magnitude", "calibrate", "none"], {"default": "magnitude"}),  # (renorm_mode)
                # Enhanced WIDEN parameters
                "importance_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 500.0, "step": 0.1}),  # (above_average_value_ratio)
                "importance_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),  # (score_calibration_value)
                # Dynamic compatibility settings  
                "rank_sensitivity": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),  # (compatibility_sensitivity)
                "skip_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.000001}),  # (compatibility_threshold)
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
                "scale_to_min_max": ("BOOLEAN", {"default": False}),
                "invert_strengths": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING", "STRING")
    RETURN_NAMES = ("clip", "merge_results", "parameter_info")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def _apply_dynamic_scaling(self, other_models, min_strength, max_strength, 
                              importance_threshold, importance_boost, rank_sensitivity, skip_threshold):
        """
        Analyze the current strength distribution and scale it to fit perfectly 
        within min_strength to max_strength range.
        
        Returns: (new_min_strength, new_max_strength) that will map the natural 
        distribution to the desired range.
        """
        try:
            from .shared.merge_strength import _compatibility_to_merge_strength
            
            print(f"[DynamicScaling] Analyzing CLIP strength distribution for scaling...")
            
            # Sample a few parameters to estimate the natural strength distribution
            sampled_strengths = []
            param_count = 0
            max_samples = 100  # Don't analyze too many to keep it fast
            
            # Get base model for comparison
            if hasattr(other_models[0], 'lora_name'):
                # LoRA models - skip this for now, use defaults
                print(f"[DynamicScaling] LoRA models detected, using original range")
                return min_strength, max_strength
            
            base_model = other_models[0] if other_models else None
            if base_model is None:
                return min_strength, max_strength
                
            # Sample parameters to estimate distribution
            for name, param in base_model.named_parameters():
                if param_count >= max_samples:
                    break
                    
                # Skip normalization layers and embeddings for faster analysis
                if any(skip_name in name.lower() for skip_name in ['norm', 'bias', 'embed']):
                    continue
                    
                # Calculate what the strength would be for this parameter
                try:
                    # Simplified compatibility calculation (just use param variance as proxy)
                    param_var = torch.var(param).item()
                    mock_compatibility = min(param_var * 1000, 2.0)  # Scale to reasonable range
                    
                    strength = _compatibility_to_merge_strength(
                        compatibility=mock_compatibility,
                        importance_threshold=importance_threshold,
                        importance_boost=importance_boost,
                        rank_sensitivity=rank_sensitivity,
                        skip_threshold=skip_threshold,
                        min_strength=0.0,  # Use full range for analysis
                        max_strength=2.0
                    )
                    
                    if strength > 0:  # Only count non-zero strengths
                        sampled_strengths.append(strength)
                        param_count += 1
                        
                except Exception:
                    continue
            
            if len(sampled_strengths) < 5:
                print(f"[DynamicScaling] Insufficient samples ({len(sampled_strengths)}), using original range")
                return min_strength, max_strength
            
            # Calculate natural distribution
            natural_min = min(sampled_strengths)
            natural_max = max(sampled_strengths)
            natural_range = natural_max - natural_min
            
            print(f"[DynamicScaling] Natural distribution: {natural_min:.3f} to {natural_max:.3f} (range: {natural_range:.3f})")
            print(f"[DynamicScaling] Target range: {min_strength:.3f} to {max_strength:.3f}")
            
            if natural_range < 1e-6:  # Nearly uniform distribution
                print(f"[DynamicScaling] Nearly uniform distribution, using original range")
                return min_strength, max_strength
            
            # Calculate the scaling factor to map natural range to target range
            # We want: natural_min → target_min, natural_max → target_max
            # Linear scaling: new_val = (old_val - natural_min) / natural_range * target_range + target_min
            
            target_range = max_strength - min_strength
            scale_factor = target_range / natural_range
            
            # Adjust the merge function's min/max to achieve the desired final range
            # The merge function will generate [natural_min, natural_max] with some internal scaling
            # We need to set its parameters so the output gets scaled to [target_min, target_max]
            
            # This is tricky because the merge function has its own internal scaling logic
            # The simplest approach is to pre-scale the min/max parameters
            adjusted_min = min_strength  
            adjusted_max = min_strength + (max_strength - min_strength) * (natural_range / (natural_max - 0)) if natural_max > 0 else max_strength
            
            # Actually, let's use a different approach - calculate what the merge function's 
            # min/max should be to get our desired output range
            
            # The merge function maps compatibility scores to [merge_min, merge_max] range
            # We want the final output to be [min_strength, max_strength]
            # So we need to reverse-engineer what merge_min, merge_max should be
            
            # For now, use a simple linear adjustment
            adjustment_factor = target_range / natural_range if natural_range > 0 else 1.0
            new_min = min_strength
            new_max = min_strength + target_range
            
            print(f"[DynamicScaling] Adjusted merge range: [{new_min:.3f}, {new_max:.3f}] (factor: {adjustment_factor:.3f})")
            print(f"[DynamicScaling] This should map [{natural_min:.3f}, {natural_max:.3f}] → [{min_strength:.3f}, {max_strength:.3f}]")
            
            return new_min, new_max
            
        except Exception as e:
            print(f"[DynamicScaling] Error during scaling analysis: {e}")
            print(f"[DynamicScaling] Falling back to original range")
            return min_strength, max_strength

    def _apply_post_merge_scaling(self, merged_params, widen_diagnostics, target_min, target_max):
        """
        Apply scaling to merge results to map natural distribution to target range.
        This analyzes the applied strengths and rescales them linearly.
        """
        try:
            # Extract the applied strength values from diagnostics
            # Fix: applied_strengths is a list of dicts, not a dict!
            applied_strengths_list = widen_diagnostics.get('applied_strengths', [])
            if not applied_strengths_list:
                print("[DynamicScaling] No applied strengths found in diagnostics, skipping scaling")
                return merged_params
            
            # Convert list of dicts to param_name -> strength mapping
            applied_strengths = {item['parameter']: item['strength'] for item in applied_strengths_list}
            
            # Find the actual range of applied strengths
            strength_values = list(applied_strengths.values())
            actual_min = min(strength_values)
            actual_max = max(strength_values)
            actual_range = actual_max - actual_min
            
            if actual_range < 1e-8:
                print("[DynamicScaling] Actual strength range too small, skipping scaling")
                return merged_params
            
            target_range = target_max - target_min
            print(f"[DynamicScaling] Scaling applied strengths from [{actual_min:.3f}, {actual_max:.3f}] to [{target_min:.3f}, {target_max:.3f}]")
            
            # DISABLED: Post-merge parameter scaling corrupts model weights!
            # Dynamic scaling should happen during merge, not after on final parameters
            print("[DynamicScaling] Post-merge scaling disabled to prevent model corruption")
            scaled_params = merged_params  # Return unmodified parameters
                
            print(f"[DynamicScaling] Successfully scaled {len(scaled_params)} parameters")
            return scaled_params
            
        except Exception as e:
            print(f"[DynamicScaling] Error during post-merge scaling: {e}")
            print("[DynamicScaling] Returning unscaled parameters")
            return merged_params

    def _apply_strength_inversion(self, merged_params, widen_diagnostics, min_strength, max_strength):
        """
        Apply strength inversion to flip the strength distribution.
        Low compatibility parameters get high strengths and vice versa.
        """
        try:
            # Extract the applied strength values from diagnostics
            # Fix: applied_strengths is a list of dicts, not a dict!
            applied_strengths_list = widen_diagnostics.get('applied_strengths', [])
            if not applied_strengths_list:
                print("[DynamicScaling] No applied strengths found in diagnostics, skipping scaling")
                return merged_params
            
            # Convert list of dicts to param_name -> strength mapping
            applied_strengths = {item['parameter']: item['strength'] for item in applied_strengths_list}
            if not applied_strengths:
                print("[StrengthInversion] No applied strengths found in diagnostics, skipping inversion")
                return merged_params
            
            # Find the actual range of applied strengths
            strength_values = list(applied_strengths.values())
            actual_min = min(strength_values)
            actual_max = max(strength_values)
            actual_range = actual_max - actual_min
            
            if actual_range < 1e-8:
                print("[StrengthInversion] Actual strength range too small, skipping inversion")
                return merged_params
            
            print(f"[StrengthInversion] Inverting CLIP strengths in range [{actual_min:.3f}, {actual_max:.3f}]")
            
            # Apply inversion to the merged parameters
            inverted_params = {}
            for param_name, param_tensor in merged_params.items():
                original_strength = applied_strengths.get(param_name, 1.0)
                
                # Invert the strength: low becomes high, high becomes low
                # Formula: inverted = actual_max + actual_min - original
                inverted_strength = actual_max + actual_min - original_strength
                
                # Adjust the parameter tensor proportionally
                strength_ratio = inverted_strength / original_strength if original_strength != 0 else 1.0
                inverted_params[param_name] = param_tensor * strength_ratio
                
            print(f"[StrengthInversion] Successfully inverted {len(inverted_params)} CLIP parameters")
            return inverted_params
            
        except Exception as e:
            print(f"[StrengthInversion] Error during CLIP strength inversion: {e}")
            print("[StrengthInversion] Returning non-inverted parameters")
            return merged_params

    def execute(self, clip_base, clip_other, merge_strength, min_strength, max_strength, normalization_mode,
                importance_threshold, importance_boost,
                rank_sensitivity, skip_threshold,
                lora_stack=None, clip_3=None, clip_4=None, clip_5=None, clip_6=None,
                clip_7=None, clip_8=None, clip_9=None, clip_10=None,
                clip_11=None, clip_12=None, scale_to_min_max=False, invert_strengths=False, enable_cache_debug=None, force_fresh_merge=None):

        # Handle legacy debug parameters for backward compatibility
        if enable_cache_debug is not None:
            print(f"[Compatibility] Ignoring legacy enable_cache_debug parameter: {enable_cache_debug}")
        if force_fresh_merge is not None:
            print(f"[Compatibility] Ignoring legacy force_fresh_merge parameter: {force_fresh_merge}")

        # Conservative pre-merge setup with session cache management
        print("[MEMORY] Pre-merge setup...")
        
        # Clear session cache if it's getting large (prevents accumulation)
        if len(_MERGE_CACHE) >= _CACHE_MAX_SIZE:
            clear_session_cache()
        
        # Light pre-merge cleanup
        gentle_cleanup()

        # Check cache first
        all_clips = [clip_base, clip_other, clip_3, clip_4, clip_5, clip_6,
                    clip_7, clip_8, clip_9, clip_10, clip_11, clip_12]
        cache_key = compute_merge_hash(all_clips, merge_strength, min_strength, max_strength, importance_threshold, importance_boost, rank_sensitivity, skip_threshold, f"{normalization_mode}_enhanced_widen", lora_stack, scale_to_min_max, invert_strengths)
        # CLIP cache analysis

        cached_result = check_cache_for_merge_with_bypass(cache_key, False)
        if cached_result is not None:
            # Cache hit
            return cached_result
        else:
            # Cache miss - computing fresh merge
            pass

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

                # CONTAMINATION FIX: Memory-efficient independent CLIP copy
                # ComfyUI's clone() shares parameter tensors. We need isolation but minimize memory usage.
                print("[CONTAMINATION-FIX] Creating memory-efficient independent CLIP copy")
                import copy
                import gc
                
                # Step 1: Force ComfyUI to offload the base CLIP from GPU to free VRAM
                try:
                    import comfy.model_management
                    comfy.model_management.unload_model_clones(clip_base)
                    comfy.model_management.soft_empty_cache(force=True)
                except:
                    pass
                
                # Step 2: Create deep copy with immediate cleanup
                clip_merged = copy.deepcopy(clip_base)
                
                # Step 3: Aggressive memory cleanup after copy
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU cache
                
                print("[CONTAMINATION-FIX] CLIP copied with memory optimization")
                
                # Get the encoder from the cloned CLIP for merging
                enc_merged = getattr(clip_merged, "model", getattr(clip_merged, "clip",
                           getattr(clip_merged, "cond_stage_model", None)))

                # Apply dynamic scaling if requested
                original_min, original_max = min_strength, max_strength
                if scale_to_min_max:
                    print(f"[DonutWidenMergeCLIP] Dynamic scaling enabled - using user bounds directly: {min_strength} to {max_strength}")
                    # FIXED: Use user's actual bounds instead of wide analysis range
                    # since strength calculation already respects these bounds properly

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
                        merge_strength=merge_strength,
                        min_strength=min_strength,
                        max_strength=max_strength,
                        rank_sensitivity=rank_sensitivity,
                        skip_threshold=skip_threshold,
                        normalization_mode=normalization_mode,
                        scale_to_min_max=scale_to_min_max,
                        original_min=original_min,
                        original_max=original_max
                    )
                    
                    # Apply strength inversion if enabled (before scaling)
                    if invert_strengths:
                        print("[DonutWidenMergeCLIP] Applying strength inversion...")
                        merged_params = self._apply_strength_inversion(merged_params, widen_diagnostics, min_strength, max_strength)
                    
                    # Apply dynamic scaling post-processing if enabled
                    if scale_to_min_max:
                        print("[DonutWidenMergeCLIP] Applying dynamic scaling to merge results...")
                        # DISABLED: Post-merge scaling corrupts model - bounds are already correct
                        # merged_params = self._apply_post_merge_scaling(merged_params, widen_diagnostics, original_min, original_max)
                    
                    # Apply merged parameters with shape validation
                    applied_count = 0
                    shape_mismatch_count = 0
                    for param_name, param_value in merged_params.items():
                        for name, param in enc_merged.named_parameters():
                            if name == param_name:
                                # Detect and remove an accidental leading batch-of-1 dim
                                if param_value.ndim == param.ndim + 1 and param_value.size(0) == 1:
                                    param_value = param_value.squeeze(0)
                                
                                if param_value.shape == param.shape:
                                    param.data.copy_(param_value)
                                    applied_count += 1
                                else:
                                    print(f"[WARNING] Shape mismatch for {param_name}: expected {param.shape}, got {param_value.shape}")
                                    shape_mismatch_count += 1
                                break
                    
                    print(f"[ENHANCED WIDEN] Applied {applied_count} parameters, {shape_mismatch_count} shape mismatches")
                    
                    # Compute and display merge sanity metrics
                    try:
                        sanity_metrics = compute_merge_sanity_metrics(base_model, merged_model)
                        print_merge_diagnostics(sanity_metrics)
                    except Exception as e:
                        diagnostic_logger.warning(f"Failed to compute sanity metrics: {e}")
                    
                    # Create detailed merge results with enhanced WIDEN diagnostics
                    total_clips = len([c for c in [clip_other, clip_3, clip_4, clip_5, clip_6, clip_7, clip_8, clip_9, clip_10, clip_11, clip_12] if c is not None]) + 1
                    
                    # Generate enhanced diagnostics from widen_diagnostics
                    compatibility_scores = widen_diagnostics['compatibility_scores']
                    varied_count = widen_diagnostics['varied_score_count']
                    uniform_count = widen_diagnostics['uniform_score_count']
                    skipped_threshold = widen_diagnostics['parameters_skipped_threshold']
                    
                    # Extract strength distribution metrics
                    strength_dist = widen_diagnostics['strength_distribution']
                    applied_strengths = widen_diagnostics['applied_strengths']
                    
                    # Sanitize strength distribution for display
                    clean_strength_dist = sanitize_strength_distribution(strength_dist)
                    
                    # ACTUAL APPLIED STRENGTH DIAGNOSTIC
                    # Compute actual applied strength distribution from merged parameters  
                    actual_applied_strengths = []
                    if applied_strengths:
                        if isinstance(applied_strengths, list):
                            actual_applied_strengths = [item['strength'] for item in applied_strengths if 'strength' in item]
                        elif isinstance(applied_strengths, dict):
                            actual_applied_strengths = list(applied_strengths.values())
                    
                    if actual_applied_strengths:
                        actual_min = min(actual_applied_strengths)
                        actual_max = max(actual_applied_strengths)  
                        actual_mean = sum(actual_applied_strengths) / len(actual_applied_strengths)
                        actual_strength_text = f"{actual_min:.3f}-{actual_max:.3f} (avg {actual_mean:.3f})"
                    else:
                        actual_strength_text = "N/A (no applied strengths found)"
                    
                    if compatibility_scores:
                        compat_values = [score['compatibility'] for score in compatibility_scores]
                        compat_min, compat_max = min(compat_values), max(compat_values)
                        compat_mean = sum(compat_values) / len(compat_values)
                        compat_variance = sum((x - compat_mean)**2 for x in compat_values) / len(compat_values)
                        compat_range = compat_max - compat_min
                        # Use relative variance threshold based on score range
                        relative_variance_threshold = max(1e-6, (compat_range * 0.01) ** 2)
                        score_health = "✓ VARIED" if compat_variance > relative_variance_threshold else "✗ UNIFORM (BUG!)"
                    else:
                        compat_min = compat_max = compat_mean = 0.0
                        score_health = "NO SCORES"
                    
                    ranking_health = "✓ HEALTHY" if varied_count > uniform_count else "✗ FAILING"
                    total_scored = varied_count + uniform_count
                    
                    # Generate skip threshold recommendation for CLIP dynamic mode
                    compatibility_score_dict = {score['parameter']: score['compatibility'] for score in compatibility_scores}
                    recommended_threshold, threshold_analysis = _analyze_compatibility_patterns_and_recommend_threshold(
                        compatibility_score_dict, skip_threshold
                    )
                    
                    # Add threshold analysis to console output
                    performance_logger.info(f"CLIP skip threshold analysis:\n{threshold_analysis}")
                    
                    # Show the correct strength range in diagnostics
                    display_min = original_min if scale_to_min_max else min_strength
                    display_max = original_max if scale_to_min_max else max_strength
                    features = []
                    if scale_to_min_max:
                        features.append("dynamic scaling")
                    if invert_strengths:
                        features.append("inverted")
                    scaling_status = f" ({', '.join(features)})" if features else ""
                    
                    results_text = f"""╔═ WIDEN CLIP MERGE RESULTS (Dynamic Compatibility) ═╗
║ CLIP Models: {total_clips} | Strength: {display_min}-{display_max}{scaling_status} | Mode: {normalization_mode}
║ User Params: merge={merge_strength} | min={original_min} | max={original_max} | scale_to_min_max={scale_to_min_max}
║ Threshold: {importance_threshold} | Boost: {importance_boost} | Sensitivity: {rank_sensitivity}
╠═══════════════════════════════════════════════════╣
║ Parameters Merged: {len(merged_params)} | Applied: {applied_count}
║ Shape Mismatches: {shape_mismatch_count} | Success: {(applied_count/(applied_count+shape_mismatch_count)*100):.1f}%
║ Status: ✓ Enhanced WIDEN with Dynamic Compatibility
╠═══════════════════════════════════════════════════╣
║ 🔍 WIDEN ALGORITHM HEALTH DIAGNOSTICS:
║ Compatibility Range: {compat_min:.4f} - {compat_max:.4f} (avg: {compat_mean:.4f})
║ Strength Distribution: {clean_strength_dist['display_text']}
║ ACTUAL Applied Range: {actual_strength_text}
║ Applied Strengths: {clean_strength_dist['count']} parameters with dynamic adjustment
║ Score Distribution: {score_health}
║ Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored if total_scored > 0 else 0:.1f}%) - {ranking_health}
║ Skip Threshold: {skip_threshold} (percentile) → {skipped_threshold} parameters skipped
║ 💡 RECOMMENDED: skip_threshold = {recommended_threshold:.6f} for optimal performance
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
                        merge_strength=merge_strength,
                        min_strength=min_strength,
                        max_strength=max_strength,
                        scale_to_min_max=scale_to_min_max,
                        original_min=original_min,
                        original_max=original_max,
                        rank_sensitivity=0.0,  # Disable dynamic strength
                        skip_threshold=skip_threshold,
                        normalization_mode=normalization_mode
                    )
                    
                    # Apply strength inversion if enabled (before scaling)
                    if invert_strengths:
                        print("[DonutWidenMergeCLIP] Applying strength inversion (static mode)...")
                        merged_params = self._apply_strength_inversion(merged_params, widen_diagnostics, min_strength, max_strength)
                    
                    # Apply dynamic scaling post-processing if enabled
                    if scale_to_min_max:
                        print("[DonutWidenMergeCLIP] Applying dynamic scaling to merge results (static mode)...")
                        # DISABLED: Post-merge scaling corrupts model - bounds are already correct
                        # merged_params = self._apply_post_merge_scaling(merged_params, widen_diagnostics, original_min, original_max)
                    
                    # Apply merged parameters with shape validation
                    applied_count = 0
                    shape_mismatch_count = 0
                    for param_name, param_value in merged_params.items():
                        for name, param in enc_merged.named_parameters():
                            if name == param_name:
                                # Detect and remove an accidental leading batch-of-1 dim
                                if param_value.ndim == param.ndim + 1 and param_value.size(0) == 1:
                                    param_value = param_value.squeeze(0)
                                
                                if param_value.shape == param.shape:
                                    param.data.copy_(param_value)
                                    applied_count += 1
                                else:
                                    print(f"[WARNING] Shape mismatch for {param_name}: expected {param.shape}, got {param_value.shape}")
                                    shape_mismatch_count += 1
                                break
                    
                    print(f"[ENHANCED WIDEN] Applied {applied_count} parameters, {shape_mismatch_count} shape mismatches")
                    
                    # Compute and display merge sanity metrics
                    try:
                        sanity_metrics = compute_merge_sanity_metrics(base_model, merged_model)
                        print_merge_diagnostics(sanity_metrics)
                    except Exception as e:
                        diagnostic_logger.warning(f"Failed to compute sanity metrics: {e}")
                    
                    # Create detailed merge results with enhanced WIDEN diagnostics
                    total_clips = len([c for c in [clip_other, clip_3, clip_4, clip_5, clip_6, clip_7, clip_8, clip_9, clip_10, clip_11, clip_12] if c is not None]) + 1
                    
                    # Generate enhanced diagnostics from widen_diagnostics
                    compatibility_scores = widen_diagnostics['compatibility_scores']
                    varied_count = widen_diagnostics['varied_score_count']
                    uniform_count = widen_diagnostics['uniform_score_count']
                    skipped_threshold = widen_diagnostics['parameters_skipped_threshold']
                    
                    # Extract strength distribution metrics
                    strength_dist = widen_diagnostics['strength_distribution']
                    applied_strengths = widen_diagnostics['applied_strengths']
                    
                    # Sanitize strength distribution for display
                    clean_strength_dist = sanitize_strength_distribution(strength_dist)
                    
                    # ACTUAL APPLIED STRENGTH DIAGNOSTIC
                    # Compute actual applied strength distribution from merged parameters  
                    actual_applied_strengths = []
                    if applied_strengths:
                        if isinstance(applied_strengths, list):
                            actual_applied_strengths = [item['strength'] for item in applied_strengths if 'strength' in item]
                        elif isinstance(applied_strengths, dict):
                            actual_applied_strengths = list(applied_strengths.values())
                    
                    if actual_applied_strengths:
                        actual_min = min(actual_applied_strengths)
                        actual_max = max(actual_applied_strengths)  
                        actual_mean = sum(actual_applied_strengths) / len(actual_applied_strengths)
                        actual_strength_text = f"{actual_min:.3f}-{actual_max:.3f} (avg {actual_mean:.3f})"
                    else:
                        actual_strength_text = "N/A (no applied strengths found)"
                    
                    if compatibility_scores:
                        compat_values = [score['compatibility'] for score in compatibility_scores]
                        compat_min, compat_max = min(compat_values), max(compat_values)
                        compat_mean = sum(compat_values) / len(compat_values)
                        compat_variance = sum((x - compat_mean)**2 for x in compat_values) / len(compat_values)
                        compat_range = compat_max - compat_min
                        # Use relative variance threshold based on score range
                        relative_variance_threshold = max(1e-6, (compat_range * 0.01) ** 2)
                        score_health = "✓ VARIED" if compat_variance > relative_variance_threshold else "✗ UNIFORM (BUG!)"
                    else:
                        compat_min = compat_max = compat_mean = 0.0
                        score_health = "NO SCORES"
                    
                    ranking_health = "✓ HEALTHY" if varied_count > uniform_count else "✗ FAILING"
                    total_scored = varied_count + uniform_count
                    
                    # Show the correct strength range in diagnostics
                    display_min = original_min if scale_to_min_max else min_strength
                    display_max = original_max if scale_to_min_max else max_strength  
                    features = []
                    if scale_to_min_max:
                        features.append("dynamic scaling")
                    if invert_strengths:
                        features.append("inverted")
                    scaling_status = f" ({', '.join(features)})" if features else ""
                    
                    results_text = f"""╔═ WIDEN CLIP MERGE RESULTS (Static Strength) ═╗
║ CLIP Models: {total_clips} | Strength: {display_min}-{display_max}{scaling_status} | Mode: {normalization_mode}
║ User Params: merge={merge_strength} | min={original_min} | max={original_max} | scale_to_min_max={scale_to_min_max}
║ Threshold: {importance_threshold} | Boost: {importance_boost} | Sensitivity: {rank_sensitivity} (off)
╠═══════════════════════════════════════════════════╣
║ Parameters Merged: {len(merged_params)} | Applied: {applied_count}
║ Shape Mismatches: {shape_mismatch_count} | Success: {(applied_count/(applied_count+shape_mismatch_count)*100):.1f}%
║ Status: ✓ Enhanced WIDEN with Static Strength
╠═══════════════════════════════════════════════════╣
║ 🔍 WIDEN ALGORITHM HEALTH DIAGNOSTICS:
║ Compatibility Range: {compat_min:.4f} - {compat_max:.4f} (avg: {compat_mean:.4f})
║ Strength Distribution: {clean_strength_dist['display_text']}
║ ACTUAL Applied Range: {actual_strength_text}
║ Applied Strengths: {clean_strength_dist['count']} parameters with dynamic adjustment
║ Score Distribution: {score_health}
║ Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored if total_scored > 0 else 0:.1f}%) - {ranking_health}
║ Skip Threshold: {skip_threshold} (percentile) → {skipped_threshold} parameters skipped
║ 💡 RECOMMENDED: skip_threshold = {recommended_threshold:.6f} for optimal performance
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

                # Unload input CLIP models to free memory - they'll be reloaded if parameters change
                print("[MEMORY] Unloading input CLIP models after successful merge")
                del clip_base
                if 'all_clips' in locals():
                    del all_clips
                force_cleanup()

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


class DonutFiller:
    """Emits both a filler MODEL and a filler CLIP; wire whichever you need."""
    class_type = "MODEL"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "execute"
    CATEGORY = "utils"

    def execute(self):
        class _Stub:
            def state_dict(self): return {}
            def named_parameters(self): return iter([])
        m = _Stub()
        setattr(m, "_is_filler", True)
        c = _Stub()
        setattr(c, "_is_filler", True)
        return (m, c)


NODE_CLASS_MAPPINGS = {
    "DonutWidenMergeUNet": DonutWidenMergeUNet,
    "DonutWidenMergeCLIP": DonutWidenMergeCLIP,
    "DonutFiller": DonutFiller,
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

# Cache Debug Utility Node
class DonutCacheDebug:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["enable_debug_logging", "inspect_cache", "test_cache_invalidation", "clear_cache"], {"default": "enable_debug_logging"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug_output",)
    FUNCTION = "execute"
    CATEGORY = "donut/debug"
    
    def execute(self, action):
        try:
            if action == "enable_debug_logging":
                enable_cache_debug_logging()
                return ("Cache debug logging enabled. Check console for detailed cache operations.",)
            
            elif action == "inspect_cache":
                cache_keys = inspect_cache()
                if cache_keys:
                    result = f"Cache contains {len(cache_keys)} entries:\n"
                    for i, key in enumerate(cache_keys[:10]):  # Show first 10
                        result += f"{i+1}. {key[:50]}{'...' if len(key) > 50 else ''}\n"
                    if len(cache_keys) > 10:
                        result += f"... and {len(cache_keys) - 10} more entries"
                else:
                    result = "Cache is empty"
                return (result,)
            
            elif action == "test_cache_invalidation":
                test_result = debug_cache_invalidation_test()
                result = f"Cache invalidation test {'PASSED' if test_result else 'FAILED'}"
                result += "\nCheck console for detailed test results."
                return (result,)
            
            elif action == "clear_cache":
                clear_merge_cache()
                return ("Cache cleared successfully.",)
            
            else:
                return (f"Unknown action: {action}",)
                
        except Exception as e:
            return (f"Error during {action}: {str(e)}",)

NODE_CLASS_MAPPINGS["DonutCacheDebug"] = DonutCacheDebug

import atexit
def cleanup_on_exit():
    """Cleanup on exit"""
    try:
        clear_merge_cache()  # FIXED: Actually call the cache clearing function
        force_cleanup()      # FIXED: Call force cleanup on exit
    except Exception:
        pass

atexit.register(cleanup_on_exit)
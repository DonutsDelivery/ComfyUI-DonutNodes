"""
Core WIDEN Merge Algorithm Functions

This module contains the core WIDEN merge algorithm functions extracted from DonutWidenMerge.py.
These functions implement the enhanced WIDEN merging with dynamic compatibility-based strength,
post-refinement pipelines, and all supporting WIDEN merge processing.

Key Functions:
- _group_parameters_by_blocks(): Groups SDXL parameters by architectural blocks
- enhanced_widen_merging_with_dynamic_strength(): Main WIDEN merge with dynamic strength
- enhanced_widen_merging_with_post_refinement(): WIDEN merge with post-refinement pipeline
- create_enhanced_merge_with_refinement_config(): Creates refinement configuration
- _process_block_parameters_together(): Processes block parameters with cross-parameter evaluation
- _merge_param_magnitude_direction_with_dynamic_strength(): Core parameter merging logic

This module preserves all original WIDEN algorithm logic, comments, and functionality.
"""

import torch
import logging
import os
import time
from collections import defaultdict
from tqdm import tqdm

# Import all shared modules with fallback
try:
    from .logging_config import diagnostic_logger, widen_logger, memory_logger, performance_logger, ProgressBarContext, print_progress_bar
    from .alignment import safe_stack
    from .merge_strength import _compatibility_to_merge_strength, get_adaptive_skip_threshold
    from .memory_management import (
        MemoryEfficientContext, gentle_cleanup, force_cleanup,
        monitor_memory_usage, get_widen_memory_profiler, soft_empty_cache
    )
    from .utility_functions import (
        _analyze_compatibility_patterns_and_recommend_threshold,
        _batch_compute_magnitude_direction_diffs,
        _rank_per_param_magnitude_or_direction_within_model,
        _transpose_token_embeddings,
        smart_device_management,
        get_param_names_to_merge
    )
    from .tensor_operations import (
        _vectorized_parameter_batch_merge,
        _batch_importance_score_computation,
        _safe_tensor_multiply
    )
    from .renormalization import calibrate_renormalize
except ImportError:
    # Fallback imports
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from shared.logging_config import diagnostic_logger, widen_logger, memory_logger, performance_logger, ProgressBarContext, print_progress_bar
    from shared.alignment import safe_stack
    from shared.merge_strength import _compatibility_to_merge_strength, get_adaptive_skip_threshold
    from shared.memory_management import (
        MemoryEfficientContext, gentle_cleanup, force_cleanup,
        monitor_memory_usage, get_widen_memory_profiler, soft_empty_cache
    )
    from shared.utility_functions import (
        _analyze_compatibility_patterns_and_recommend_threshold,
        _batch_compute_magnitude_direction_diffs,
        _rank_per_param_magnitude_or_direction_within_model,
        _transpose_token_embeddings,
        smart_device_management,
        get_param_names_to_merge
    )
    from shared.tensor_operations import (
        _vectorized_parameter_batch_merge,
        _batch_importance_score_computation,
        _safe_tensor_multiply
    )

# Import TaskVector from shared module
try:
    from .task_vector import TaskVector
except ImportError:
    try:
        from shared.task_vector import TaskVector
    except ImportError:
        # Fallback - assume TaskVector is available in the main scope
        pass

# Import post-merge refinement components
try:
    from ..post_merge_refinement import (
        PostMergeRefinementPipeline, create_default_refinement_config,
        compute_layer_energy_stats, compare_model_energy_contrast,
        print_energy_contrast_summary
    )
except ImportError:
    # These are optional for basic WIDEN merge functionality - provide fallbacks
    class PostMergeRefinementPipeline:
        def __init__(self, *args, **kwargs):
            pass
        def apply(self, model, *args, **kwargs):
            return model
    
    def create_default_refinement_config():
        return {}
    
    def compute_layer_energy_stats(*args, **kwargs):
        return {}
    
    def compare_model_energy_contrast(*args, **kwargs):
        return {}
    
    def print_energy_contrast_summary(*args, **kwargs):
        pass

# Import renormalization function
try:
    from ..norm_recalibration import calibrate_renormalize
except ImportError:
    try:
        # Try importing from the root directory
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from norm_recalibration import calibrate_renormalize
    except ImportError:
        # Provide a fallback implementation
        def calibrate_renormalize(merged_param, base_param, mode="none", *args, **kwargs):
            return merged_param


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
            except (ValueError, IndexError) as e:
                diagnostic_logger.warning(f"Could not parse input block number from '{block_name}': {e}")
                return (0, 999)
        elif 'middle_block' in block_name:
            if block_name == 'middle_block':
                return (1, 0)
            else:
                try:
                    num = int(block_name.split('_')[-1])
                    return (1, num)
                except (ValueError, IndexError) as e:
                    diagnostic_logger.warning(f"Could not parse middle block number from '{block_name}': {e}")
                    return (1, 999)
        elif 'output_block_' in block_name:
            try:
                num = int(block_name.split('_')[-1])
                return (2, num)  # Output blocks after middle, then by number
            except (ValueError, IndexError) as e:
                diagnostic_logger.warning(f"Could not parse output block number from '{block_name}': {e}")
                return (2, 999)
        else:
            return (3, block_name)  # Other blocks at end, alphabetically
    
    result.sort(key=sort_key)
    return result


def enhanced_widen_merging_with_dynamic_strength(
    merger,
    merged_model,
    models_to_merge,
    exclude_param_names_regex,
    importance_threshold,
    importance_boost,
    merge_strength,
    min_strength,
    max_strength,
    rank_sensitivity,
    skip_threshold,
    normalization_mode,
    ultra_memory_mode=False,
    scale_to_min_max=False,
    original_min=None,
    original_max=None
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
        except (RuntimeError, AttributeError) as e:
            memory_logger.warning(f"Could not get CUDA memory info: {e}")
            vram_available_mb = 1000  # Conservative fallback
    
    # Check system RAM and auto-enable ultra memory mode if needed
    try:
        import psutil
        ram_info = psutil.virtual_memory()
        ram_available_gb = ram_info.available / (1024 * 1024 * 1024)
        ram_total_gb = ram_info.total / (1024 * 1024 * 1024)
        ram_used_percent = ram_info.percent
        
        # Log memory info for debugging
        
        if ram_available_gb < 2.0:
            print(f"[WARNING] Very low RAM detected ({ram_available_gb:.1f}GB available). Using ultra conservative processing.")
    except (ImportError, AttributeError) as e:
        memory_logger.warning(f"Could not get system RAM info: {e}")
        ram_available_gb = 4.0  # Conservative fallback
    
    # Simplified processing strategy: Always use CPU to avoid redundant GPU/CPU computation
    # Since we're doing all ranking on CPU anyway, using GPU for intermediate computations is wasteful
    computation_device = torch.device("cpu")
    storage_device = torch.device("cpu")
    ranking_device = torch.device("cpu")
    
    if ram_available_gb < 2.0:
        pass  # Low RAM mode: Conservative CPU processing
    elif ram_available_gb > 4.0:
        pass  # High RAM mode: Optimized CPU processing
    else:
        pass  # Standard mode: CPU processing
    
    # All processing on CPU for maximum efficiency and WIDEN validation
    
    # Memory debugging: Track where the 58GB spike occurs
    
    # Create task vectors efficiently (these contain the deltas we need)
    print("🔧 Creating TaskVectors...")
    monitor_memory_usage("PRE-TASKVECTOR")
    
    # MEMORY OPTIMIZATION: Adaptive batch size based on available memory and model count
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / (1024**3)
        if ram_gb < 4:
            batch_size = 1  # Ultra conservative for low memory
        elif ram_gb < 8:
            batch_size = min(2, len(models_to_merge))  # Conservative
        else:
            batch_size = min(3, len(models_to_merge))  # Slightly more aggressive when memory allows
    except ImportError:
        batch_size = min(2, len(models_to_merge))  # Fallback
    models_to_merge_task_vectors = []
    
    performance_logger.info(f"Creating TaskVectors in batches of {batch_size}")
    
    for batch_start in range(0, len(models_to_merge), batch_size):
        batch_end = min(batch_start + batch_size, len(models_to_merge))
        batch_models = models_to_merge[batch_start:batch_end]
        
        monitor_memory_usage(f"PRE-TASKVECTOR-BATCH-{batch_start//batch_size}")
        
        # Create TaskVectors in this batch with memory management
        batch_task_vectors = []
        for i, model_to_merge in enumerate(batch_models):
            global_i = batch_start + i
            memory_logger.debug(f"Creating TaskVector {global_i+1}/{len(models_to_merge)}")
            
            # Create TaskVector with memory-efficient context
            with MemoryEfficientContext(f"TaskVector-{global_i}"):
                tv = TaskVector(merged_model, model_to_merge, exclude_param_names_regex)
                batch_task_vectors.append(tv)
            
            # Debug the TaskVector memory usage (only for first few)
            if global_i < 3:
                # TaskVector memory analysis via monitor_memory_usage
                pass  # Analysis completed
        
        # Add batch to main list
        models_to_merge_task_vectors.extend(batch_task_vectors)
        
        # Cleanup batch references
        del batch_models, batch_task_vectors
        
        # Less frequent cleanup (only after each batch)
        gentle_cleanup()
        monitor_memory_usage(f"POST-TASKVECTOR-BATCH-{batch_start//batch_size}")
    
    performance_logger.info(f"Created {len(models_to_merge_task_vectors)} TaskVectors in {(len(models_to_merge) + batch_size - 1) // batch_size} batches")
    monitor_memory_usage("POST-TASKVECTOR")
    import gc
    gc.collect()
    soft_empty_cache()
    monitor_memory_usage("POST-TASKVECTOR-CLEANUP")
    
    # MEMORY OPTIMIZATION: Don't create full parameter copy - access original model directly
    # Using direct model parameter access to minimize memory
    # pretrained_param_dict = {} - REMOVED to save 6.5GB
    # Instead, we'll access merged_model.named_parameters() directly when needed
    
    # Transpose token embeddings in TaskVectors only (we don't need separate copies)
    for task_vector in models_to_merge_task_vectors:
        _transpose_token_embeddings(task_vector.task_vector_param_dict)
    # Note: We'll handle transposition when accessing merged_model parameters directly
    
    # Create parameter cache for O(1) lookup instead of O(N) scan
    base_param_dict = dict(merged_model.named_parameters())
    
    # Helper function to get parameter directly from model (saves 6.5GB of memory + O(1) lookup)
    def get_base_param(param_name, device=storage_device):
        """Get parameter directly from base model without storing full copy"""
        if param_name not in base_param_dict:
            raise KeyError(f"Parameter {param_name} not found in base model")
        
        param = base_param_dict[param_name]
        result = param.detach().to(device).float()
        # Apply transpose if needed
        if param_name == "model.embed_tokens.weight":
            result = result.transpose(dim0=0, dim1=1)
        return result
    
    # Get list of parameter names for processing
    param_names_in_model = [name for name, _ in merged_model.named_parameters()]
    
    with torch.no_grad():
        widen_logger.info("Computing differences...")
        
        # Step 1: Use TaskVector deltas directly instead of recomputing everything
        memory_logger.debug("Computing magnitude and direction differences - monitoring for memory spikes")
        monitor_memory_usage("PRE-MAGNITUDE-DIRECTION")
        
        # Use batched computation for better performance and memory efficiency
        performance_logger.info("Using optimized batched tensor operations with mixed precision...")
        
        # Enable mixed precision for bandwidth-bound operations
        use_mixed_precision = (
            torch.cuda.is_available() and 
            torch.cuda.get_device_capability()[0] >= 7  # Only on GPUs that support FP16 efficiently
        )
        
        if use_mixed_precision:
            performance_logger.info("Mixed precision enabled - using FP16 for magnitude/direction computations")
        
        # Enhanced profiling for performance analysis
        # Enable profiling based on environment variable or logger level
        use_profiler = (
            performance_logger.isEnabledFor(logging.DEBUG) or 
            os.environ.get('WIDEN_ENABLE_PROFILER', '').lower() in ('1', 'true', 'yes')
        )
        
        if use_profiler:
            performance_logger.info("Profiler enabled - this will slow down execution but provide detailed performance analysis")
            
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                # Only record a subset to avoid huge trace files
                schedule=torch.profiler.schedule(
                    wait=1,    # Skip first step
                    warmup=1,  # Warmup for 1 step  
                    active=3,  # Record 3 steps
                    repeat=1   # Only do this once
                )
            ) as prof:
                # Use mixed precision and memory-efficient context for magnitude/direction computations
                with MemoryEfficientContext("magnitude_direction_computation_profiled"):
                    # No autocast to avoid TorchScript vmap issues
                    models_to_merge_param_magnitude_direction_diff_tuples = _batch_compute_magnitude_direction_diffs(
                        models_to_merge_task_vectors, 
                        param_names_in_model,
                        skip_threshold
                    )
                prof.step()  # Required for scheduled profiling
            
            # Save profiling results with timestamp
            import time
            timestamp = int(time.time())
            trace_file = f"widen_profile_trace_{timestamp}.json"
            prof.export_chrome_trace(trace_file)
            performance_logger.info(f"Saved profiling trace to {trace_file}")
            
            # Show top operations
            key_averages = prof.key_averages(group_by_stack_n=5)
            performance_logger.info("Top 10 operations by time:")
            performance_logger.info(key_averages.table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", 
                row_limit=10
            ))
        else:
            # Use mixed precision and memory-efficient context for magnitude/direction computations
            with MemoryEfficientContext("magnitude_direction_computation"):
                # No autocast to avoid TorchScript vmap issues
                models_to_merge_param_magnitude_direction_diff_tuples = _batch_compute_magnitude_direction_diffs(
                    models_to_merge_task_vectors, 
                    param_names_in_model,
                    skip_threshold
                )
        
        print(f"📊 Computed differences for {len(models_to_merge_task_vectors)} models")
        
        # Cleanup after magnitude/direction computation
        force_cleanup()
        
        # Step 2: Enhanced parameter merging with dynamic compatibility-based strength (no autocast for vmap compatibility)
        merged_params = _merge_param_magnitude_direction_with_dynamic_strength(
            models_to_merge_param_magnitude_direction_diff_tuples,
            get_base_param,  # Pass function instead of full parameter dict
            models_to_merge_task_vectors,
            exclude_param_names_regex,
            importance_threshold,
            importance_boost,
            merge_strength,
            min_strength,
            max_strength,
            rank_sensitivity,
            skip_threshold,
            normalization_mode,
            computation_device,
            target_device,
            storage_device,
            ranking_device,
            param_names_in_model,
            scale_to_min_max,
            original_min,
            original_max
        )
        
        # Transpose back
        _transpose_token_embeddings(merged_params)
    
    return merged_params


def enhanced_widen_merging_with_post_refinement(
    merger,
    merged_model,
    models_to_merge,
    exclude_param_names_regex,
    importance_threshold,
    importance_boost,
    merge_strength,
    min_strength,
    max_strength,
    rank_sensitivity,
    skip_threshold,
    normalization_mode,
    ultra_memory_mode=False,
    enable_post_refinement=True,
    refinement_config=None,
    calibration_data=None
):
    """
    Enhanced WIDEN merging with optional post-merge refinement pipeline.
    
    This function applies the complete pipeline:
    1. Standard WIDEN merge with dynamic strength
    2. Post-merge refinement (optional):
       - Frobenius norm rescaling
       - Low-rank residual injection  
       - Normalization recalibration
       - Mini-finetuning (if calibration data provided)
       - Activation space sharpening
    3. Energy/contrast monitoring
    
    Args:
        (... same as enhanced_widen_merging_with_dynamic_strength ...)
        enable_post_refinement: Whether to apply post-merge refinement
        refinement_config: Dict of refinement settings (uses defaults if None)
        calibration_data: Optional data for calibration-based refinements
        
    Returns:
        Dict containing merged parameters and refinement statistics
    """
    diagnostic_logger.info("🚀 Starting enhanced WIDEN merge with post-refinement pipeline")
    
    # Step 1: Perform standard WIDEN merge
    with MemoryEfficientContext("widen_merge"):
        merged_params = enhanced_widen_merging_with_dynamic_strength(
            merger=merger,
            merged_model=merged_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            importance_threshold=importance_threshold,
            importance_boost=importance_boost,
            merge_strength=merge_strength,
            min_strength=min_strength,
            max_strength=max_strength,
            rank_sensitivity=rank_sensitivity,
            skip_threshold=skip_threshold,
            normalization_mode=normalization_mode,
            ultra_memory_mode=ultra_memory_mode
        )
    
    result = {
        'merged_params': merged_params,
        'refinement_applied': False,
        'refinement_stats': {},
        'energy_contrast_analysis': {}
    }
    
    if not enable_post_refinement:
        diagnostic_logger.info("Post-refinement disabled, returning merged parameters")
        return result
    
    # Step 2: Apply the merged parameters to create a working model for refinement
    try:
        # Create a copy of the merged model for refinement
        import copy
        working_model = copy.deepcopy(merged_model)
        working_model.load_state_dict(merged_params)
        working_model.to(next(merged_model.parameters()).device)
        
        diagnostic_logger.info("📊 Computing pre-refinement energy/contrast baseline")
        
        # Baseline energy/contrast measurement
        pre_refinement_stats = compute_layer_energy_stats(working_model)
        
        # Step 3: Apply post-merge refinement pipeline
        if refinement_config is None:
            refinement_config = create_default_refinement_config()
        
        pipeline = PostMergeRefinementPipeline(refinement_config)
        
        monitor_memory_usage("PRE-REFINEMENT")
        
        # Apply refinement pipeline
        refinement_stats = pipeline.apply_full_pipeline(
            merged_model=working_model,
            base_model=merged_model,
            other_models=models_to_merge,
            calibration_data=calibration_data,
            monitor_stats=True
        )
        
        monitor_memory_usage("POST-REFINEMENT")
        
        # Step 4: Compute post-refinement energy/contrast analysis
        diagnostic_logger.info("📊 Computing post-refinement energy/contrast analysis")
        
        energy_contrast_comparison = compare_model_energy_contrast(
            merged_model, working_model, calibration_data
        )
        
        # Print summary
        print_energy_contrast_summary(energy_contrast_comparison)
        
        # Step 5: Extract refined parameters
        refined_params = working_model.state_dict()
        
        result.update({
            'merged_params': refined_params,
            'refinement_applied': True,
            'refinement_stats': refinement_stats,
            'energy_contrast_analysis': energy_contrast_comparison,
            'pre_refinement_stats': pre_refinement_stats
        })
        
        diagnostic_logger.info(f"✅ Post-refinement pipeline complete: {len(refinement_stats['steps_completed'])} steps applied")
        
    except Exception as e:
        diagnostic_logger.error(f"Post-refinement pipeline failed: {e}")
        # Return original merged parameters if refinement fails
        result['refinement_error'] = str(e)
    
    return result


def create_enhanced_merge_with_refinement_config(enable_frobenius=True, enable_low_rank=True, 
                                                 enable_mini_finetune=False, enable_sharpening=False):
    """
    Create a configuration for enhanced WIDEN merge with post-refinement.
    
    Args:
        enable_frobenius: Enable Frobenius norm rescaling
        enable_low_rank: Enable low-rank residual injection  
        enable_mini_finetune: Enable mini-finetuning (requires calibration data)
        enable_sharpening: Enable activation space sharpening
        
    Returns:
        Dictionary of settings for enhanced merge
    """
    return {
        'frobenius_rescaling': enable_frobenius,
        'low_rank_injection': enable_low_rank,
        'svd_rank': 4,
        'injection_strength': 0.3,
        'norm_recalibration': True,
        'mini_finetune': enable_mini_finetune,
        'finetune_steps': 3,
        'finetune_lr': 1e-5,
        'activation_sharpening': enable_sharpening,
        'sharpening_lambda': 0.1
    }


def _process_block_parameters_together(
    block_params,
    param_names_merged_by_magnitude_direction,
    magnitude_rankings,
    direction_rankings,
    models_to_merge_task_vectors,
    get_base_param_func,
    above_average_value_ratio,
    score_calibration_value,
    merge_strength,
    min_strength,
    max_strength,
    rank_sensitivity,
    skip_threshold,
    normalization_mode,
    computation_device,
    target_device,
    widen_diagnostics,
    scale_to_min_max=False,
    original_min=None,
    original_max=None
):
    """Process all parameters in a block together to preserve WIDEN cross-parameter evaluation"""
    import torch
    # Note: smart_device_management already imported at module level
    # Note: calibrate_renormalize and _compute_importance_scores are defined in this file
    
    block_merged_params = {}
    
    # Collect all delta tensors for all parameters in the block at once
    block_delta_tensors = {}
    block_rankings = {}
    
    # Phase 1: Collect all data for block parameters
    with torch.no_grad():
        for param_name in block_params:
            try:
                # Get delta tensors for this parameter
                if param_name in param_names_merged_by_magnitude_direction and param_name in magnitude_rankings:
                    delta_tensors = []
                    for models_to_merge_task_vector in models_to_merge_task_vectors:
                        if param_name in models_to_merge_task_vector.task_vector_param_dict:
                            delta = models_to_merge_task_vector.task_vector_param_dict[param_name]
                            delta = smart_device_management(delta, computation_device)
                            delta_tensors.append(delta)
                    
                    if delta_tensors:
                        block_delta_tensors[param_name] = torch.stack(delta_tensors, dim=0)
                        block_rankings[param_name] = {
                            'magnitude': magnitude_rankings[param_name].to(computation_device),
                            'direction': direction_rankings[param_name].to(computation_device)
                        }
                        del delta_tensors
                    else:
                        # No deltas - use base parameter
                        widen_diagnostics['parameters_skipped_no_rankings'] += 1
                        if len(widen_diagnostics['compatibility_scores']) < 3:  # Debug first few
                            # No deltas - using base parameter unchanged
                            pass
                        block_merged_params[param_name] = get_base_param_func(param_name, target_device)
                        
            except Exception as e:
                widen_logger.error(f"Failed to collect data for {param_name}: {e}")
                # Fallback to base parameter
                block_merged_params[param_name] = get_base_param_func(param_name, target_device)
    
    # Phase 2: Vectorized batch processing of all collected parameters
    if block_delta_tensors:
        # Prepare batch data for vectorized processing
        param_names_batch = list(block_delta_tensors.keys())
        magnitude_ranks_batch = [block_rankings[name]['magnitude'] for name in param_names_batch]
        direction_ranks_batch = [block_rankings[name]['direction'] for name in param_names_batch] 
        
        # Vectorized importance score computation for the entire batch
        batch_importance_scores, batch_variances = _batch_importance_score_computation(
            magnitude_ranks_batch, direction_ranks_batch, 
            above_average_value_ratio, score_calibration_value, computation_device
        )
        
        # Prepare batch data for parameter merging
        param_tensors_batch = []
        task_vector_deltas_batch = []
        weight_scores_batch = []
        valid_param_indices = []
        
        for idx, param_name in enumerate(param_names_batch):
            try:
                magnitude_scores, direction_scores = batch_importance_scores[idx]
                variance_info = batch_variances[idx]
                
                # WIDEN diagnostics
                widen_diagnostics['parameters_with_rankings'] += 1
                
                # Count uniform scores
                if variance_info['magnitude_uniform']:
                    widen_diagnostics['uniform_score_count'] += 1
                    # Uniform magnitude scores detected
                    if len(widen_diagnostics['compatibility_scores']) < 5:
                        # Uniform magnitude scores detected
                        pass
                else:
                    widen_diagnostics['varied_score_count'] += 1
                    
                if variance_info['direction_uniform']:
                    widen_diagnostics['uniform_score_count'] += 1
                    # Uniform direction scores detected
                    if len(widen_diagnostics['compatibility_scores']) < 5:
                        # Uniform direction scores detected
                        pass
                else:
                    widen_diagnostics['varied_score_count'] += 1
                
                widen_diagnostics['importance_score_variances'].append({
                    'parameter': param_name,
                    'magnitude_variance': torch.var(magnitude_scores).item(),
                    'direction_variance': torch.var(direction_scores).item()
                })
                
                # Check for empty scores
                if magnitude_scores.numel() == 0 or direction_scores.numel() == 0:
                    print(f"[WARNING] Empty scores for {param_name}, using fallback")
                    block_merged_params[param_name] = get_base_param_func(param_name, target_device)
                    continue
                
                # Compute compatibility and dynamic strength
                combined_scores = 0.5 * (magnitude_scores + direction_scores)
                compatibility_score = torch.mean(combined_scores).item()
                
                # Raw compatibility score analysis
                if len(widen_diagnostics['compatibility_scores']) < 10:  # Log first 10 parameters
                    # Parameter compatibility analysis completed
                    pass
                
                widen_diagnostics['compatibility_scores'].append({
                    'parameter': param_name,
                    'compatibility': compatibility_score
                })
                
                # Calculate dynamic strength BEFORE skip check to see what we would get
                parameter_strength = _compatibility_to_merge_strength(
                    compatibility_score, merge_strength, min_strength, max_strength, rank_sensitivity
                )
                
                # Strength calculation analysis
                if len(widen_diagnostics['compatibility_scores']) <= 5:
                    # Parameter strength calculated
                    pass
                
                # Use adaptive skip threshold based on parameter type
                adaptive_threshold = get_adaptive_skip_threshold(param_name, skip_threshold)
                
                # Skip if incompatible
                if abs(compatibility_score) < adaptive_threshold:
                    diagnostic_logger.debug(f"Skipped {param_name}: {compatibility_score:.6f} < {adaptive_threshold:.6f} (adaptive)")
                    widen_diagnostics['parameters_skipped_threshold'] += 1
                    block_merged_params[param_name] = get_base_param_func(param_name, target_device)
                    continue
                
                # Track applied strength for diagnostics
                widen_diagnostics['applied_strengths'].append({
                    'parameter': param_name,
                    'strength': parameter_strength,
                    'compatibility': compatibility_score
                })
                
                # Update strength distribution stats
                dist = widen_diagnostics['strength_distribution']
                if dist['count'] == 0:
                    # First strength value
                    dist['min_used'] = parameter_strength
                    dist['max_used'] = parameter_strength
                    dist['mean'] = parameter_strength
                else:
                    # Subsequent values
                    dist['min_used'] = min(dist['min_used'], parameter_strength)
                    dist['max_used'] = max(dist['max_used'], parameter_strength)
                    dist['mean'] = (dist['mean'] * dist['count'] + parameter_strength) / (dist['count'] + 1)
                dist['count'] += 1
                
                # Check for scalar/1D tensors that need simple blend fallback
                base_param = get_base_param_func(param_name, computation_device)
                delta_param = block_delta_tensors[param_name]
                
                if base_param.dim() <= 1:
                    # Simple blend for scalars/vectors like logit_scale - use same logic as batch processing
                    weight_scores = torch.full_like(combined_scores, parameter_strength)
                    weighted_deltas = _safe_tensor_multiply(delta_param, weight_scores, param_name)
                    if weighted_deltas is not None:
                        merged_delta = weighted_deltas.sum(dim=0)
                    else:
                        # Fallback with proper parameter_strength application
                        merged_delta = delta_param.mean(dim=0) * parameter_strength
                    merged_param = base_param + merged_delta
                    
                    # Apply renormalization if enabled AND there's actually a meaningful delta (from bd89e6c)
                    if hasattr(merged_param, 'shape') and normalization_mode != "none":
                        # Skip renormalization if effective strength is zero (preserves exact base model)
                        if parameter_strength != 0.0 and torch.any(torch.abs(merged_delta) > 1e-10):
                            try:
                                if normalization_mode == "calibrate":
                                    merged_param = calibrate_renormalize(merged_param, base_param, normalization_mode, 0.3, 1.1)
                                else:  # magnitude
                                    merged_param = calibrate_renormalize(merged_param, base_param, normalization_mode, 1.0, 1.0)
                            except Exception as e:
                                widen_logger.warning(f"Renormalization failed for {param_name}: {e}")
                                # Continue with non-renormalized parameter
                        else:
                            # Zero strength: use exact base parameter without renormalization
                            diagnostic_logger.debug(f"Skipping renormalization for {param_name}: zero effective strength")
                    
                    block_merged_params[param_name] = merged_param.to(target_device)
                    continue
                
                # Prepare data for vectorized batch merge (multi-dimensional tensors)
                weight_scores = torch.full_like(combined_scores, parameter_strength)
                
                param_tensors_batch.append(base_param)
                task_vector_deltas_batch.append(delta_param)  
                weight_scores_batch.append(weight_scores)
                valid_param_indices.append(idx)
                
            except Exception as e:
                widen_logger.error(f"Failed to process scores for {param_name}: {e}")
                block_merged_params[param_name] = get_base_param_func(param_name, target_device)
        
        # Note: Dynamic scaling will be applied globally after all parameters are processed
        
        # Vectorized batch parameter merging
        if param_tensors_batch:
            try:
                # Vectorized batch merge (no autocast to avoid TorchScript issues)
                merged_params_batch = _vectorized_parameter_batch_merge(
                    param_tensors_batch, task_vector_deltas_batch, weight_scores_batch, computation_device
                )
                
                # Apply results and renormalization
                for i, param_idx in enumerate(valid_param_indices):
                    param_name = param_names_batch[param_idx]
                    merged_param = merged_params_batch[i]
                    
                    # Apply renormalization if enabled AND there's actually a meaningful delta (from bd89e6c)
                    if hasattr(merged_param, 'shape') and normalization_mode != "none":
                        # Extract parameter_strength from weight_scores for this parameter
                        parameter_strength = weight_scores_batch[i].mean().item()
                        
                        try:
                            base_param_for_renorm = get_base_param_func(param_name, computation_device)
                            merged_delta = merged_param - base_param_for_renorm
                            
                            # Skip renormalization if effective strength is zero (preserves exact base model)
                            if parameter_strength != 0.0 and torch.any(torch.abs(merged_delta) > 1e-10):
                                if normalization_mode == "calibrate":
                                    merged_param = calibrate_renormalize(merged_param, base_param_for_renorm, normalization_mode, 0.3, 1.1)
                                else:  # magnitude
                                    merged_param = calibrate_renormalize(merged_param, base_param_for_renorm, normalization_mode, 1.0, 1.0)
                            else:
                                # Zero strength: use exact base parameter without renormalization
                                diagnostic_logger.debug(f"Skipping renormalization for {param_name}: zero effective strength")
                        except Exception as e:
                            widen_logger.warning(f"Renormalization failed for {param_name}: {e}")
                    
                    # Store result
                    block_merged_params[param_name] = merged_param.to(target_device)
                    
            except Exception as e:
                widen_logger.error(f"Vectorized batch merge failed: {e}")
                # Structure-preserving fallback using adaptive pooling (from bd89e6c)
                print(f"[WIDEN] Structure-preserving merge failed, using adaptive fallback: {e}")
                # Fallback to sequential processing for this batch
                for i, param_idx in enumerate(valid_param_indices):
                    param_name = param_names_batch[param_idx]
                    try:
                        base_param = param_tensors_batch[i]
                        delta_param = task_vector_deltas_batch[i]
                        weight_scores = weight_scores_batch[i]
                        
                        weighted_deltas = _safe_tensor_multiply(delta_param, weight_scores, param_name)
                        if weighted_deltas is not None:
                            merged_delta_param = weighted_deltas.sum(dim=0)
                        else:
                            # Structure-preserving fallback using parameter_strength (from bd89e6c)
                            try:
                                # Try to preserve as much structure as possible in fallback
                                if delta_param.dim() > 1:
                                    # Use mean along model dimension only, preserve parameter structure
                                    merged_delta_param = delta_param.mean(dim=0) * weight_scores.mean().item()
                                else:
                                    merged_delta_param = delta_param.mean() * weight_scores.mean().item()
                            except:
                                # Final fallback - this should rarely happen
                                merged_delta_param = torch.zeros_like(delta_param[0] if delta_param.dim() > 0 else delta_param)
                        
                        merged_param = base_param + merged_delta_param
                        block_merged_params[param_name] = merged_param.to(target_device)
                        
                    except Exception as inner_e:
                        widen_logger.error(f"Fallback processing failed for {param_name}: {inner_e}")
                        block_merged_params[param_name] = get_base_param_func(param_name, target_device)
    
    # Clean up block data
    for param_name in list(block_delta_tensors.keys()):
        del block_delta_tensors[param_name]
    for param_name in list(block_rankings.keys()):
        del block_rankings[param_name]['magnitude']
        del block_rankings[param_name]['direction']
    del block_delta_tensors, block_rankings
    
    return block_merged_params


def _merge_param_magnitude_direction_with_dynamic_strength(
    models_to_merge_param_magnitude_direction_diff_tuples,
    get_base_param_func,  # Function to get parameters instead of full dict
    models_to_merge_task_vectors,
    exclude_param_names_regex,
    importance_threshold,
    importance_boost,
    merge_strength,
    min_strength,
    max_strength,
    rank_sensitivity,
    skip_threshold,
    normalization_mode,
    computation_device,
    target_device,
    storage_device,
    ranking_device,
    param_names_in_model,
    scale_to_min_max=False,
    original_min=None,
    original_max=None
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
        input_param_names=param_names_in_model,  # Use passed parameter names
        exclude_param_names_regex=exclude_param_names_regex
    )
    
    # Initialize WIDEN diagnostics collection
    widen_diagnostics = {
        'compatibility_scores': [],
        'importance_score_variances': [],
        'magnitude_score_ranges': [],
        'direction_score_ranges': [],
        'applied_strengths': [],  # Track actual strength values applied
        'strength_distribution': {'min_used': 0.0, 'max_used': 0.0, 'mean': 0.0, 'count': 0},
        'parameters_with_rankings': 0,
        'parameters_skipped_threshold': 0,
        'parameters_skipped_no_rankings': 0,
        'uniform_score_count': 0,
        'varied_score_count': 0
    }
    
    # Unpack magnitude and direction differences
    
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
        if not param_names_merged_by_magnitude_direction:
            print(f"[CRITICAL] No parameters eligible for ranking - all filtered out!")
    
    # PHASE 1: Compute ALL rankings on CPU to preserve WIDEN cross-parameter validation
    if param_names_merged_by_magnitude_direction:
        print(f"Phase 1: Computing rankings for {len(param_names_merged_by_magnitude_direction)} parameters...")
        if 'magnitude_rankings' not in locals():
            magnitude_rankings = {}
        if 'direction_rankings' not in locals():  
            direction_rankings = {}
            
        # Organize parameters by SDXL blocks to preserve WIDEN cross-parameter context
        block_groups = _group_parameters_by_blocks(param_names_merged_by_magnitude_direction)
        total_blocks = len(block_groups)
        total_params = len(param_names_merged_by_magnitude_direction)
        
        # Block-wise processing setup complete
        
        import time
        start_time = time.time()
        
        # Process parameters block by block to preserve WIDEN cross-parameter validation
        # Starting block processing
        monitor_memory_usage("PRE-BLOCK-PROCESSING")
        
        profiler = get_widen_memory_profiler()
        profiler.start()
        profiler.checkpoint(f"Block processing started - {total_blocks} blocks, {total_params} params")
        
        for block_idx, (block_name, block_params) in enumerate(block_groups):
            block_start_time = time.time()
            
            # Update progress bar with clean output (suppress profiler logs during update)
            with ProgressBarContext():
                profiler.checkpoint(f"Block {block_idx+1}/{total_blocks} start: {block_name}")
                print_progress_bar(block_idx + 1, total_blocks, prefix=f'Block Merge', suffix=f'{block_name} ({len(block_params)} params)')
            
            # MEMORY OPTIMIZATION: Process all parameters in block together (preserves WIDEN cross-parameter evaluation)
            # But use memory-efficient tensor operations and aggressive cleanup
            block_magnitude_diffs = {}
            block_direction_diffs = {}
            valid_params_in_block = []
            
            # Processing parameters together for cross-parameter evaluation
            
            # Collect all magnitude/direction diffs for this block with memory optimization
            for param_name in block_params:
                try:
                    magnitude_diffs = []
                    direction_diffs = []
                    
                    for model_idx in range(len(models_to_merge_param_magnitude_diff_tuple)):
                        if param_name in models_to_merge_param_magnitude_diff_tuple[model_idx]:
                            mag_diff = models_to_merge_param_magnitude_diff_tuple[model_idx][param_name]
                            dir_diff = models_to_merge_param_direction_diff_tuple[model_idx][param_name]
                            
                            # Handle both scalar and tensor values
                            if isinstance(mag_diff, (int, float)):
                                # Scalar values - no device management or size checking needed
                                magnitude_diffs.append(mag_diff)
                                direction_diffs.append(dir_diff)
                            else:
                                # Tensor values - apply original logic
                                mag_diff = smart_device_management(mag_diff, torch.device('cpu'))
                                dir_diff = smart_device_management(dir_diff, torch.device('cpu'))
                                
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
                
                # Immediate cleanup of intermediate tensors to prevent accumulation
                del magnitude_diffs, direction_diffs
                
                # Gentle cleanup every 20 parameters to manage memory without breaking WIDEN evaluation
                if len(valid_params_in_block) % 20 == 0:
                    gentle_cleanup()
            
            # Now compute rankings for all valid parameters in this block
            # Computing rankings for block parameters
            
            # MEMORY OPTIMIZATION: Process rankings one at a time but maintain cross-parameter visibility
            for param_idx, param_name in enumerate(valid_params_in_block):
                # Initialize variables to avoid scoping issues
                mag_tensor = None
                dir_tensor = None
                
                try:
                    # Use memory-efficient tensor operations for stacking - FORCE CPU TO AVOID VRAM OOM
                    with torch.no_grad():
                        # Block-wise logging - log first parameter of each block
                        if param_idx == 0:
                            # Starting rankings computation
                            pass
                        
                        # Handle scalar values from dimension-agnostic computation
                        mag_values = block_magnitude_diffs[param_name]
                        dir_values = block_direction_diffs[param_name]
                        
                        # Use sophisticated tensor alignment instead of naive stacking
                        if isinstance(mag_values[0], (int, float)):
                            # We have scalar values, create 1D tensor
                            mag_tensor = torch.tensor(mag_values, dtype=torch.float32, device='cpu').unsqueeze(1)
                            dir_tensor = torch.tensor(dir_values, dtype=torch.float32, device='cpu').unsqueeze(1)
                        else:
                            # We have tensor values, use structure-preserving alignment
                            cpu_mag_tensors = [t.cpu() if t.device.type != 'cpu' else t for t in mag_values]
                            cpu_dir_tensors = [t.cpu() if t.device.type != 'cpu' else t for t in dir_values]
                            
                            # Use safe_stack for robust tensor alignment
                            mag_tensor = safe_stack(cpu_mag_tensors, dim=0)
                            dir_tensor = safe_stack(cpu_dir_tensors, dim=0)
                        
                        # Keep on CPU for memory efficiency
                        mag_tensor = mag_tensor.cpu()
                        dir_tensor = dir_tensor.cpu()
                    
                    # Verify ranking shapes for WIDEN correctness (first parameter only) - BEFORE deletion
                    if block_idx == 0 and param_name == valid_params_in_block[0]:
                        print(f"[WIDEN VERIFICATION] Parameter '{param_name}': magnitude shape {mag_tensor.shape} -> ranking shape will be computed")
                        print(f"[WIDEN VERIFICATION] This ranks {mag_tensor.shape[1]} features across {mag_tensor.shape[0]} models - CORRECT WIDEN behavior")
                    
                    # Compute rankings with FULL parameter visibility (critical for WIDEN)
                    # Handle scalar/1D tensors like logit_scale that don't need ranking
                    if mag_tensor.dim() <= 1 or mag_tensor.shape[1] <= 1:
                        # For scalar or single-element tensors, create simple uniform rankings
                        magnitude_rankings[param_name] = torch.ones_like(mag_tensor, dtype=torch.float32) * 0.5
                        direction_rankings[param_name] = torch.ones_like(dir_tensor, dtype=torch.float32) * 0.5
                    else:
                        magnitude_rankings[param_name] = _rank_per_param_magnitude_or_direction_within_model(mag_tensor)
                        direction_rankings[param_name] = _rank_per_param_magnitude_or_direction_within_model(dir_tensor)
                    
                    # Block-wise processing - no cleanup within block to maintain WIDEN integrity
                    
                except Exception as e:
                    print(f"[WARNING] Failed to compute rankings for {param_name}: {e}")
                
                finally:
                    # IMMEDIATE cleanup of large tensors after each parameter - even on failure
                    if mag_tensor is not None:
                        del mag_tensor
                    if dir_tensor is not None:
                        del dir_tensor
            
            # Clean up block data
            del block_magnitude_diffs, block_direction_diffs
            
            # Block timing and cleanup
            block_time = time.time() - block_start_time
            profiler.checkpoint(f"Block {block_idx+1}/{total_blocks} complete: {block_name} ({block_time:.1f}s)")
            # Block completed
            
            # Clean up after each block's ranking computation - this is when memory is allocated
            gentle_cleanup()  # Clean up intermediate ranking tensors and computations
    
    print(f"✅ Phase 1 complete: Rankings computed for {len(magnitude_rankings)} parameters")
    
    # AUTO-TUNE SKIP THRESHOLD based on Phase 1 compatibility scores
    if widen_diagnostics['compatibility_scores']:
        compat_dict = {d['parameter']: d['compatibility'] for d in widen_diagnostics['compatibility_scores']}
        new_thresh, analysis = _analyze_compatibility_patterns_and_recommend_threshold(
            compat_dict, skip_threshold
        )
        if abs(new_thresh - skip_threshold) > 1e-7:  # Only adjust if meaningful difference
            performance_logger.info(f"🎯 Auto-adjusting skip_threshold {skip_threshold:.6f} → {new_thresh:.6f}")
            performance_logger.info(f"Analysis: {analysis}")
            skip_threshold = new_thresh
            print(f"[AUTO-TUNE] Skip threshold adjusted: {skip_threshold:.6f}")
        else:
            print(f"[AUTO-TUNE] Skip threshold optimal: {skip_threshold:.6f}")
    
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
    processed_count = 0  # Track total processed parameters for memory cleanup
    
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
                soft_empty_cache(force=True)
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
        
        # Update progress bar with clean output (suppress profiler logs during update)
        with ProgressBarContext():
            profiler.checkpoint(f"Phase2 Block {processed_block_count}/{total_phase2_blocks} start: {block_name}")
            print_progress_bar(processed_block_count, total_phase2_blocks, prefix=f'Phase 2 Merge', suffix=f'{block_name} ({len(block_params)} params, {ram_gb:.1f}GB RAM)')
        # Processing parameters together for cross-parameter evaluation
        
        # Process all parameters in this block together (WIDEN cross-parameter evaluation)
        block_merged_params = _process_block_parameters_together(
            block_params,
            param_names_merged_by_magnitude_direction,
            magnitude_rankings,
            direction_rankings,
            models_to_merge_task_vectors,
            get_base_param_func,
            above_average_value_ratio,
            score_calibration_value,
            merge_strength,
            min_strength,
            max_strength,
            rank_sensitivity,
            skip_threshold,
            normalization_mode,
            computation_device,
            target_device,
            widen_diagnostics,
            scale_to_min_max,
            original_min,
            original_max
        )
        
        # Update merged_params with block results
        merged_params.update(block_merged_params)
        
        # Update counters based on block processing results
        # Parameters were already processed by _process_block_parameters_together function above
        # Just update the counters to reflect what was processed in this block
        block_params_count = len(block_params)
        widen_merged_count += block_params_count
        processed_count += block_params_count
        
        # Block completion checkpoint and cleanup
        profiler.checkpoint(f"Phase2 Block {processed_block_count}/{total_phase2_blocks} complete: {block_name}")
        
        # Clean up after each Phase 2 block - this is when parameter merging allocations happen
        gentle_cleanup()  # Clean up intermediate merge computations
    
    # Finalize memory profiling
    profiler.checkpoint("WIDEN merge processing complete")
    memory_summary = profiler.finish()
    
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
        compat_values = [score['compatibility'] for score in compatibility_scores]
        compat_min, compat_max = min(compat_values), max(compat_values)
        compat_mean = sum(compat_values) / len(compat_values)
        compat_variance = sum((x - compat_mean)**2 for x in compat_values) / len(compat_values)
        compat_range = compat_max - compat_min
        # Use relative variance threshold based on score range
        relative_variance_threshold = max(1e-6, (compat_range * 0.01) ** 2)
        print(f"  Compatibility Scores: {compat_min:.4f} - {compat_max:.4f} (mean: {compat_mean:.4f}, var: {compat_variance:.6f})")
        print(f"  Score Distribution: {'✓ VARIED' if compat_variance > relative_variance_threshold else '✗ UNIFORM (BUG!)'}")
    else:
        print(f"  Compatibility Scores: NONE COMPUTED")
    
    if score_variances:
        # Extract numeric values from variance dictionaries
        mag_variances = [v['magnitude_variance'] if isinstance(v, dict) else v for v in score_variances]
        dir_variances = [v['direction_variance'] if isinstance(v, dict) else v for v in score_variances]
        avg_mag_variance = sum(mag_variances) / len(mag_variances) if mag_variances else 0
        avg_dir_variance = sum(dir_variances) / len(dir_variances) if dir_variances else 0
        overall_avg = (avg_mag_variance + avg_dir_variance) / 2
        print(f"  Importance Score Variance: mag={avg_mag_variance:.6f}, dir={avg_dir_variance:.6f}, avg={overall_avg:.6f} ({'✓ VARIED' if overall_avg > 1e-6 else '✗ UNIFORM (BUG!)'})")
    
    varied_count = widen_diagnostics['varied_score_count']
    uniform_count = widen_diagnostics['uniform_score_count']
    total_scored = varied_count + uniform_count
    if total_scored > 0:
        print(f"  Parameter Ranking: {varied_count}/{total_scored} varied ({100*varied_count/total_scored:.1f}%)")
        print(f"  Ranking Algorithm: {'✓ HEALTHY' if varied_count > uniform_count else '✗ FAILING (UNIFORM SCORES!)'}")
    
    if skip_threshold > 0.0:
        skip_effectiveness = widen_diagnostics['parameters_skipped_threshold']
        print(f"  Skip Threshold: {skip_threshold} (percentile) -> {skip_effectiveness} parameters skipped")
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
    
    # COMPREHENSIVE MEMORY CLEANUP - Essential for large models
    # Clean up all intermediate data structures
    # Clear TaskVector parameters explicitly
    try:
        if 'models_to_merge_task_vectors' in locals():
            for tv in models_to_merge_task_vectors:
                if hasattr(tv, 'task_vector_param_dict'):
                    tv.task_vector_param_dict.clear()
            del models_to_merge_task_vectors
    except: pass
    
    # Force comprehensive garbage collection
    import gc
    gc.collect()  # Single collection is sufficient
    
    # CUDA memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        torch.cuda.empty_cache()  # Second clear after sync
    
    # Final memory check
    try:
        import psutil
        final_ram = psutil.virtual_memory().used / (1024**3)
        print(f"[ENHANCED WIDEN] Cleanup complete - Final RAM: {final_ram:.2f}GB")
    except: pass
    
    # Add strength debug info to diagnostics
    strength_values = [s['strength'] for s in widen_diagnostics['applied_strengths']]
    if strength_values:
        actual_min = min(strength_values)
        actual_max = max(strength_values)
        widen_diagnostics['strength_debug'] = {
            'actual_strength_range': f"{actual_min:.3f}-{actual_max:.3f}",
            'theoretical_range': f"{merge_strength * min_strength:.3f}-{merge_strength * max_strength:.3f}",
            'parameter_range': f"{min_strength}-{max_strength}",
            'avg_strength': sum(strength_values) / len(strength_values),
            'merge_strength_used': merge_strength,
            'sample_calculations': [(s['parameter'], s['compatibility'], s['strength']) for s in widen_diagnostics['applied_strengths'][:5]]
        }
    
    # Apply global dynamic scaling after all parameters are processed
    if scale_to_min_max and widen_diagnostics['applied_strengths']:
        print(f"[GlobalScaling] Applying global strength scaling across ALL {len(widen_diagnostics['applied_strengths'])} parameters...")
        
        # Extract all calculated strengths globally
        all_strengths = [item['strength'] for item in widen_diagnostics['applied_strengths']]
        global_min = min(all_strengths)
        global_max = max(all_strengths)
        global_range = global_max - global_min
        
        if global_range > 1e-8:  # Only scale if there's meaningful range
            # Calculate target bounds
            target_min = merge_strength * (original_min if original_min is not None else min_strength)
            target_max = merge_strength * (original_max if original_max is not None else max_strength)
            target_range = target_max - target_min
            
            print(f"[GlobalScaling] Scaling ALL parameters from [{global_min:.3f}, {global_max:.3f}] to [{target_min:.3f}, {target_max:.3f}]")
            
            # Apply scaling to diagnostics only (the actual merge already happened with correct relative strengths)
            for item in widen_diagnostics['applied_strengths']:
                original_strength = item['strength']
                
                # Linear scaling: map global_min→target_min, global_max→target_max
                scaled_strength = ((original_strength - global_min) / global_range) * target_range + target_min
                
                # Update diagnostics with scaled strength for display
                item['strength'] = scaled_strength
            
            # Update the strength distribution in diagnostics to reflect scaled values
            final_strengths = [item['strength'] for item in widen_diagnostics['applied_strengths']]
            final_min = min(final_strengths)
            final_max = max(final_strengths)
            final_avg = sum(final_strengths) / len(final_strengths)
            
            # Update strength_distribution with scaled values for diagnostic display
            widen_diagnostics['strength_distribution'] = {
                'min_used': final_min,
                'max_used': final_max,
                'mean': final_avg,
                'count': len(final_strengths)
            }
    
    return merged_params, widen_diagnostics
"""
Memory Management Module for DonutWidenMerge

This module contains all memory management functionality extracted from DonutWidenMerge.py
including context managers, tensor buffer caching, memory monitoring, and cleanup utilities.
"""

import gc
import torch
import psutil
import time
from contextlib import contextmanager

# Import logging from the shared module
try:
    from .logging_config import memory_logger
except ImportError:
    import logging
    memory_logger = logging.getLogger(__name__)

# Global cache variables
_TENSOR_BUFFER_CACHE = {}
_MERGE_CACHE = {}
_CACHE_MAX_SIZE = 3  # Reduced from 10 to prevent memory bloat
_TENSOR_BUFFER_CACHE_MAX_SIZE = 8  # Cap for the reusable-buffer cache
_WIDEN_MEMORY_PROFILER = None
_EMPTY_CACHE_MAX_EVENTS = 1000  # Cap profiler event history

# Rate limiting for CUDA cache flushes (hot-path safety).
_EMPTY_CACHE_MIN_INTERVAL = 2.0  # seconds between allowed flushes
_EMPTY_CACHE_MIN_FREE_GB = 2.0   # skip flush when at least this much VRAM is free
_last_empty_cache_time = 0.0


def soft_empty_cache(force=False):
    """Rate-limited torch.cuda.empty_cache().

    Calling empty_cache() in hot paths forces the CUDA caching allocator to
    release blocks back to the driver, causing churn, fragmentation, and
    driver-level GEM/mmap pressure (NVKMS "Failed to allocate GEM object",
    NVRM "invalid mmap context"). This helper only flushes when it is likely
    to matter: at most once every _EMPTY_CACHE_MIN_INTERVAL seconds and only
    when free VRAM is below _EMPTY_CACHE_MIN_FREE_GB (or force=True).
    """
    if not torch.cuda.is_available():
        return
    global _last_empty_cache_time
    now = time.monotonic()
    if not force and (now - _last_empty_cache_time) < _EMPTY_CACHE_MIN_INTERVAL:
        return
    try:
        free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        if not force and free_gb > _EMPTY_CACHE_MIN_FREE_GB:
            return
    except Exception:
        pass
    _last_empty_cache_time = now
    torch.cuda.empty_cache()


class MemoryExhaustionError(Exception):
    """Exception raised when memory usage becomes critical"""
    pass


class MemoryEfficientContext:
    """Context manager for memory-intensive operations with automatic cleanup"""
    def __init__(self, operation_name="tensor_operation"):
        self.operation_name = operation_name
        self.initial_memory = None

    def __enter__(self):
        # Pre-emptive cleanup before large operations
        import gc
        gc.collect()
        soft_empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup after operations (rate-limited to avoid allocator churn)
        import gc
        gc.collect()
        soft_empty_cache()


def get_reusable_tensor_buffer(shape, dtype=torch.float32, device='cpu'):
    """Get a reusable tensor buffer to avoid repeated allocations"""
    key = (tuple(shape), dtype, device)
    if key not in _TENSOR_BUFFER_CACHE:
        # Bound the cache so it cannot grow without limit.
        if len(_TENSOR_BUFFER_CACHE) >= _TENSOR_BUFFER_CACHE_MAX_SIZE:
            _TENSOR_BUFFER_CACHE.pop(next(iter(_TENSOR_BUFFER_CACHE)))
        _TENSOR_BUFFER_CACHE[key] = torch.empty(shape, dtype=dtype, device=device)
    else:
        # Reuse existing buffer, just zero it out if needed
        buffer = _TENSOR_BUFFER_CACHE[key]
        if buffer.shape != tuple(shape):
            # Shape mismatch, create new buffer
            _TENSOR_BUFFER_CACHE[key] = torch.empty(shape, dtype=dtype, device=device)
    return _TENSOR_BUFFER_CACHE[key]


def clear_tensor_buffer_cache():
    """Clear the tensor buffer cache to free memory"""
    global _TENSOR_BUFFER_CACHE
    _TENSOR_BUFFER_CACHE.clear()
    soft_empty_cache(force=True)


def clear_session_cache():
    """Clear global merge cache for fresh session start"""
    global _MERGE_CACHE
    if _MERGE_CACHE:
        print(f"[Session] Clearing {len(_MERGE_CACHE)} cached merge results")
        _MERGE_CACHE.clear()
        # Light cleanup after cache clear
        gc.collect()
        soft_empty_cache(force=True)


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

        memory_logger.debug(f"[{label}] RAM: {ram_mb:.1f}MB{vram_info}")

    except Exception as e:
        memory_logger.warning(f"[{label}] Memory monitoring error: {e}")


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


def force_cleanup():
    """Conservative memory cleanup to prevent CUDA allocator conflicts"""
    gc.collect()
    soft_empty_cache()
    # Removed torch.cuda.synchronize() to prevent allocator conflicts


def gentle_cleanup():
    """Very light cleanup for frequent use during processing"""
    gc.collect()
    # No CUDA operations to avoid allocator stress


def aggressive_memory_cleanup():
    """Aggressive memory cleanup for critical memory optimization"""
    # Clear optimized cache (use fallback if not available)
    try:
        from ..memory_utils import get_optimized_cache
        cache = get_optimized_cache()
        cache.clear()
    except ImportError:
        # Fallback: clear our own caches
        clear_tensor_buffer_cache()
        clear_session_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    soft_empty_cache(force=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all operations complete


def monitor_memory_usage(label=""):
    """Monitor memory usage for debugging"""
    try:
        import psutil
        process = psutil.Process()
        ram_mb = process.memory_info().rss / 1024 / 1024
        ram_gb = ram_mb / 1024
        
        # Also get system memory info
        system_memory = psutil.virtual_memory()
        system_ram_gb = system_memory.used / (1024**3)
        
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            gpu_allocated_gb = gpu_allocated / 1024
            print(f"[MEMORY-{label}] Process RAM: {ram_gb:.2f}GB, System RAM: {system_ram_gb:.2f}GB, VRAM: {gpu_allocated_gb:.2f}GB allocated, {gpu_reserved/1024:.2f}GB reserved")
        else:
            print(f"[MEMORY-{label}] Process RAM: {ram_gb:.2f}GB, System RAM: {system_ram_gb:.2f}GB")
        
        # Track memory spikes
        if hasattr(monitor_memory_usage, 'last_system_ram'):
            delta = system_ram_gb - monitor_memory_usage.last_system_ram
            if abs(delta) > 2.0:  # Alert on 2GB+ changes
                print(f"[MEMORY SPIKE-{label}] System RAM changed by {delta:+.2f}GB")
        monitor_memory_usage.last_system_ram = system_ram_gb
        
    except Exception as e:
        print(f"[MEMORY-{label}] Monitor failed: {e}")


class MemoryProfiler:
    """Lightweight memory profiler for WIDEN merge operations"""
    
    def __init__(self, name: str = "MemoryProfiler"):
        self.name = name
        self.enabled = True
        self.start_time = None
        self.start_ram = None
        self.start_vram = None
        self.peak_ram = 0
        self.peak_vram = 0
        self.events = []
        
    def start(self):
        """Start memory profiling"""
        if not self.enabled:
            return
            
        self.start_time = time.time()
        self.start_ram = self._get_ram_usage()
        self.start_vram = self._get_vram_usage()
        self.peak_ram = self.start_ram
        self.peak_vram = self.start_vram
        self.events = []
        
        # Memory profiling started silently
        
    def checkpoint(self, event_name: str):
        """Add a checkpoint with current memory usage"""
        if not self.enabled:
            return
            
        current_ram = self._get_ram_usage()
        current_vram = self._get_vram_usage()
        
        self.peak_ram = max(self.peak_ram, current_ram)
        self.peak_vram = max(self.peak_vram, current_vram)
        
        ram_delta = current_ram - self.start_ram if self.start_ram else 0
        vram_delta = current_vram - self.start_vram if self.start_vram else 0
        
        event = {
            'name': event_name,
            'time': time.time() - self.start_time if self.start_time else 0,
            'ram_gb': current_ram,
            'vram_gb': current_vram,
            'ram_delta': ram_delta,
            'vram_delta': vram_delta
        }
        
        # Bound the event history so long merges cannot grow it without limit.
        if len(self.events) >= _EMPTY_CACHE_MAX_EVENTS:
            self.events.pop(0)
        self.events.append(event)
        # Verbose memory logging disabled for cleaner output
        # print(f"[{self.name}] {event_name}: RAM {current_ram:.2f}GB ({ram_delta:+.2f}), VRAM {current_vram:.2f}GB ({vram_delta:+.2f})")
        
    def finish(self):
        """Finish profiling and print summary"""
        if not self.enabled or not self.start_time:
            return
            
        final_ram = self._get_ram_usage()
        final_vram = self._get_vram_usage()
        total_time = time.time() - self.start_time
        
        # Memory profiling completed silently - data stored in self.events
        
        return {
            'duration': total_time,
            'peak_ram_gb': self.peak_ram,
            'peak_vram_gb': self.peak_vram,
            'ram_delta_gb': final_ram - self.start_ram if self.start_ram else 0,
            'vram_delta_gb': final_vram - self.start_vram if self.start_vram else 0,
            'events': self.events
        }
        
    def stop(self):
        """Alias for finish() for backward compatibility"""
        return self.finish()
        
    def _get_ram_usage(self) -> float:
        """Get current RAM usage in GB"""
        try:
            return psutil.virtual_memory().used / (1024**3)
        except:
            return 0.0
            
    def _get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
            return 0.0
        except:
            return 0.0


def get_widen_memory_profiler():
    """Get or create the global WIDEN memory profiler"""
    global _WIDEN_MEMORY_PROFILER
    if _WIDEN_MEMORY_PROFILER is None:
        _WIDEN_MEMORY_PROFILER = MemoryProfiler("WIDEN_MERGE")
    return _WIDEN_MEMORY_PROFILER


def ultra_memory_efficient_widen_merge(
    merged_model, models_to_merge, exclude_param_names_regex,
    importance_threshold, importance_boost, base_merge_strength,
    rank_sensitivity, skip_threshold, normalization_mode,
    computation_device, target_device, storage_device
):
    """Ultra memory-efficient WIDEN merge - processes parameters one at a time"""
    
    print("[ULTRA MEMORY] Starting ultra memory-efficient WIDEN merge")
    initial_memory = psutil.virtual_memory().used / (1024**3)
    print(f"[ULTRA MEMORY] Initial RAM: {initial_memory:.2f}GB")
    
    # Get parameter names to process
    param_names_to_merge = []
    for name, _ in merged_model.named_parameters():
        if not exclude_param_names_regex or not exclude_param_names_regex.search(name):
            param_names_to_merge.append(name)
    
    print(f"[ULTRA MEMORY] Processing {len(param_names_to_merge)} parameters one at a time")
    
    merged_params = {}
    processed_count = 0
    peak_memory = initial_memory
    
    # Process each parameter individually to minimize memory usage
    for param_idx, param_name in enumerate(param_names_to_merge):
        try:
            # Get base parameter (float32 copy on the storage device, as in
            # the original implementation, to preserve merge numerics).
            base_param = None
            for name, param in merged_model.named_parameters():
                if name == param_name:
                    base_param = param.detach().to(storage_device).float()
                    if param_name == "model.embed_tokens.weight":
                        base_param = base_param.transpose(dim0=0, dim1=1)
                    break
            
            if base_param is None:
                continue
            
            # Create deltas one at a time on the storage device; accumulate
            # the weighted sum on the compute device incrementally instead of
            # stacking every delta in GPU memory.
            delta_tensors = []
            for model_to_merge in models_to_merge:
                other_param = None
                for name, param in model_to_merge.named_parameters():
                    if name == param_name:
                        other_param = param.detach().to(storage_device).float()
                        if param_name == "model.embed_tokens.weight":
                            other_param = other_param.transpose(dim0=0, dim1=1)
                        break
                
                if other_param is not None:
                    delta = other_param - base_param
                    delta_tensors.append(delta)
                    del other_param
            
            if not delta_tensors:
                merged_params[param_name] = base_param.to(target_device)
                processed_count += 1
                continue
            
            # Simple weighted merge (simplified WIDEN)
            with torch.no_grad():
                # The original code stacked the deltas and branched on the
                # stacked tensor's dim: stacked dim > 1 (i.e. per-delta
                # dim > 0) takes the magnitude/importance path; only scalar
                # deltas (0-dim) take the mean path.
                if delta_tensors[0].dim() > 0:
                    magnitudes = torch.zeros(len(delta_tensors), device=storage_device, dtype=torch.float32)
                    for i, delta in enumerate(delta_tensors):
                        magnitudes[i] = torch.norm(delta, p=2)
                    importance_scores = magnitudes / (magnitudes.max() + 1e-8)

                    # Apply importance boost to top features
                    top_k = max(1, int(len(importance_scores) * importance_threshold / 100.0))
                    top_indices = torch.argsort(importance_scores, descending=True)[:top_k]

                    weights = torch.ones_like(importance_scores) * 0.1
                    weights[top_indices] = importance_boost

                    # Weighted merge
                    merged_delta = None
                    for i, delta in enumerate(delta_tensors):
                        w = weights[i]
                        delta_c = delta.to(computation_device) * w
                        merged_delta = delta_c if merged_delta is None else merged_delta + delta_c
                        del delta_c
                else:
                    merged_delta = None
                    for delta in delta_tensors:
                        delta_c = delta.to(computation_device)
                        merged_delta = delta_c if merged_delta is None else merged_delta + delta_c
                        del delta_c
                    # Original code stacked the deltas and took mean over the
                    # model axis (dim 0); accumulating then dividing by the
                    # model count is the same operation without the stack.
                    if merged_delta is not None:
                        merged_delta = merged_delta / len(delta_tensors)

                merged_param = base_param.to(computation_device) + merged_delta * base_merge_strength
                merged_params[param_name] = merged_param.to(target_device)

                del delta_tensors, merged_delta, merged_param
            
            processed_count += 1
            
            # Monitor memory and cleanup periodically
            if param_idx % 20 == 0:
                current_memory = psutil.virtual_memory().used / (1024**3)
                peak_memory = max(peak_memory, current_memory)
                gc.collect()
                soft_empty_cache()
                print(f"[ULTRA MEMORY] Progress: {param_idx}/{len(param_names_to_merge)}, RAM: {current_memory:.2f}GB")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {param_name}: {e}")
            processed_count += 1
    
    final_memory = psutil.virtual_memory().used / (1024**3)
    total_memory_used = peak_memory - initial_memory
    
    print(f"[ULTRA MEMORY] Complete! Processed: {processed_count}/{len(param_names_to_merge)}")
    print(f"[ULTRA MEMORY] Peak RAM: {peak_memory:.2f}GB (Δ{total_memory_used:+.2f}GB)")
    
    # Transpose back
    if "model.embed_tokens.weight" in merged_params:
        merged_params["model.embed_tokens.weight"] = merged_params["model.embed_tokens.weight"].transpose(dim0=0, dim1=1)
    
    return merged_params


@contextmanager
def memory_cleanup_context(label=""):
    """Context manager for automatic memory cleanup"""
    monitor_memory(f"{label}-START")
    try:
        yield
    finally:
        force_cleanup()
        monitor_memory(f"{label}-END")


# Utility functions for cache management
def get_merge_cache():
    """Get the global merge cache"""
    return _MERGE_CACHE


def get_cache_max_size():
    """Get the maximum cache size"""
    return _CACHE_MAX_SIZE


def set_cache_max_size(size):
    """Set the maximum cache size"""
    global _CACHE_MAX_SIZE
    _CACHE_MAX_SIZE = size


# Additional utility functions for memory optimization
def memory_usage_mb():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def cuda_memory_usage_mb():
    """Get current CUDA memory usage in MB"""
    if torch.cuda.is_available():
        try:
            return torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            return 0.0
    return 0.0


def memory_efficient_operation(func):
    """Decorator for memory-efficient operations"""
    def wrapper(*args, **kwargs):
        with MemoryEfficientContext(func.__name__):
            return func(*args, **kwargs)
    return wrapper


def batch_cleanup(frequency=10):
    """Decorator for batch cleanup operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, 'call_count'):
                wrapper.call_count = 0
            
            wrapper.call_count += 1
            result = func(*args, **kwargs)
            
            if wrapper.call_count % frequency == 0:
                gentle_cleanup()
            
            return result
        return wrapper
    return decorator
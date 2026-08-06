"""
Memory monitoring and optimization utilities for WIDEN merge
"""

import torch
import psutil
import gc
import time
from contextlib import contextmanager
from typing import Dict, Optional
import threading
import warnings

# Rate limiting for CUDA cache flushes (hot-path safety).
# Calling empty_cache() in hot paths forces the caching allocator to release
# blocks to the driver, causing churn and driver-level GEM/mmap pressure.
_EMPTY_CACHE_MIN_INTERVAL = 2.0  # seconds between allowed flushes
_EMPTY_CACHE_MIN_FREE_GB = 2.0   # skip flush when at least this much VRAM is free
_last_empty_cache_time = 0.0


def _soft_empty_cache(force=False):
    """Rate-limited torch.cuda.empty_cache() for hot paths."""
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
        
        ram_delta = current_ram - self.start_ram
        vram_delta = current_vram - self.start_vram
        
        event = {
            'name': event_name,
            'time': time.time() - self.start_time if self.start_time else 0,
            'ram_gb': current_ram,
            'vram_gb': current_vram,
            'ram_delta': ram_delta,
            'vram_delta': vram_delta
        }
        
        # Bound the event history so long merges cannot grow it without limit.
        if len(self.events) >= 1000:
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
            'ram_delta_gb': final_ram - self.start_ram,
            'vram_delta_gb': final_vram - self.start_vram,
            'events': self.events
        }
        
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

@contextmanager
def memory_profiler(name: str = "Operation"):
    """Context manager for memory profiling"""
    profiler = MemoryProfiler(name)
    profiler.start()
    try:
        yield profiler
    finally:
        profiler.finish()

def optimize_tensor_operations(func):
    """Decorator to optimize tensor operations for memory efficiency"""
    def wrapper(*args, **kwargs):
        # Force garbage collection before operation
        gc.collect()
        _soft_empty_cache()
            
        # Run operation with memory optimization
        with torch.no_grad():
            result = func(*args, **kwargs)
            
        # Cleanup after operation (rate-limited to avoid allocator churn)
        gc.collect()
        _soft_empty_cache()
            
        return result
    return wrapper

def smart_device_management(tensor: torch.Tensor, target_device: Optional[torch.device] = None) -> torch.Tensor:
    """Smart device management to minimize VRAM usage"""
    if target_device is None:
        # Use CPU for large tensors, GPU for small ones
        if tensor.numel() > 1_000_000:  # 1M elements threshold
            target_device = torch.device('cpu')
        else:
            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move tensor efficiently
    if tensor.device != target_device:
        tensor = tensor.to(target_device, non_blocking=True)
    
    return tensor

def batch_process_parameters(param_dict: Dict[str, torch.Tensor], 
                           batch_size: int = 50,
                           process_fn = None) -> Dict[str, torch.Tensor]:
    """Process parameters in batches to reduce memory usage"""
    if process_fn is None:
        return param_dict
        
    result_dict = {}
    param_items = list(param_dict.items())
    
    # Processing parameters in batches
    
    for i in range(0, len(param_items), batch_size):
        batch_items = param_items[i:i + batch_size]
        
        # Process batch
        batch_dict = {name: tensor for name, tensor in batch_items}
        batch_result = process_fn(batch_dict)
        
        # Update result
        result_dict.update(batch_result)
        
        # Cleanup between batches
        del batch_dict
        gc.collect()
        _soft_empty_cache()
            
        if i % (batch_size * 5) == 0:  # Progress every 5 batches
            pass  # Progress tracking disabled for cleaner output
    
    return result_dict

def memory_efficient_tensor_ops(tensor_a: torch.Tensor, 
                               tensor_b: torch.Tensor,
                               operation: str = "subtract") -> torch.Tensor:
    """Memory-efficient tensor operations with automatic device management"""
    # Ensure tensors are on optimal device
    if tensor_a.numel() > 10_000_000:  # Large tensors -> CPU
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move tensors to optimal device
    tensor_a = tensor_a.to(device, non_blocking=True)
    tensor_b = tensor_b.to(device, non_blocking=True)
    
    # Perform operation in-place when possible
    with torch.no_grad():
        if operation == "subtract":
            if tensor_a.shape == tensor_b.shape:
                result = tensor_a.sub_(tensor_b)  # In-place subtraction
            else:
                result = tensor_a - tensor_b
        elif operation == "add":
            if tensor_a.shape == tensor_b.shape:
                result = tensor_a.add_(tensor_b)  # In-place addition
            else:
                result = tensor_a + tensor_b
        elif operation == "multiply":
            result = tensor_a * tensor_b
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    return result

class MemoryOptimizedCache:
    """Memory-optimized cache with automatic cleanup"""
    
    def __init__(self, max_size: int = 3, max_memory_gb: float = 8.0):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.max_memory_gb = max_memory_gb
        
    def get(self, key: str):
        """Get item from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
        
    def put(self, key: str, value):
        """Put item in cache with memory management"""
        # Check memory usage before adding
        current_memory = self._get_cache_memory_usage()
        if current_memory > self.max_memory_gb:
            self._cleanup_by_memory()
            
        # Check size limit
        if len(self.cache) >= self.max_size:
            self._cleanup_by_size()
            
        self.cache[key] = value
        self.access_times[key] = time.time()
        
    def _get_cache_memory_usage(self) -> float:
        """Estimate cache memory usage in GB"""
        try:
            total_size = 0
            for value in self.cache.values():
                if hasattr(value, 'element_size') and hasattr(value, 'nelement'):
                    total_size += value.element_size() * value.nelement()
                elif hasattr(value, '__sizeof__'):
                    total_size += value.__sizeof__()
            return total_size / (1024**3)
        except:
            return 0.0
            
    def _cleanup_by_memory(self):
        """Cleanup cache based on memory usage"""
        # Cache cleanup in progress
        # Remove oldest half of entries
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        keys_to_remove = sorted_keys[:len(sorted_keys)//2]
        
        for key in keys_to_remove:
            del self.cache[key]
            del self.access_times[key]
            
        gc.collect()
        
    def _cleanup_by_size(self):
        """Cleanup cache based on size limit"""
        # Cache size limit reached
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_times.clear()
        gc.collect()

# Global optimized cache instance
_OPTIMIZED_CACHE = MemoryOptimizedCache()

def get_optimized_cache():
    """Get the global optimized cache instance"""
    return _OPTIMIZED_CACHE
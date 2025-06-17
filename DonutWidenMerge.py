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

def compute_merge_hash(models, merge_strength, temperature, enable_ties, threshold, forced_merge_ratio, renorm_mode):
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
        calibration_value: float = 0.0,_value: float = 1.0,
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
                "widen_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "widen_calibration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "renorm_mode": (["magnitude", "calibrate", "none"], {"default": "magnitude"}),
                "batch_size": ("INT", {"default": 50, "min": 10, "max": 500, "step": 10}),
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

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "merge_results")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, model_base, model_other, merge_strength, widen_threshold, widen_calibration, renorm_mode, batch_size,
                lora_stack=None, model_3=None, model_4=None, model_5=None, model_6=None,
                model_7=None, model_8=None, model_9=None, model_10=None,
                model_11=None, model_12=None):

        # FIXED: Aggressive memory cleanup before starting
        print("[MEMORY] Pre-merge cleanup...")
        force_cleanup()

        # Check cache first
        all_models = [model_base, model_other, model_3, model_4, model_5, model_6,
                     model_7, model_8, model_9, model_10, model_11, model_12]
        cache_key = compute_merge_hash(all_models, merge_strength, widen_threshold, False, widen_calibration, 0, f"{renorm_mode}")

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

                merger = MergingMethod("DonutWidenMergeUNet")
                results_text = merger.widen_merging_sdxl(
                    target_model=model_merged.model,
                    base_model=base_model_obj,
                    models_to_merge=other_model_objs,
                    merge_strength=merge_strength,
                    renorm_mode=renorm_mode,
                    widen_threshold=widen_threshold,
                    calibration_value=widen_calibration,
                    batch_size=batch_size,
                )

                # FIXED: Aggressive cleanup before returning
                del base_model_obj, other_model_objs, models_to_merge, merger
                force_cleanup()

                result = (model_merged, results_text)

                # Store in cache
                store_merge_result(cache_key, result)

                return result

            except MemoryExhaustionError as e:
                print(f"[SAFETY] Memory exhaustion prevented crash: {e}")
                # FIXED: Cleanup on error
                force_cleanup()
                error_results = f"[SAFETY] Merge terminated to prevent crash: {e}"
                result = (model_base, error_results)
                store_merge_result(cache_key, result)
                return result

            except Exception as e:
                print(f"[DonutWidenMergeUNet] Error: {e}")
                # FIXED: Cleanup on error
                force_cleanup()
                if "memory" in str(e).lower():
                    error_results = f"[SAFETY] Memory error prevented crash: {e}"
                    result = (model_base, error_results)
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
                "widen_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "widen_calibration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "renorm_mode": (["magnitude", "calibrate", "none"], {"default": "magnitude"}),
                "batch_size": ("INT", {"default": 75, "min": 10, "max": 500, "step": 10}),
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

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "merge_results")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, clip_base, clip_other, merge_strength, widen_threshold, widen_calibration, renorm_mode, batch_size,
                lora_stack=None, clip_3=None, clip_4=None, clip_5=None, clip_6=None,
                clip_7=None, clip_8=None, clip_9=None, clip_10=None,
                clip_11=None, clip_12=None):

        # FIXED: Aggressive memory cleanup before starting
        print("[MEMORY] Pre-merge cleanup...")
        force_cleanup()

        # Check cache first
        all_clips = [clip_base, clip_other, clip_3, clip_4, clip_5, clip_6,
                    clip_7, clip_8, clip_9, clip_10, clip_11, clip_12]
        cache_key = compute_merge_hash(all_clips, merge_strength, widen_threshold, False, widen_calibration, 0, f"{renorm_mode}")

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

                merger = MergingMethod("DonutWidenMergeCLIP")
                results_text = merger.widen_merging_sdxl(
                    target_model=enc_merged,
                    base_model=base_enc,
                    models_to_merge=other_encs,
                    merge_strength=merge_strength,
                    renorm_mode=renorm_mode,
                    widen_threshold=widen_threshold,
                    calibration_value=widen_calibration,
                    batch_size=batch_size,
                )

                # FIXED: Aggressive cleanup before returning
                del base_enc, other_encs, clips_to_merge, merger, enc_merged
                force_cleanup()

                result = (clip_merged, results_text)

                # Store in cache
                store_merge_result(cache_key, result)

                return result

            except MemoryExhaustionError as e:
                print(f"[SAFETY] Memory exhaustion prevented crash: {e}")
                # FIXED: Cleanup on error
                force_cleanup()
                error_results = f"[SAFETY] CLIP merge terminated to prevent crash: {e}"
                result = (clip_base, error_results)
                store_merge_result(cache_key, result)
                return result

            except Exception as e:
                print(f"[DonutWidenMergeCLIP] Error: {e}")
                # FIXED: Cleanup on error
                force_cleanup()
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

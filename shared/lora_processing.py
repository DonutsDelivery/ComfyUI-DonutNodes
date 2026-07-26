"""
LoRA Processing Module for WIDEN Merge

This module contains memory-efficient LoRA delta storage and processing classes
extracted from DonutWidenMerge.py. These classes handle the conversion of LoRA
stacks into delta representations for efficient WIDEN merging operations.

Classes:
    LoRADelta: Memory-efficient LoRA delta storage for WIDEN merging
    LoRAStackProcessor: Process LoRA stacks efficiently for WIDEN merging

The LoRADelta class computes and stores only the parameter differences between
base and LoRA-enhanced models, significantly reducing memory usage while
maintaining full model functionality through delta application.

The LoRAStackProcessor handles the complex process of applying LoRA stacks
to base models, supporting both UNet and CLIP processing with proper block
weight handling and memory management.
"""

import torch
import gc
import re
from collections import defaultdict

# Import shared modules
try:
    # Try relative imports first (when running as part of ComfyUI)
    from .memory_management import MemoryEfficientContext
except ImportError:
    # Fallback to absolute imports (when running standalone)
    from memory_management import MemoryEfficientContext


def _named_parameter_module(value):
    """Resolve a ComfyUI wrapper to the module that owns its parameters."""
    pending = [value]
    seen = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if callable(getattr(candidate, "named_parameters", None)):
            return candidate
        for attribute in ("model", "cond_stage_model", "clip"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                pending.append(nested)
    raise TypeError(f"{type(value).__name__} does not expose a parameter module")


def _default_block_vector(lora):
    """Choose an all-enabled vector that matches the LoRA's DiT layout."""
    layer_numbers = set()
    block_numbers = set()
    for key in lora:
        if "layers." in key:
            match = re.search(r"layers\.(\d+)", key)
            if match:
                layer_numbers.add(int(match.group(1)))
        elif "blocks." in key:
            match = re.search(r"(?<![a-z_])blocks\.(\d+)", key)
            if match:
                block_numbers.add(int(match.group(1)))

    numbers = block_numbers or layer_numbers
    if numbers:
        return ",".join(["1"] * (max(numbers) + 2))  # non-block bucket + blocks
    return ",".join(["1"] * 13)  # non-block bucket + SDXL's 12 blocks


class LoRADelta:
    """Memory-efficient LoRA delta storage for WIDEN merging"""
    def __init__(self, base_model, lora_model, lora_name="unknown"):
        self.base_model = _named_parameter_module(base_model)  # Reference to base parameters (no copy)
        self.lora_name = lora_name
        self.deltas = {}  # Only store the differences
        self.param_metadata = {}
        
        print(f"[LoRADelta] Computing deltas for {lora_name}...")
        self._compute_deltas(lora_model)
        
    def _compute_deltas(self, lora_model):
        """Compute only the parameter differences between base and LoRA-enhanced model"""
        try:
            base_params = dict(self.base_model.named_parameters())
            lora_params = dict(_named_parameter_module(lora_model).named_parameters())
            
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
                try:
                    from ..lora_block_weight import LoraLoaderBlockWeight
                except ImportError:
                    # Fallback for standalone execution
                    class LoraLoaderBlockWeight:
                        def __init__(self): pass
                        def load_lora(self, *args, **kwargs): return None, None
                
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
                vector = block_vector if block_vector else _default_block_vector(lora)
                
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
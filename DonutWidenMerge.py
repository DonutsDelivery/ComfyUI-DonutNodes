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

def monitor_memory(label=""):
    """Print current memory usage"""
    try:
        process = psutil.Process()
        ram_mb = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"[MEMORY-{label}] RAM: {ram_mb:.1f}MB, VRAM: {vram_mb:.1f}MB")
        else:
            print(f"[MEMORY-{label}] RAM: {ram_mb:.1f}MB")
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

class TrueMemoryEfficientModel:
    """Model wrapper that loads parameters on-demand to truly save memory"""
    def __init__(self, model):
        self.model = model
        self._param_names = [name for name, _ in model.named_parameters()]

    def get_parameter(self, name):
        """Load a single parameter on demand"""
        for param_name, param in self.model.named_parameters():
            if param_name == name:
                return param.detach().clone().cpu().float()
        return None

    def get_parameters_batch(self, names):
        """Load only the requested parameters"""
        params = {}
        state_dict = self.model.state_dict()
        for name in names:
            if name in state_dict:
                params[name] = state_dict[name].detach().clone().cpu().float()
        del state_dict
        return params

    def named_parameters(self):
        """Iterator over parameter names (without loading tensors)"""
        for name in self._param_names:
            yield name, None

class MergingMethod:
    def __init__(self, merging_method_name: str):
        self.method = merging_method_name

    def widen_merging_zero_accumulation(
        self,
        target_model,
        base_model,
        models_to_merge,
        merge_strength: float = 1.0,
        temperature: float = 0.8,
        enable_ties: bool = True,
        threshold: float = 0.00005,
        batch_size: int = 20,
        forced_merge_ratio: float = 0.1,
    ):
        """Zero accumulation merge - writes directly to target model to prevent any memory buildup"""

        algorithm_name = "TIES+WIDEN" if enable_ties else "WIDEN"
        results_text = f"[{self.method}] Starting zero-accumulation merge with {algorithm_name}\n"

        # Initial safety check
        safe, ram_percent, available_gb = check_memory_safety()
        if not safe:
            error_msg = f"[SAFETY] Cannot start - memory critical! RAM: {ram_percent*100:.1f}%, Available: {available_gb:.1f}GB"
            print(error_msg)
            raise MemoryExhaustionError(error_msg)

        print(f"[{self.method}] Initial memory check: {ram_percent*100:.1f}% used, {available_gb:.1f}GB available")

        # Wrap models for on-demand loading
        base_wrapper = TrueMemoryEfficientModel(base_model)
        model_wrappers = [TrueMemoryEfficientModel(model) for model in models_to_merge]

        # Get target model's state dict (this is where we'll write directly)
        target_state_dict = target_model.state_dict()

        # Get common parameter names WITHOUT loading any tensors
        print(f"[{self.method}] Finding common parameters...")
        common_names = set(base_wrapper._param_names)
        for wrapper in model_wrappers:
            common_names &= set(wrapper._param_names)
        common_names = list(common_names)

        print(f"[{self.method}] Found {len(common_names)} common parameters across {len(models_to_merge) + 1} models")

        # Adjust batch size based on memory - more aggressive now that we have zero accumulation
        if available_gb < 4:
            batch_size = min(batch_size, 10)
            print(f"[MEMORY] Critical memory - batch size: {batch_size}")
        elif available_gb < 8:
            batch_size = min(batch_size, 25)
            print(f"[MEMORY] Low memory - batch size: {batch_size}")
        elif available_gb > 30:
            batch_size = max(batch_size, 500)  # Very large batches for high memory systems
            print(f"[MEMORY] Excellent memory - batch size: {batch_size}")
        elif available_gb > 20:
            batch_size = max(batch_size, 200)  # Large batches when we have memory
            print(f"[MEMORY] Very good memory - batch size: {batch_size}")
        elif available_gb > 15:
            batch_size = max(batch_size, 100)
            print(f"[MEMORY] Good memory - batch size: {batch_size}")
        else:
            batch_size = max(batch_size, 50)
            print(f"[MEMORY] Adequate memory - batch size: {batch_size}")

        # Calculate importance scores with minimal memory footprint
        importance_scores = {}
        forced_merge_set = set()

        if forced_merge_ratio > 0:
            print(f"[{self.method}] Calculating importance scores with minimal memory...")
            sample_size = min(50, len(common_names))

            for i, name in enumerate(common_names[:sample_size]):
                if i % 10 == 0:
                    print(f"  Sampling {i+1}/{sample_size}...")

                try:
                    base_param = base_wrapper.get_parameter(name)
                    if base_param is None:
                        importance_scores[name] = 0.0
                        continue

                    total_diff = 0.0
                    for wrapper in model_wrappers:
                        other_param = wrapper.get_parameter(name)
                        if other_param is not None and base_param.shape == other_param.shape:
                            diff = torch.norm(other_param - base_param).item()
                            total_diff += diff
                        del other_param

                    importance_scores[name] = total_diff
                    del base_param

                    if i % 5 == 0:
                        force_cleanup()

                except Exception:
                    importance_scores[name] = 0.0

            # Determine forced merge parameters
            if importance_scores:
                sorted_names = sorted(importance_scores.items(), key=lambda x: -x[1])
                min_merge_count = max(1, int(len(common_names) * forced_merge_ratio))
                forced_merge_set = set(n for n, _ in sorted_names[:min_merge_count])
                print(f"[{self.method}] Will force merge {len(forced_merge_set)} parameters ({forced_merge_ratio*100:.1f}%)")

        # Zero-accumulation processing - update target model in-place
        fell_back = 0
        actually_merged = 0
        total_batches = (len(common_names) + batch_size - 1) // batch_size

        print(f"[{self.method}] Zero-accumulation processing {len(common_names)} parameters in {total_batches} batches")

        for batch_start in range(0, len(common_names), batch_size):
            batch_end = min(batch_start + batch_size, len(common_names))
            batch_names = common_names[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1

            print(f"[PROGRESS] Batch {batch_num}/{total_batches} ({len(batch_names)} params) - {batch_num/total_batches*100:.1f}% complete")

            # Memory check every batch
            safe, ram_percent, available_gb = check_memory_safety()
            print(f"  Memory: {ram_percent*100:.1f}% used, {available_gb:.1f}GB available")

            if not safe:
                print(f"[EMERGENCY] Memory critical at batch {batch_num}! Stopping safely...")
                partial_results = f"""
[PARTIAL] Emergency stop at batch {batch_num}/{total_batches}:
  - Processed: {(batch_num-1)*batch_size}/{len(common_names)} parameters ({((batch_num-1)*batch_size)/len(common_names)*100:.1f}%)"""
                return results_text + partial_results

            try:
                print(f"  Loading batch parameters...")
                batch_base_params = base_wrapper.get_parameters_batch(batch_names)
                batch_other_params = []

                for wrapper in model_wrappers:
                    other_batch = wrapper.get_parameters_batch(batch_names)
                    batch_other_params.append(other_batch)

                print(f"  Loaded {len(batch_base_params)} base params and {len(batch_other_params)} model sets")

                # Process each parameter individually and update target model immediately
                for name in batch_names:
                    if name not in batch_base_params:
                        fell_back += 1
                        continue

                    try:
                        base_param = batch_base_params[name]

                        # Compute task vectors for this single parameter
                        task_vectors = []
                        for other_batch in batch_other_params:
                            if name in other_batch:
                                task_vectors.append(other_batch[name] - base_param)

                        if not task_vectors:
                            fell_back += 1
                            continue

                        # Apply TIES to this single parameter
                        if enable_ties and len(task_vectors) > 0:
                            # Trim small values
                            trimmed_vectors = []
                            for tv in task_vectors:
                                mask = torch.abs(tv) >= threshold
                                trimmed_vectors.append(tv * mask.float())

                            # Sign election if multiple models
                            if len(trimmed_vectors) > 1:
                                deltas = torch.stack(trimmed_vectors)
                                signs = torch.sign(deltas)
                                sign_sum = signs.sum(dim=0)
                                elected_sign = torch.sign(sign_sum)

                                mask = torch.where(elected_sign == 0,
                                                 torch.ones_like(signs),
                                                 (signs * elected_sign.unsqueeze(0)) >= 0)

                                final_delta = (deltas * mask.float()).sum(0)
                            else:
                                final_delta = trimmed_vectors[0]
                        else:
                            # No TIES, just stack
                            final_delta = torch.stack(task_vectors).sum(0)

                        # Decide if we should merge
                        should_merge = False
                        merge_type = "SKIP"

                        if name in forced_merge_set:
                            should_merge = True
                            merge_type = "FORCED"
                        else:
                            delta_magnitude = torch.norm(final_delta).item()
                            if delta_magnitude > threshold:
                                should_merge = True
                                merge_type = "THRESHOLD"

                        if should_merge:
                            # Apply merge
                            strength = merge_strength if merge_type == "FORCED" else merge_strength * 0.5
                            strength *= (1.0 / temperature)

                            merged = base_param + final_delta * strength

                            # CRITICAL: Update target model directly, no intermediate storage!
                            target_state_dict[name].copy_(merged.to(target_state_dict[name].device))
                            actually_merged += 1

                            if actually_merged <= 3:
                                diff_magnitude = torch.norm(merged - base_param).item()
                                print(f"    [{merge_type}] {name}: strength={strength:.2f}, diff={diff_magnitude:.6f}")

                        # Clean up immediately after each parameter
                        del base_param, task_vectors
                        if 'trimmed_vectors' in locals():
                            del trimmed_vectors
                        if 'final_delta' in locals():
                            del final_delta

                    except Exception as e:
                        print(f"    [WARNING] Failed to process {name}: {e}")
                        fell_back += 1

                # CRITICAL: Immediately delete all batch data to free memory
                del batch_base_params
                del batch_other_params

                # Force aggressive cleanup after each batch
                force_cleanup()

                print(f"  Batch {batch_num} complete - memory cleaned")

            except Exception as e:
                print(f"[ERROR] Batch {batch_num} failed: {e}")
                if "memory" in str(e).lower():
                    print("[EMERGENCY] Memory error - merge partially complete")
                force_cleanup()
                continue

        # Final results
        total = len(common_names)
        skipped_small = total - actually_merged - fell_back

        merge_results = f"""
[RESULTS] Zero-accumulation merge complete:
  - Successfully merged: {actually_merged}/{total} parameters
  - Failed: {fell_back}
  - Skipped (below threshold): {skipped_small}
  - Forced merges: {len(forced_merge_set)}"""

        print(merge_results)
        results_text += merge_results

        force_cleanup()
        return results_text

    # Keep the old method name for backward compatibility
    def widen_merging_true_batched(self, base_model, models_to_merge, **kwargs):
        """Backward compatibility - redirects to zero accumulation method"""
        print("[DEPRECATED] widen_merging_true_batched is deprecated, using widen_merging_zero_accumulation")

        temp_model = base_model
        results_text = self.widen_merging_zero_accumulation(
            target_model=temp_model,
            base_model=base_model,
            models_to_merge=models_to_merge,
            **kwargs
        )

        return temp_model.state_dict(), results_text

    def widen_merging_streaming(self, base_model, models_to_merge, target_state_dict, **kwargs):
        """Streaming method - now redirects to zero accumulation"""
        print("[INFO] Using zero-accumulation approach for better memory efficiency")

        class TempModel:
            def __init__(self, state_dict):
                self._state_dict = state_dict
            def state_dict(self):
                return self._state_dict

        temp_model = TempModel(target_state_dict)
        results_text = self.widen_merging_zero_accumulation(
            target_model=temp_model,
            base_model=base_model,
            models_to_merge=models_to_merge,
            **kwargs
        )

        return target_state_dict, results_text


class DonutWidenMergeUNet:
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_base": ("MODEL",),
                "model_other": ("MODEL",),
                "merge_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 10.0, "step": 0.1}),
                "enable_ties": ("BOOLEAN", {"default": True}),
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
                "threshold": ("FLOAT", {"default": 0.00005, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "batch_size": ("INT", {"default": 75, "min": 10, "max": 9999, "step": 10}),
                "forced_merge_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "merge_results")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, model_base, model_other, merge_strength, temperature, enable_ties,
                model_3=None, model_4=None, model_5=None, model_6=None,
                model_7=None, model_8=None, model_9=None, model_10=None,
                model_11=None, model_12=None,
                threshold=0.00005, batch_size=75, forced_merge_ratio=0.1):

        with memory_cleanup_context("DonutWidenMergeUNet"):
            import copy

            all_models = [model_other, model_3, model_4, model_5, model_6,
                         model_7, model_8, model_9, model_10, model_11, model_12]
            models_to_merge = [m for m in all_models
                             if m is not None and not getattr(m, "_is_filler", False)]

            print(f"[DonutWidenMergeUNet] Zero-accumulation merging {len(models_to_merge)} models")

            try:
                base_model_obj = model_base.model
                other_model_objs = [model.model for model in models_to_merge]

                model_merged = copy.deepcopy(model_base)

                merger = MergingMethod("DonutWidenMergeUNet")
                results_text = merger.widen_merging_zero_accumulation(
                    target_model=model_merged.model,
                    base_model=base_model_obj,
                    models_to_merge=other_model_objs,
                    merge_strength=merge_strength,
                    temperature=temperature,
                    enable_ties=enable_ties,
                    threshold=threshold,
                    batch_size=batch_size,
                    forced_merge_ratio=forced_merge_ratio,
                )

                force_cleanup()
                return (model_merged, results_text)

            except MemoryExhaustionError as e:
                print(f"[SAFETY] Memory exhaustion prevented crash: {e}")
                error_results = f"[SAFETY] Merge terminated to prevent crash: {e}"
                return (model_base, error_results)

            except Exception as e:
                print(f"[DonutWidenMergeUNet] Error: {e}")
                if "memory" in str(e).lower():
                    error_results = f"[SAFETY] Memory error prevented crash: {e}"
                    return (model_base, error_results)
                else:
                    raise


class DonutWidenMergeCLIP:
    class_type = "CLIP"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_base": ("CLIP",),
                "clip_other": ("CLIP",),
                "merge_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 10.0, "step": 0.1}),
                "enable_ties": ("BOOLEAN", {"default": True}),
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
                "threshold": ("FLOAT", {"default": 0.00005, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "batch_size": ("INT", {"default": 100, "min": 10, "max": 9999, "step": 10}),
                "forced_merge_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "merge_results")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, clip_base, clip_other, merge_strength, temperature, enable_ties,
                clip_3=None, clip_4=None, clip_5=None, clip_6=None,
                clip_7=None, clip_8=None, clip_9=None, clip_10=None,
                clip_11=None, clip_12=None,
                threshold=0.00005, batch_size=100, forced_merge_ratio=0.1):

        with memory_cleanup_context("DonutWidenMergeCLIP"):
            import copy

            all_clips = [clip_other, clip_3, clip_4, clip_5, clip_6,
                        clip_7, clip_8, clip_9, clip_10, clip_11, clip_12]
            clips_to_merge = [c for c in all_clips
                            if c is not None and not getattr(c, "_is_filler", False)]

            print(f"[DonutWidenMergeCLIP] Zero-accumulation merging {len(clips_to_merge)} CLIP models")

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
                results_text = merger.widen_merging_zero_accumulation(
                    target_model=enc_merged,
                    base_model=base_enc,
                    models_to_merge=other_encs,
                    merge_strength=merge_strength,
                    temperature=temperature,
                    enable_ties=enable_ties,
                    threshold=threshold,
                    batch_size=batch_size,
                    forced_merge_ratio=forced_merge_ratio,
                )

                force_cleanup()
                return (clip_merged, results_text)

            except MemoryExhaustionError as e:
                print(f"[SAFETY] Memory exhaustion prevented crash: {e}")
                error_results = f"[SAFETY] CLIP merge terminated to prevent crash: {e}"
                return (clip_base, error_results)

            except Exception as e:
                print(f"[DonutWidenMergeCLIP] Error: {e}")
                if "memory" in str(e).lower():
                    error_results = f"[SAFETY] CLIP memory error prevented crash: {e}"
                    return (clip_base, error_results)
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
    pass

import atexit
def cleanup_on_exit():
    """Cleanup on exit"""
    try:
        pass
    except Exception:
        pass

atexit.register(cleanup_on_exit)

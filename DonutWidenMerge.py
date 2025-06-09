import torch
import torch.nn as nn
from tqdm import tqdm

# use package-relative path for ComfyUI
from .utils.sdxl_safetensors import ensure_same_device

class TaskVector:
    def __init__(self, base_model, finetuned_model):
        base_state = base_model.state_dict()
        finetuned_state = finetuned_model.state_dict()

        self.task_vector_param_dict = {}
        for k in base_state:
            if k in finetuned_state:
                base_param = base_state[k].detach().cpu()
                finetuned_param = finetuned_state[k].detach().cpu()
                self.task_vector_param_dict[k] = finetuned_param - base_param

        print(f"[TaskVector] param count: {len(self.task_vector_param_dict)}")

class MergingMethod:
    def __init__(self, merging_method_name: str):
        self.method = merging_method_name

    def widen_merging(
        self,
        merged_model: nn.Module,
        models_to_merge: list,
        merge_strength: float = 1.0,
        temperature: float = 0.8,
        enable_ties: bool = True,
        threshold: float = 0.00005,
        batch_size: int = 30,
        forced_merge_ratio: float = 0.1,
    ):
        """Memory-efficient TIES+WIDEN with batch processing"""
        device = torch.device("cpu")
        algorithm_name = "TIES+WIDEN" if enable_ties else "WIDEN"
        print(f"[{self.method}] merging on {device} with {algorithm_name} (Memory Optimized)")

        # Get parameter names without loading all tensors
        base_param_names = list(merged_model.named_parameters())
        other_param_names = [list(model.named_parameters()) for model in models_to_merge]

        # Find common parameters
        common_names = set(name for name, _ in base_param_names)
        for other_params in other_param_names:
            common_names &= set(name for name, _ in other_params)
        common_names = list(common_names)

        print(f"[{self.method}] Processing {len(common_names)} parameters in batches of {batch_size}")

        merged_params = {}
        fell_back = 0
        actually_merged = 0

        # Calculate importance scores for forcing merges (lightweight operation)
        importance_scores = {}

        if forced_merge_ratio > 0:
            print("[Memory] Calculating parameter importance scores...")
            for name in tqdm(common_names[:100], desc="[Importance] Sampling"):
                try:
                    base_param = dict(merged_model.named_parameters())[name].detach().cpu()
                    other_params = [dict(model.named_parameters())[name].detach().cpu() for model in models_to_merge]

                    total_diff = 0.0
                    for other_param in other_params:
                        if base_param.shape == other_param.shape:
                            diff = torch.norm(other_param - base_param).item()
                            total_diff += diff

                    importance_scores[name] = total_diff

                    # Clean up immediately
                    del base_param, other_params
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                except Exception:
                    importance_scores[name] = 0.0

        # Determine forced merge parameters
        forced_merge_set = set()
        if forced_merge_ratio > 0:
            sorted_names = sorted(importance_scores.items(), key=lambda x: -x[1])
            min_merge_count = max(1, int(len(common_names) * forced_merge_ratio))
            forced_merge_set = set(n for n, _ in sorted_names[:min_merge_count])
            print(f"[{self.method}] Will force merge {len(forced_merge_set)} most important parameters ({forced_merge_ratio*100:.1f}%)")
        else:
            print(f"[{self.method}] Forced merging disabled")

        # Process parameters in batches
        for batch_start in range(0, len(common_names), batch_size):
            batch_end = min(batch_start + batch_size, len(common_names))
            batch_names = common_names[batch_start:batch_end]

            print(f"[Memory] Processing batch {batch_start//batch_size + 1}/{(len(common_names) + batch_size - 1)//batch_size} ({len(batch_names)} params)")

            # Load only current batch into memory
            batch_base_params = {}
            batch_other_params = [{}] * len(models_to_merge)

            for name in batch_names:
                try:
                    base_param = dict(merged_model.named_parameters())[name].detach().cpu().float()
                    batch_base_params[name] = base_param

                    for i, model in enumerate(models_to_merge):
                        other_param = dict(model.named_parameters())[name].detach().cpu().float()
                        if batch_other_params[i] == {}:
                            batch_other_params[i] = {}
                        batch_other_params[i][name] = other_param

                except Exception as e:
                    print(f"[WARNING] Failed to load {name}: {e}")
                    continue

            # Apply TIES processing to the batch
            batch_task_vectors = []
            for i in range(len(models_to_merge)):
                tv_dict = {}
                for name in batch_names:
                    if name in batch_base_params and name in batch_other_params[i]:
                        tv_dict[name] = batch_other_params[i][name] - batch_base_params[name]
                batch_task_vectors.append(tv_dict)

            # Apply TIES trimming to batch if enabled
            if enable_ties and batch_task_vectors:
                batch_task_vectors = self._apply_batch_ties_trimming(batch_task_vectors, threshold)
                if len(batch_task_vectors) > 1:
                    batch_deltas = self._apply_batch_sign_election(batch_task_vectors, batch_names)
                else:
                    batch_deltas = {name: torch.stack([tv[name] for tv in batch_task_vectors])
                                  for name in batch_names if name in batch_task_vectors[0]}
            else:
                batch_deltas = {name: torch.stack([tv[name] for tv in batch_task_vectors])
                              for name in batch_names if all(name in tv for tv in batch_task_vectors)}

            # Process the batch
            for name in batch_names:
                if name not in batch_base_params or name not in batch_deltas:
                    merged_params[name] = dict(merged_model.named_parameters())[name].detach().cpu()
                    fell_back += 1
                    continue

                try:
                    base_param = batch_base_params[name]
                    delta = batch_deltas[name]

                    # Check if this parameter should be merged
                    should_merge = False
                    merge_type = "SKIP"

                    if name in forced_merge_set:
                        # Always merge forced parameters
                        should_merge = True
                        merge_type = "FORCED"
                    else:
                        # Check threshold for non-forced parameters
                        delta_magnitude = torch.norm(delta).item()
                        if delta_magnitude > threshold:
                            should_merge = True
                            merge_type = "THRESHOLD"

                    if should_merge:
                        # Apply merging with temperature and strength
                        if merge_type == "FORCED":
                            # Use full strength for forced parameters
                            strength = merge_strength
                        else:
                            # Use reduced strength for threshold parameters
                            strength = merge_strength * 0.5

                        # Temperature scaling
                        strength *= (1.0 / temperature)

                        merged = base_param + delta.sum(0) * strength
                        merged_params[name] = merged
                        actually_merged += 1

                        if actually_merged <= 10:
                            diff_magnitude = torch.norm(merged - base_param).item()
                            print(f"[{merge_type}] {name}: strength={strength:.2f}, diff_mag={diff_magnitude:.6f}")
                    else:
                        # Skip merging, keep original parameter
                        merged_params[name] = base_param

                except Exception as e:
                    print(f"[WARNING] Failed to process {name}: {e}")
                    merged_params[name] = batch_base_params[name]
                    fell_back += 1

            # Clean up batch memory immediately
            del batch_base_params, batch_other_params, batch_task_vectors
            if 'batch_deltas' in locals():
                del batch_deltas
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Force garbage collection
            import gc
            gc.collect()

        total = len(common_names)
        skipped_small = total - actually_merged - fell_back

        # Create results string and print it
        results_text = f"""[{self.method}] Results:
  - Successfully merged: {actually_merged} / {total} parameters
  - Failed (errors): {fell_back}
  - Skipped (below threshold): {skipped_small}
  - Forced merges: {len(forced_merge_set)}"""

        print(results_text)

        merged_params = ensure_same_device(merged_params, "cpu")

        # Return both merged params and results string
        return merged_params, results_text

    def _apply_batch_ties_trimming(self, task_vectors, threshold):
        """Apply TIES trimming to a batch of task vectors"""
        trimmed_vectors = []
        for tv_dict in task_vectors:
            trimmed_dict = {}
            for name, delta in tv_dict.items():
                mask = torch.abs(delta) >= threshold
                trimmed_dict[name] = delta * mask.float()
            trimmed_vectors.append(trimmed_dict)
        return trimmed_vectors

    def _apply_batch_sign_election(self, task_vectors, param_names):
        """Apply TIES sign election to a batch"""
        elected_deltas = {}
        for name in param_names:
            if all(name in tv for tv in task_vectors):
                deltas = torch.stack([tv[name] for tv in task_vectors])
                signs = torch.sign(deltas)
                sign_sum = signs.sum(dim=0)
                elected_sign = torch.sign(sign_sum)

                mask = torch.where(elected_sign == 0,
                                 torch.ones_like(signs),
                                 (signs * elected_sign.unsqueeze(0)) >= 0)

                elected_deltas[name] = deltas * mask.float()
        return elected_deltas


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
                "threshold": ("FLOAT", {"default": 0.00005, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "batch_size": ("INT", {"default": 30, "min": 10, "max": 100, "step": 10}),
                "forced_merge_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "merge_results")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, model_base, model_other, merge_strength, temperature, enable_ties,
                threshold=0.00005, batch_size=30, forced_merge_ratio=0.1):
        # Create isolated copies to avoid modifying originals
        import copy

        base_state = model_base.model.state_dict()
        other_state = model_other.model.state_dict()

        # Create completely independent copies
        isolated_base_state = {k: v.clone().detach() for k, v in base_state.items()}
        isolated_other_state = {k: v.clone().detach() for k, v in other_state.items()}

        class TempModel:
            def __init__(self, state_dict):
                self._state_dict = state_dict

            def named_parameters(self):
                for name, param in self._state_dict.items():
                    yield name, param

            def state_dict(self):
                return self._state_dict

        temp_base_model = TempModel(isolated_base_state)
        temp_other_model = TempModel(isolated_other_state)

        merger = MergingMethod("DonutWidenMergeUNet")
        merged_dict, results_text = merger.widen_merging(
            merged_model=temp_base_model,
            models_to_merge=[temp_other_model],
            merge_strength=merge_strength,
            temperature=temperature,
            enable_ties=enable_ties,
            threshold=threshold,
            batch_size=batch_size,
            forced_merge_ratio=forced_merge_ratio,
        )

        # Create merged model
        model_merged = copy.deepcopy(model_base)
        model_merged.model.load_state_dict(merged_dict, strict=False)

        return (model_merged, results_text)


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
                "threshold": ("FLOAT", {"default": 0.00005, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "batch_size": ("INT", {"default": 30, "min": 10, "max": 100, "step": 10}),
                "forced_merge_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "merge_results")
    FUNCTION = "execute"
    CATEGORY = "donut/merge"

    def execute(self, clip_base, clip_other, merge_strength, temperature, enable_ties,
                threshold=0.00005, batch_size=30, forced_merge_ratio=0.1):
        # Create isolated copies to avoid modifying originals
        import copy

        # Handle different CLIP model structures
        enc_a = getattr(clip_base, "model", getattr(clip_base, "clip", getattr(clip_base, "cond_stage_model", None)))
        enc_b = getattr(clip_other, "model", getattr(clip_other, "clip", getattr(clip_other, "cond_stage_model", None)))

        if enc_a is None or enc_b is None:
            raise AttributeError("Could not locate encoder model in CLIP structure.")

        base_state = enc_a.state_dict()
        other_state = enc_b.state_dict()

        # Create completely independent copies
        isolated_base_state = {k: v.clone().detach() for k, v in base_state.items()}
        isolated_other_state = {k: v.clone().detach() for k, v in other_state.items()}

        class TempModel:
            def __init__(self, state_dict):
                self._state_dict = state_dict

            def named_parameters(self):
                for name, param in self._state_dict.items():
                    yield name, param

            def state_dict(self):
                return self._state_dict

        temp_base_model = TempModel(isolated_base_state)
        temp_other_model = TempModel(isolated_other_state)

        merger = MergingMethod("DonutWidenMergeCLIP")
        merged_dict, results_text = merger.widen_merging(
            merged_model=temp_base_model,
            models_to_merge=[temp_other_model],
            merge_strength=merge_strength,
            temperature=temperature,
            enable_ties=enable_ties,
            threshold=threshold,
            batch_size=batch_size,
            forced_merge_ratio=forced_merge_ratio,
        )

        # Create merged CLIP model
        clip_merged = copy.deepcopy(clip_base)

        # Get the encoder from the merged clip and load the merged parameters
        enc_merged = getattr(clip_merged, "model", getattr(clip_merged, "clip", getattr(clip_merged, "cond_stage_model", None)))
        if enc_merged is not None:
            enc_merged.load_state_dict(merged_dict, strict=False)

        return (clip_merged, results_text)


NODE_CLASS_MAPPINGS = {
    "DonutWidenMergeUNet": DonutWidenMergeUNet,
    "DonutWidenMergeCLIP": DonutWidenMergeCLIP,
}

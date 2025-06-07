
import torch
import torch.nn as nn
from tqdm import tqdm

# use package-relative path for ComfyUI
from .utils.sdxl_safetensors import ensure_same_device

# Dummy placeholders for required utility classes that should be part of the node
class TaskVector:
    def __init__(self, base_model, finetuned_model, exclude_param_names_regex):
        self.task_vector_param_dict = {
            k: finetuned_model.state_dict()[k] - base_model.state_dict()[k]
            for k in base_model.state_dict()
            if k in finetuned_model.state_dict()
        }
        print(f"[TaskVector] param count: {len(self.task_vector_param_dict)}")

class MergingMethod:
    def __init__(self, merging_method_name: str, vram_limit_bytes: int = None):
        self.method = merging_method_name
        self.vram_limit = vram_limit_bytes

    def _choose_device(self):
        if torch.cuda.is_available() and self.vram_limit is not None:
            dev = torch.cuda.current_device()
            free, _ = torch.cuda.mem_get_info(dev)
            if free >= self.vram_limit:
                return torch.device("cuda")
        return torch.device("cpu")

    def widen_merging(
        self,
        merged_model: nn.Module,
        models_to_merge: list,
        exclude_param_names_regex: list,
        above_average_value_ratio: float = 1.0,
        score_calibration_value: float = 1.0,
        min_merge_percent: float = 0.0,
    ):
        device = self._choose_device()
        print(f"[{self.method}] merging on {device}")

        # initialize the dictionary that will hold the merged weights early so
        # it always exists even if an exception occurs later in the method
        merged_params = {}

        pre_params = {n: p.detach().to(device=device, dtype=torch.float32).clone()
                      for n, p in merged_model.named_parameters()}
        finetuned_dicts = [
            {n: p.detach().to(device=device, dtype=torch.float32).clone()
             for n, p in m.named_parameters()}
            for m in models_to_merge
        ]

        def transpose_tok(d):
            if "model.embed_tokens.weight" in d:
                d["model.embed_tokens.weight"] = d["model.embed_tokens.weight"].T

        transpose_tok(pre_params)
        for d in finetuned_dicts:
            transpose_tok(d)

        task_vectors = [
            TaskVector(merged_model, m, exclude_param_names_regex)
            for m in models_to_merge
        ]

        def compute_mag_dir(param_dict, desc):
            mags, dirs = {}, {}
            for name, tensor in tqdm(param_dict.items(), desc=desc):
                try:
                    if tensor.dim() < 1:
                        continue
                    flat = tensor.view(tensor.shape[0], -1) if tensor.dim() == 4 else tensor
                    mag = flat.norm(dim=0)
                    dir = flat / (mag.unsqueeze(0) + 1e-8)
                    dirs[name] = dir.view(tensor.shape) if tensor.dim()==4 else dir
                    mags[name] = mag
                except Exception:
                    continue
            return mags, dirs

        pre_mag, pre_dir = compute_mag_dir(pre_params, "[mag/dir] pretrained")
        diff_list = []
        for fin in finetuned_dicts:
            fin_mag, fin_dir = compute_mag_dir(fin, "[mag/dir] finetuned")
            mag_diff = {k:(fin_mag[k] - pre_mag[k]).abs() for k in pre_mag if k in fin_mag}
            dir_diff = {k:1 - torch.cosine_similarity(fin_dir[k], pre_dir[k], dim=0)
                        for k in pre_dir if k in fin_dir}
            diff_list.append((mag_diff, dir_diff))

        def rank_sig(diff: torch.Tensor):
            if diff.ndim != 2: raise IndexError
            n,dim = diff.shape
            flat = diff.reshape(n,-1)
            idx = torch.argsort(flat, dim=1)
            L = flat.shape[1]
            sig = torch.arange(L, device=flat.device)/L
            base = sig.unsqueeze(0).repeat(n,1)
            return base.scatter(1, idx, sig)

        def importance(sig: torch.Tensor):
            sc = torch.softmax(sig, dim=0)
            avg= sig.mean(1, keepdim=True)
            mask = sig > avg * above_average_value_ratio
            sc[mask] = score_calibration_value
            return sc

        def merge_param(delta, base, mag_rank, dir_rank):
            try:
                ms = importance(mag_rank)
                ds = importance(dir_rank)
                w = 0.5 * (ms + ds)
                w = w.view(delta.shape[0], *([1]*(delta.dim()-2)), delta.shape[-1])
                merged = base + (delta * w).sum(0)
                return merged if merged.shape == base.shape else base
            except Exception:
                return base

        fell_back = 0
        common = set(pre_mag.keys())
        for tv in task_vectors:
            common &= set(tv.task_vector_param_dict.keys())

        # determine parameters to force merge
        importance_scores = {}
        for name in common:
            score_sum = 0.0
            for mag_diff, dir_diff in diff_list:
                if name in mag_diff and name in dir_diff:
                    score_sum += mag_diff[name].mean().item() + dir_diff[name].mean().item()
            importance_scores[name] = score_sum

        sorted_names = sorted(importance_scores.items(), key=lambda x: -x[1])
        min_merge_count = int(len(sorted_names) * min_merge_percent)
        forced_merge_set = set(n for n, _ in sorted_names[:min_merge_count])

        for name in tqdm(common, desc=f"[{self.method}] merging"):
            try:
                delta = torch.stack([tv.task_vector_param_dict[name] for tv in task_vectors])
                magd  = torch.stack([d[0][name] for d in diff_list])
                dird  = torch.stack([d[1][name] for d in diff_list])
                rankm = rank_sig(magd)
                rankd = rank_sig(dird)
            except Exception:
                merged_params[name] = pre_params[name]
                fell_back += 1
                continue
            merged = merge_param(delta, pre_params[name], rankm, rankd)
            if name in forced_merge_set or not torch.allclose(merged, pre_params[name]):
                merged_params[name] = merged
            else:
                merged_params[name] = pre_params[name]
                fell_back += 1

        total = len(common)
        print(f"[{self.method}] merged {total - fell_back} / {total} parameters")
        merged_params = ensure_same_device(merged_params, "cpu")
        return merged_params


class DonutWidenMergeUNet:
    class_type = "MODEL"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_base": ("MODEL",),
                "model_other": ("MODEL",),
                "above_average_value_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "score_calibration_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "min_merge_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "donut"

    def execute(self, model_base, model_other, above_average_value_ratio, score_calibration_value, min_merge_percent):
        base_state = model_base.model.state_dict()
        other_state = model_other.model.state_dict()
        diffs = [
            k
            for k in base_state
            if k in other_state
            and not torch.equal(
                base_state[k].detach().cpu(), other_state[k].detach().cpu()
            )
        ]
        print(f"[WidenMergeUNet] Different parameters: {len(diffs)} / {len(base_state)}")

        merging = MergingMethod("WidenMergeUNet")

        merged_params = merging.widen_merging(
            merged_model=model_base.model,
            models_to_merge=[model_other.model],
            exclude_param_names_regex=[],
            above_average_value_ratio=above_average_value_ratio,
            score_calibration_value=score_calibration_value,
            min_merge_percent=min_merge_percent,
        ) or {}
        model_base.model.load_state_dict(merged_params, strict=False)
        return (model_base,)


class DonutWidenMergeCLIP:
    class_type = "CLIP"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_base": ("CLIP",),
                "clip_other": ("CLIP",),
                "above_average_value_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "score_calibration_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "min_merge_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "execute"
    CATEGORY = "donut"

    def execute(self, clip_base, clip_other, above_average_value_ratio, score_calibration_value, min_merge_percent):
        base_state = clip_base.cond_stage_model.state_dict()
        other_state = clip_other.cond_stage_model.state_dict()
        diffs = [
            k
            for k in base_state
            if k in other_state
            and not torch.equal(
                base_state[k].detach().cpu(), other_state[k].detach().cpu()
            )
        ]
        print(f"[WidenMergeCLIP] Different parameters: {len(diffs)} / {len(base_state)}")

        merging = MergingMethod("WidenMergeCLIP")
        merged_params = merging.widen_merging(
            merged_model=clip_base.cond_stage_model,
            models_to_merge=[clip_other.cond_stage_model],
            exclude_param_names_regex=[],
            above_average_value_ratio=above_average_value_ratio,
            score_calibration_value=score_calibration_value,
            min_merge_percent=min_merge_percent,
        )
        clip_base.cond_stage_model.load_state_dict(merged_params, strict=False)
        return (clip_base,)


NODE_CLASS_MAPPINGS = {
    "DonutWidenMergeUNet": DonutWidenMergeUNet,
    "DonutWidenMergeCLIP": DonutWidenMergeCLIP,
}

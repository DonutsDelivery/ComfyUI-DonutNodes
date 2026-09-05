"""Donut Extract LoRA - raw MODEL + patched MODEL -> low-rank LoRA safetensors.

Unlike ComfyUI's ModelSubtract-based extractor, this node compares effective
ModelPatcher tensors in float32 and explicitly includes Donut runtime-only
features. Donut Experimental-bypass LoRA adapters normally exist only as forward
hooks, and Donut Model Merge Krea2 Experimental bypass keeps exact model2 swaps
as source-model forwards rather than model1 weight patches. Ordinary state-dict
subtraction cannot see either behavior.

The extractor processes one model parameter at a time on CPU. It does not need
the full patched model or a dense full-model difference resident in VRAM, making
it more practical on lower-VRAM GPUs. Large matrices use randomized low-rank
SVD; small matrices use exact SVD. Bias and one-dimensional weight differences
are stored as Comfy-compatible direct-diff entries so model merges are not
silently reduced to matrix-only changes.

Quantized ComfyUI models may expose auxiliary state-dict entries such as
``weight_scale``/``input_scale`` that are not real module attributes. We never
feed those entries through ModelPatcher.get_key_patches(); instead we enumerate
only actual ``*.weight``/``*.bias`` targets and reproduce get_key_patches for
those keys.
"""

import logging
import os

import folder_paths
import torch

import comfy.lora
import comfy.model_patcher
import comfy.utils

try:
    from .donut_bypass_materialization import (
        BYPASS_INJECTION_KEY,
        get_bypass_components,
    )
    from .donut_krea2_merge_serialization import (
        KREA2_MERGE_INJECTION_KEY,
        get_krea2_merge_bypass_info,
    )
except ImportError:
    from donut_bypass_materialization import (
        BYPASS_INJECTION_KEY,
        get_bypass_components,
    )
    from donut_krea2_merge_serialization import (
        KREA2_MERGE_INJECTION_KEY,
        get_krea2_merge_bypass_info,
    )


OUTPUT_DTYPES = ("fp16", "bf16", "fp32")
OUTPUT_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _runtime_injection_keys(model):
    injections = getattr(model, "injections", None)
    if not isinstance(injections, dict):
        return set()

    # Krea2's composable injection list is intentionally falsey so the Donut
    # bypass-LoRA compatibility guard can wrap it. Presence of the key still
    # means runtime behavior exists and must not be ignored by extraction.
    return {
        key
        for key, value in injections.items()
        if value or key == KREA2_MERGE_INJECTION_KEY
    }


def _identity(value, **kwargs):
    return value


def _parameter_target_keys(model, bypass_components=None):
    """Return real diffusion parameters a Comfy LoRA can target.

    Quantized state dicts can contain entries such as ``*.weight_scale`` and
    ``*.input_scale``. Those are serialization metadata, not normal LoRA target
    parameters, and asking ModelPatcher.get_key_patches() to resolve the entire
    state dict can fail when they are not live module attributes. Restricting the
    scan to canonical ``*.weight`` and ``*.bias`` keys avoids that failure.
    """
    keys = set()

    def accepted(key):
        return (
            isinstance(key, str)
            and key.startswith("diffusion_model.")
            and (key.endswith(".weight") or key.endswith(".bias"))
        )

    try:
        state_dict = model.model.state_dict()
        keys.update(key for key in state_dict.keys() if accepted(key))
    except Exception:
        logging.exception("[DonutExtractLoRA] Could not enumerate model state dict")

    patches = getattr(model, "patches", {})
    if isinstance(patches, dict):
        keys.update(key for key in patches.keys() if accepted(key))

    if isinstance(bypass_components, dict):
        # Donut bypass adapters are currently weight-only, but keep this generic
        # in case Comfy gains additive bias adapters later.
        keys.update(key for key in bypass_components.keys() if accepted(key))

    return keys


def _single_key_patches(model, key):
    """Reproduce ModelPatcher.get_key_patches() for one real parameter key.

    Doing this one key at a time avoids ComfyUI's quantization bookkeeping keys
    while preserving physically patched backups, hook backups, conversion
    functions, and the ordered regular patch stack.
    """
    try:
        weight, _set_func, convert_func = comfy.model_patcher.get_key_weight(
            model.model, key
        )
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not resolve model parameter {key}: {exc}") from exc

    backup = getattr(model, "backup", {}).get(key)
    if backup is not None:
        weight = backup.weight

    hook_backup = getattr(model, "hook_backup", {}).get(key)
    if hook_backup is not None:
        weight = hook_backup[0]

    if convert_func is None:
        convert_func = _identity

    result = [(weight, convert_func)]
    patches = getattr(model, "patches", {})
    if key in patches:
        result.extend(patches[key])
    return result


def _get_extractable_key_patches(model, bypass_components=None):
    """Return get_key_patches-style entries for actual LoRA target parameters."""
    output = {}
    for key in sorted(_parameter_target_keys(model, bypass_components)):
        try:
            output[key] = _single_key_patches(model, key)
        except Exception as exc:
            # A model may expose a serialization-only parameter-like key without
            # a resolvable live module. Do not let one exotic layer abort every
            # otherwise extractable LoRA layer.
            logging.warning(
                "[DonutExtractLoRA] Skipping unresolved parameter %s: %s",
                key,
                exc,
            )
    return output


def _convert_base_weight(base_weight, convert_func):
    """Convert a normal or quantized model parameter to CPU float32."""
    value = base_weight

    # Comfy quantized modules commonly provide convert_weight(), which
    # dequantizes QuantizedTensor. Apply the model's own conversion before doing
    # any dtype conversion ourselves so per-format scale information is honored.
    if convert_func is not None:
        try:
            value = convert_func(value, inplace=False)
        except TypeError:
            try:
                value = convert_func(value, inplace=True)
            except TypeError:
                value = convert_func(value)

    dequantize = getattr(value, "dequantize", None)
    if callable(dequantize):
        value = dequantize()

    if not torch.is_tensor(value):
        raise TypeError(f"Unsupported base parameter type: {type(value).__name__}")

    return value.detach().to(device="cpu", dtype=torch.float32).clone()


def _zero_parameter_patches(key_patches):
    """Build an effective zero parameter matching an existing parameter shape."""
    if not key_patches:
        return None
    first = key_patches[0]
    if not isinstance(first, (tuple, list)) or len(first) < 2:
        return None
    value = _convert_base_weight(first[0], first[1])
    return [(torch.zeros_like(value), _identity)]


def _get_effective_key_patches(model, bypass_components=None):
    """Resolve the parameter sources that actually execute at inference time.

    Normal and partial Krea2 merge components stay on ``model`` and therefore
    use its ordinary patch stack. Exact ratio-0 Krea2 Experimental-bypass plans
    execute the retained model2 source module instead. Replace those parameter
    sources with model2 here before subtraction so extraction represents the
    live merged model, not model1-with-LoRAs.

    Returns ``(key_patches, swapped_keys, swap_count)``.
    """
    output = _get_extractable_key_patches(model, bypass_components)
    merge_info = get_krea2_merge_bypass_info(model)
    if merge_info is None:
        return output, set(), 0

    source_model, plans, _attachment_keys = merge_info
    source_keys = _get_extractable_key_patches(source_model)
    swapped_keys = set()

    for module_path, weight_key, _ratio in plans:
        source_weight = source_keys.get(weight_key)
        if source_weight is None:
            raise RuntimeError(
                "Donut Extract LoRA could not resolve the model2 source weight "
                f"for Krea2 bypassed module '{module_path}' ({weight_key})."
            )

        # The runtime hook calls model2's complete Linear module, so its weight
        # and bias (if present) replace model1. Later Donut bypass LoRAs are not
        # added here; _effective_weight applies the final model's bypass adapter
        # on top of this model2 weight, matching inference order.
        output[weight_key] = source_weight
        swapped_keys.add(weight_key)

        bias_key = f"{module_path}.bias"
        if bias_key in source_keys:
            output[bias_key] = source_keys[bias_key]
            swapped_keys.add(bias_key)
        elif bias_key in output:
            # A source Linear with bias=None executes with zero bias. If model1
            # has a bias, represent that effective zero so the emitted diff_b can
            # cancel the raw bias instead of silently keeping it.
            zero_patches = _zero_parameter_patches(output[bias_key])
            if zero_patches is not None:
                output[bias_key] = zero_patches
                swapped_keys.add(bias_key)

    return output, swapped_keys, len(plans)


def _effective_weight(key, key_patches, bypass_components):
    """Materialize one effective parameter in float32 without final fp8 rounding."""
    if not key_patches:
        return None

    first = key_patches[0]
    if not isinstance(first, (tuple, list)) or len(first) < 2:
        raise TypeError(f"Unexpected key-patch layout for {key}")

    base_weight, convert_func = first[0], first[1]
    weight = _convert_base_weight(base_weight, convert_func)
    patches = list(key_patches[1:])

    # Donut's experimental bypass is activation-side at inference time, but for
    # the bypass-compatible additive adapters Donut permits, its exact
    # serialization equivalent is an ordinary weight-adapter patch at the same
    # strength. For Krea2 exact swaps, key_patches above already points at the
    # model2 source weight, so this correctly produces model2 + later bypass LoRA.
    for adapter, strength in bypass_components or ():
        strength = float(strength)
        if strength != 0.0:
            patches.append((strength, adapter, 1.0, None, None))

    if patches:
        weight = comfy.lora.calculate_weight(
            patches,
            weight,
            key,
            intermediate_dtype=torch.float32,
            original_weights={key: key_patches},
        )
    return weight.detach().float().cpu()


def _factorize_delta(delta, rank):
    """Return balanced LoRA up/down factors approximating one weight delta."""
    if delta.ndim < 2:
        return None

    original_shape = tuple(delta.shape)
    out_dim = original_shape[0]
    matrix = delta.reshape(out_dim, -1).float().contiguous()
    rows, cols = matrix.shape
    effective_rank = min(int(rank), rows, cols)
    if effective_rank < 1:
        return None

    if matrix.numel() == 0 or float(matrix.abs().max()) <= 1e-12:
        return None

    min_dim = min(rows, cols)
    use_exact = matrix.numel() <= 4_000_000 or min_dim <= effective_rank + 8

    if use_exact:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        u = u[:, :effective_rank]
        s = s[:effective_rank]
        vh = vh[:effective_rank, :]
    else:
        # Randomized low-rank SVD avoids the very large temporary allocations of
        # a full SVD on transformer projection matrices. Keep it deterministic.
        q = min(min_dim, effective_rank + 8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            u, s, v = torch.svd_lowrank(matrix, q=q, niter=2)
        order = torch.argsort(s, descending=True)[:effective_rank]
        u = u[:, order]
        s = s[order]
        vh = v[:, order].transpose(0, 1)

    # Split singular values symmetrically. This reconstructs the same truncated
    # SVD as (U*S)@Vh but keeps both LoRA factors numerically well-scaled.
    root_s = torch.sqrt(torch.clamp(s, min=0.0))
    up = u * root_s.unsqueeze(0)
    down = root_s.unsqueeze(1) * vh

    if len(original_shape) > 2:
        up = up.reshape(out_dim, effective_rank, *([1] * (len(original_shape) - 2)))
        down = down.reshape(effective_rank, *original_shape[1:])

    return up.contiguous(), down.contiguous(), effective_rank


def _is_zero_delta(delta):
    return delta.numel() == 0 or float(delta.abs().max()) <= 1e-12


class DonutExtractLoRA:
    """Extract a diffusion-model LoRA from RAW MODEL and PATCHED MODEL."""

    DESCRIPTION = (
        "Extracts a LoRA representing PATCHED MODEL - RAW MODEL. Understands "
        "Donut Experimental-bypass LoRAs and Donut Model Merge Krea2 Experimental "
        "bypass, including model2 hard-swapped weights. Matrix differences are "
        "rank-limited by SVD; bias/1D weight differences are stored as exact "
        "Comfy-compatible direct diffs. Quantization metadata is ignored safely."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_model": ("MODEL", {
                    "tooltip": "The unmodified/base diffusion model before LoRAs or merges are applied.",
                }),
                "patched_model": ("MODEL", {
                    "tooltip": "The effective target model after LoRAs/patches/Krea2 merge bypasses.",
                }),
                "rank": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Maximum SVD rank per matrix weight. Higher ranks preserve more model-merge detail and create larger LoRAs.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "loras/Donut_extracted_lora",
                }),
                "dtype": (OUTPUT_DTYPES, {
                    "default": "fp16",
                    "tooltip": "Storage dtype for extracted factors/direct diffs. Computation is float32.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "report")
    FUNCTION = "extract"
    OUTPUT_NODE = True
    CATEGORY = "advanced/model_merging"

    def extract(self, raw_model, patched_model, rank, filename_prefix, dtype="fp16"):
        rank = int(rank)
        output_dtype = OUTPUT_DTYPE_MAP[dtype]

        raw_bypass = get_bypass_components(raw_model)
        patched_bypass = get_bypass_components(patched_model)
        raw_keys, raw_swapped_keys, raw_swap_count = _get_effective_key_patches(
            raw_model,
            raw_bypass,
        )
        patched_keys, patched_swapped_keys, patched_swap_count = _get_effective_key_patches(
            patched_model,
            patched_bypass,
        )

        raw_injections = _runtime_injection_keys(raw_model)
        patched_injections = _runtime_injection_keys(patched_model)
        if BYPASS_INJECTION_KEY in raw_injections and not raw_bypass:
            raise RuntimeError(
                "RAW MODEL has Donut Experimental-bypass injections, but their adapters "
                "could not be recovered. Restart ComfyUI with the updated DonutNodes and retry."
            )
        if BYPASS_INJECTION_KEY in patched_injections and not patched_bypass:
            raise RuntimeError(
                "PATCHED MODEL has Donut Experimental-bypass injections, but their adapters "
                "could not be recovered. Restart ComfyUI with the updated DonutNodes and retry."
            )

        supported_injections = {
            BYPASS_INJECTION_KEY,
            KREA2_MERGE_INJECTION_KEY,
        }
        unsupported_injections = (
            raw_injections | patched_injections
        ) - supported_injections

        common_keys = sorted(set(raw_keys).intersection(patched_keys))
        output_sd = {}
        low_rank_layers = 0
        direct_diff_layers = 0
        skipped_zero = 0
        failed_layers = []

        same_base = getattr(raw_model, "clone_base_uuid", None) == getattr(
            patched_model, "clone_base_uuid", object()
        )

        for key in common_keys:
            try:
                raw_weight = _effective_weight(
                    key,
                    raw_keys[key],
                    raw_bypass.get(key),
                )
                patched_weight = _effective_weight(
                    key,
                    patched_keys[key],
                    patched_bypass.get(key),
                )
                if raw_weight is None or patched_weight is None:
                    continue
                if raw_weight.shape != patched_weight.shape:
                    failed_layers.append(
                        f"{key}: shape {tuple(raw_weight.shape)} != {tuple(patched_weight.shape)}"
                    )
                    continue

                delta = patched_weight - raw_weight
                del raw_weight, patched_weight

                if _is_zero_delta(delta):
                    skipped_zero += 1
                    del delta
                    continue

                if key.endswith(".bias"):
                    base = key[:-len(".bias")]
                    output_sd[f"{base}.diff_b"] = delta.to(output_dtype).contiguous().cpu()
                    direct_diff_layers += 1
                    del delta
                    continue

                if not key.endswith(".weight"):
                    del delta
                    continue

                base = key[:-len(".weight")]
                if delta.ndim < 2:
                    output_sd[f"{base}.diff"] = delta.to(output_dtype).contiguous().cpu()
                    direct_diff_layers += 1
                    del delta
                    continue

                factors = _factorize_delta(delta, rank)
                del delta
                if factors is None:
                    skipped_zero += 1
                    continue

                up, down, actual_rank = factors
                output_sd[f"{base}.lora_up.weight"] = up.to(output_dtype).cpu()
                output_sd[f"{base}.lora_down.weight"] = down.to(output_dtype).cpu()
                # alpha/rank = 1, so applying the extracted LoRA at strength 1
                # reconstructs the truncated SVD delta directly.
                output_sd[f"{base}.alpha"] = torch.tensor(
                    float(actual_rank),
                    dtype=torch.float32,
                )
                low_rank_layers += 1
                del up, down
            except Exception as exc:
                logging.exception("[DonutExtractLoRA] Failed extracting %s", key)
                failed_layers.append(f"{key}: {exc}")

        extracted_layers = low_rank_layers + direct_diff_layers
        if extracted_layers == 0:
            detail = ""
            if failed_layers:
                detail = " First failure: " + failed_layers[0]
            raise RuntimeError(
                "No non-zero diffusion-model LoRA layers were extracted." + detail
            )

        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
            )
        output_path = os.path.join(
            full_output_folder,
            f"{filename}_{counter:05}_.safetensors",
        )

        metadata = {
            "donut.extractor": "DonutExtractLoRA",
            "donut.rank": str(rank),
            "donut.dtype": dtype,
            "donut.bypass_aware": "true",
            "donut.krea2_merge_aware": "true",
            "donut.quant_metadata_safe": "true",
            "donut.low_rank_layers": str(low_rank_layers),
            "donut.direct_diff_layers": str(direct_diff_layers),
        }
        comfy.utils.save_torch_file(output_sd, output_path, metadata=metadata)

        report_parts = [
            f"Extracted {low_rank_layers} low-rank matrix layer(s) at rank <= {rank}",
            f"{direct_diff_layers} exact bias/1D diff layer(s)",
            f"skipped {skipped_zero} zero layer(s)",
        ]
        if patched_swap_count:
            report_parts.append(
                f"included {patched_swap_count} Krea2 model2 hard swap(s) "
                f"covering {len(patched_swapped_keys)} extractable parameter(s)"
            )
        if raw_swap_count:
            report_parts.append(
                f"RAW MODEL also contains {raw_swap_count} Krea2 hard swap(s)"
            )
        if patched_bypass:
            report_parts.append(
                f"included Donut Experimental bypass on {len(patched_bypass)} weight(s)"
            )
        if not same_base:
            report_parts.append(
                "RAW and PATCHED do not share clone_base_uuid; the LoRA includes all model differences, not only attached LoRAs"
            )
        if unsupported_injections:
            report_parts.append(
                "WARNING: unsupported runtime injections were ignored: "
                + ", ".join(sorted(unsupported_injections))
            )
        if failed_layers:
            report_parts.append(f"{len(failed_layers)} layer(s) failed; see console")

        report = "; ".join(report_parts)
        print(f"[DonutExtractLoRA] {report}")
        if patched_swap_count:
            print(
                "[DonutExtractLoRA] Krea2 Experimental-bypass source weights "
                "were included before SVD."
            )
        print(f"[DonutExtractLoRA] Saved to {output_path}")
        return output_path, report


NODE_CLASS_MAPPINGS = {
    "DonutExtractLoRA": DonutExtractLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutExtractLoRA": "Donut Extract LoRA (Raw → Patched)",
}

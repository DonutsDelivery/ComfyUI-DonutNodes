"""Donut Extract LoRA - raw MODEL + patched MODEL -> low-rank LoRA safetensors.

Unlike ComfyUI's ModelSubtract-based extractor, this node compares ModelPatcher
weights in float32 directly and explicitly includes Donut Experimental-bypass
LoRA adapters. Bypass adapters normally exist only as forward hooks, so ordinary
state-dict subtraction cannot see them.

The extractor processes one model weight at a time on CPU. It does not need the
full patched model or a dense full-model difference resident in VRAM, making it
more practical on lower-VRAM GPUs. Large matrices use randomized low-rank SVD;
small matrices use exact SVD.

Quantized ComfyUI models may expose auxiliary state-dict entries such as
``weight_scale``/``input_scale`` that are not real module attributes. We never
feed those entries through ModelPatcher.get_key_patches(); instead we enumerate
only actual ``*.weight`` targets and reproduce get_key_patches for those keys.
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
except ImportError:
    from donut_bypass_materialization import (
        BYPASS_INJECTION_KEY,
        get_bypass_components,
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
    return {key for key, value in injections.items() if value}


def _identity(value, **kwargs):
    return value


def _weight_target_keys(model, bypass_components=None):
    """Return real diffusion-model weight targets, excluding quant metadata.

    Quantized state dicts can contain entries such as ``*.weight_scale`` and
    ``*.input_scale``. Those are serialization metadata, not LoRA targets, and
    calling ModelPatcher.get_key_patches() on the full state dict can try to
    resolve them as normal module attributes. Restricting the scan to the
    canonical ``*.weight`` keys avoids that failure and is also exactly what a
    standard LoRA can represent.
    """
    keys = set()

    try:
        state_dict = model.model.state_dict()
        keys.update(
            key for key in state_dict.keys()
            if key.startswith("diffusion_model.") and key.endswith(".weight")
        )
    except Exception:
        logging.exception("[DonutExtractLoRA] Could not enumerate model state dict")

    patches = getattr(model, "patches", {})
    if isinstance(patches, dict):
        keys.update(
            key for key in patches.keys()
            if isinstance(key, str)
            and key.startswith("diffusion_model.")
            and key.endswith(".weight")
        )

    if isinstance(bypass_components, dict):
        keys.update(
            key for key in bypass_components.keys()
            if isinstance(key, str)
            and key.startswith("diffusion_model.")
            and key.endswith(".weight")
        )

    return keys


def _single_key_patches(model, key):
    """Reproduce ModelPatcher.get_key_patches() for one real weight key.

    Doing this one key at a time avoids ComfyUI's quantization bookkeeping keys
    while preserving physically patched backups, hook backups, conversion
    functions, and the ordered regular patch stack.
    """
    try:
        weight, _set_func, convert_func = comfy.model_patcher.get_key_weight(
            model.model, key
        )
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not resolve model weight {key}: {exc}") from exc

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
    """Return get_key_patches-style entries for only actual LoRA weight targets."""
    output = {}
    for key in sorted(_weight_target_keys(model, bypass_components)):
        try:
            output[key] = _single_key_patches(model, key)
        except Exception as exc:
            # A model may expose a serialization-only weight-like key without a
            # resolvable live module. Do not let one exotic layer abort every
            # otherwise extractable LoRA layer.
            logging.warning("[DonutExtractLoRA] Skipping unresolved weight %s: %s", key, exc)
    return output


def _convert_base_weight(base_weight, convert_func):
    """Convert a normal or quantized model weight to CPU float32."""
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
        raise TypeError(f"Unsupported base weight type: {type(value).__name__}")

    return value.detach().to(device="cpu", dtype=torch.float32).clone()


def _effective_weight(key, key_patches, bypass_components):
    """Materialize one effective model weight in float32 without final fp8 rounding."""
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
    # strength. Append those adapters before computing the effective weight so
    # extraction sees what inference actually used.
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


class DonutExtractLoRA:
    """Extract a diffusion-model LoRA from RAW MODEL and PATCHED MODEL."""

    DESCRIPTION = (
        "Extracts a standard low-rank LoRA representing PATCHED MODEL - RAW MODEL. "
        "Understands Donut Experimental bypass, whose adapter effect is invisible "
        "to ordinary ModelSubtract/state-dict LoRA extractors. Quantization metadata "
        "is ignored safely, and large SVDs run in CPU memory so the full merged "
        "model does not need to fit in VRAM."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_model": ("MODEL", {
                    "tooltip": "The unmodified/base diffusion model before LoRAs or merges are applied.",
                }),
                "patched_model": ("MODEL", {
                    "tooltip": "The model after LoRAs/patches. Donut Experimental bypass adapters are included.",
                }),
                "rank": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Maximum SVD rank per weight. Higher ranks preserve more of the model difference and create larger LoRAs.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "loras/Donut_extracted_lora",
                }),
                "dtype": (OUTPUT_DTYPES, {
                    "default": "fp16",
                    "tooltip": "Storage dtype for the extracted LoRA factors. Computation is float32.",
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
        raw_keys = _get_extractable_key_patches(raw_model, raw_bypass)
        patched_keys = _get_extractable_key_patches(patched_model, patched_bypass)

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

        unsupported_injections = (raw_injections | patched_injections) - {BYPASS_INJECTION_KEY}

        common_keys = sorted(set(raw_keys).intersection(patched_keys))
        output_sd = {}
        extracted_layers = 0
        skipped_zero = 0
        failed_layers = []

        same_base = getattr(raw_model, "clone_base_uuid", None) == getattr(
            patched_model, "clone_base_uuid", object()
        )

        for key in common_keys:
            try:
                raw_weight = _effective_weight(key, raw_keys[key], raw_bypass.get(key))
                patched_weight = _effective_weight(key, patched_keys[key], patched_bypass.get(key))
                if raw_weight is None or patched_weight is None:
                    continue
                if raw_weight.shape != patched_weight.shape:
                    failed_layers.append(f"{key}: shape {tuple(raw_weight.shape)} != {tuple(patched_weight.shape)}")
                    continue

                delta = patched_weight - raw_weight
                factors = _factorize_delta(delta, rank)
                del raw_weight, patched_weight, delta
                if factors is None:
                    skipped_zero += 1
                    continue

                up, down, actual_rank = factors
                base = key[:-len(".weight")]
                output_sd[f"{base}.lora_up.weight"] = up.to(output_dtype).cpu()
                output_sd[f"{base}.lora_down.weight"] = down.to(output_dtype).cpu()
                # alpha/rank = 1, so applying the extracted LoRA at strength 1
                # reconstructs the truncated SVD delta directly.
                output_sd[f"{base}.alpha"] = torch.tensor(float(actual_rank), dtype=torch.float32)
                extracted_layers += 1
                del up, down
            except Exception as exc:
                logging.exception("[DonutExtractLoRA] Failed extracting %s", key)
                failed_layers.append(f"{key}: {exc}")

        if extracted_layers == 0:
            detail = ""
            if failed_layers:
                detail = " First failure: " + failed_layers[0]
            raise RuntimeError(
                "No non-zero diffusion-model LoRA layers were extracted." + detail
            )

        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
        output_path = os.path.join(
            full_output_folder, f"{filename}_{counter:05}_.safetensors"
        )

        metadata = {
            "donut.extractor": "DonutExtractLoRA",
            "donut.rank": str(rank),
            "donut.dtype": dtype,
            "donut.bypass_aware": "true",
            "donut.quant_metadata_safe": "true",
        }
        comfy.utils.save_torch_file(output_sd, output_path, metadata=metadata)

        report_parts = [
            f"Extracted {extracted_layers} layer(s) at rank <= {rank}",
            f"skipped {skipped_zero} zero/non-matrix layer(s)",
        ]
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
        print(f"[DonutExtractLoRA] Saved to {output_path}")
        return output_path, report


NODE_CLASS_MAPPINGS = {
    "DonutExtractLoRA": DonutExtractLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutExtractLoRA": "Donut Extract LoRA (Raw → Patched)",
}

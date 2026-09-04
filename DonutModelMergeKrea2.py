"""Krea2 model merge with an optional quantized-weight hybrid bypass path.

The regular mode mirrors ComfyUI's built-in ``ModelMergeKrea2`` node. The
experimental mode bypasses only exact model2 swaps (ratio 0.0) for compatible
linear layers, keeping those quantized modules intact at inference time.
Partial blends remain on ComfyUI's normal patch path so inference uses one
materialized linear forward instead of evaluating both full model layers.
"""

from __future__ import annotations

import math
import uuid
import weakref
from typing import Any

import torch

try:
    import comfy.weight_adapter as comfy_weight_adapter
except ImportError:  # Older ComfyUI versions keep the regular merge available.
    comfy_weight_adapter = None

try:
    from comfy.patcher_extension import PatcherInjection
except ImportError:  # Older ComfyUI versions keep the regular merge available.
    PatcherInjection = None


_EXECUTION_MODES = ("Comfy patches", "Experimental bypass")
_DIFFUSION_PREFIX = "diffusion_model."
_INJECTION_KEY = "donut_krea2_model_merge_bypass"
_SOURCE_MODELS_KEY = "donut_krea2_model_merge_sources"
_CACHE_ATTACHMENT_PREFIX = "donut_krea2_model_merge_identity:"
_WEIGHT_ADAPTER_BASE = getattr(comfy_weight_adapter, "WeightAdapterBase", None)
_BYPASS_MANAGER = getattr(comfy_weight_adapter, "BypassInjectionManager", None)


class _ComposableInjectionList(list):
    """Runtime injections that intentionally compose with Donut LoRA bypass.

    ``DonutApplyLoRAStack`` conservatively treats any truthy pre-existing
    injection list as incompatible with LoRA forward adapters. Krea2 hard-swap
    hooks are an exception: they replace the base linear forward with model2,
    and a later additive LoRA hook can safely wrap that swapped forward. The
    list therefore stays iterable for ModelPatcher but is falsey for that
    compatibility guard. ``copy`` preserves the behavior across patcher clones.
    """

    def __bool__(self):
        return False

    def copy(self):
        return type(self)(self)


def _module_for_path(model_root: Any, module_path: str):
    """Resolve a dotted module path, including numeric ModuleList indices."""
    module = model_root
    try:
        for part in module_path.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return module


def _module_for_weight_key(model_root: Any, key: str):
    if not isinstance(key, str) or not key.endswith(".weight"):
        return None
    return _module_for_path(model_root, key[:-7])


def _is_supported_linear(module: Any) -> bool:
    """Match the conservative linear-layer subset used by Donut bypass nodes."""
    if module is None:
        return False
    if isinstance(module, torch.nn.Linear):
        return True

    module_type = type(module)
    return (
        module_type.__module__ == "comfy.ops"
        and module_type.__name__ == "Linear"
        and callable(getattr(module, "_forward", None))
    )


def _linear_compatibility_error(base_module: Any, source_module: Any):
    if not _is_supported_linear(base_module):
        return "model1 target is not a supported linear layer"
    if not _is_supported_linear(source_module):
        return "model2 target is not a supported linear layer"
    if base_module is source_module:
        return "model1 and model2 resolve to the same layer object"

    for attribute in ("in_features", "out_features"):
        base_value = getattr(base_module, attribute, None)
        source_value = getattr(source_module, attribute, None)
        if (
            base_value is not None
            and source_value is not None
            and int(base_value) != int(source_value)
        ):
            return f"linear {attribute} differs between the two models"
    return None


def _direct_module_state_keys(keys, module_path: str):
    """Return state keys owned directly by a module, not by its children.

    Quantized Comfy linear layers can own buffers such as ``weight_scale``,
    ``input_scale`` and ``pre_quant_scale`` in addition to weight/bias. When a
    layer is swapped at runtime, all direct keys stay with their original model
    so the source module uses internally consistent quantization metadata.
    """
    prefix = f"{module_path}."
    direct = set()
    for key in keys:
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if suffix and "." not in suffix:
            direct.add(key)
    return direct


def _ratio_for_key(key: str, ratios: dict[str, float]) -> float:
    """Resolve the longest Krea2 component-prefix match, like core ComfyUI."""
    default_ratio = float(ratios.get("first.", next(iter(ratios.values()), 1.0)))
    ratio = default_ratio
    key_without_prefix = (
        key[len(_DIFFUSION_PREFIX):]
        if key.startswith(_DIFFUSION_PREFIX)
        else key
    )

    longest = -1
    for prefix, value in ratios.items():
        if key_without_prefix.startswith(prefix) and len(prefix) > longest:
            ratio = float(value)
            longest = len(prefix)

    if not math.isfinite(ratio):
        raise ValueError(f"Non-finite merge ratio for {key}: {ratio}")
    return ratio


def _has_runtime_injections(model) -> bool:
    if bool(getattr(model, "is_injected", False)):
        return True
    injections = getattr(model, "injections", {})
    if isinstance(injections, dict):
        # Injection keys indicate active runtime behavior even when a compatible
        # injection list intentionally reports false to DonutApplyLoRAStack.
        return bool(injections)
    return bool(injections)


def _bypass_prerequisite_error(model1, model2):
    if _WEIGHT_ADAPTER_BASE is None or _BYPASS_MANAGER is None or PatcherInjection is None:
        return "this ComfyUI build does not expose weight-adapter bypass support"
    required_methods = (
        "clone",
        "get_key_patches",
        "set_injections",
        "set_additional_models",
        "get_additional_models_with_key",
        "set_attachments",
    )
    for label, model in (("model1", model1), ("model2", model2)):
        missing = [name for name in required_methods if not callable(getattr(model, name, None))]
        if missing:
            return f"{label} patcher lacks required methods: {', '.join(missing)}"

    if getattr(model1, "model", None) is getattr(model2, "model", None):
        return "model1 and model2 share the same underlying model"
    is_clone = getattr(model1, "is_clone", None)
    if callable(is_clone) and is_clone(model2):
        return "model1 and model2 are patcher clones of the same model"
    if _has_runtime_injections(model1) or _has_runtime_injections(model2):
        return "one of the input models already has runtime injections"

    device1 = getattr(model1, "load_device", None)
    device2 = getattr(model2, "load_device", None)
    if device1 is not None and device2 is not None and device1 != device2:
        return "model1 and model2 use different load devices"
    return None


if _WEIGHT_ADAPTER_BASE is not None:
    class _ModelBlendBypassAdapter(_WEIGHT_ADAPTER_BASE):
        """Swap one compatible linear forward to model2 without rebuilding weight."""

        name = "donut_model_merge"

        def __init__(self, source_patcher, module_path: str, model1_ratio: float):
            self.source_patcher = source_patcher
            self.module_path = module_path
            self.model1_ratio = float(model1_ratio)
            self.loaded_keys = set()
            self.weights = ()
            self._source_module = None

        def _get_source_module(self):
            if self._source_module is None:
                source_root = getattr(self.source_patcher, "model", None)
                source_module = _module_for_path(source_root, self.module_path)
                if not _is_supported_linear(source_module):
                    raise RuntimeError(
                        "Donut Krea2 merge bypass could not resolve a supported "
                        f"model2 linear layer at '{self.module_path}'"
                    )
                self._source_module = source_module
            return self._source_module

        def bypass_forward(self, original_forward, x, *args, **kwargs):
            ratio = self.model1_ratio
            if ratio >= 1.0:
                return original_forward(x, *args, **kwargs)
            if ratio != 0.0:
                raise RuntimeError(
                    "Donut Krea2 merge bypass received a partial ratio. Partial "
                    "blends must stay on Comfy's materialized patch path."
                )
            return self._get_source_module()(x, *args, **kwargs)
else:
    class _ModelBlendBypassAdapter:  # pyright: ignore[reportRedeclaration]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Experimental bypass requires comfy.weight_adapter")


def _eject_abandoned_bypass_injections(inner_injections):
    """Restore forwards when ComfyUI drops a loaded merge clone.

    ComfyUI keeps loaded model patchers through weak references. If a merge
    output is disconnected, the clone can be collected before the next model
    load and ComfyUI then switches the loaded entry back to the clone's parent.
    Bypass hooks live on the shared underlying modules, so without this
    finalizer the parent model would continue using model2 forwards.
    """
    for inner in reversed(inner_injections):
        try:
            # BypassInjectionManager eject callbacks do not need a live patcher;
            # they restore the original module forwards held by their hooks.
            inner.eject(None)
        except Exception as error:
            print(
                "[DonutModelMergeKrea2] Failed to eject an abandoned bypass "
                f"injection: {error}"
            )


def _make_dynamic_bypass_injection(plans):
    """Build a clone-safe injection that resolves the paired model at load time."""
    if PatcherInjection is None or _BYPASS_MANAGER is None:
        raise RuntimeError("Experimental bypass is unavailable")

    plans = tuple(plans)
    active_injections = weakref.WeakKeyDictionary()

    def inject(model_patcher):
        if model_patcher in active_injections:
            return

        source_models = model_patcher.get_additional_models_with_key(_SOURCE_MODELS_KEY)
        if len(source_models) != 1:
            raise RuntimeError(
                "Donut Krea2 merge bypass expected exactly one paired model2 "
                f"source, found {len(source_models)}"
            )
        source_patcher = source_models[0]

        manager = _BYPASS_MANAGER()
        for module_path, weight_key, ratio in plans:
            base_module = _module_for_path(model_patcher.model, module_path)
            source_module = _module_for_path(source_patcher.model, module_path)
            reason = _linear_compatibility_error(base_module, source_module)
            if reason is not None:
                raise RuntimeError(
                    f"Donut Krea2 merge bypass target '{module_path}' changed "
                    f"after node execution: {reason}"
                )
            adapter = _ModelBlendBypassAdapter(source_patcher, module_path, ratio)
            manager.add_adapter(weight_key, adapter, strength=1.0)

        inner_injections = tuple(manager.create_injections(model_patcher.model))
        hook_count = manager.get_hook_count()
        if hook_count != len(plans):
            raise RuntimeError(
                "Donut Krea2 merge bypass could not create every planned hook "
                f"({hook_count}/{len(plans)})"
            )

        injected = []
        try:
            for inner in inner_injections:
                injected.append(inner)
                inner.inject(model_patcher)
        except Exception:
            for inner in reversed(injected):
                inner.eject(model_patcher)
            raise

        cleanup = weakref.finalize(
            model_patcher,
            _eject_abandoned_bypass_injections,
            inner_injections,
        )
        cleanup.atexit = False
        active_injections[model_patcher] = (inner_injections, cleanup)

    def eject(model_patcher):
        active = active_injections.pop(model_patcher, None)
        if active is None:
            return
        inner_injections, cleanup = active
        cleanup.detach()
        for inner in reversed(inner_injections):
            inner.eject(model_patcher)

    return PatcherInjection(inject=inject, eject=eject)


def _regular_merge(model1, model2, ratios):
    """Mirror ComfyUI's ModelMergeBlocks patch orientation exactly."""
    merged = model1.clone()
    patches = model2.get_key_patches(_DIFFUSION_PREFIX)
    for key, patch in patches.items():
        ratio = _ratio_for_key(key, ratios)
        merged.add_patches({key: patch}, 1.0 - ratio, ratio)
    return merged


def _build_bypass_plans(base_model, source_model, patch_keys, ratios):
    """Plan only exact model2 swaps; partial ratios are deliberately materialized."""
    plans = []
    bypassed_keys = set()
    seen_paths = set()

    for key in patch_keys:
        if not isinstance(key, str) or not key.endswith(".weight"):
            continue
        module_path = key[:-7]
        if module_path in seen_paths:
            continue
        seen_paths.add(module_path)

        ratio = _ratio_for_key(key, ratios)
        if ratio != 0.0:
            continue

        base_module = _module_for_weight_key(base_model.model, key)
        source_module = _module_for_weight_key(source_model.model, key)
        if _linear_compatibility_error(base_module, source_module) is not None:
            continue

        direct_keys = _direct_module_state_keys(patch_keys, module_path)
        if key not in direct_keys:
            continue

        plans.append((module_path, key, ratio))
        bypassed_keys.update(direct_keys)

    return plans, bypassed_keys


class DonutModelMergeKrea2:
    """ComfyUI ModelMergeKrea2 plus an opt-in hybrid runtime bypass."""

    class_type = "CUSTOM"
    aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model1": ("MODEL",),
            "model2": ("MODEL",),
        }
        argument = ("FLOAT", {
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "tooltip": "1.0 keeps model1; 0.0 uses model2 for this Krea2 component.",
        })

        required["first."] = argument
        required["tmlp."] = argument
        required["txtmlp."] = argument
        required["tproj."] = argument

        for index in range(2):
            required[f"txtfusion.layerwise_blocks.{index}."] = argument
        required["txtfusion.projector."] = argument
        for index in range(2):
            required[f"txtfusion.refiner_blocks.{index}."] = argument
        for index in range(28):
            required[f"blocks.{index}."] = argument
        required["last."] = argument

        return {
            "required": required,
            "optional": {
                "execution_mode": (list(_EXECUTION_MODES), {
                    "default": "Comfy patches",
                    "tooltip": (
                        "Experimental bypass is hybrid: exact 0.0 model2 swaps use "
                        "runtime forwarding for compatible linear layers, while "
                        "partial ratios use normal Comfy merged weights so inference "
                        "runs one linear forward instead of two. Exact 1.0 keeps "
                        "model1 unchanged. Runtime swaps are not materialized when "
                        "saving a checkpoint."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "model/merging/model specific"
    DESCRIPTION = (
        "Krea2 component merge with the same controls as ComfyUI's built-in node. "
        "Experimental bypass accelerates hard model1/model2 component swaps while "
        "keeping partial blends on Comfy's single-forward materialized merge path."
    )

    def merge(self, model1, model2, execution_mode="Comfy patches", **ratios):
        if execution_mode not in _EXECUTION_MODES:
            raise ValueError(f"Unknown Krea2 merge execution mode: {execution_mode}")
        if execution_mode == "Comfy patches":
            return (_regular_merge(model1, model2, ratios),)

        prerequisite_error = _bypass_prerequisite_error(model1, model2)
        if prerequisite_error is not None:
            print(
                "[DonutModelMergeKrea2] Experimental bypass used the regular "
                f"compatibility path: {prerequisite_error}"
            )
            return (_regular_merge(model1, model2, ratios),)

        merged = model1.clone()
        source = model2.clone()
        patches = model2.get_key_patches(_DIFFUSION_PREFIX)
        plans, bypassed_keys = _build_bypass_plans(
            merged,
            source,
            tuple(patches.keys()),
            ratios,
        )

        regular_count = 0
        partial_count = 0
        for key, patch in patches.items():
            if key in bypassed_keys:
                continue
            ratio = _ratio_for_key(key, ratios)
            if ratio >= 1.0:
                continue
            if 0.0 < ratio < 1.0:
                partial_count += 1
            merged.add_patches({key: patch}, 1.0 - ratio, ratio)
            regular_count += 1

        if not plans:
            print(
                "[DonutModelMergeKrea2] Experimental bypass used Comfy patches "
                f"for {partial_count} partial state key(s); no runtime swaps needed"
            )
            return (merged,)

        merged.set_additional_models(_SOURCE_MODELS_KEY, [source])
        merged.set_injections(
            _INJECTION_KEY,
            _ComposableInjectionList([_make_dynamic_bypass_injection(plans)]),
        )
        identity_key = f"{_CACHE_ATTACHMENT_PREFIX}{uuid.uuid4().hex}"
        merged.set_attachments(identity_key, tuple(plans))
        if hasattr(merged, "patches_uuid"):
            merged.patches_uuid = uuid.uuid4()

        print(
            "[DonutModelMergeKrea2] Experimental hybrid attached "
            f"{len(plans)} exact model2 swap hook(s); kept {regular_count} "
            f"state key(s) on Comfy's regular path ({partial_count} partial)"
        )
        return (merged,)


NODE_CLASS_MAPPINGS = {
    "DonutModelMergeKrea2": DonutModelMergeKrea2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutModelMergeKrea2": "Donut Model Merge Krea2",
}

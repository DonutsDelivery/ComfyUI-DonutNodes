"""Safe Krea2 LoRA stack application.

Overrides DonutApplyLoRAStack with an opt-in per-block RMS energy limiter for
Krea2 LoRAs and optional fusion-aware budgeting for the 12-column text-fusion
projector. Safety remains off by default so old workflows retain their exact
behaviour.
"""

import copy
import math
import re
import weakref
import uuid

import comfy.sd
import comfy.utils
import folder_paths
import torch

try:
    import comfy.weight_adapter as comfy_weight_adapter
except ImportError:  # Older ComfyUI releases retain the standard patch path.
    comfy_weight_adapter = None

from .donut_lora_nodes import (
    _TEXT_MERGE_VECTOR,
    _lora_has_real_text_encoder,
    _split_fused_text,
)
from .lora_block_weight import LoraLoaderBlockWeight
from .donut_model_patch_routing import add_model_patch_components
from .donut_krea2_merge_serialization import KREA2_MERGE_SOURCE_KEY, get_krea2_merge_bypass_info
from .donut_lora_execution import (
    EXECUTION_MODES,
    publish_execution_mode,
    resolve_execution_mode,
)


_KREA_BLOCK_RE = re.compile(r"(?<![a-z_])blocks\.(\d+)")
_KREA_BLOCK_COUNT = 28
_KREA_VECTOR_SIZE = _KREA_BLOCK_COUNT + 1  # non-block bucket + 28 blocks
_SAFE_ENERGY_BUDGET = 1.0
_FUSION_BUDGET_KEY = "donut_krea2_fusion_budget"
_FUSION_AWARE_MODES = ("Off", "Attenuate only", "Use headroom")
_EXECUTION_MODES = EXECUTION_MODES
_KREA_TEXT_RE = re.compile(r"txt(?:fusion|mlp)|text_(?:fusion|mlp)")
_PROJECTOR_RE = re.compile(r"(?:txtfusion|text_fusion).*projector")
_PROJECTOR_COLUMN_COUNT = 12


def _krea2_block_indices(lora):
    """Return the Krea2 single-stream block indices referenced by a LoRA."""
    block_nums = set()
    for key in lora.keys():
        match = _KREA_BLOCK_RE.search(key)
        if match:
            block_nums.add(int(match.group(1)))
    return block_nums


def _is_krea2_lora(lora):
    """Return True when a LoRA contains Krea2-style single-stream blocks."""
    block_nums = _krea2_block_indices(lora)
    return bool(block_nums) and max(block_nums) < _KREA_BLOCK_COUNT


def _is_krea2_text_lora(lora):
    """Return True for Krea2's diffusion-side text-fusion adapter weights."""
    return any(_KREA_TEXT_RE.search(key.lower()) for key in lora)


def _split_projector_text(lora_text):
    """Separate the 12-column projector adapter from other text-fusion keys."""
    projector, other = {}, {}
    for key, value in lora_text.items():
        (projector if _PROJECTOR_RE.search(key.lower()) else other)[key] = value
    return projector, other


def _read_fusion_budget(model):
    """Read Fusion Control metadata without depending on its implementation."""
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        return None
    transformer_options = model_options.get("transformer_options")
    if not isinstance(transformer_options, dict):
        return None
    metadata = transformer_options.get(_FUSION_BUDGET_KEY)
    if not isinstance(metadata, dict) or int(metadata.get("version", 0)) != 1:
        return None
    gains = metadata.get("projector_gains")
    if not isinstance(gains, (list, tuple)) or len(gains) != _PROJECTOR_COLUMN_COUNT:
        return None
    try:
        gains = tuple(float(value) for value in gains)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in gains):
        return None
    output = dict(metadata)
    output["projector_gains"] = gains
    return output


def _nominal_projector_gains(metadata):
    """Resolve static gains used for scalar-energy accounting.

    tensor_rms adds a prompt-dependent common multiplier at runtime.  Dividing
    by the profile RMS gives a neutral-energy nominal profile for attenuation,
    but its unknown runtime multiplier makes automatic headroom boosts unsafe.
    """
    gains = tuple(float(value) for value in metadata["projector_gains"])
    dynamic = metadata.get("projector_normalization") == "tensor_rms"
    if dynamic:
        rms = math.sqrt(sum(value * value for value in gains) / len(gains))
        if rms > 1e-12:
            gains = tuple(value / rms for value in gains)
    return gains, dynamic


def _projector_column_scales(
    entries,
    gains,
    mode,
    max_boost,
    dynamic=False,
    budget=_SAFE_ENERGY_BUDGET,
):
    """Budget projector LoRA scalar energy after Fusion Control's 12 gains."""
    participants = [
        idx for idx, entry in enumerate(entries)
        if entry["is_krea_text"] and entry["projector_text"] and float(entry["cw"]) != 0.0
    ]
    if not participants:
        return (1.0,) * _PROJECTOR_COLUMN_COUNT, None

    energy = math.sqrt(sum(float(entries[idx]["cw"]) ** 2 for idx in participants))
    allow_boost = mode == "Use headroom" and not dynamic
    max_boost = max(1.0, float(max_boost))
    scales = []
    for gain in gains:
        effective_energy = abs(float(gain)) * energy
        if effective_energy <= 1e-12:
            scale = max_boost if allow_boost else 1.0
        else:
            scale = budget / effective_energy
            scale = min(max_boost if allow_boost else 1.0, scale)
        scales.append(scale)

    return tuple(scales), {
        "raw_energy": energy,
        "dynamic": dynamic,
        "boosted": any(scale > 1.0 + 1e-12 for scale in scales),
        "limited": any(scale < 1.0 - 1e-12 for scale in scales),
    }


def _scale_projector_lora_columns(lora, scales):
    """Scale projector LoRA delta columns without scaling the base projector.

    Standard LoRA/PEFT adapters are adjusted on their input/down matrix. Direct
    ``.diff`` projector patches are adjusted directly. Unsupported adapter
    families are returned unchanged so the caller can use a conservative
    uniform strength fallback.
    """
    if len(scales) != _PROJECTOR_COLUMN_COUNT:
        raise ValueError(f"Expected {_PROJECTOR_COLUMN_COUNT} projector scales")

    adjusted = dict(lora)
    transformed = False
    for key, value in lora.items():
        if not torch.is_tensor(value) or not value.is_floating_point() or value.ndim < 2:
            continue
        lower = key.lower()
        if not _PROJECTOR_RE.search(lower) or value.shape[-1] != _PROJECTOR_COLUMN_COUNT:
            continue

        is_down = lower.endswith("lora_down.weight") or lower.endswith("lora_a.weight")
        is_diff = lower.endswith(".diff")
        is_direct_weight = lower.endswith("projector.weight") and value.shape[-2] == 1
        if not (is_down or is_diff or is_direct_weight):
            continue

        scale = torch.tensor(scales, device=value.device, dtype=value.dtype)
        adjusted[key] = value * scale.reshape(*([1] * (value.ndim - 1)), -1)
        transformed = True

    return adjusted, transformed


def _parse_numeric_vector(vector, required_size=1):
    """Parse a numeric Krea2 vector into a full safety-analysis vector.

    Donut's normal auto-vector path intentionally sizes a Krea2 vector only up
    to the highest block actually present in that LoRA. A LoRA that only has
    blocks 0..15 therefore has a valid 17-value vector (base + 16 blocks), not
    a 29-value vector. Safe Stack must accept that rather than requiring all 28
    architectural blocks.

    Returns ``(padded_values, original_size, tail)``. Missing higher Krea2 blocks are
    zero-filled for the energy calculation because the LoRA has no weights for
    them. The caller later trims back to ``original_size`` before handing the
    vector to the normal Donut block loader, preserving its original shape.
    """
    if not vector:
        values = [1.0] * max(1, min(required_size, _KREA_VECTOR_SIZE))
        return values + [0.0] * (_KREA_VECTOR_SIZE - len(values)), len(values), []

    parts = [part.strip() for part in vector.split(",")]
    if len(parts) < required_size:
        return None

    try:
        values = [float(part) for part in parts]
    except (TypeError, ValueError):
        return None

    original_size = len(values)
    tail = values[_KREA_VECTOR_SIZE:]
    values = values[:_KREA_VECTOR_SIZE]
    values.extend([0.0] * (_KREA_VECTOR_SIZE - len(values)))
    return values, original_size, tail


def _format_vector(values):
    def fmt(value):
        if abs(value) < 1e-12:
            return "0"
        if abs(value - 1.0) < 1e-12:
            return "1"
        return f"{value:.6g}"

    return ",".join(fmt(value) for value in values)


def _normalise_krea_vectors(entries, budget=_SAFE_ENERGY_BUDGET):
    """Cap effective per-block RMS energy while preserving relative strengths.

    Each effective contribution is model_weight * block_weight. If the root
    sum of squares of all Krea2 contributions in a block exceeds ``budget``,
    every LoRA touching that block is attenuated by the same factor. Blocks
    below budget are left equivalent apart from numeric vector formatting.
    """
    vectors = []
    original_sizes = []
    vector_tails = []
    eligible = []

    for entry in entries:
        if not entry["is_krea"]:
            vectors.append(None)
            original_sizes.append(0)
            vector_tails.append([])
            eligible.append(False)
            continue

        block_indices = entry["krea_blocks"]
        required_size = (max(block_indices) + 2) if block_indices else 1
        parsed = _parse_numeric_vector(entry["vector"], required_size=required_size)
        if parsed is None:
            vectors.append(None)
            original_sizes.append(0)
            vector_tails.append([])
            eligible.append(False)
            print(
                f"[DonutApplyLoRAStack] Safe Stack: '{entry['name']}' uses a "
                "non-numeric vector or one too short for its populated Krea2 "
                "blocks; leaving it unchanged"
            )
            continue

        values, original_size, tail = parsed
        vectors.append(values)
        original_sizes.append(original_size)
        vector_tails.append(tail)
        eligible.append(True)

    scales = [[1.0] * _KREA_VECTOR_SIZE for _ in entries]
    limited_blocks = []

    for block_idx in range(_KREA_VECTOR_SIZE):
        energy_sq = 0.0
        participants = []
        for idx, entry in enumerate(entries):
            if not eligible[idx]:
                continue
            effective = float(entry["mw"]) * vectors[idx][block_idx]
            if effective == 0.0:
                continue
            energy_sq += effective * effective
            participants.append(idx)

        energy = math.sqrt(energy_sq)
        if energy > budget and participants:
            scale = budget / energy
            limited_blocks.append((block_idx, energy, scale))
            for idx in participants:
                scales[idx][block_idx] = scale

    adjusted = []
    for idx, entry in enumerate(entries):
        if not eligible[idx]:
            adjusted.append(entry["vector"])
            continue

        adjusted_full = [
            value * scales[idx][block_idx]
            for block_idx, value in enumerate(vectors[idx])
        ]
        kept = adjusted_full[:min(original_sizes[idx], _KREA_VECTOR_SIZE)]
        adjusted.append(_format_vector(kept + vector_tails[idx]))

    return adjusted, limited_blocks


def _normalise_fused_text_weights(entries, component=None, budget=_SAFE_ENERGY_BUDGET):
    """RMS-limit a Krea2 fused-text component as one shared scalar bucket."""
    participants = [
        idx for idx, entry in enumerate(entries)
        if (
            entry["is_krea_text"]
            and entry["fused_text"]
            and (component is None or bool(entry[component]))
            and float(entry["cw"]) != 0.0
        )
    ]
    if not participants:
        return [float(entry["cw"]) for entry in entries], None

    energy = math.sqrt(sum(float(entries[idx]["cw"]) ** 2 for idx in participants))
    scale = min(1.0, budget / energy) if energy > 0.0 else 1.0
    weights = [float(entry["cw"]) for entry in entries]
    if scale < 1.0:
        for idx in participants:
            weights[idx] *= scale
        return weights, (energy, scale)
    return weights, None


_WEIGHT_ADAPTER_BASE = getattr(comfy_weight_adapter, "WeightAdapterBase", None)


if _WEIGHT_ADAPTER_BASE is not None:
    class _CompositeBypassAdapter(_WEIGHT_ADAPTER_BASE):
        """Sum multiple additive bypass adapters for one model module."""

        name = "donut_composite"

        def __init__(self, components):
            self.components = tuple(components)
            self.loaded_keys = set().union(*(
                getattr(adapter, "loaded_keys", set())
                for adapter, _ in self.components
            ))
            self._weight_layout = []
            for adapter, _ in self.components:
                weights = adapter.weights
                if isinstance(weights, tuple):
                    self._weight_layout.append(("tuple", len(weights)))
                elif isinstance(weights, list):
                    self._weight_layout.append(("list", len(weights)))
                else:
                    self._weight_layout.append(("scalar", 1))

        @property
        def weights(self):
            flattened = []
            for adapter, _ in self.components:
                weights = adapter.weights
                if isinstance(weights, (tuple, list)):
                    flattened.extend(weights)
                else:
                    flattened.append(weights)
            return tuple(flattened)

        @weights.setter
        def weights(self, flattened):
            offset = 0
            for (adapter, _), (kind, size) in zip(self.components, self._weight_layout):
                values = flattened[offset:offset + size]
                offset += size
                if kind == "tuple":
                    adapter.weights = tuple(values)
                elif kind == "list":
                    adapter.weights = list(values)
                else:
                    adapter.weights = values[0]
            if offset != len(flattened):
                raise ValueError("Composite bypass adapter weight layout changed unexpectedly")

        def h(self, x, base_out):
            total = None
            outer_multiplier = float(getattr(self, "multiplier", 1.0))
            shared_attributes = (
                "is_conv",
                "conv_dim",
                "kernel_size",
                "in_channels",
                "out_channels",
                "kw_dict",
            )
            for adapter, strength in self.components:
                adapter.multiplier = outer_multiplier * float(strength)
                for attribute in shared_attributes:
                    if hasattr(self, attribute):
                        setattr(adapter, attribute, getattr(self, attribute))
                contribution = adapter.h(x, base_out)
                total = contribution if total is None else total + contribution
            return total
else:
    class _CompositeBypassAdapter:  # pyright: ignore[reportRedeclaration]
        def __init__(self, components):
            raise RuntimeError(
                "Experimental bypass requires comfy.weight_adapter support"
            )


_LOKR_ADAPTER_BASE = getattr(comfy_weight_adapter, "LoKrAdapter", None)
if _LOKR_ADAPTER_BASE is not None:
    class _LinearLoKrBypassAdapter(_LOKR_ADAPTER_BASE):
        """Keep native patch/save math, with matching rank scaling in forward."""

        def __init__(self, adapter):
            self.loaded_keys = set(adapter.loaded_keys)
            self.weights = adapter.weights

        def h(self, x, base_out):
            w1, w2, alpha, a, b, c, d, _, _ = self.weights
            rank = None
            if w1 is None:
                rank = b.shape[0]
                w1 = a.to(dtype=x.dtype) @ b.to(dtype=x.dtype)
            else:
                w1 = w1.to(dtype=x.dtype)
            grouped = x.reshape(*x.shape[:-1], w1.shape[1], -1)
            if w2 is None:
                # calculate_weight uses w2's rank when both are decomposed.
                rank = d.shape[0]
                hidden = torch.nn.functional.linear(grouped, d.to(dtype=x.dtype))
                hidden = torch.nn.functional.linear(hidden, c.to(dtype=x.dtype))
            else:
                hidden = torch.nn.functional.linear(grouped, w2.to(dtype=x.dtype))
            out = torch.nn.functional.linear(hidden.transpose(-1, -2), w1)
            out = out.transpose(-1, -2).flatten(-2)
            scale = alpha / rank if alpha is not None and rank is not None else 1.0
            return out * (scale * getattr(self, "multiplier", 1.0))


def _module_for_weight_key(model_root, key):
    """Resolve a state-dict weight key to its owning model module."""
    if not isinstance(key, str) or not key.endswith(".weight"):
        return None
    module = model_root
    try:
        for part in key[:-7].split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return module


def _bypass_compatibility_error(adapter, module):
    """Return why an adapter cannot use Donut's conservative bypass subset."""
    if comfy_weight_adapter is None:
        return "comfy.weight_adapter is unavailable"
    module_type = type(module)
    is_torch_linear = module is not None and isinstance(module, torch.nn.Linear)
    is_comfy_linear = (
        module is not None
        and module_type.__module__ == "comfy.ops"
        and module_type.__name__ == "Linear"
        and callable(getattr(module, "_forward", None))
    )
    conv_types = tuple(getattr(torch.nn, name) for name in ("Conv1d", "Conv2d", "Conv3d")
                       if hasattr(torch.nn, name))
    is_conv = isinstance(module, conv_types)
    if not (is_torch_linear or is_comfy_linear or is_conv):
        return "target module is not a supported linear layer or convolution"
    if is_conv and (module.groups != 1 or module.padding_mode != "zeros"):
        return "grouped or nonzero-padding-mode convolutions are not supported"
    if getattr(module, "pre_quant_scale", None) is not None:
        return "target applies an input pre-quantization scale"

    adapter_type = type(adapter)
    lora_type = getattr(comfy_weight_adapter, "LoRAAdapter", None)
    lokr_type = getattr(comfy_weight_adapter, "LoKrAdapter", None)
    weights = getattr(adapter, "weights", None)
    if lora_type is not None and adapter_type is lora_type:
        if not isinstance(weights, (tuple, list)) or len(weights) != 6:
            return "unexpected LoRA weight layout"
        up, down, _, mid, dora_scale, reshape = weights
        if dora_scale is not None:
            return "DoRA normalization is weight-dependent"
        if reshape is not None:
            return "reshape_weight is not activation-additive"
        if mid is not None:
            return "mid/Tucker LoRA bypass has not been parity-validated"
        if is_conv:
            dim = len(module.kernel_size)
            if not all(torch.is_tensor(value) and value.ndim in (2, dim + 2)
                       for value in (up, down)):
                return "unexpected convolutional LoRA factor dimensions"
            rank = down.shape[0]
            if (rank == 0 or up.shape[0] != module.out_channels or up.shape[1] != rank
                    or up.numel() != module.out_channels * rank
                    or down.numel() != rank * module.in_channels * math.prod(module.kernel_size)):
                return "convolutional LoRA factors do not match target kernel"
            if up.ndim != 2 and any(size != 1 for size in up.shape[2:]):
                return "convolutional LoRA up factor must use a pointwise kernel"
            if down.ndim != 2 and tuple(down.shape[1:]) != (module.in_channels, *module.kernel_size):
                return "convolutional LoRA down factor has incompatible kernel shape"
        return None

    if is_conv:
        return "convolutional bypass currently supports plain LoRA/LoCon only"

    loha_type = getattr(comfy_weight_adapter, "LoHaAdapter", None)
    if loha_type is not None and adapter_type is loha_type:
        if not isinstance(weights, (tuple, list)) or len(weights) != 8:
            return "unexpected LoHa weight layout"
        a, b, _, c, d, t1, t2, dora = weights
        if dora is not None:
            return "DoRA normalization is weight-dependent"
        if t1 is not None or t2 is not None:
            return "Tucker LoHa bypass has not been parity-validated"
        if not all(torch.is_tensor(value) and value.ndim == 2 for value in (a, b, c, d)):
            return "LoHa requires matrix factors"
        if (a.shape[1] != b.shape[0] or c.shape[1] != d.shape[0]
                or b.shape[0] == 0 or d.shape[0] == 0
                or (a.shape[0], b.shape[1]) != (c.shape[0], d.shape[1])):
            return "LoHa has incompatible factor shapes"
        return None

    if lokr_type is not None and adapter_type in (lokr_type, _LinearLoKrBypassAdapter):
        if not isinstance(weights, (tuple, list)) or len(weights) != 9:
            return "unexpected LoKr weight layout"
        w1, w2, _, w1_a, w1_b, w2_a, w2_b, t2, dora_scale = weights
        if dora_scale is not None:
            return "DoRA normalization is weight-dependent"
        if t2 is not None:
            return "Tucker LoKr bypass has not been parity-validated"
        for direct, a, b in ((w1, w1_a, w1_b), (w2, w2_a, w2_b)):
            if direct is not None:
                if torch.is_tensor(direct) and direct.ndim != 2:
                    return "linear LoKr requires matrix factors"
                continue
            if not all(torch.is_tensor(value) and value.ndim == 2 for value in (a, b)):
                return "LoKr decomposition requires two matrix factors"
            if a.shape[1] != b.shape[0] or b.shape[0] == 0:
                return "LoKr decomposition has incompatible factor ranks"
        return None

    return f"unsupported adapter type {adapter_type.__name__}"


def _partition_bypass_targets(model_root, model_keys, patches_by_key):
    """Keep a target wholly bypassed or wholly on the ordered regular path."""
    bypass_targets = {}
    regular_targets = {}
    fallback_reasons = {}
    adapter_base = _WEIGHT_ADAPTER_BASE or ()

    for key, components in patches_by_key.items():
        module = _module_for_weight_key(model_root, key) if key in model_keys else None
        reasons = []
        for patch_data, _ in components:
            if not isinstance(patch_data, adapter_base):
                reasons.append(f"regular patch type {type(patch_data).__name__}")
                continue
            reason = _bypass_compatibility_error(patch_data, module)
            if reason is not None:
                reasons.append(reason)

        if reasons:
            regular_targets[key] = components
            fallback_reasons[key] = tuple(dict.fromkeys(reasons))
        else:
            bypass_targets[key] = components

    return bypass_targets, regular_targets, fallback_reasons


def _register_bypass_adapters(manager, adapters_by_key):
    """Register one adapter or one additive composite for each model key."""
    for key, components in adapters_by_key.items():
        components = [
            (_LinearLoKrBypassAdapter(adapter) if _LOKR_ADAPTER_BASE is not None
             and type(adapter) is _LOKR_ADAPTER_BASE else adapter, strength)
            for adapter, strength in components
        ]
        if len(components) == 1:
            adapter, strength = components[0]
            manager.add_adapter(key, adapter, strength=strength)
            continue

        manager.add_adapter(key, _CompositeBypassAdapter(components), strength=1.0)



def _copy_runtime_adapter(adapter):
    """Keep runtime device moves/multipliers separate from saved components."""
    if isinstance(adapter, _CompositeBypassAdapter):
        return _CompositeBypassAdapter([
            (_copy_runtime_adapter(child), strength)
            for child, strength in adapter.components
        ])
    return copy.copy(adapter)


def _trace_lokr_calls(manager):
    """Report actual LoKr execution, including children of stacked adapters."""
    pending = set()
    for key, (adapter, _strength) in manager.adapters.items():
        components = adapter.components if isinstance(adapter, _CompositeBypassAdapter) else ((adapter, 1.),)
        for index, (component, _scale) in enumerate(components):
            if _LOKR_ADAPTER_BASE is None or not isinstance(component, _LOKR_ADAPTER_BASE):
                continue
            token = (key, index)
            pending.add(token)
            original = component.h

            def traced(x, base_out, _original=original, _token=token):
                result = _original(x, base_out)
                if _token in pending:
                    pending.remove(_token)
                    if not pending:
                        print(f"[DonutApplyLoRAStack] LoKr forward coverage: {total}/{total} component(s) executed")
                return result

            component.h = traced
    total = len(pending)
    return pending


def _eject_runtime_bypass(manager, injections, model_patcher=None):
    # Core ejects injection groups in insertion order, not reverse order. An
    # earlier Krea2 merge hook may already have restored this layer's base
    # forward. Do not resurrect that now-ejected swap from our saved forward.
    for hook in manager.hooks:
        if (hook.original_forward is not None
                and hook.module.forward != hook._bypass_forward):
            hook.original_forward = None
    for injection in reversed(injections):
        injection.eject(model_patcher)


def _make_rebinding_bypass_injections(manager, model_root):
    """Bind fresh hooks to the sampling clone, rather than the loader's root.

    The template manager also retains the canonical components for the save /
    extraction bridge. Its hooks are only used to validate the initial plan.
    """
    templates = manager.create_injections(model_root)
    expected = len(manager.adapters)
    if manager.get_hook_count() != expected or not templates:
        raise RuntimeError("Donut bypass could not create every planned forward hook")
    injection_type = type(templates[0])
    active = weakref.WeakKeyDictionary()
    reported = False

    def inject(model_patcher):
        nonlocal reported
        if model_patcher in active:
            return
        # A copied patcher can share the same physical modules. Retire any
        # previous runtime from this plan before installing the copied one.
        for owner, (root, runtime, inner, cleanup) in list(active.items()):
            if root() is model_patcher.model:
                cleanup.detach()
                _eject_runtime_bypass(runtime, inner, owner)
                del active[owner]

        runtime = type(manager)()
        for key, (adapter, strength) in manager.adapters.items():
            runtime.add_adapter(key, _copy_runtime_adapter(adapter), strength=strength)
        _trace_lokr_calls(runtime)
        inner = tuple(runtime.create_injections(model_patcher.model))
        if runtime.get_hook_count() != expected:
            raise RuntimeError(
                f"Donut bypass sampling model has {runtime.get_hook_count()}/{expected} planned hooks"
            )
        try:
            for injection in inner:
                injection.inject(model_patcher)
        except Exception:
            _eject_runtime_bypass(runtime, inner, model_patcher)
            raise
        cleanup = weakref.finalize(model_patcher, _eject_runtime_bypass, runtime, inner)
        cleanup.atexit = False
        active[model_patcher] = (weakref.ref(model_patcher.model), runtime, inner, cleanup)
        if not reported:
            print(f"[DonutApplyLoRAStack] Experimental bypass activated {expected} forward hook(s) on the sampling model")
            reported = True

    def eject(model_patcher):
        entry = active.pop(model_patcher, None)
        if entry is not None:
            _, runtime, inner, cleanup = entry
            cleanup.detach()
            _eject_runtime_bypass(runtime, inner, model_patcher)

    return [injection_type(inject=inject, eject=eject)]


def _apply_bypass_applications(model, applications):
    """Apply block-weighted components as forward adapters.

    ``applications`` contains ``(lora, strength, block_vector)`` tuples.  Main
    and fused-text components can therefore share one injection manager while
    retaining the exact ratios produced by Donut's block-weight loader.

    The upstream bypass manager stores one adapter per model key. Overlapping
    additive adapters are therefore wrapped in one composite that sums their
    independently scaled h(x) contributions. Adapters with output transforms
    or custom bypass-forward semantics remain guarded because their ordering
    cannot be represented by a simple additive sum.
    """
    patches_by_key = {}

    for lora, strength, block_vector in applications:
        block_weights, _, _ = LoraLoaderBlockWeight.load_lbw(
            model,
            None,
            lora,
            inverse=False,
            seed=0,
            A=1.0,
            B=1.0,
            block_vector=block_vector,
        )
        for key, (patch_data, ratio) in block_weights.items():
            final_strength = float(strength) * float(ratio)
            if final_strength == 0.0:
                continue
            patches_by_key.setdefault(key, []).append((patch_data, final_strength))

    return _apply_bypass_components(model, patches_by_key)


def _apply_bypass_components(model, patches_by_key):
    manager_type = getattr(comfy_weight_adapter, "BypassInjectionManager", None)
    existing_injections = getattr(model, "injections", {})
    # Rebuild our own additive adapter group when a downstream loader adds a LoRA.
    if existing_injections.get("donut_bypass_lora") and not getattr(model, "is_injected", False):
        from .donut_bypass_materialization import get_bypass_components, BYPASS_ATTACHMENT_KEY
        previous = get_bypass_components(model)
        if previous:
            model = model.clone()
            model.remove_injections("donut_bypass_lora")
            model.remove_attachments(BYPASS_ATTACHMENT_KEY)
            combined = {key: list(value) for key, value in previous.items()}
            for key, value in patches_by_key.items():
                combined.setdefault(key, []).extend(value)
            patches_by_key = combined
            existing_injections = getattr(model, "injections", {})
    force_regular_reason = None
    if _WEIGHT_ADAPTER_BASE is None or manager_type is None:
        force_regular_reason = "comfy.weight_adapter bypass support is unavailable"
    elif getattr(model, "is_injected", False) or any(value for key, value in existing_injections.items()
                                                        if key != "donut_krea2_model_merge_bypass"):
        force_regular_reason = "the input model already has runtime injections"

    new_model = model.clone()
    model_keys = set(new_model.model.state_dict().keys())
    merge_info = get_krea2_merge_bypass_info(model)
    if merge_info:
        swapped = {path for path, _key, _ratio in merge_info[1]}
        source_components, main_components = {}, {}
        for key, components in patches_by_key.items():
            target = key[0] if isinstance(key, tuple) else key
            owner = source_components if target.rpartition('.')[0] in swapped else main_components
            owner[key] = components
        if source_components:
            source = _apply_bypass_components(merge_info[0], source_components)
            new_model.set_additional_models(KREA2_MERGE_SOURCE_KEY, [source])
            new_model.set_attachments('donut_lora_source_bypass:' + uuid.uuid4().hex, len(source_components))
            new_model.patches_uuid = uuid.uuid4()
        patches_by_key = main_components

    if force_regular_reason is None:
        adapters_by_key, regular_targets, fallback_reasons = _partition_bypass_targets(
            new_model.model,
            model_keys,
            patches_by_key,
        )
    else:
        adapters_by_key = {}
        regular_targets = patches_by_key
        fallback_reasons = {
            key: (force_regular_reason,)
            for key in patches_by_key
        }
    regular_count = sum(len(components) for components in regular_targets.values())
    if regular_targets:
        add_model_patch_components(new_model, regular_targets)

    if adapters_by_key:
        assert manager_type is not None
        manager = manager_type()
        _register_bypass_adapters(manager, adapters_by_key)
        injections = _make_rebinding_bypass_injections(manager, new_model.model)
        new_model.set_injections("donut_bypass_lora", injections)
    if regular_count:
        print(
            "[DonutApplyLoRAStack] Experimental bypass kept "
            f"{regular_count} patch component(s) across {len(regular_targets)} "
            "target(s) on the ordered regular path"
        )
        for key, reasons in fallback_reasons.items():
            print(
                "[DonutApplyLoRAStack] Regular fallback for "
                f"{key}: {'; '.join(reasons)}"
            )
    if adapters_by_key:
        print(
            "[DonutApplyLoRAStack] Experimental bypass attached "
            f"{sum(len(components) for components in adapters_by_key.values())} adapter "
            f"component(s) across {len(adapters_by_key)} forward hook(s) with "
            "block-vector strengths"
        )
    elif regular_count:
        print(
            "[DonutApplyLoRAStack] Experimental bypass used the compatibility "
            "path for this stack; all model patches remain regular Comfy patches"
        )
    return new_model


class DonutApplyLoRAStackSafe:
    """Drop-in replacement for DonutApplyLoRAStack with optional Krea2 safety."""

    class_type = "CUSTOM"
    aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_stack": ("LORA_STACK",),
                "safe_stack": (["Off", "On"], {
                    "default": "Off",
                    "tooltip": (
                        "Krea2 only. RMS-limits overlapping LoRA strength per block "
                        "so stacked LoRAs cannot collectively exceed one full-strength "
                        "LoRA worth of block energy. Off preserves legacy behaviour."
                    ),
                }),
                "fusion_aware": (list(_FUSION_AWARE_MODES), {
                    "default": "Off",
                    "tooltip": (
                        "Requires the model output from Donut Krea2 Fusion Control. "
                        "Budgets projector LoRA columns against the resolved 12-channel "
                        "projector gains. Use headroom can boost quiet columns; dynamic "
                        "tensor_rms profiles are attenuation-only."
                    ),
                }),
                "max_fusion_boost": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Maximum per-column LoRA boost in Use headroom mode.",
                }),
                "safe_limit": ("FLOAT", {
                    "default": _SAFE_ENERGY_BUDGET,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": (
                        "Maximum combined RMS energy for overlapping Krea2 LoRAs. "
                        "1.0 equals one full-strength LoRA; lower values are stricter. "
                        "Set Safe Stack to Off to disable limiting completely."
                    ),
                }),
            },
            "optional": {
                "execution_mode": (list(_EXECUTION_MODES), {
                    "default": "Comfy patches",
                    "tooltip": (
                        "Global for the connected Donut model path: downstream edit and "
                        "UncensorFix nodes inherit this selection. "
                        "Experimental bypass computes base(x) + LoRA(x) without rebuilding "
                        "quantized model weights. Supports linear LoRA, LoHa, LoKr, and "
                        "ungrouped zero-padded Conv1d/2d/3d LoRA/LoCon. LoHa builds a dense "
                        "delta each forward and can be slower. Unsupported targets stay ordered on "
                        "Comfy's regular path."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "show_help")
    FUNCTION = "apply_stack"
    CATEGORY = "Comfyanonymous/LoRA"

    def apply_stack(
        self,
        model,
        clip,
        lora_stack=None,
        safe_stack="Off",
        fusion_aware="Off",
        max_fusion_boost=2.0,
        safe_limit=_SAFE_ENERGY_BUDGET,
        execution_mode="Comfy patches",
    ):
        help_url = (
            "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/"
            "wiki/LoRA-Nodes#cr-apply-lora-stack"
        )

        execution_mode = resolve_execution_mode(model, execution_mode)
        model = model.clone()
        model.model_options = dict(getattr(model, "model_options", {}))
        publish_execution_mode(model, execution_mode)
        if lora_stack is None or len(lora_stack) == 0:
            return (model, clip, help_url)
        if fusion_aware not in _FUSION_AWARE_MODES:
            raise ValueError(f"Unknown fusion-aware safety mode: {fusion_aware}")
        if execution_mode not in _EXECUTION_MODES:
            raise ValueError(f"Unknown LoRA execution mode: {execution_mode}")
        max_fusion_boost = float(max_fusion_boost)
        if not math.isfinite(max_fusion_boost) or max_fusion_boost < 1.0:
            raise ValueError("max_fusion_boost must be finite and at least 1.0")
        if safe_stack == "On":
            safe_limit = float(safe_limit)
            if not math.isfinite(safe_limit) or safe_limit < 0.0:
                raise ValueError("safe_limit must be finite and non-negative")

        # Preserve DonutApplyLoRAStack's duplicate semantics before doing any
        # safety calculation, otherwise duplicate entries would consume budget
        # even though the apply pass would skip them.
        entries = []
        seen = set()
        for name, mw, cw, bv in lora_stack:
            if mw == 0.0 and cw == 0.0:
                continue
            if name in seen:
                print(
                    f"[DonutApplyLoRAStack] Skipping duplicate LoRA '{name}' "
                    "(already applied this run)"
                )
                continue
            seen.add(name)

            path = folder_paths.get_full_path("loras", name)
            lora = comfy.utils.load_torch_file(path, safe_load=True)
            krea_blocks = _krea2_block_indices(lora)

            # Match the original apply node's automatic vector behaviour.
            vector = bv
            if not vector:
                block_nums = set()
                for key in lora.keys():
                    match = _KREA_BLOCK_RE.search(key)
                    if match:
                        block_nums.add(int(match.group(1)))
                    elif "layers." in key:
                        layer_match = re.search(r"layers\.(\d+)", key)
                        if layer_match:
                            block_nums.add(int(layer_match.group(1)))

                if block_nums:
                    vector = ",".join(["1"] * (max(block_nums) + 2))
                else:
                    vector = ",".join(["1"] * 13)

            has_real_te = _lora_has_real_text_encoder(lora)
            lora_main, lora_text = _split_fused_text(lora)
            fused_text = bool(lora_text) and not has_real_te
            projector_text, other_text = _split_projector_text(lora_text)

            entries.append({
                "name": name,
                "mw": float(mw),
                "cw": float(cw),
                "vector": vector,
                "lora": lora,
                "lora_main": lora_main,
                "lora_text": lora_text,
                "projector_text": projector_text,
                "other_text": other_text,
                "has_projector_text": bool(projector_text),
                "has_other_text": bool(other_text),
                "fused_text": fused_text,
                "is_krea": bool(krea_blocks) and max(krea_blocks) < _KREA_BLOCK_COUNT,
                "is_krea_text": _is_krea2_text_lora(lora),
                "krea_blocks": krea_blocks,
            })

        fusion_metadata = _read_fusion_budget(model) if fusion_aware != "Off" else None
        fusion_aware_active = safe_stack == "On" and fusion_metadata is not None and fusion_aware != "Off"
        if fusion_aware != "Off" and safe_stack != "On":
            print("[DonutApplyLoRAStack] Fusion-aware safety requires safe_stack=On; using legacy behavior")
        elif fusion_aware != "Off" and fusion_metadata is None:
            print(
                "[DonutApplyLoRAStack] Fusion-aware safety found no Fusion Control metadata; "
                "connect Donut Krea2 Fusion Control's model output before this node"
            )

        if safe_stack == "On":
            adjusted_vectors, limited_blocks = _normalise_krea_vectors(
                entries,
                budget=safe_limit,
            )
            for idx, entry in enumerate(entries):
                entry["vector"] = adjusted_vectors[idx]

            if limited_blocks:
                block_labels = ["base" if idx == 0 else str(idx - 1) for idx, _, _ in limited_blocks]
                print(
                    "[DonutApplyLoRAStack] Safe Stack: limited Krea2 block energy "
                    f"in {len(limited_blocks)} bucket(s): {', '.join(block_labels)}"
                )

            if fusion_aware_active:
                other_weights, other_limit = _normalise_fused_text_weights(
                    entries,
                    component="has_other_text",
                    budget=safe_limit,
                )
                projector_gains, dynamic = _nominal_projector_gains(fusion_metadata)
                projector_scales, projector_report = _projector_column_scales(
                    entries,
                    projector_gains,
                    fusion_aware,
                    max_fusion_boost,
                    dynamic=dynamic,
                    budget=safe_limit,
                )

                unsupported = []
                for idx, entry in enumerate(entries):
                    entry["fusion_split"] = bool(entry["fused_text"] and entry["is_krea_text"])
                    entry["effective_cw"] = entry["cw"]
                    entry["effective_other_cw"] = other_weights[idx]
                    entry["effective_projector_cw"] = entry["cw"]
                    entry["effective_projector_text"] = entry["projector_text"]
                    if not (entry["is_krea_text"] and entry["projector_text"]):
                        continue

                    adjusted_projector, transformed = _scale_projector_lora_columns(
                        entry["projector_text"],
                        projector_scales,
                    )
                    if transformed:
                        entry["effective_projector_text"] = adjusted_projector
                    else:
                        # A scalar fallback must use the most restrictive column
                        # scale to keep every projector column within budget.
                        uniform_scale = min(projector_scales)
                        entry["effective_projector_cw"] = entry["cw"] * uniform_scale
                        unsupported.append(entry["name"])

                if other_limit:
                    energy, scale = other_limit
                    print(
                        "[DonutApplyLoRAStack] Safe Stack: limited non-projector "
                        f"Krea2 fused-text energy {energy:.3f}x -> {safe_limit:.3f}x "
                        f"(scale {scale:.3f})"
                    )
                if projector_report:
                    print(
                        "[DonutApplyLoRAStack] Fusion-aware projector budget: "
                        f"LoRA energy={projector_report['raw_energy']:.3f}x, "
                        f"column scales={min(projector_scales):.3f}..{max(projector_scales):.3f}, "
                        f"mode={fusion_aware}"
                    )
                    if projector_report["dynamic"] and fusion_aware == "Use headroom":
                        print(
                            "[DonutApplyLoRAStack] Fusion-aware projector budget: "
                            "tensor_rms is prompt-dependent, so automatic boosts were disabled"
                        )
                if unsupported:
                    print(
                        "[DonutApplyLoRAStack] Fusion-aware projector budget used a "
                        "conservative scalar fallback for unsupported adapter format(s): "
                        + ", ".join(unsupported)
                    )
            else:
                adjusted_text_weights, text_limit = _normalise_fused_text_weights(
                    entries,
                    budget=safe_limit,
                )
                for idx, entry in enumerate(entries):
                    entry["fusion_split"] = False
                    entry["effective_cw"] = adjusted_text_weights[idx]
                if text_limit:
                    energy, scale = text_limit
                    print(
                        "[DonutApplyLoRAStack] Safe Stack: limited Krea2 fused-text "
                        f"energy {energy:.3f}x -> {safe_limit:.3f}x "
                        f"(scale {scale:.3f})"
                    )
        else:
            for entry in entries:
                entry["fusion_split"] = False
                entry["effective_cw"] = entry["cw"]

        unet, text_enc = model, clip
        loader = LoraLoaderBlockWeight()
        bypass_applications = []

        for entry in entries:
            mw = entry["mw"]
            lora = entry["lora"]
            lora_main = entry["lora_main"]
            lora_text = entry["lora_text"]
            fused_text = entry["fused_text"]
            vector = entry["vector"]

            # 1) block-weighted diffusion-model merge.
            merge_main = lora_main if fused_text else lora
            if mw != 0.0 and merge_main:
                if execution_mode == "Experimental bypass":
                    bypass_applications.append((merge_main, mw, vector))
                else:
                    unet, _, _ = loader.load_lora_for_models(
                        unet,
                        None,
                        merge_main,
                        strength_model=mw,
                        strength_clip=0.0,
                        inverse=False,
                        seed=0,
                        A=1.0,
                        B=1.0,
                        block_vector=vector,
                    )

            # 2) text handling, matching the original DonutApplyLoRAStack.
            if fused_text:
                text_applications = (
                    (
                        (entry["other_text"], entry["effective_other_cw"]),
                        (entry["effective_projector_text"], entry["effective_projector_cw"]),
                    )
                    if entry["fusion_split"]
                    else ((lora_text, entry["effective_cw"]),)
                )
                for text_lora, text_strength in text_applications:
                    if not text_lora or text_strength == 0.0:
                        continue
                    if execution_mode == "Experimental bypass":
                        bypass_applications.append(
                            (text_lora, text_strength, _TEXT_MERGE_VECTOR)
                        )
                    else:
                        unet, _, _ = loader.load_lora_for_models(
                            unet,
                            None,
                            text_lora,
                            strength_model=text_strength,
                            strength_clip=0.0,
                            inverse=False,
                            seed=0,
                            A=1.0,
                            B=1.0,
                            block_vector=_TEXT_MERGE_VECTOR,
                        )
            elif entry["effective_cw"] != 0.0:
                _, text_enc = comfy.sd.load_lora_for_models(
                    unet,
                    text_enc,
                    lora,
                    0.0,
                    entry["effective_cw"],
                )

        if execution_mode == "Experimental bypass" and bypass_applications:
            unet = _apply_bypass_applications(model, bypass_applications)

        return (unet, text_enc, help_url)


NODE_CLASS_MAPPINGS = {
    "DonutApplyLoRAStack": DonutApplyLoRAStackSafe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutApplyLoRAStack": "Donut Apply LoRA Stack",
}

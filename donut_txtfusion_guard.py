"""Experimental checkpoint-referenced txtfusion residual guard.

A separate safetensors reference, never a snapshot of live patched weights.
Only attention/MLP contributions are scaled, before their residual additions.
The checkpoint projector, incoming conditioning, body and NAG math are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
from pathlib import Path
import re

import torch

LOGGER = logging.getLogger(__name__)
NAG_KEY = "krea2_normalized_attention_guidance"
EXPERIMENT_KEY = "donut_nag_text_energy_experiment"
PREFIX = "diffusion_model.txtfusion."
BYPASS_KEY = "donut_bypass_lora_components_v1"
_COMPONENT = re.compile(r"^(layerwise_blocks|refiner_blocks)\.(\d+)\.(attn|mlp)\.")
_PLAIN_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _rms(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2 or not value.is_floating_point() or value.numel() == 0:
        raise ValueError("txtfusion guard requires nonempty floating-point batched tensors")
    return value.float().square().mean(tuple(range(1, value.ndim)), keepdim=True).sqrt()


def match_contribution(reference, patched, *, batch_size, max_gain=4.0):
    """One RMS ratio per original sample (not per flattened text token).

    Returning the scaled *contribution*, not the residual sum, is intentional.
    Zero-energy cases stay unchanged. Nonfinite input fails instead of hiding it.
    """
    if reference.shape != patched.shape or reference.device != patched.device:
        raise ValueError("Reference and patched contributions must have matching shape/device")
    if batch_size < 1 or patched.shape[0] % batch_size:
        raise ValueError("Invalid original batch size for txtfusion contribution")
    if not math.isfinite(max_gain) or max_gain < 1:
        raise ValueError("max_gain must be finite and at least 1")
    ref = reference.reshape(batch_size, -1)
    out = patched.reshape(batch_size, -1)
    r0, r1 = _rms(ref), _rms(out)
    if not bool((torch.isfinite(r0).all() & torch.isfinite(r1).all()).item()):
        raise RuntimeError("Nonfinite txtfusion contribution; guard cannot repair NaN/Inf")
    ratio = torch.where((r0 > 1e-12) & (r1 > 1e-12), r0 / r1.clamp_min(1e-12), torch.ones_like(r1))
    gain = ratio.clamp(1.0 / max_gain, max_gain)
    adjusted = (out.float() * gain).to(patched.dtype).reshape_as(patched)
    if not bool(torch.isfinite(adjusted).all().item()):
        raise RuntimeError("txtfusion guard overflow after restoring output dtype")
    return adjusted, r0.detach(), r1.detach(), gain.detach(), ratio.ne(gain).sum().item()


def _looks_like_weight_adapter(value) -> bool:
    """Distinguish LoRA/adapter patches from ordinary merge patch payloads."""
    return hasattr(value, "weights") or hasattr(value, "loaded_keys")


def active_adapter_keys(model) -> set[str]:
    """Return active txtfusion adapter keys on one patcher without merge patches."""
    keys = set()
    for key, entries in getattr(model, "patches", {}).items():
        if not isinstance(key, str):
            continue
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            strength, adapter = float(entry[0]), entry[1]
            if strength != 0.0 and _looks_like_weight_adapter(adapter):
                keys.add(key)
                break
    getter = getattr(model, "get_attachment", None)
    bypass = getter(BYPASS_KEY) if callable(getter) else None
    if bypass is not None and not isinstance(bypass, dict):
        raise RuntimeError("Unrecognized Donut bypass metadata")
    if "donut_bypass_lora" in getattr(model, "injections", {}) and not bypass:
        raise RuntimeError("Guard requires recorded Donut bypass adapters; rebuild the LoRA model after restart")
    for key, entries in (bypass or {}).items():
        if any(float(strength) != 0 for _, strength in entries):
            keys.add(key)
    return {key for key in keys if key.startswith(PREFIX)}


def _components_from_keys(keys) -> set[str]:
    result = set()
    for key in keys:
        match = _COMPONENT.match(key[len(PREFIX):]) if key.startswith(PREFIX) else None
        if match:
            result.add(".".join(match.groups()))
    return result


def affected_components(model) -> set[str]:
    """Find active native or recorded Donut bypass adapters on one patcher."""
    return _components_from_keys(active_adapter_keys(model))


def _merge_support(model):
    """Resolve Donut's runtime model2 swaps without accepting unknown injections.

    Returns (source_model_or_none, plans, ownership) where ownership maps each
    guarded txtfusion attention/MLP component to "primary" or "retained model2".
    A component is source-owned only when all of its linear submodules are hard
    swapped; mixed ownership is rejected because one checkpoint reference cannot
    represent that component faithfully.
    """
    try:
        from .donut_krea2_merge_serialization import (
            KREA2_MERGE_INJECTION_KEY, get_krea2_merge_bypass_info,
        )
    except ImportError:
        from donut_krea2_merge_serialization import (
            KREA2_MERGE_INJECTION_KEY, get_krea2_merge_bypass_info,
        )

    injections = getattr(model, "injections", {})
    injection_keys = set(injections) if isinstance(injections, dict) else set()
    unknown = injection_keys - {"donut_bypass_lora", KREA2_MERGE_INJECTION_KEY}
    if unknown:
        raise ValueError(
            "Guard does not support unrelated model injections: "
            + ", ".join(sorted(map(str, unknown)))
        )

    info = get_krea2_merge_bypass_info(model) if KREA2_MERGE_INJECTION_KEY in injection_keys else None
    if info is None:
        return None, (), {}

    source, plans, _ = info
    plan_paths = {module_path for module_path, _key, ratio in plans if float(ratio) == 0.0}
    linear_suffixes = {
        "attn": ("wq", "wk", "wv", "gate", "wo"),
        "mlp": ("gate", "up", "down"),
    }
    ownership = {}
    candidates = affected_components(model) | affected_components(source)
    for component in candidates:
        kind = component.rsplit(".", 1)[-1]
        expected = {PREFIX + component + "." + suffix for suffix in linear_suffixes[kind]}
        swapped = expected & plan_paths
        if swapped and swapped != expected:
            missing = ", ".join(sorted(expected - swapped))
            raise ValueError(
                "Checkpoint txtfusion guard cannot represent a partially swapped "
                f"merge component {component!r}; unswapped linear(s): {missing}"
            )
        ownership[component] = "retained model2" if swapped == expected else "primary"
    return source, tuple(plans), ownership


def load_reference_states(path, components, *, opener=None):
    """Read only selected txtfusion submodules from an explicit checkpoint.

    Initial version accepts ordinary FP16/BF16/FP32 txtfusion tensors. Quantized
    txtfusion (including FP8 scales/metadata) is rejected, never guessed. The
    remainder of the diffusion checkpoint may be quantized independently.
    """
    from safetensors import safe_open

    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Reference must be a safetensors diffusion checkpoint")
    before = path.stat()
    states = {}
    digest = hashlib.sha256()
    with (opener or safe_open)(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        candidates = [p for p in ("txtfusion.", "diffusion_model.txtfusion.", "model.diffusion_model.txtfusion.")
                      if all(any(k.startswith(p + c + ".") for k in keys) for c in components)]
        if len(candidates) != 1:
            raise ValueError("Reference has missing/ambiguous native Krea2 txtfusion keys")
        prefix = candidates[0]
        metadata = handle.metadata() or {}
        quantization = metadata.get("_quantization_metadata")
        if quantization:
            layers = json.loads(quantization).get("layers", {})
            for layer in layers:
                canonical = layer.removeprefix("model.").removeprefix("diffusion_model.")
                if any(canonical.startswith("txtfusion." + c + ".") for c in components):
                    raise ValueError("Quantized txtfusion metadata is not supported by this experimental guard")
        for component in sorted(components):
            stem = prefix + component + "."
            state = {}
            for key in sorted(k for k in keys if k.startswith(stem)):
                short = key[len(stem):]
                if any(token in short for token in ("scale_weight", "weight_scale", "input_scale", "quant", "scale_input")):
                    raise ValueError("Quantized txtfusion reference is not supported by this experimental guard")
                value = handle.get_tensor(key)
                if value.dtype not in _PLAIN_DTYPES or not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"Unsupported/nonfinite reference tensor: {key} ({value.dtype})")
                # Force independent ownership, including memory-mapped safetensors.
                value = value.detach().clone()
                digest.update(key.encode()); digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.view(torch.uint8).numpy().tobytes())
                state[short] = value
            if not state:
                raise ValueError(f"Empty checkpoint reference: {component}")
            states[component] = state
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Reference checkpoint changed while reading")
    return states, digest.hexdigest()


def make_reference(live, state, kind):
    """Instantiate Comfy's native component, populated exclusively from the file.

    Fresh CPU manual-cast parameters avoid retaining extra reference weights on
    GPU between calls. No live module is copied or temporarily unpatched.
    """
    import comfy.ops
    from comfy.ldm.krea2.model import Attention, SwiGLU

    ops = comfy.ops.manual_cast
    if kind == "attn":
        ref = Attention(live.wq.in_features, live.heads, live.kvheads,
                        bias=live.wq.bias is not None, device="meta", operations=ops)
    else:
        # Use the native nonlinear forward, but infer the checkpoint's exact
        # intermediate width instead of assuming a multiplier/rounding recipe.
        ref = SwiGLU.__new__(SwiGLU)
        torch.nn.Module.__init__(ref)
        for name in ("gate", "up", "down"):
            source = getattr(live, name)
            setattr(ref, name, ops.Linear(source.in_features, source.out_features,
                                         bias=source.bias is not None, device="meta"))
    expected = ref.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
        raise ValueError(f"Reference checkpoint does not match the live {kind} component")
    ref.load_state_dict(state, strict=True, assign=True)
    ref.requires_grad_(False)
    ref.eval()
    return ref


def validate_structure(fusion):
    from comfy.ldm.krea2.model import Attention, SwiGLU, TextFusionBlock, TextFusionTransformer

    if type(fusion) is not TextFusionTransformer:
        raise ValueError("Guard supports native Krea2 txtfusion only")
    for module in [fusion, *fusion.layerwise_blocks, *fusion.refiner_blocks]:
        if "forward" in module.__dict__ or module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("Custom txtfusion/block forwards or hooks are not supported by the guard")
    for block in [*fusion.layerwise_blocks, *fusion.refiner_blocks]:
        if type(block) is not TextFusionBlock or type(block.attn) is not Attention or type(block.mlp) is not SwiGLU:
            raise ValueError("Guard requires native Krea2 attention/MLP blocks")
    for name, parameter in fusion.named_parameters():
        # Do not silently substitute manual-cast reference math for packed or
        # FP8 txtfusion kernels. Body-only quantization does not hit this check.
        if type(parameter) not in (torch.Tensor, torch.nn.Parameter) or parameter.dtype not in _PLAIN_DTYPES:
            raise ValueError(f"Quantized/dynamic txtfusion parameter unsupported by guard: {name}")


@dataclass
class GuardRun:
    references: dict
    digest: str
    max_gain: float = 4.0
    forward_calls: int = 0
    contribution_calls: int = 0
    reports: dict = field(default_factory=dict)

    def contribution(self, name, module, value, batch_size, **kwargs):
        patched = module(value, **kwargs)
        reference_module = self.references.get(name)
        if reference_module is None:
            return patched
        with torch.no_grad():
            reference = reference_module(value, **kwargs)
        adjusted, before, after, gain, clipped = match_contribution(
            reference, patched, batch_size=batch_size, max_gain=self.max_gain)
        self.contribution_calls += 1
        if self.forward_calls <= 2:
            # NAG normally calls positive text first and negative text second.
            # Label call order rather than infer semantic identity from shapes.
            self.reports.setdefault(name, {})[f"text_call_{self.forward_calls}"] = {
                "checkpoint_rms": before.flatten().cpu().tolist(),
                "patched_rms": after.flatten().cpu().tolist(),
                "gain": gain.flatten().cpu().tolist(), "clipped": clipped,
            }
        return adjusted

    def close(self):
        self.references.clear()


class GuardedFusion:
    """Native text-fusion order with component correction before residual add."""
    def __init__(self, fusion, run):
        self.fusion, self.run = fusion, run

    def __call__(self, value, mask=None, transformer_options=None):
        options = transformer_options or {}
        batch, length, taps, width = value.shape
        self.run.forward_calls += 1
        x = value.reshape(batch * length, taps, width)
        for index, block in enumerate(self.fusion.layerwise_blocks):
            x = x.contiguous()
            stem = f"layerwise_blocks.{index}"
            x = x + self.run.contribution(stem + ".attn", block.attn, block.prenorm(x), batch,
                                           mask=None, transformer_options=options)
            x = x + self.run.contribution(stem + ".mlp", block.mlp, block.postnorm(x), batch)
        # Same layout/strides as native einops rearrange; projector is unchanged.
        x = x.reshape(batch, length, taps, width).permute(0, 1, 3, 2)
        x = self.fusion.projector(x).squeeze(-1)
        for index, block in enumerate(self.fusion.refiner_blocks):
            stem = f"refiner_blocks.{index}"
            x = x + self.run.contribution(stem + ".attn", block.attn, block.prenorm(x), batch,
                                           mask=mask, transformer_options=options)
            x = x + self.run.contribution(stem + ".mlp", block.mlp, block.postnorm(x), batch)
        return x


class _ModelView:
    def __init__(self, model, fusion):
        self._model, self.txtfusion = model, fusion

    def __getattr__(self, name):
        return getattr(self._model, name)


class _ExecutorView:
    def __init__(self, executor, model):
        self._executor, self.class_obj = executor, model

    def __getattr__(self, name):
        return getattr(self._executor, name)

    def __call__(self, *args, **kwargs):
        return self._executor(*args, **kwargs)


def install_guard(model, reference_path, *, reference_factory=make_reference):
    """Replace one NAG callable on a clone without changing upstream modules."""
    import comfy.patcher_extension as pe

    kind = pe.WrappersMP.DIFFUSION_MODEL
    table = getattr(model, "wrappers", {}).get(kind, {})
    selected = table.get(NAG_KEY, ())
    if len(selected) != 1 or EXPERIMENT_KEY in table or "krea2_edit_normalized_attention_guidance" in table:
        raise ValueError("Internal guard requires exactly one standard non-edit NAG wrapper; disable other NAG experiments")
    budget = model.model_options.get("transformer_options", {}).get("donut_krea2_fusion_budget", {})
    if budget.get("nag_text_energy_compensation") or budget.get("nag_batch_txtfusion"):
        raise ValueError("Disable midpoint RMS compensation and txtfusion batching for this isolated experiment")
    source, _merge_plans, ownership = _merge_support(model)
    primary_components = affected_components(model)
    source_components = affected_components(source) if source is not None else set()
    if source is not None:
        # Adapters already present on model2 only matter for components whose
        # forwards are actually swapped to model2. Primary-side bypass adapters
        # may intentionally wrap those swapped forwards, so their storage
        # location does not determine checkpoint ownership.
        source_components = {
            component for component in source_components
            if ownership.get(component, "primary") == "retained model2"
        }
    components = primary_components | source_components
    if not components:
        LOGGER.info("[Donut txtfusion guard] no active attention/MLP adapters; unchanged")
        return model, None
    if source is not None:
        owners = {ownership.get(component, "primary") for component in components}
        if len(owners) != 1:
            raise ValueError(
                "Checkpoint txtfusion guard needs one effective txtfusion checkpoint, "
                "but active adapter components span both primary and retained model2"
            )
        owner = next(iter(owners))
        LOGGER.info("[Donut txtfusion guard] merge-aware reference owner: %s", owner)
    if reference_path is None:
        raise ValueError("Select the same checkpoint in the V5 txtfusion reference control before enabling the guard")
    fusion = model.get_model_object("diffusion_model").txtfusion
    validate_structure(fusion)
    states, digest = load_reference_states(reference_path, components)
    references = {name: reference_factory(fusion.get_submodule(name), state, name.rsplit(".", 1)[1])
                  for name, state in states.items()}
    run = GuardRun(references, digest)
    original = selected[0]

    def wrapper(executor, *args, **kwargs):
        live_fusion = executor.class_obj.txtfusion
        # An outer runtime wrapper may have changed a forward after install.
        # Refuse that path instead of silently bypassing its custom operation.
        validate_structure(live_fusion)
        view = _ModelView(executor.class_obj, GuardedFusion(live_fusion, run))
        return original(_ExecutorView(executor, view), *args, **kwargs)

    patched = model.clone()
    # Replace in-place in the *clone's registry* to retain wrapper ordering.
    patched.wrappers = {k: {key: list(values) for key, values in entries.items()}
                        for k, entries in patched.wrappers.items()}
    patched.wrappers[kind][NAG_KEY] = [wrapper]
    # A previously sampled clone may contain the old prepared runtime copy too.
    options = dict(patched.model_options)
    transformer = dict(options.get("transformer_options", {}))
    wrappers = {k: dict(v) for k, v in transformer.get("wrappers", {}).items()}
    if NAG_KEY in wrappers.get(kind, {}):
        wrappers[kind].pop(NAG_KEY)
    if wrappers:
        transformer["wrappers"] = wrappers
    options["transformer_options"] = transformer
    patched.model_options = options
    LOGGER.info("[Donut txtfusion guard] base-pass only; reference=%s; sha256=%s; components=%s",
                Path(reference_path).name, digest, ",".join(sorted(components)))
    return patched, run

"""Model-local txtfusion RMS guard, independent of NAG and sampler selection.

The reference is an independent reconstruction of the effective checkpoint/merge
BEFORE adapters. Native component forward hooks constrain attention/MLP residual
contributions and the projector output. No diffusion forward is replaced.
"""
from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
import copy
from dataclasses import dataclass, field, fields, is_dataclass, replace
import logging
import math
import uuid
import weakref

import torch

LOGGER = logging.getLogger(__name__)
KEY = "donut_txtfusion_model_rms_guard_v1"
PREFIX = "diffusion_model.txtfusion"
_MERGE_KEY = "donut_krea2_model_merge_bypass"
_BYPASS_KEY = "donut_bypass_lora"
_CURRENT = ContextVar("donut_txtfusion_rms_session", default=None)
_SCOPES = ContextVar("donut_txtfusion_rms_scopes", default=())


def _cpu_copy(tensor):
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
        raise ValueError("Cannot establish txtfusion reference from unavailable/meta weights; reload the source model")
    # Kitchen's clone() copies qdata but shares Params. Copy scales, block
    # scales and other tensor metadata too, or live requantization could change
    # the supposedly independent reference. No format conversion is involved.
    if callable(getattr(tensor, "_copy_with", None)) and torch.is_tensor(getattr(tensor, "_qdata", None)):
        return tensor._copy_with(qdata=_cpu_copy(tensor._qdata), params=_copy_metadata(tensor._params))
    return tensor.detach().to(device="cpu", copy=True).clone()


def _copy_metadata(value):
    if isinstance(value, torch.Tensor):
        return _cpu_copy(value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: _copy_metadata(getattr(value, f.name)) for f in fields(value) if f.init})
    if isinstance(value, dict):
        return {k: _copy_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_metadata(v) for v in value)
    return copy.deepcopy(value)


def _checkpoint_patches(entries):
    """Keep standard/Donut merge recipes; discard added adapters recursively.

    Core model merging represents a source as [(weight, converter), *patches].
    All other entries (LoRA/LoKr/DoRA objects, diff/set adapters) are additions,
    not the checkpoint merge. Removing an adapter removes its model-scale too.
    """
    clean = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 5:
            raise ValueError("Unrecognized txtfusion patch recipe; cannot establish its checkpoint reference")
        strength, payload, model_scale, offset, function = entry
        if not isinstance(payload, list):
            continue
        if not payload or not isinstance(payload[0], (tuple, list)) or len(payload[0]) != 2:
            raise ValueError("Malformed txtfusion model-merge source")
        weight, converter = payload[0]
        if not isinstance(weight, torch.Tensor) or not callable(converter):
            raise ValueError("Malformed txtfusion checkpoint weight/converter")
        clean.append((strength, [(weight, converter), *_checkpoint_patches(payload[1:])],
                      model_scale, offset, function))
    return clean


def _materialize(recipe, key):
    """Use core merge arithmetic, on an independent tensor, without live unpatching."""
    if not recipe or len(recipe[0]) != 2:
        raise ValueError(f"Missing checkpoint provenance for {key}")
    original, convert = recipe[0]
    base = _cpu_copy(original)
    merges = _checkpoint_patches(recipe[1:])
    if not merges:
        return base
    import comfy.lora
    import comfy.float
    import comfy.utils

    floating = convert(base.to(dtype=torch.float32, copy=True), inplace=True)
    if hasattr(floating, "dequantize") and (getattr(floating, "is_quantized", False)
                                           or type(floating).__name__ == "QuantizedTensor"):
        floating = floating.dequantize()
    # calculate_weight may recursively convert source weights; it only receives
    # copies and never changes a live model parameter.
    merged = comfy.lora.calculate_weight(merges, floating, key, intermediate_dtype=torch.float32)
    seed = comfy.utils.string_to_seed(key)
    if hasattr(base, "requantize_from_float"):
        return base.requantize_from_float(merged, scale="recalculate", stochastic_rounding=seed)
    if getattr(base, "is_quantized", False):
        raise ValueError("This packed merge format has no core requantization method")
    return comfy.float.stochastic_rounding(merged, base.dtype, seed=seed)


def _recipes(model):
    # Existing Donut helper repairs core's quantization export-key mismatch and
    # honors backup/hook-backup precedence. Do not substitute live state_dict.
    try:
        from .DonutModelMergeKrea2 import _get_merge_key_patches
    except ImportError:
        from DonutModelMergeKrea2 import _get_merge_key_patches
    return _get_merge_key_patches(model, PREFIX + ".")


def _merge_info(model):
    try:
        from .donut_krea2_merge_serialization import get_krea2_merge_bypass_info
    except ImportError:
        from donut_krea2_merge_serialization import get_krea2_merge_bypass_info
    return get_krea2_merge_bypass_info(model)


# All of these are runtime loading/adapter caches, not checkpoint state. They
# must not make the reference fault the live model's patched DynamicVRAM region.
_RUNTIME_FIELDS = {
    "forward", "_compiled_call_impl", "_v", "_v_signature", "_v_weight", "_v_bias",
    "_prefetch", "_pin_state", "weight_lowvram_function", "bias_lowvram_function",
    "prev_comfy_cast_weights", "comfy_patched_weights", "_donutGuardRuntime",
}


def _copy_shell(module):
    result = copy.copy(module)
    result.__dict__ = module.__dict__.copy()
    for name in _RUNTIME_FIELDS:
        result.__dict__.pop(name, None)
    result._parameters = OrderedDict()
    result._buffers = OrderedDict()
    result._modules = OrderedDict()
    for name, value in list(result.__dict__.items()):
        if "hook" in name and isinstance(value, (dict, OrderedDict)):
            result.__dict__[name] = type(value)()
        elif name in ("weight_function", "bias_function"):
            result.__dict__[name] = []
        elif isinstance(value, torch.Tensor):
            # E.g. calibrated FP8 input scales on custom operation classes.
            result.__dict__[name] = _cpu_copy(value)
    result._non_persistent_buffers_set = set(module._non_persistent_buffers_set)
    if hasattr(module, "comfy_cast_weights"):
        result.comfy_cast_weights = True
    result.training = False
    return result


def component_names(fusion):
    names = []
    for group in ("layerwise_blocks", "refiner_blocks"):
        blocks = getattr(fusion, group, None)
        if blocks is None:
            raise ValueError("Txtfusion RMS guard needs Krea2 text-fusion blocks")
        for index, block in enumerate(blocks):
            for part in ("attn", "mlp"):
                if not isinstance(getattr(block, part, None), torch.nn.Module):
                    raise ValueError(f"Missing txtfusion {group}.{index}.{part}")
                names.append(f"{group}.{index}.{part}")
    if not isinstance(getattr(fusion, "projector", None), torch.nn.Module):
        raise ValueError("Missing Krea2 txtfusion projector")
    return tuple(names) + ("projector",)


def capture_reference(model, *, recipe_reader=None, merge_reader=None):
    """Independent effective merged checkpoint, including per-linear hard swaps.

    Called during node execution, before new adapters are applied. Existing
    native adapters are excluded using core's unpatched-weight/merge recipes;
    bypass hooks are not copied. Primary/source mixtures and partial regular
    merges are resolved per parameter rather than choosing one whole checkpoint.
    """
    recipe_reader = recipe_reader or _recipes
    merge_reader = merge_reader or _merge_info
    tables, sources = {}, {}
    visiting = set()

    def collect(patcher):
        marker = id(patcher)
        if marker in visiting:
            raise ValueError("Cyclic txtfusion model-merge reference")
        if marker in tables:
            return
        visiting.add(marker)
        unknown = set(getattr(patcher, "injections", {})) - {_MERGE_KEY, _BYPASS_KEY}
        if unknown:
            # Capture cannot pretend arbitrary executable injections are a
            # checkpoint. Later injections (including SDA) are guarded normally.
            raise ValueError("Capture txtfusion RMS reference before these unrecorded model injections: "
                             + ", ".join(sorted(unknown)))
        info = merge_reader(patcher)
        swaps = {}
        if info is not None:
            source, plans, _ = info
            collect(source)
            for path, weight_key, ratio in plans:
                if ratio != 0 or weight_key != path + ".weight":
                    raise ValueError("Invalid txtfusion hard-swap plan")
                if path.startswith(PREFIX + "."):
                    swaps[path] = source
        tables[marker] = recipe_reader(patcher)
        sources[marker] = swaps
        visiting.remove(marker)

    collect(model)

    def resolve(patcher, path):
        # The merge swaps direct modules, not an entire text transformer.
        source = sources[id(patcher)].get(path)
        return resolve(source, path) if source is not None else patcher

    def module_at(patcher, path):
        getter = getattr(patcher, "get_model_object", None)
        if callable(getter):
            return getter(path)
        return patcher.model.get_submodule(path)

    def clone_module(patcher, path):
        owner = resolve(patcher, path)
        module = module_at(owner, path)
        out = _copy_shell(module)
        table = tables[id(owner)]
        for name, parameter in module._parameters.items():
            if parameter is None:
                out._parameters[name] = None
                continue
            key = path + "." + name
            if key not in table:
                raise ValueError(f"No checkpoint reference recipe for {key}")
            value = _materialize(table[key], key)
            out._parameters[name] = torch.nn.Parameter(value, requires_grad=False)
        for name, buffer in module._buffers.items():
            key = path + "." + name
            out._buffers[name] = _materialize(table[key], key) if key in table else _cpu_copy(buffer)
        for name, child in module._modules.items():
            out._modules[name] = None if child is None else clone_module(owner, path + "." + name)
        return out

    reference = clone_module(model, PREFIX)
    component_names(reference)
    return reference


def match_rms(reference, patched, batch, max_gain=4.0):
    if not isinstance(reference, torch.Tensor) or not isinstance(patched, torch.Tensor):
        raise TypeError("Txtfusion component output is not a tensor")
    if reference.shape != patched.shape or reference.device != patched.device:
        raise ValueError("Txtfusion reference/output shape or device mismatch")
    if batch < 1 or patched.numel() == 0 or patched.shape[0] % batch:
        raise ValueError("Invalid original batch grouping in txtfusion")
    r = reference.reshape(batch, -1).float().square().mean(1, keepdim=True).sqrt()
    p = patched.reshape(batch, -1).float().square().mean(1, keepdim=True).sqrt()
    if not bool((torch.isfinite(r).all() & torch.isfinite(p).all()).item()):
        raise RuntimeError("Nonfinite txtfusion contribution; RMS guard will not hide NaN/Inf")
    # A zero base/patch has no meaningful direction to scale. Leave it alone.
    ratio = torch.where((r > 1e-12) & (p > 1e-12), r / p.clamp_min(1e-12), torch.ones_like(p))
    gain = ratio.clamp(1.0 / max_gain, max_gain)
    # Return the original object on a true no-op, including an unmodified model.
    if bool(gain.eq(1).all().item()):
        out = patched
    else:
        out = (patched.reshape(batch, -1).float() * gain).to(patched.dtype).reshape_as(patched)
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError("Txtfusion guard overflow after restoring compute dtype")
    return out, r.detach(), p.detach(), gain.detach()


class ReferenceModel(torch.nn.Module):
    """Only txtfusion is duplicated; core's additional-model manager owns loading."""
    def __init__(self, fusion, dtype):
        super().__init__()
        self.txtfusion, self.dtype = fusion, dtype
    def get_dtype(self):
        return self.dtype
    def memory_required(self, input_shape):
        return 0  # Weights are separately budgeted by ModelPatcher.model_size().


@dataclass(eq=False)
class Session:
    patcher_ref: object
    reference: object
    identity: str
    max_gain: float
    handles: list = field(default_factory=list)
    frames: ContextVar = field(default_factory=lambda: ContextVar("txtfusion_frames", default=()))
    calls: int = 0
    contributions: int = 0
    reports: dict = field(default_factory=dict)
    token: object = None
    closed: bool = False
    finalizer: object = None

    def start(self, fusion):
        names = component_names(fusion)
        if names != component_names(self.reference):
            raise ValueError("Txtfusion structure changed after checkpoint reference capture")
        self.token = _CURRENT.set(self)
        try:
            self.handles.append(fusion.register_forward_pre_hook(self.enter, with_kwargs=True))
            # always_call unwinds a failed/cancelled txtfusion call too.
            self.handles.append(fusion.register_forward_hook(self.leave, with_kwargs=True, always_call=True))
            for name in names:
                module = fusion.get_submodule(name)
                ref = self.reference.get_submodule(name)
                self.handles.append(module.register_forward_hook(self.hook(name, ref), with_kwargs=True))
        except BaseException:
            self.close()
            raise

    def enter(self, module, args, kwargs):
        patcher = self.patcher_ref()
        current = getattr(getattr(patcher, "model", None), "current_patcher", patcher)
        active = _CURRENT.get() is self and (current is None or current is patcher)
        options = kwargs.get("transformer_options")
        if options is None and len(args) > 2:
            options = args[2]
        if isinstance(options, dict) and options.get(KEY, self.identity) != self.identity:
            active = False
        value = args[0] if args else kwargs.get("x")
        batch = value.shape[0] if active and torch.is_tensor(value) else None
        self.frames.set(self.frames.get() + (({"batch": batch, "seen": set()} if batch is not None else None),))
        if batch is not None:
            self.calls += 1

    def leave(self, module, args, kwargs, output):
        frames = self.frames.get()
        if frames:
            frame = frames[-1]
            self.frames.set(frames[:-1])
            if frame is not None and output is not None:
                missing = set(component_names(module)) - frame["seen"]
                if missing:
                    raise RuntimeError("Txtfusion RMS guard did not observe these components: " + ", ".join(sorted(missing)))

    def hook(self, name, reference):
        def apply(module, args, kwargs, output):
            frames = self.frames.get()
            if not frames or frames[-1] is None or self.closed:
                return output
            with torch.no_grad():
                base = reference(*args, **kwargs)
            result, r, p, gain = match_rms(base, output, frames[-1]["batch"], self.max_gain)
            frames[-1]["seen"].add(name)
            self.contributions += 1
            if self.calls <= 2:
                self.reports[f"{self.calls}:{name}"] = {
                    "base_rms": r.flatten().cpu().tolist(),
                    "patched_rms": p.flatten().cpu().tolist(),
                    "gain": gain.flatten().cpu().tolist(),
                }
            return result
        return apply

    def close(self):
        if self.closed:
            return
        self.closed = True
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.frames.set(())
        if self.token is not None:
            try:
                _CURRENT.reset(self.token)
            except ValueError:
                # Finalizers can run outside the inference Context. Hooks have
                # already been removed; do not disturb another thread's state.
                pass
            self.token = None
        if self.finalizer is not None:
            self.finalizer.detach()
        self.reference = None


class Binding:
    def __init__(self, reference_patcher, max_gain):
        self.reference_patcher = reference_patcher
        self.max_gain = max_gain
        self.identity = uuid.uuid4().hex
        self.sessions = weakref.WeakKeyDictionary()
        self.last_report = None

    def pre_run(self, patcher):
        self.cleanup(patcher)
        sources = patcher.get_additional_models_with_key(KEY)
        if len(sources) != 1:
            raise RuntimeError("Txtfusion checkpoint reference was lost while cloning the model")
        session = Session(weakref.ref(patcher), sources[0].model.txtfusion, self.identity, self.max_gain)
        self.sessions[patcher] = session
        fusion = patcher.get_model_object(PREFIX)
        session.start(fusion)
        session.finalizer = weakref.finalize(patcher, session.close)
        scopes = _SCOPES.get()
        if scopes:
            scopes[-1].append(session)
        LOGGER.info("[Donut txtfusion RMS] model-level guard active; reference=%s; components=%d; NAG-independent",
                    self.identity, len(component_names(fusion)))

    def cleanup(self, patcher, *unused):
        session = self.sessions.pop(patcher, None)
        if session is None:
            return
        # Remove hooks BEFORE reporting, so logging failure cannot leak state.
        session.close()
        self.last_report = {"calls": session.calls, "contributions": session.contributions,
                            "stats": session.reports}
        LOGGER.info("[Donut txtfusion RMS] calls=%d contributions=%d stats=%s",
                    session.calls, session.contributions, session.reports)

    def outer(self, executor, *args, **kwargs):
        # Backup cleanup even when an earlier third-party cleanup callback fails.
        sessions = []
        token = _SCOPES.set(_SCOPES.get() + (sessions,))
        try:
            return executor(*args, **kwargs)
        finally:
            for session in reversed(sessions):
                session.close()
            _SCOPES.reset(token)


def remove_model_guard(model):
    """Remove only this feature from a clone; an unrelated model is a true no-op."""
    if getattr(model, "get_attachment", lambda key: None)(KEY) is None:
        return model
    from comfy.patcher_extension import CallbacksMP, WrappersMP
    out = model.clone()
    for event in (CallbacksMP.ON_PRE_RUN, CallbacksMP.ON_CLEANUP, CallbacksMP.ON_DETACH):
        out.remove_callbacks_with_key(event, KEY)
    out.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, KEY)
    out.remove_additional_models(KEY)
    out.remove_attachments(KEY)
    # Some sampled clones already contain a prepared runtime wrapper copy.
    options = copy.deepcopy(out.model_options) if not isinstance(out.model_options, dict) else dict(out.model_options)
    transformer = dict(options.get("transformer_options", {}))
    transformer.pop(KEY, None)
    wrappers = {kind: dict(entries) for kind, entries in transformer.get("wrappers", {}).items()}
    for entries in wrappers.values():
        entries.pop(KEY, None)
    if "wrappers" in transformer:
        transformer["wrappers"] = wrappers
    options["transformer_options"] = transformer
    out.model_options = options
    return out


def attach_model_guard(model, *, max_gain=4.0, reference_factory=None, patcher_factory=None):
    """Clone-persistent guard for EVERY txtfusion caller; no NAG checks here."""
    if not math.isfinite(max_gain) or max_gain < 1:
        raise ValueError("Txtfusion RMS max gain must be finite and >= 1")
    existing = getattr(model, "get_attachment", lambda key: None)(KEY)
    if isinstance(existing, Binding) and existing.max_gain == max_gain:
        return model
    from comfy.patcher_extension import CallbacksMP, WrappersMP
    from comfy.model_patcher import ModelPatcher

    clean = remove_model_guard(model)
    reference = (reference_factory or capture_reference)(clean)
    dtype = clean.model_dtype() or torch.float32
    root = ReferenceModel(reference, dtype)
    ref_patcher = (patcher_factory or ModelPatcher)(root, load_device=clean.load_device,
                                                  offload_device=clean.offload_device)
    binding = Binding(ref_patcher, max_gain)
    out = clean.clone()
    out.set_attachments(KEY, binding)
    out.set_additional_models(KEY, [ref_patcher])
    out.add_callback_with_key(CallbacksMP.ON_PRE_RUN, KEY, binding.pre_run)
    out.add_callback_with_key(CallbacksMP.ON_CLEANUP, KEY, binding.cleanup)
    out.add_callback_with_key(CallbacksMP.ON_DETACH, KEY, binding.cleanup)
    out.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, binding.outer)
    options = dict(out.model_options)
    transformer = dict(options.get("transformer_options", {}))
    transformer[KEY] = binding.identity
    options["transformer_options"] = transformer
    out.model_options = options
    return out

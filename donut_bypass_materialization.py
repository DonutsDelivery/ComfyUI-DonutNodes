"""Helpers for serializing Donut's experimental bypass LoRA adapters.

Experimental bypass keeps adapter math in forward hooks instead of changing the
ModelPatcher weight state. That is ideal for quantized/low-VRAM inference, but a
plain state-dict save or ModelSubtract cannot see those adapters.

New Donut bypass applications record adapter components as a ModelPatcher
attachment. For compatibility with already-created bypass models, these helpers
can also recover the BypassInjectionManager captured by ComfyUI's
PatcherInjection closures. Consumers then clone the patcher, remove the runtime
injection, and register the exact same adapters on ComfyUI's ordinary patch path.
This lets LazyCastingParam materialize one weight at a time without constructing
a dense full-model delta in memory.
"""

BYPASS_ATTACHMENT_KEY = "donut_bypass_lora_components_v1"
BYPASS_INJECTION_KEY = "donut_bypass_lora"


def _flatten_adapter(adapter, strength):
    """Expand Donut's composite bypass adapter back into its components."""
    components = getattr(adapter, "components", None)
    if isinstance(components, (list, tuple)) and components:
        flattened = []
        valid = True
        for item in components:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                valid = False
                break
            child, child_strength = item
            flattened.append((child, float(strength) * float(child_strength)))
        if valid:
            return flattened
    return [(adapter, float(strength))]


def _get_attached_components(model):
    getter = getattr(model, "get_attachment", None)
    if not callable(getter):
        return {}
    value = getter(BYPASS_ATTACHMENT_KEY)
    return value if isinstance(value, dict) else {}


def _discover_components_from_injections(model):
    """Recover bypass adapters from ComfyUI's injection closure when possible."""
    getter = getattr(model, "get_injections", None)
    if callable(getter):
        injections = getter(BYPASS_INJECTION_KEY) or []
    else:
        injections = getattr(model, "injections", {}).get(BYPASS_INJECTION_KEY, [])

    managers = []
    seen_managers = set()
    for injection in injections:
        for callback_name in ("inject", "eject"):
            callback = getattr(injection, callback_name, None)
            closure = getattr(callback, "__closure__", None)
            if not closure:
                continue
            for cell in closure:
                try:
                    candidate = cell.cell_contents
                except ValueError:
                    continue
                adapters = getattr(candidate, "adapters", None)
                if not isinstance(adapters, dict):
                    continue
                marker = id(candidate)
                if marker in seen_managers:
                    continue
                seen_managers.add(marker)
                managers.append(candidate)

    discovered = {}
    for manager in managers:
        for module_key, adapter_entry in manager.adapters.items():
            if not isinstance(adapter_entry, (list, tuple)) or len(adapter_entry) != 2:
                continue
            adapter, strength = adapter_entry
            weight_key = module_key if str(module_key).endswith(".weight") else f"{module_key}.weight"
            discovered.setdefault(weight_key, []).extend(_flatten_adapter(adapter, strength))
    return discovered


def get_bypass_components(model):
    """Return {weight_key: [(adapter, strength), ...]} for Donut bypass LoRAs."""
    attached = _get_attached_components(model)
    if attached:
        return attached

    # Older bypass models did not persist explicit extraction metadata. The
    # injection manager itself still owns the exact adapter objects, so recover
    # them from its closure rather than pretending the state dict contains them.
    return _discover_components_from_injections(model)


def attach_bypass_components(model, adapters_by_key):
    """Merge bypass components into clone-persistent ModelPatcher metadata."""
    if not adapters_by_key:
        return model

    # Only merge the explicit attachment here. Calling get_bypass_components()
    # would rediscover the same injection manager and duplicate every component.
    merged = {
        key: list(components)
        for key, components in _get_attached_components(model).items()
    }
    for key, components in adapters_by_key.items():
        merged.setdefault(key, []).extend(list(components))

    setter = getattr(model, "set_attachments", None)
    if not callable(setter):
        raise RuntimeError("This ComfyUI ModelPatcher does not support attachments.")
    setter(BYPASS_ATTACHMENT_KEY, merged)
    return model


def install_bypass_recording_patch():
    """Make future Donut bypass results persist their adapter components.

    The existing apply implementation already constructs ComfyUI's
    BypassInjectionManager. Wrapping it after import lets us recover that exact
    manager once and store its adapters as clone-persistent metadata without
    changing the inference implementation itself.
    """
    try:
        from . import DonutSafeApplyLoRAStack as safe_module
    except ImportError:
        try:
            import DonutSafeApplyLoRAStack as safe_module
        except ImportError:
            return False

    original = getattr(safe_module, "_apply_bypass_applications", None)
    if not callable(original):
        return False
    if getattr(original, "_donut_records_bypass_components", False):
        return True

    def wrapped(model, applications):
        result = original(model, applications)
        discovered = _discover_components_from_injections(result)
        if discovered:
            attach_bypass_components(result, discovered)
        return result

    wrapped._donut_records_bypass_components = True
    wrapped._donut_original = original
    safe_module._apply_bypass_applications = wrapped
    return True


def clone_with_bypass_as_regular_patches(model):
    """Clone a model and turn Donut bypass adapters into normal weight patches.

    Callers should preferably invoke this while ``model.use_ejected()`` is active
    if the source model is currently injected. The returned clone no longer owns
    Donut's bypass injection/metadata, so serialization cannot double-apply it.
    """
    components_by_key = get_bypass_components(model)
    if not components_by_key:
        return model

    converted = model.clone()

    remover = getattr(converted, "remove_injections", None)
    if callable(remover):
        remover(BYPASS_INJECTION_KEY)

    attachment_remover = getattr(converted, "remove_attachments", None)
    if callable(attachment_remover):
        attachment_remover(BYPASS_ATTACHMENT_KEY)

    for key, components in components_by_key.items():
        for adapter, strength in components:
            strength = float(strength)
            if strength == 0.0:
                continue
            converted.add_patches({key: adapter}, strength)

    return converted

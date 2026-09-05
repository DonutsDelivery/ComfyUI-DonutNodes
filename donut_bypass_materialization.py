"""Helpers for serializing Donut's experimental bypass LoRA adapters.

Experimental bypass keeps adapter math in forward hooks instead of changing the
ModelPatcher weight state. That is ideal for quantized/low-VRAM inference, but a
plain state-dict save or ModelSubtract cannot see those adapters.

Donut records the lightweight adapter objects on the ModelPatcher as an
attachment. Consumers that need real weights (model saving or LoRA extraction)
can clone the patcher, remove the runtime injection, and register the exact same
adapters on ComfyUI's ordinary patch path. LazyCastingParam can then materialize
one weight at a time without constructing a dense full-model delta in memory.
"""

BYPASS_ATTACHMENT_KEY = "donut_bypass_lora_components_v1"
BYPASS_INJECTION_KEY = "donut_bypass_lora"


def get_bypass_components(model):
    """Return {weight_key: [(adapter, strength), ...]} recorded on a model."""
    getter = getattr(model, "get_attachment", None)
    if not callable(getter):
        return {}
    value = getter(BYPASS_ATTACHMENT_KEY)
    if not isinstance(value, dict):
        return {}
    return value


def attach_bypass_components(model, adapters_by_key):
    """Merge newly attached bypass components into clone-persistent metadata."""
    if not adapters_by_key:
        return model

    merged = {
        key: list(components)
        for key, components in get_bypass_components(model).items()
    }
    for key, components in adapters_by_key.items():
        merged.setdefault(key, []).extend(list(components))

    setter = getattr(model, "set_attachments", None)
    if not callable(setter):
        raise RuntimeError("This ComfyUI ModelPatcher does not support attachments.")
    setter(BYPASS_ATTACHMENT_KEY, merged)
    return model


def clone_with_bypass_as_regular_patches(model):
    """Clone a model and turn recorded bypass adapters into normal weight patches.

    Callers should preferably invoke this while ``model.use_ejected()`` is active
    if the source model is currently injected. The returned clone no longer owns
    Donut's bypass injection/metadata, so serialization cannot double-apply it.
    """
    components_by_key = get_bypass_components(model)
    converted = model.clone()

    if not components_by_key:
        return converted

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

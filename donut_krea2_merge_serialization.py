"""Serialization helpers for Donut Model Merge Krea2 Experimental bypass.

The Krea2 merge node keeps exact ratio=0 model2 swaps as runtime forward hooks.
Those hooks intentionally do not modify model1's stored weights, so a normal
state-dict save would silently serialize model1 for every bypassed component.

This module reconstructs the effective save state without materializing a dense
merged model in VRAM:

* normal and partial-blend keys stay on the final model1 patcher
* exact bypassed modules are taken from the retained model2 source patcher
* all direct module state (weight, bias, quant scales/buffers) is swapped
* later Donut bypass-LoRA adapters can be applied to model2's swapped weights
  before the final state dict is produced

The final composed UNet state dict still uses ComfyUI LazyCastingParam wrappers,
so tensors are materialized one at a time during saving.
"""

KREA2_MERGE_INJECTION_KEY = "donut_krea2_model_merge_bypass"
KREA2_MERGE_SOURCE_KEY = "donut_krea2_model_merge_sources"
KREA2_MERGE_PLAN_PREFIX = "donut_krea2_model_merge_identity:"


def _get_injections(model):
    injections = getattr(model, "injections", None)
    return injections if isinstance(injections, dict) else {}


def _get_additional_models(model, key):
    getter = getattr(model, "get_additional_models_with_key", None)
    if callable(getter):
        return list(getter(key) or [])
    additional = getattr(model, "additional_models", None)
    if isinstance(additional, dict):
        return list(additional.get(key, []) or [])
    return []


def _get_attachments(model):
    attachments = getattr(model, "attachments", None)
    return attachments if isinstance(attachments, dict) else {}


def get_krea2_merge_bypass_info(model):
    """Return ``(source_model, plans, attachment_keys)`` or ``None``.

    A plan item is ``(module_path, weight_key, model1_ratio)``. Experimental
    bypass currently emits only ratio 0.0 plans. If runtime merge hooks exist
    but the source/plan metadata is missing, raise instead of allowing a silent
    model1-only save.
    """
    injections = _get_injections(model)
    has_runtime = KREA2_MERGE_INJECTION_KEY in injections

    plans = []
    attachment_keys = []
    for key, value in _get_attachments(model).items():
        if not isinstance(key, str) or not key.startswith(KREA2_MERGE_PLAN_PREFIX):
            continue
        attachment_keys.append(key)
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            module_path, weight_key, ratio = item
            if not isinstance(module_path, str) or not isinstance(weight_key, str):
                continue
            plans.append((module_path, weight_key, float(ratio)))

    if not has_runtime and not plans:
        return None
    if not plans:
        raise RuntimeError(
            "Donut Krea2 Experimental-bypass merge is active, but its save plan "
            "metadata is missing. Refusing to save model1 without the bypassed model2 blocks."
        )

    # Preserve order while removing duplicate plan tuples inherited through clones.
    unique_plans = []
    seen = set()
    for plan in plans:
        if plan in seen:
            continue
        seen.add(plan)
        unique_plans.append(plan)

    for module_path, weight_key, ratio in unique_plans:
        if ratio != 0.0:
            raise RuntimeError(
                "Donut Krea2 save encountered a non-zero runtime bypass ratio for "
                f"{weight_key}: {ratio}. Partial blends must be regular Comfy patches."
            )
        if not weight_key.endswith(".weight"):
            raise RuntimeError(f"Invalid Krea2 bypass weight key: {weight_key}")
        if weight_key[:-7] != module_path:
            raise RuntimeError(
                f"Krea2 bypass plan mismatch: module '{module_path}' vs '{weight_key}'"
            )

    sources = _get_additional_models(model, KREA2_MERGE_SOURCE_KEY)
    if len(sources) != 1:
        raise RuntimeError(
            "Donut Krea2 Experimental-bypass save expected exactly one retained "
            f"model2 source, found {len(sources)}. Refusing a lossy save."
        )

    return sources[0], tuple(unique_plans), tuple(attachment_keys)


def clone_without_krea2_merge_runtime(model, info=None):
    """Clone the final model and remove only Krea2 merge runtime plumbing."""
    if info is None:
        info = get_krea2_merge_bypass_info(model)
    if info is None:
        return model

    _source, _plans, attachment_keys = info
    converted = model.clone()

    remove_injections = getattr(converted, "remove_injections", None)
    if callable(remove_injections):
        remove_injections(KREA2_MERGE_INJECTION_KEY)
    elif isinstance(getattr(converted, "injections", None), dict):
        converted.injections.pop(KREA2_MERGE_INJECTION_KEY, None)

    remove_additional = getattr(converted, "remove_additional_models", None)
    if callable(remove_additional):
        remove_additional(KREA2_MERGE_SOURCE_KEY)
    elif callable(getattr(converted, "set_additional_models", None)):
        converted.set_additional_models(KREA2_MERGE_SOURCE_KEY, [])
    elif isinstance(getattr(converted, "additional_models", None), dict):
        converted.additional_models.pop(KREA2_MERGE_SOURCE_KEY, None)

    remove_attachment = getattr(converted, "remove_attachments", None)
    for key in attachment_keys:
        if callable(remove_attachment):
            remove_attachment(key)
        elif isinstance(getattr(converted, "attachments", None), dict):
            converted.attachments.pop(key, None)

    return converted


def clone_with_regular_components(model, components_by_key, allowed_keys=None):
    """Clone ``model`` and register selected bypass adapters as normal patches."""
    if not components_by_key:
        return model.clone()

    allowed = None if allowed_keys is None else set(allowed_keys)
    converted = model.clone()
    for key, components in components_by_key.items():
        if allowed is not None and key not in allowed:
            continue
        for adapter, strength in components:
            strength = float(strength)
            if strength == 0.0:
                continue
            converted.add_patches({key: adapter}, strength)
    return converted


def _direct_module_state_keys(state_dict, module_path):
    prefix = f"{module_path}."
    direct = set()
    for key in state_dict.keys():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if suffix and "." not in suffix:
            direct.add(key)
    return direct


def compose_krea2_merge_unet_state_dict(base_model, source_model, plans):
    """Compose the lazy UNet state dict matching Experimental-bypass inference.

    ``base_model`` is the final model with runtime merge plumbing removed and all
    normal patches preserved. ``source_model`` is the retained model2 patcher,
    optionally with later bypass-LoRA adapters registered as ordinary patches.
    """
    base_sd = base_model.model_state_dict_for_saving(
        base_model.model.diffusion_model,
        "diffusion_model.",
    )
    source_sd = source_model.model_state_dict_for_saving(
        source_model.model.diffusion_model,
        "diffusion_model.",
    )

    swapped_modules = 0
    swapped_keys = 0
    for module_path, weight_key, _ratio in plans:
        base_keys = _direct_module_state_keys(base_sd, module_path)
        source_keys = _direct_module_state_keys(source_sd, module_path)

        if weight_key not in source_keys:
            raise RuntimeError(
                "Donut Krea2 save could not find the model2 source weight for "
                f"bypassed module '{module_path}'."
            )
        if not source_keys:
            raise RuntimeError(
                f"Donut Krea2 save found no model2 state for '{module_path}'."
            )

        # Runtime bypass calls the complete source module, not merely its weight.
        # Remove every direct model1 state entry and replace it with model2's set,
        # preserving quant metadata such as weight_scale/input_scale as well as bias.
        for key in base_keys:
            base_sd.pop(key, None)
        for key in source_keys:
            base_sd[key] = source_sd[key]

        swapped_modules += 1
        swapped_keys += len(source_keys)

    return base_sd, swapped_modules, swapped_keys

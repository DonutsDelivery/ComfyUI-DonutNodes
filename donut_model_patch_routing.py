"""Route regular LoRA patches to the weights actually used by Krea2 swaps."""

import uuid

from .donut_krea2_merge_serialization import (
    KREA2_MERGE_SOURCE_KEY,
    get_krea2_merge_bypass_info,
)

_SOURCE_PATCH_ID_PREFIX = "donut_lora_source_patches:"


def add_model_patch_components(model, components_by_key):
    """Append ordered ``[(adapter, strength), ...]`` lists to a caller's clone.

    A runtime Krea2 swap executes the retained source module, ignoring regular
    patches on the outer model for that module. Clone and patch that source
    instead. Preserve tuple keys (offset/function) and all upstream components.
    """
    info = get_krea2_merge_bypass_info(model)
    swapped_modules = {path for path, _key, _ratio in info[1]} if info else set()
    source = None
    accepted = set()
    redirected = 0

    for key, components in components_by_key.items():
        target = key[0] if isinstance(key, tuple) else key
        on_source = isinstance(target, str) and target.rpartition('.')[0] in swapped_modules
        for adapter, strength in components:
            strength = float(strength)
            if strength == 0.0:
                continue
            if on_source:
                if source is None:
                    source = info[0].clone()
                loaded = source.add_patches({key: adapter}, strength)
                if key not in (loaded or ()):
                    raise RuntimeError(f"LoRA patch could not be applied to active Krea2 merge source: {target}")
                redirected += 1
            else:
                loaded = model.add_patches({key: adapter}, strength)
            accepted.update(loaded or ())

    if source is not None:
        model.set_additional_models(KREA2_MERGE_SOURCE_KEY, [source])
        # Core clone comparison may ignore additional-model contents and return
        # True for empty outer patch lists before looking at patches_uuid.
        # A distinct attachment key invalidates that comparison for source-only
        # changes while ordinary clones preserve the same identity.
        model.set_attachments(_SOURCE_PATCH_ID_PREFIX + uuid.uuid4().hex, redirected)
        if hasattr(model, 'patches_uuid'):
            model.patches_uuid = uuid.uuid4()
        print(f"[DonutApplyLoRAStack] Comfy patches routed {redirected} component(s) to the active Krea2 merge source")

    return accepted

"""Small compatibility helpers for ComfyUI model lifecycle operations."""


def offload_models(model_management, *models):
    """Offload only the supplied model families, with a legacy global fallback."""
    models = [model for model in models if model is not None]
    if not models:
        return

    unload_model = getattr(model_management, "unload_model_and_clones", None)
    if unload_model is None:
        model_management.unload_all_models()
        return

    seen = set()
    for model in models:
        clone_id = getattr(model, "clone_base_uuid", None)
        identity = clone_id if clone_id is not None else id(model)
        if identity in seen:
            continue
        seen.add(identity)
        unload_model(model)

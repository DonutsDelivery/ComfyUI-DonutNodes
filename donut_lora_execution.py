"""Shared execution-mode policy for the Donut model path.

The first Donut node that selects a LoRA execution mode publishes it on the
model. Downstream nodes inherit that choice so a workflow cannot accidentally
stack regular Comfy patches and runtime bypass adapters on different branches.
"""

EXECUTION_MODES = ("Comfy patches", "Experimental bypass")
MODEL_OPTIONS_KEY = "donut_lora_execution_mode"


def get_execution_mode(model):
    """Return a valid mode already published by ``model``, if any."""
    options = getattr(model, "model_options", None)
    if not isinstance(options, dict):
        return None
    mode = options.get(MODEL_OPTIONS_KEY)
    return mode if mode in EXECUTION_MODES else None


def resolve_execution_mode(model, requested=None):
    """Resolve one mode for this model path.

    An upstream model selection is authoritative. A downstream widget is kept
    for old workflows and user visibility, but it cannot create a mixed-mode
    path when the model already carries a valid selection.
    """
    inherited = get_execution_mode(model)
    if inherited is not None:
        return inherited
    if requested is None:
        return EXECUTION_MODES[0]
    if requested not in EXECUTION_MODES:
        raise ValueError(f"Unknown LoRA execution mode: {requested}")
    return requested


def publish_execution_mode(model, mode):
    """Store the path-wide mode on a model clone or output object."""
    if mode not in EXECUTION_MODES:
        raise ValueError(f"Unknown LoRA execution mode: {mode}")
    options = getattr(model, "model_options", None)
    if not isinstance(options, dict):
        options = {}
        try:
            model.model_options = options
        except (AttributeError, TypeError):
            return model
    options[MODEL_OPTIONS_KEY] = mode
    return model

"""Compatibility for saved sampler-local switches; new workflows use the MODEL switch.

No alpha, NAG, edit, SDA or sampling-mode gate. Normalization is implemented by
model-local txtfusion hooks shared with Fusion Control, never by a NAG forward.
"""
from copy import deepcopy
from inspect import signature
import logging

from .donut_grounding_schedule import DonutSampler as _Base
from .donut_txtfusion_model_guard import attach_model_guard

LOGGER = logging.getLogger(__name__)


class DonutSampler(_Base):
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        schema = deepcopy(super().INPUT_TYPES())
        optional = schema.setdefault("optional", {})
        # Keep both old positions and values readable. New global control is on
        # Fusion Control so the same guarded MODEL also reaches finish stages.
        optional["txtfusion_internal_guard"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "Legacy sampler-local switch. Now NAG-independent, including alpha 0, editing and SDA. For all connected stages use Models > Txtfusion RMS guard instead.",
        })
        names = [name for name in folder_paths.get_filename_list("diffusion_models")
                 if name.lower().endswith(".safetensors")]
        optional["txtfusion_reference_checkpoint"] = (["None", *names], {
            "default": "None",
            "tooltip": "Deprecated compatibility field, no file is read. Reference is now captured automatically from the effective checkpoint/merge before adapters, for both native and bypass execution.",
        })
        return schema

    def sample(self, *args, txtfusion_internal_guard=False,
               txtfusion_reference_checkpoint="None", **kwargs):
        if type(txtfusion_internal_guard) is not bool:
            raise ValueError("txtfusion_internal_guard must be a boolean")
        if not txtfusion_internal_guard:
            # Do not remove a model-wide guard selected upstream.
            return super().sample(*args, **kwargs)
        if txtfusion_reference_checkpoint != "None":
            LOGGER.warning("[Donut txtfusion RMS] legacy reference filename is not used; "
                           "the effective model/merge supplies the checkpoint reference")
        bound = signature(_Base.sample).bind(self, *args, **kwargs)
        for name in ("model", "model_2", "model_3"):
            if bound.arguments.get(name) is not None:
                bound.arguments[name] = attach_model_guard(bound.arguments[name])
        # BoundArguments preserves positional/keyword/VAR_KEYWORD behavior.
        return _Base.sample(*bound.args, **bound.kwargs)


NODE_CLASS_MAPPINGS = {"DonutSampler": DonutSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutSampler": "DonutSampler"}

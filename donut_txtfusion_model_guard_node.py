"""The model-level switch, appended to Fusion Control without shifting widgets."""
from copy import deepcopy

from .donut_krea2_fusion_experiments import DonutKrea2FusionControl as _Fusion
from .donut_txtfusion_model_guard import attach_model_guard, remove_model_guard

TIP = ("EXPERIMENT: normalize internal txtfusion attention/MLP contributions and the "
       "projector (bounded 0.25x-4x gains) against an independent pre-adapter reference of the effective model/merge. "
       "Applies to ALL callers of this model: base, upscale and detailer, with or without "
       "NAG (including alpha 0). Rebalance conditioning is not changed. Adds reference "
       "weights and computation. Default Off. Does not guarantee removal of image artifacts.")


class DonutKrea2FusionControl(_Fusion):
    @classmethod
    def INPUT_TYPES(cls):
        schema = deepcopy(super().INPUT_TYPES())
        schema.setdefault("optional", {})["txtfusion_rms_guard"] = ("BOOLEAN", {
            "default": False, "tooltip": TIP,
        })
        return schema

    def apply(self, *args, txtfusion_rms_guard=False, **kwargs):
        if type(txtfusion_rms_guard) is not bool:
            raise ValueError("txtfusion_rms_guard must be a boolean")
        model = args[0] if args else kwargs.get("model")
        # Establish reference BEFORE the parent adds UncensorFix. Earlier native
        # adapters are stripped from checkpoint recipes; late live snapshots are
        # never the reference. Off introduces no reference allocation/callbacks.
        prepared = attach_model_guard(model) if txtfusion_rms_guard else remove_model_guard(model)
        if args:
            args = (prepared, *args[1:])
        else:
            kwargs = {**kwargs, "model": prepared}
        result = list(super().apply(*args, **kwargs))
        if txtfusion_rms_guard:
            result[-1] += "\ntxtfusion_rms_guard=model-wide; reference=effective checkpoint/merge before adapters; nag_independent=true"
        return tuple(result)


class DonutTxtfusionRMSGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "enabled": ("BOOLEAN", {"default": False, "tooltip": TIP})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/patches/Krea2"
    DESCRIPTION = TIP

    def apply(self, model, enabled=False):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        return (attach_model_guard(model) if enabled else remove_model_guard(model),)


NODE_CLASS_MAPPINGS = {"DonutKrea2FusionControl": DonutKrea2FusionControl,
                       "DonutTxtfusionRMSGuard": DonutTxtfusionRMSGuard}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutKrea2FusionControl": "Donut Krea2 Fusion Control",
                              "DonutTxtfusionRMSGuard": "Donut Txtfusion RMS Guard (model-wide)"}

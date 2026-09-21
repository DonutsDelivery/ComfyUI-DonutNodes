"""Opt-in, first-pass-only txtfusion guard on the existing DonutSampler node."""
from contextvars import ContextVar
from copy import deepcopy
from inspect import signature
import logging

from .donut_grounding_schedule import DonutSampler as _Base
from .donut_txtfusion_guard import install_guard

_REQUEST = ContextVar("donut_txtfusion_internal_guard_request", default=None)
LOGGER = logging.getLogger(__name__)


class DonutSampler(_Base):
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        schema = deepcopy(super().INPUT_TYPES())
        optional = schema.setdefault("optional", {})
        # Append after ALL inherited widgets; never shift serialized positions.
        optional["txtfusion_internal_guard"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "EXPERIMENT, first pass only: RMS-match adapter-modified txtfusion attention/MLP contributions to checkpoint references BEFORE residual addition. Does not normalize Rebalance. Requires simple non-edit NAG and floating-point txtfusion.",
        })
        names = [name for name in folder_paths.get_filename_list("diffusion_models")
                 if name.lower().endswith(".safetensors")]
        optional["txtfusion_reference_checkpoint"] = (["None", *names], {
            "default": "None",
            "tooltip": "Fallback file reference for plain txtfusion. V5 grouped merge with Fusion ratio 0 automatically uses the retained Secondary/model2 checkpoint forward as the reference, preserving its quantized kernel, so None is valid there. Other paths may require selecting the effective txtfusion checkpoint. Mixed primary/model2 ownership is rejected.",
        })
        return schema

    def sample(self, *args, txtfusion_internal_guard=False,
               txtfusion_reference_checkpoint="None", **kwargs):
        if type(txtfusion_internal_guard) is not bool:
            raise ValueError("txtfusion_internal_guard must be a boolean")
        token = _REQUEST.set(None)
        try:
            if txtfusion_internal_guard:
                # Inspect the actual inherited signature without changing what
                # the parent receives, including promoted positional arguments.
                bound = signature(_Base.sample).bind(self, *args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                nag = values.get("nag_options", {})
                if values.get("mode", "simple") != "simple" or values.get("edit_mode") or values.get("sda_enabled"):
                    raise ValueError("Internal txtfusion guard currently supports simple non-edit NAG only; no SDA/multi-model/edit mode")
                if not nag.get("nag_enabled", False) or float(nag.get("nag_phi", 4.0)) == 0:
                    raise ValueError("Internal txtfusion guard requires NAG enabled; enable NAG without changing phi")
                if float(nag.get("nag_alpha", .25)) == 0:
                    raise ValueError("Internal txtfusion guard executes inside NAG's active forward path; set NAG alpha above 0 (e.g. 0.45)")
                _REQUEST.set((self, txtfusion_reference_checkpoint))
            return super().sample(*args, **kwargs)
        finally:
            _REQUEST.reset(token)

    def run_simple(self, model, *args, **kwargs):
        request = _REQUEST.get()
        if request is None or request[0] is not self:
            return super().run_simple(model, *args, **kwargs)
        import folder_paths

        name = request[1]
        reference = None
        if name != "None":
            # Only catalog entries, not arbitrary paths from an API prompt.
            if name not in folder_paths.get_filename_list("diffusion_models"):
                raise ValueError("Unknown txtfusion reference checkpoint")
            reference = folder_paths.get_full_path_or_raise("diffusion_models", name)
        patched, run = install_guard(model, reference)
        try:
            result = super().run_simple(patched, *args, **kwargs)
            if run is not None and run.contribution_calls == 0:
                raise RuntimeError("Internal guard was requested but never executed; check NAG's sigma window and wrapper path")
            return result
        finally:
            if run is not None:
                try:
                    LOGGER.info("[Donut txtfusion guard] forwards=%d; contributions=%d; first-observed stats=%s",
                                run.forward_calls, run.contribution_calls, run.reports)
                finally:
                    # Interruption, model errors and logging failures all release
                    # the same references. No live weight needs restoring.
                    run.close()


NODE_CLASS_MAPPINGS = {"DonutSampler": DonutSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutSampler": "DonutSampler"}

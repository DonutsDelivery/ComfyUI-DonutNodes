"""Native Krea2 Turbo SDA diversity integration for DonutSampler.

The upstream SDA adapter is intentionally active for only the first two steps of
Krea2 Turbo's eight-step schedule.  DonutSampler already has an exact latent-
continuation multi-model path, so this module wraps the existing sampler instead
of adding a second scheduler implementation.
"""

from copy import deepcopy
import math

import comfy.utils
import folder_paths

from .DonutKSamplerCFGLinear import DonutSampler as _BaseDonutSampler
from .donut_lora_execution import publish_execution_mode, resolve_execution_mode
from .lora_block_weight import LoraLoaderBlockWeight


SDA_LORA_NAME = "krea2/krea2_turbo_sda_v1.0_comfy.safetensors"
SDA_SUPPORTED_STEPS = 8
SDA_GATE_STEPS = 2
_KREA2_FULL_VECTOR = ",".join(["1"] * 29)


def _sda_path():
    path = folder_paths.get_full_path("loras", SDA_LORA_NAME)
    if not path:
        raise FileNotFoundError(
            "Krea2 SDA diversity is enabled but "
            f"'{SDA_LORA_NAME}' is missing. Use Donut's Download missing control "
            "or install the ComfyUI-format SDA LoRA under models/loras/krea2/."
        )
    return path


def apply_krea2_sda(model, strength=1.0):
    """Return a model carrying the SDA adapter in the path's execution mode."""
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0.0 or strength > 2.0:
        raise ValueError("SDA strength must be finite and between 0 and 2.")

    execution_mode = resolve_execution_mode(model)
    lora = comfy.utils.load_torch_file(_sda_path(), safe_load=True)

    if execution_mode == "Experimental bypass":
        from .DonutSafeApplyLoRAStack import _apply_bypass_applications
        result = _apply_bypass_applications(
            model, [(lora, strength, _KREA2_FULL_VECTOR)]
        )
    else:
        result, _clip, _vector = LoraLoaderBlockWeight.load_lora_for_models(
            model,
            None,
            lora,
            strength,
            0.0,
            False,
            0,
            1.0,
            1.0,
            _KREA2_FULL_VECTOR,
        )

    # Keep the execution policy explicit on the temporary phase model.  This is
    # especially important when SDA is the first adapter added to a model path.
    publish_execution_mode(result, execution_mode)
    return result, execution_mode


def _validate_sda_sampling(kwargs):
    if not kwargs.get("turbo_mode", False):
        raise ValueError("SDA diversity requires Turbo mode.")

    steps = int(kwargs.get("steps", 20))
    if steps != SDA_SUPPORTED_STEPS:
        raise ValueError(
            f"SDA diversity is trained for the {SDA_SUPPORTED_STEPS}-step Krea2 Turbo "
            f"schedule; set Steps to {SDA_SUPPORTED_STEPS}."
        )

    denoise = float(kwargs.get("denoise", 1.0))
    if denoise < 0.999999:
        raise ValueError(
            "SDA diversity is only valid on a full-denoise base generation. "
            "Disable SDA for partial-denoise refinement, upscaling, or detailing."
        )

    if kwargs.get("edit_mode", False):
        raise ValueError(
            "SDA diversity is a composition adapter for text-to-image generation and "
            "is disabled for Krea2 editing/inpainting."
        )

    if kwargs.get("mode", "simple") == "multi_model" or kwargs.get("model_2") is not None or kwargs.get("model_3") is not None:
        raise ValueError(
            "Native SDA uses DonutSampler's two-model phase internally. Disconnect "
            "manual model_2/model_3 inputs and use simple or advanced mode."
        )

    if int(kwargs.get("start_at_step", 0)) != 0:
        raise ValueError("SDA diversity must start at step 0 of the Turbo schedule.")
    end = int(kwargs.get("end_at_step", 10000))
    if end < SDA_SUPPORTED_STEPS:
        raise ValueError("SDA diversity requires the complete 8-step Turbo schedule.")


class DonutSampler(_BaseDonutSampler):
    """DonutSampler with an optional native, correctly gated Krea2 SDA phase."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(super().INPUT_TYPES())
        optional = inputs.setdefault("optional", {})
        # APPENDED only: old saved DonutSampler positional widget arrays retain
        # their exact ordering and acquire these controls at their defaults.
        optional["sda_enabled"] = ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "Krea2 Turbo only. Restores seed/composition diversity with the F16 "
                "SDA LoRA for exactly the first 2 of 8 denoise steps, then returns "
                "to the clean model automatically. Uses the model path's existing "
                "Comfy-patches or Experimental-bypass execution mode."
            ),
        })
        optional["sda_strength"] = ("FLOAT", {
            "default": 1.0,
            "min": 0.0,
            "max": 2.0,
            "step": 0.05,
            "tooltip": "SDA adapter strength. 1.0 is the upstream recommended value.",
        })
        return inputs

    def sample(self, model, sda_enabled=False, sda_strength=1.0, **kwargs):
        if not sda_enabled:
            return super().sample(model=model, **kwargs)

        _validate_sda_sampling(kwargs)
        clean_model = model
        sda_model, execution_mode = apply_krea2_sda(clean_model, sda_strength)

        # Reuse DonutSampler's existing latent-continuation phase engine:
        #   steps 0..1 -> SDA model
        #   steps 2..7 -> identical clean model
        # Noise is added only in phase 1 and the seed is intentionally unchanged.
        native = dict(kwargs)
        native.update(
            mode="multi_model",
            model_2=clean_model,
            model_3=None,
            switch_at_step_1=SDA_GATE_STEPS,
            switch_at_step_2=SDA_SUPPORTED_STEPS - 1,
            randomize_seed_per_model="disable",
        )
        latent, info = super().sample(model=sda_model, **native)
        prefix = (
            f"SDA diversity: native gate {SDA_GATE_STEPS}/{SDA_SUPPORTED_STEPS}, "
            f"strength={float(sda_strength):.2f}, execution={execution_mode}"
        )
        return latent, f"{prefix}\n{info}"


NODE_CLASS_MAPPINGS = {"DonutSampler": DonutSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutSampler": "DonutSampler"}

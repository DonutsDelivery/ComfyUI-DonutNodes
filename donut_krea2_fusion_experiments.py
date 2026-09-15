"""Experimental Krea2 fusion presets for prompt-coherence / rendering A/B tests.

This module deliberately leaves the established Rebalance and Balanced presets
unchanged.  It only extends the preset selector with a few diagnostic recipes
that can be compared with identical prompts/seeds.
"""

try:
    from . import DonutKrea2FusionPreset as stable
except ImportError:
    import DonutKrea2FusionPreset as stable


PRESET_EXP_MEAN = "Experiment · NAG-friendly mean"
PRESET_EXP_STATIC_RMS = "Experiment · NAG-friendly static RMS"
PRESET_EXP_POWER_060 = "Experiment · NAG-friendly power 0.60"
PRESET_EXP_TENSOR_075 = "Experiment · soft tensor RMS 0.75"
EXPERIMENTAL_PRESETS = (
    PRESET_EXP_MEAN,
    PRESET_EXP_STATIC_RMS,
    PRESET_EXP_POWER_060,
    PRESET_EXP_TENSOR_075,
)

# These values are also mirrored by the frontend extension so choosing a preset
# writes the visible widgets exactly like the established compatibility presets.
EXPERIMENTAL_SETTINGS = {
    PRESET_EXP_MEAN: {
        "tap_method": "Donut 12-tap gains",
        "tap_profile": "classic",
        "tap_strength": 1.0,
        "tap_formula": "scale_around_1",
        "tap_normalization": "mean_gain",
        "projector_method": "Donut projector-input gains",
        "projector_profile": "off",
        "projector_strength": 1.0,
        "projector_formula": "scale_around_1",
        "projector_normalization": "none",
        "fusion_method": "Standard Krea2 fusion",
        "fusion_strength": 1.0,
    },
    PRESET_EXP_STATIC_RMS: {
        "tap_method": "Donut 12-tap gains",
        "tap_profile": "classic",
        "tap_strength": 1.0,
        "tap_formula": "scale_around_1",
        "tap_normalization": "rms_gain",
        "projector_method": "Donut projector-input gains",
        "projector_profile": "off",
        "projector_strength": 1.0,
        "projector_formula": "scale_around_1",
        "projector_normalization": "none",
        "fusion_method": "Standard Krea2 fusion",
        "fusion_strength": 1.0,
    },
    PRESET_EXP_POWER_060: {
        "tap_method": "Donut 12-tap gains",
        "tap_profile": "classic",
        "tap_strength": 0.60,
        "tap_formula": "geometric_power",
        "tap_normalization": "none",
        "projector_method": "Donut projector-input gains",
        "projector_profile": "off",
        "projector_strength": 1.0,
        "projector_formula": "scale_around_1",
        "projector_normalization": "none",
        "fusion_method": "Standard Krea2 fusion",
        "fusion_strength": 1.0,
    },
    PRESET_EXP_TENSOR_075: {
        "tap_method": "Donut 12-tap gains",
        "tap_profile": "classic",
        "tap_strength": 0.75,
        "tap_formula": "scale_around_1",
        "tap_normalization": "tensor_rms",
        "projector_method": "Donut projector-input gains",
        "projector_profile": "off",
        "projector_strength": 1.0,
        "projector_formula": "scale_around_1",
        "projector_normalization": "none",
        "fusion_method": "Standard Krea2 fusion",
        "fusion_strength": 1.0,
    },
}


class DonutKrea2FusionControl(stable.DonutKrea2FusionControl):
    """Stable Fusion Control plus opt-in experiment presets."""

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema["required"])
        spec = required["compatibility_preset"]
        values = list(spec[0])
        insertion = values.index(stable.PRESET_UNCENSORFIX) if stable.PRESET_UNCENSORFIX in values else len(values)
        for offset, name in enumerate(EXPERIMENTAL_PRESETS):
            if name not in values:
                values.insert(insertion + offset, name)
        settings = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
        settings["tooltip"] = (
            settings.get("tooltip", "")
            + " Experimental presets are diagnostic A/B recipes; NAG-friendly means the tap transform uses "
              "prompt-independent fixed gains, not that every NAG parameter is guaranteed artifact-free."
        ).strip()
        required["compatibility_preset"] = (values, settings)
        return {**schema, "required": required}

    def apply(self, *args, **kwargs):
        preset = kwargs.get("compatibility_preset")
        if preset not in EXPERIMENTAL_PRESETS:
            return super().apply(*args, **kwargs)

        # Presets are UI helpers in the existing architecture: the frontend has
        # already copied the recipe into visible widgets.  Delegate as Custom so
        # the stable server-side validation accepts it, then restore the chosen
        # experimental label in diagnostics.
        delegated = dict(kwargs)
        delegated["compatibility_preset"] = stable.PRESET_CUSTOM
        result = list(super().apply(*args, **delegated))
        result[-1] = stable._rewrite_preset_diagnostics(
            result[-1], stable.PRESET_CUSTOM, preset
        )
        return tuple(result)


NODE_CLASS_MAPPINGS = {"DonutKrea2FusionControl": DonutKrea2FusionControl}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutKrea2FusionControl": "Donut Krea2 Fusion Control"}

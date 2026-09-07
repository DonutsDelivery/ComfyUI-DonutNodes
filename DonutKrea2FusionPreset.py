"""Self-contained UncensorFix preset and Simple/Advanced fusion UI.

The existing DonutKrea2FusionControl node ID is retained. UncensorFix uses numerical
factors embedded in Python source. It never reads safetensors, searches LoRA
folders, or downloads weights. All other presets delegate to the base node.
"""

import math

from . import DonutKrea2FusionControl as base


UI_MODE_SIMPLE = "Simple"
UI_MODE_ADVANCED = "Advanced"
UI_MODES = (UI_MODE_SIMPLE, UI_MODE_ADVANCED)

PRESET_CUSTOM = "Custom"
PRESET_BYPASS_2 = "Bypass 2"
PRESET_BYPASS_3 = "Bypass 3"
PRESET_REBALANCE = "Rebalance"
PRESET_ENHANCER = "Enhancer"
PRESET_REBALANCE_ENHANCER = "Rebalance + Enhancer"
PRESET_REBALANCE_BYPASS_2 = "Rebalance + Bypass 2"
PRESET_REBALANCE_BYPASS_3 = "Rebalance + Bypass 3"
PRESET_BALANCED = "Balanced"
PRESET_BALANCED_ENHANCER = "Balanced + Enhancer"
PRESET_UNCENSORFIX = "UncensorFix"
LEGACY_TEACHERFIX = "DONUT settings: Krea2 C33 TeacherFix EMA5000"
LEGACY_TEACHERFIX_SHORT = "TeacherFix"

SIMPLE_PRESET_TO_LEGACY = {
    PRESET_CUSTOM: base.PRESET_MANUAL,
    PRESET_BYPASS_2: base.PRESET_BYPASS_2,
    PRESET_BYPASS_3: base.PRESET_BYPASS_3,
    PRESET_REBALANCE: base.PRESET_REBALANCE,
    PRESET_ENHANCER: base.PRESET_ENHANCER,
    PRESET_REBALANCE_ENHANCER: base.PRESET_REBALANCE_ENHANCER,
    PRESET_REBALANCE_BYPASS_2: base.PRESET_REBALANCE_BYPASS_2,
    PRESET_REBALANCE_BYPASS_3: base.PRESET_REBALANCE_BYPASS_3,
    PRESET_BALANCED: base.PRESET_DONUT_BALANCED,
    PRESET_BALANCED_ENHANCER: base.PRESET_DONUT_BALANCED_ENHANCER,
}
SIMPLE_PRESETS = tuple(SIMPLE_PRESET_TO_LEGACY) + (PRESET_UNCENSORFIX,)
LEGACY_PRESET_TO_SIMPLE = {legacy: simple for simple, legacy in SIMPLE_PRESET_TO_LEGACY.items()}
LEGACY_PRESET_TO_SIMPLE[LEGACY_TEACHERFIX] = PRESET_UNCENSORFIX
LEGACY_PRESET_TO_SIMPLE[LEGACY_TEACHERFIX_SHORT] = PRESET_UNCENSORFIX

UNCENSORFIX_TARGET_COUNT = 33


def _preset_combo_spec(spec, tooltip=None):
    settings = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
    settings["default"] = PRESET_CUSTOM
    if tooltip is not None:
        settings["tooltip"] = tooltip
    return (list(SIMPLE_PRESETS), settings)


def _uncensorfix_factors():
    # Lazy import: other presets and strength zero never import/decode the data.
    from .uncensorfix_weights import get_uncensorfix_factors
    return get_uncensorfix_factors()


def _apply_uncensorfix(model, strength):
    """Construct model patches from embedded tensors, without any LoRA loader."""
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError("UncensorFix strength must be finite")
    if strength == 0.0:
        return model, 0, "embedded (strength 0)"

    # Use the same weight arithmetic as ComfyUI's ordinary patch route, but
    # construct adapters directly. No load_lora, load_torch_file, key-map scan,
    # safetensors parser, or external asset is involved.
    from comfy.weight_adapter.lora import LoRAAdapter

    factors = _uncensorfix_factors()
    expected = {key for key, _, _, _ in factors}
    if len(factors) != UNCENSORFIX_TARGET_COUNT or len(expected) != UNCENSORFIX_TARGET_COUNT:
        raise RuntimeError("UncensorFix requires exactly 33 unique embedded targets")

    model_state = model.model.state_dict()
    for key, up, down, alpha in factors:
        weight = model_state.get(key)
        if weight is None or tuple(weight.shape) != (up.shape[0], down.shape[1]):
            raise RuntimeError(
                f"UncensorFix target missing or wrong shape: {key}. "
                "The input MODEL must be a compatible Krea 2 model."
            )
    del model_state

    # Fresh adapters/tensor copies isolate cached factors from other patches.
    patches = {
        key: LoRAAdapter(set(), (up.clone(), down.clone(), alpha, None, None, None))
        for key, up, down, alpha in factors
    }
    patched = model.clone()
    loaded = patched.add_patches(patches, strength_patch=strength)
    if set(loaded) != expected:
        raise RuntimeError(f"UncensorFix could not patch every target: {len(loaded)}/33 loaded")
    return patched, len(loaded), "embedded"


def _rewrite_preset_diagnostics(diagnostics, legacy_name, simple_name):
    return str(diagnostics).replace(f"preset_label={legacy_name}", f"preset_label={simple_name}", 1)


def _rewrite_uncensorfix_diagnostics(diagnostics, strength, loaded_count, relative_name):
    diagnostics = str(diagnostics).replace(
        f"preset_label={base.PRESET_MANUAL}; preset_is_ui_only=true",
        f"preset_label={PRESET_UNCENSORFIX}; preset_is_ui_only=false", 1,
    )
    return diagnostics.replace(
        "external_files_loaded=none",
        f"uncensorfix_source=embedded; uncensorfix_strength={float(strength):g}; "
        f"uncensorfix_targets={loaded_count}; external_files_loaded=none", 1,
    )


class DonutKrea2FusionControl(base.DonutKrea2FusionControl):
    """Existing Krea2 Fusion Control with UncensorFix factors embedded in source."""

    DESCRIPTION = (
        "Krea 2 text-fusion controls with Simple/Advanced UI modes and short preset names. "
        "UncensorFix uses numerical factors embedded in Python; no external LoRA "
        "installation, file selection or download is required."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema["required"])
        required["compatibility_preset"] = _preset_combo_spec(
            required["compatibility_preset"],
            "Select a preset. UncensorFix uses embedded weights and stays selected when "
            "its controls are edited. Select Custom or another preset to turn it off.",
        )
        spec = required["tap_strength"]
        settings = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
        settings["tooltip"] = (
            "Tap strength. For UncensorFix this scales its embedded weights in both modes. "
            "In Simple mode it also drives the selected preset's projector/fusion strength."
        )
        required["tap_strength"] = (spec[0], settings)
        # Append only: do not shift saved workflows' legacy widget positions.
        required["ui_mode"] = (list(UI_MODES), {
            "default": UI_MODE_ADVANCED,
            "tooltip": "Simple shows mode, preset and tap_strength. Advanced restores the full controls.",
        })
        return {**schema, "required": required}

    def apply(self, *args, ui_mode=UI_MODE_ADVANCED, **kwargs):
        if ui_mode not in UI_MODES:
            raise ValueError(f"Unknown Krea2 Fusion UI mode: {ui_mode}")
        preset = kwargs.get("compatibility_preset", PRESET_CUSTOM)
        if preset in (PRESET_UNCENSORFIX, LEGACY_TEACHERFIX, LEGACY_TEACHERFIX_SHORT):
            strength = float(kwargs.get("tap_strength", 1.0))
            delegated = dict(kwargs)
            delegated["compatibility_preset"] = base.PRESET_MANUAL
            result = list(super().apply(*args, **delegated))
            patched_model, loaded_count, relative_name = _apply_uncensorfix(result[0], strength)
            result[0] = patched_model
            result[-1] = _rewrite_uncensorfix_diagnostics(result[-1], strength, loaded_count, relative_name)
            return tuple(result)

        legacy_preset = SIMPLE_PRESET_TO_LEGACY.get(preset, preset)
        delegated = dict(kwargs)
        delegated["compatibility_preset"] = legacy_preset
        result = list(super().apply(*args, **delegated))
        if preset in SIMPLE_PRESET_TO_LEGACY:
            result[-1] = _rewrite_preset_diagnostics(result[-1], legacy_preset, preset)
        return tuple(result)


NODE_CLASS_MAPPINGS = {"DonutKrea2FusionControl": DonutKrea2FusionControl}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutKrea2FusionControl": "Donut Krea2 Fusion Control"}

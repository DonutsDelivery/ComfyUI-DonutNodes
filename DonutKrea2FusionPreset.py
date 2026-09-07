"""Self-contained UncensorFix preset and Simple/Advanced fusion UI.

The existing DonutKrea2FusionControl node ID is retained. UncensorFix uses numerical
factors embedded in Python source. It never reads safetensors, searches LoRA
folders, or downloads weights. All other presets delegate to the base node.
"""

import math
import uuid

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
_UNCENSORFIX_SOURCE_ID_PREFIX = "donut_uncensorfix_source_identity:"


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

    # Hard model2 swaps execute a retained source model, not the main model's
    # patched weights. Resolve the same plan used by Donut save/extraction.
    # In particular, do not test the truth value of the composable injection
    # list: it deliberately reports False while containing live swap hooks.
    from .donut_krea2_merge_serialization import (
        KREA2_MERGE_INJECTION_KEY,
        KREA2_MERGE_SOURCE_KEY,
        get_krea2_merge_bypass_info,
    )

    merge_info = get_krea2_merge_bypass_info(model)
    source_model, source_keys = None, set()
    if merge_info is not None:
        if KREA2_MERGE_INJECTION_KEY not in getattr(model, "injections", {}):
            raise RuntimeError("UncensorFix found Krea2 swap plans without their runtime injection")
        source_model, plans, _ = merge_info
        source_keys = expected.intersection(key for _, key, _ in plans)

    model_state = model.model.state_dict()
    source_state = source_model.model.state_dict() if source_keys else {}
    for key, up, down, alpha in factors:
        state = source_state if key in source_keys else model_state
        weight = state.get(key)
        if weight is None or tuple(weight.shape) != (up.shape[0], down.shape[1]):
            owner = "merge-bypass model2" if key in source_keys else "main model"
            raise RuntimeError(
                f"UncensorFix target missing or wrong shape on {owner}: {key}. "
                "The input MODEL must be a compatible Krea 2 model."
            )
    del model_state, source_state

    # Fresh adapters/tensor copies isolate cached factors from other patches.
    patches = {
        key: LoRAAdapter(set(), (up.clone(), down.clone(), alpha, None, None, None))
        for key, up, down, alpha in factors
    }
    main_patches = {key: patch for key, patch in patches.items() if key not in source_keys}
    source_patches = {key: patches[key] for key in sorted(source_keys)}
    patched = model.clone()
    loaded = set()
    if main_patches:
        accepted = set(patched.add_patches(main_patches, strength_patch=strength))
        if accepted != set(main_patches):
            raise RuntimeError(
                f"UncensorFix could not patch every target on main model: {len(accepted)}/{len(main_patches)} loaded"
            )
        loaded.update(accepted)

    if source_patches:
        # Never mutate model2 or replace its existing patch lists. Its cloned
        # patch stack includes earlier LoRAs. The main model's runtime LoRA
        # injections, merge plans, and other additional models remain intact.
        source = source_model.clone()
        accepted = set(source.add_patches(source_patches, strength_patch=strength))
        if accepted != source_keys:
            raise RuntimeError(
                f"UncensorFix could not patch every target on merge-bypass model2: {len(accepted)}/{len(source_keys)} loaded"
            )
        loaded.update(accepted)
        patched.set_additional_models(KREA2_MERGE_SOURCE_KEY, [source])

        # ModelPatcher.clone_has_same_weights does not compare the contents of
        # additional_models. It can also return True for two empty main patch
        # stacks before checking patches_uuid. A fresh attachment KEY forces
        # the outer swap injection to reload and resolve this cloned source.
        patched.set_attachments(
            _UNCENSORFIX_SOURCE_ID_PREFIX + uuid.uuid4().hex, tuple(sorted(source_keys))
        )
        patched.patches_uuid = uuid.uuid4()

    if loaded != expected:
        raise RuntimeError(f"UncensorFix could not patch every target: {len(loaded)}/33 loaded")
    source_details = "embedded"
    if source_keys:
        source_details += (
            f"; uncensorfix_bypass_source_targets={len(source_keys)}"
            f"; uncensorfix_model_targets={len(main_patches)}"
        )
    return patched, len(loaded), source_details


def _rewrite_preset_diagnostics(diagnostics, legacy_name, simple_name):
    return str(diagnostics).replace(f"preset_label={legacy_name}", f"preset_label={simple_name}", 1)


def _rewrite_uncensorfix_diagnostics(diagnostics, strength, loaded_count, source_details):
    diagnostics = str(diagnostics).replace(
        f"preset_label={base.PRESET_MANUAL}; preset_is_ui_only=true",
        f"preset_label={PRESET_UNCENSORFIX}; preset_is_ui_only=false", 1,
    )
    return diagnostics.replace(
        "external_files_loaded=none",
        f"uncensorfix_source={source_details}; uncensorfix_strength={float(strength):g}; "
        f"uncensorfix_targets={loaded_count}; external_files_loaded=none", 1,
    )


class DonutKrea2FusionControl(base.DonutKrea2FusionControl):
    """Existing Krea2 Fusion Control with UncensorFix factors embedded in source."""

    DESCRIPTION = (
        "Krea 2 text-fusion controls with Simple/Advanced UI modes and short preset names. "
        "UncensorFix uses numerical factors embedded in Python; no external LoRA "
        "installation, file selection or download is required. Krea2 Experimental "
        "merge-bypass targets are patched on their retained model2 source."
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
            patched_model, loaded_count, source_details = _apply_uncensorfix(result[0], strength)
            result[0] = patched_model
            result[-1] = _rewrite_uncensorfix_diagnostics(result[-1], strength, loaded_count, source_details)
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

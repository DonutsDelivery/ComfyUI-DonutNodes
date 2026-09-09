"""Self-contained UncensorFix preset and Simple/Advanced fusion UI.

The existing DonutKrea2FusionControl node ID is retained. UncensorFix uses numerical
factors bundled in assets/uncensorfix.f32. It never reads safetensors, searches LoRA
folders, or downloads weights. Off is a pure pass-through; other presets delegate
to the base node.
"""

import math
import uuid

from . import DonutKrea2FusionControl as base


UI_MODE_SIMPLE = "Simple"
UI_MODE_ADVANCED = "Advanced"
UI_MODES = (UI_MODE_SIMPLE, UI_MODE_ADVANCED)

PRESET_OFF = "Off"
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
SIMPLE_PRESETS = (PRESET_OFF,) + tuple(SIMPLE_PRESET_TO_LEGACY) + (PRESET_UNCENSORFIX,)
LEGACY_PRESET_TO_SIMPLE = {legacy: simple for simple, legacy in SIMPLE_PRESET_TO_LEGACY.items()}
LEGACY_PRESET_TO_SIMPLE[LEGACY_TEACHERFIX] = PRESET_UNCENSORFIX
LEGACY_PRESET_TO_SIMPLE[LEGACY_TEACHERFIX_SHORT] = PRESET_UNCENSORFIX

UNCENSORFIX_TARGET_COUNT = 33
UNCENSORFIX_LORA_ONLY = "LoRA only"
UNCENSORFIX_WITH_CONTROLS = "LoRA + fusion controls"
UNCENSORFIX_CONTROL_MODES = (UNCENSORFIX_LORA_ONLY, UNCENSORFIX_WITH_CONTROLS)
FUSION_ONLY = "Fusion only"
FUSION_WITH_LORA = "Fusion + LoRA"
FUSION_WITH_WEIGHTS = "Fusion + UncensorFix weights"
UNCENSORFIX_EXECUTION_MODES = ("Comfy patches", "Experimental bypass")
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


def _apply_uncensorfix(model, strength, execution_mode="Comfy patches"):
    """Apply embedded factors using the selected Donut LoRA execution path."""
    if execution_mode not in UNCENSORFIX_EXECUTION_MODES:
        raise ValueError(f"Unknown UncensorFix execution mode: {execution_mode}")
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError("UncensorFix strength must be finite")
    if strength == 0.0:
        return model, 0, "embedded (strength 0)"

    # Native mode constructs the same adapters as Comfy's LoRA loader. Bypass
    # mode hands an in-memory state dict to Donut Apply's shared execution
    # helper. Neither mode reads a LoRA file or changes the cached factors.
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
        if execution_mode == "Experimental bypass":
            from .donut_uncensorfix_lora import apply_embedded_bypass
            patched = apply_embedded_bypass(model, main_patches, strength)
            accepted = set(main_patches)
        else:
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
        if execution_mode == "Experimental bypass":
            from .donut_uncensorfix_lora import apply_embedded_bypass
            source = apply_embedded_bypass(source_model, source_patches, strength)
            accepted = set(source_patches)
        else:
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
        f"uncensorfix_targets={loaded_count}; bundled_weight_asset=assets/uncensorfix.f32; external_files_loaded=none", 1,
    )


class DonutKrea2FusionControl(base.DonutKrea2FusionControl):
    """Existing Krea2 Fusion Control with UncensorFix factors bundled as raw numerical data."""

    DESCRIPTION = (
        "Krea 2 text-fusion controls with Simple/Advanced UI modes and short preset names. "
        "UncensorFix uses bundled numerical factors; no external LoRA "
        "installation, file selection or download is required. Krea2 Experimental "
        "merge-bypass targets are patched on their retained model2 source. "
        "LoRA only ignores this node's stored fusion controls. Match execution_mode "
        "and Donut Apply's text_weight for an equivalent LoRA comparison."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema["required"])
        required["compatibility_preset"] = _preset_combo_spec(
            required["compatibility_preset"],
            "Select a preset. UncensorFix uses embedded weights and stays selected when "
            "its controls are edited. Off passes the input model and conditioning through "
            "unchanged, preserving upstream LoRAs and stored control values.",
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
            "tooltip": "Simple hides individual fusion controls. UncensorFix composition and execution remain explicit.",
        })
        # Optional, appended widgets preserve old workflow widget indices.
        optional = dict(schema.get("optional", {}))
        optional["uncensorfix_controls"] = ([FUSION_ONLY, FUSION_WITH_WEIGHTS, FUSION_WITH_LORA, *UNCENSORFIX_CONTROL_MODES], {
            "default": FUSION_ONLY,
            "tooltip": "Fusion only applies the visible fusion settings. Fusion + UncensorFix weights also applies "
                       "embedded UncensorFix weights with any preset. Old LoRA modes remain for saved workflows.",
        })
        optional["execution_mode"] = (list(UNCENSORFIX_EXECUTION_MODES), {
            "default": "Comfy patches",
            "tooltip": "UncensorFix only: match Donut Apply LoRA Stack's execution_mode. "
                       "Experimental bypass reuses its forward-adapter path and compatibility fallbacks. "
                       "The two execution modes are not numerically interchangeable, especially with quantization.",
        })
        optional["uncensorfix_strength"] = ("FLOAT", {
            "default": 1.0, "min": -20.0, "max": 20.0, "step": 0.05,
            "tooltip": "UncensorFix weight strength, independent of tap/fusion strength.",
        })
        return {**schema, "required": required, "optional": optional}

    def apply(
        self, *args, ui_mode=UI_MODE_ADVANCED,
        uncensorfix_controls=UNCENSORFIX_LORA_ONLY,
        execution_mode="Comfy patches", uncensorfix_strength=None, **kwargs,
    ):
        if ui_mode not in UI_MODES:
            raise ValueError(f"Unknown Krea2 Fusion UI mode: {ui_mode}")
        if args:
            # Resolve ALL legacy positional inputs before inspecting preset or
            # strength, not only conditioning routes in the Off branch.
            from inspect import signature
            kwargs = signature(super().apply).bind_partial(*args, **kwargs).arguments
            args = ()
        preset = kwargs.get("compatibility_preset", PRESET_CUSTOM)
        if uncensorfix_controls in (FUSION_ONLY, FUSION_WITH_LORA, FUSION_WITH_WEIGHTS):
            # New controls are authoritative: the preset label cannot enable
            # or disable either operation. Legacy workflows use the old path.
            delegated = dict(kwargs)
            delegated["compatibility_preset"] = base.PRESET_MANUAL
            result = list(super().apply(**delegated))
            result[-1] = _rewrite_preset_diagnostics(result[-1], base.PRESET_MANUAL, preset)
            if uncensorfix_controls in (FUSION_WITH_LORA, FUSION_WITH_WEIGHTS):
                fallback = kwargs.get("tap_strength", 1.0) if uncensorfix_controls == FUSION_WITH_LORA else 1.0
                strength = float(fallback if uncensorfix_strength is None else uncensorfix_strength)
                result[0], count, source = _apply_uncensorfix(result[0], strength, execution_mode)
                result[-1] += (f"\nuncensorfix_targets={count}; uncensorfix_strength={strength:g}; "
                               f"uncensorfix_source={source}; uncensorfix_execution_mode={execution_mode}")
            result[-1] += f"\nuncensorfix_controls={uncensorfix_controls}"
            return tuple(result)
        if preset == PRESET_OFF:
            # Do not call the base node: hidden tap/projector/fusion settings
            # from the last active preset can still be non-neutral. Return the
            # original objects, including upstream patches, injections and
            # conditioning metadata. Off only disables this node's changes.
            inputs = kwargs
            conditionings = (
                inputs["conditioning_in_1"],
                inputs.get("conditioning_in_2"),
                inputs.get("conditioning_in_3"),
                inputs.get("conditioning_in_4"),
            )
            diagnostics = (
                "preset_label=Off; preset_is_ui_only=false\n"
                "fusion_control=off; uncensorfix_targets=0\n"
                f"conditioning_routes={sum(value is not None for value in conditionings)}/4\n"
                "external_files_loaded=none"
            )
            return (inputs["model"], *conditionings, diagnostics)

        if preset in (PRESET_UNCENSORFIX, LEGACY_TEACHERFIX, LEGACY_TEACHERFIX_SHORT):
            if uncensorfix_controls not in UNCENSORFIX_CONTROL_MODES:
                raise ValueError(f"Unknown UncensorFix controls mode: {uncensorfix_controls}")
            strength = float(kwargs.get("tap_strength", 1.0) if uncensorfix_strength is None else uncensorfix_strength)
            if uncensorfix_controls == UNCENSORFIX_LORA_ONLY:
                # Enforce parity server-side, including API workflows and
                # stale/hidden widget values. Do not depend on JS resetting
                # controls and do not clear any upstream wrappers or patches.
                conditionings = tuple(kwargs.get(f"conditioning_in_{i}") for i in range(1, 5))
                diagnostics = (
                    f"preset_label={base.PRESET_MANUAL}; preset_is_ui_only=true\n"
                    "fusion_control=lora_only; conditioning=unchanged\n"
                    f"conditioning_routes={sum(value is not None for value in conditionings)}/4\n"
                    "external_files_loaded=none"
                )
                result = [kwargs["model"], *conditionings, diagnostics]
            else:
                delegated = dict(kwargs)
                delegated["compatibility_preset"] = base.PRESET_MANUAL
                result = list(super().apply(*args, **delegated))
            patched_model, loaded_count, source_details = _apply_uncensorfix(
                result[0], strength, execution_mode=execution_mode,
            )
            result[0] = patched_model
            result[-1] = _rewrite_uncensorfix_diagnostics(result[-1], strength, loaded_count, source_details)
            result[-1] += (
                f"\nuncensorfix_controls={uncensorfix_controls}; "
                f"uncensorfix_execution_mode={execution_mode}; "
                f"reference_text_weight={strength:g}"
            )
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

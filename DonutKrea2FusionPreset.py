"""Self-contained TeacherFix preset and Simple/Advanced fusion UI.

The existing DonutKrea2FusionControl node ID is retained. TeacherFix reads only
its original, bundled safetensors asset; it never searches ComfyUI's LoRA
folders or downloads weights. All other presets delegate to the base node.
"""

import hashlib
import importlib
import math
from pathlib import Path

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
PRESET_TEACHERFIX = "TeacherFix"
LEGACY_TEACHERFIX = "DONUT settings: Krea2 C33 TeacherFix EMA5000"

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
SIMPLE_PRESETS = tuple(SIMPLE_PRESET_TO_LEGACY) + (PRESET_TEACHERFIX,)
LEGACY_PRESET_TO_SIMPLE = {legacy: simple for simple, legacy in SIMPLE_PRESET_TO_LEGACY.items()}
LEGACY_PRESET_TO_SIMPLE[LEGACY_TEACHERFIX] = PRESET_TEACHERFIX

TEACHERFIX_FILENAME = "krea2_c33_teacherfix_ema5000.safetensors"
TEACHERFIX_TARGET_COUNT = 33
TEACHERFIX_SIZE_BYTES = 3_470_548
TEACHERFIX_SHA256 = "db3c2b7612828120e7ef9cc8fe77124c6fd8de2e38f150599e62abd9695f6beb"
TEACHERFIX_PATH = Path(__file__).resolve().parent / "assets" / TEACHERFIX_FILENAME

_TEACHERFIX_CACHE = None


def _preset_combo_spec(spec, tooltip=None):
    settings = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
    settings["default"] = PRESET_CUSTOM
    if tooltip is not None:
        settings["tooltip"] = tooltip
    return (list(SIMPLE_PRESETS), settings)


def _validate_teacherfix_state(state):
    """Check all factor pairs and their scope before handing them to ComfyUI."""
    import torch

    suffix = ".lora_down.weight"
    targets = {key[:-len(suffix)] for key in state if key.endswith(suffix)}
    if len(targets) != TEACHERFIX_TARGET_COUNT or len(state) != 3 * TEACHERFIX_TARGET_COUNT:
        raise RuntimeError("Bundled TeacherFix must contain exactly 33 complete factor pairs and alphas")
    expected_keys = set()
    for target in targets:
        if not target.startswith("diffusion_model.txtfusion."):
            raise RuntimeError(f"Bundled TeacherFix has a non-text-fusion target: {target}")
        keys = [target + ending for ending in (suffix, ".lora_up.weight", ".alpha")]
        expected_keys.update(keys)
        if any(key not in state for key in keys):
            raise RuntimeError(f"Bundled TeacherFix has an incomplete target: {target}")
        down, up, alpha = (state[key] for key in keys)
        if any(not torch.is_tensor(value) or not value.is_floating_point() for value in (down, up, alpha)):
            raise RuntimeError(f"Bundled TeacherFix has an invalid tensor: {target}")
        if (down.ndim != 2 or up.ndim != 2 or down.shape[0] != 4
                or up.shape[1] != 4 or alpha.numel() != 1 or float(alpha.item()) != 4.0):
            raise RuntimeError(f"Bundled TeacherFix has unexpected rank/alpha: {target}")
        if any(not torch.isfinite(value).all().item() for value in (down, up, alpha)):
            raise RuntimeError(f"Bundled TeacherFix contains non-finite values: {target}")
    if set(state) != expected_keys:
        raise RuntimeError("Bundled TeacherFix contains unexpected tensors")


def _load_teacherfix_lora():
    """Decode the bundled original once, verifying the bytes actually decoded."""
    global _TEACHERFIX_CACHE
    if _TEACHERFIX_CACHE is not None:
        return _TEACHERFIX_CACHE

    try:
        payload = TEACHERFIX_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            "The bundled TeacherFix asset is missing or unreadable. Reinstall the complete "
            "DonutNodes package, including assets/. No file in models/loras is required."
        ) from exc
    if len(payload) != TEACHERFIX_SIZE_BYTES or hashlib.sha256(payload).hexdigest() != TEACHERFIX_SHA256:
        raise RuntimeError("Bundled TeacherFix asset failed its size/SHA-256 check; reinstall DonutNodes")

    from safetensors.torch import load
    state = load(payload)
    _validate_teacherfix_state(state)
    _TEACHERFIX_CACHE = (state, f"assets/{TEACHERFIX_FILENAME}")
    return _TEACHERFIX_CACHE


def _apply_teacherfix(model, strength):
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError("TeacherFix strength must be finite")
    if strength == 0.0:
        return model, 0, "not loaded (strength 0)"

    comfy_lora = getattr(base.comfy, "lora", None)
    if comfy_lora is None:
        comfy_lora = importlib.import_module("comfy.lora")

    state, relative_name = _load_teacherfix_lora()
    key_map = comfy_lora.model_lora_keys_unet(model.model)
    patches = comfy_lora.load_lora(dict(state), key_map, log_missing=False)
    expected = {key[:-len(".lora_down.weight")] + ".weight"
                for key in state if key.endswith(".lora_down.weight")}
    if set(patches) != expected:
        raise RuntimeError(
            "TeacherFix could not map all 33 bundled text-fusion targets. "
            "Make sure the input MODEL is a compatible Krea 2 model."
        )

    patched = model.clone()
    loaded = patched.add_patches(patches, strength_patch=strength)
    if set(loaded) != expected:
        raise RuntimeError(f"TeacherFix could not patch every target: {len(loaded)}/33 loaded")
    return patched, len(loaded), relative_name


def _rewrite_preset_diagnostics(diagnostics, legacy_name, simple_name):
    return str(diagnostics).replace(f"preset_label={legacy_name}", f"preset_label={simple_name}", 1)


def _rewrite_teacherfix_diagnostics(diagnostics, strength, loaded_count, relative_name):
    diagnostics = str(diagnostics).replace(
        f"preset_label={base.PRESET_MANUAL}; preset_is_ui_only=true",
        f"preset_label={PRESET_TEACHERFIX}; preset_is_ui_only=false", 1,
    )
    return diagnostics.replace(
        "external_files_loaded=none",
        f"teacherfix_file={relative_name}; teacherfix_source=bundled; "
        f"teacherfix_sha256={TEACHERFIX_SHA256}; teacherfix_strength={float(strength):g}; "
        f"teacherfix_targets={loaded_count}; external_files_loaded=none", 1,
    )


class DonutKrea2FusionControl(base.DonutKrea2FusionControl):
    """Existing Krea2 Fusion Control with original TeacherFix weights included."""

    DESCRIPTION = (
        "Krea 2 text-fusion controls with Simple/Advanced UI modes and short preset names. "
        "TeacherFix uses the original weights bundled with DonutNodes; no external LoRA "
        "installation, file selection or download is required."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema["required"])
        required["compatibility_preset"] = _preset_combo_spec(
            required["compatibility_preset"],
            "Select a preset. TeacherFix uses bundled weights and stays selected when "
            "its controls are edited. Select Custom or another preset to turn it off.",
        )
        spec = required["tap_strength"]
        settings = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
        settings["tooltip"] = (
            "Tap strength. For TeacherFix this scales its bundled weights in both modes. "
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
        if preset in (PRESET_TEACHERFIX, LEGACY_TEACHERFIX):
            strength = float(kwargs.get("tap_strength", 1.0))
            delegated = dict(kwargs)
            delegated["compatibility_preset"] = base.PRESET_MANUAL
            result = list(super().apply(*args, **delegated))
            patched_model, loaded_count, relative_name = _apply_teacherfix(result[0], strength)
            result[0] = patched_model
            result[-1] = _rewrite_teacherfix_diagnostics(result[-1], strength, loaded_count, relative_name)
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

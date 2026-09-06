"""Krea 2 TeacherFix preset and Simple/Advanced UI mode.

This module intentionally overrides the existing ``DonutKrea2FusionControl``
node id instead of registering a second node. The base implementation remains
the source of truth for all existing tap/projector/fusion controls; this layer
adds one exact-file TeacherFix preset plus a UI-mode selector.
"""

from pathlib import Path
import hashlib
import importlib
import math
import os

from . import DonutKrea2FusionControl as base


UI_MODE_SIMPLE = "Simple"
UI_MODE_ADVANCED = "Advanced"
UI_MODES = (UI_MODE_SIMPLE, UI_MODE_ADVANCED)

PRESET_TEACHERFIX = "DONUT settings: Krea2 C33 TeacherFix EMA5000"
TEACHERFIX_FILENAME = "krea2_c33_teacherfix_ema5000.safetensors"
TEACHERFIX_ALIASES = (
    TEACHERFIX_FILENAME,
    "krea2_c33_teacherfix_ema5000(1).safetensors",
)
TEACHERFIX_TARGET_COUNT = 33
TEACHERFIX_SIZE_BYTES = 3_470_548
TEACHERFIX_SHA256 = "db3c2b7612828120e7ef9cc8fe77124c6fd8de2e38f150599e62abd9695f6beb"

_TEACHERFIX_CACHE = None


def _copy_combo_spec(spec, extra_option=None, tooltip=None):
    options = list(spec[0])
    if extra_option is not None and extra_option not in options:
        options.append(extra_option)
    settings = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if tooltip is not None:
        settings["tooltip"] = tooltip
    return (options, settings)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_teacherfix_file():
    """Find the exact uploaded TeacherFix LoRA in ComfyUI's normal LoRA folder.

    The file may be renamed or live in a subfolder. We first use cheap file-size
    filtering, then verify SHA256, so a large LoRA library is not fully hashed.
    """
    folder_paths = importlib.import_module("folder_paths")
    names = list(folder_paths.get_filename_list("loras"))
    alias_basenames = {value.lower() for value in TEACHERFIX_ALIASES}

    def priority(name):
        basename = os.path.basename(str(name)).lower()
        if basename in alias_basenames:
            return (0, basename)
        if "krea2" in basename and "teacherfix" in basename and "ema5000" in basename:
            return (1, basename)
        return (2, basename)

    for name in sorted(names, key=priority):
        path = folder_paths.get_full_path("loras", name)
        if not path or not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) != TEACHERFIX_SIZE_BYTES:
                continue
        except OSError:
            continue
        if _sha256_file(path) == TEACHERFIX_SHA256:
            return str(name), str(path)

    aliases = " or ".join(TEACHERFIX_ALIASES)
    raise RuntimeError(
        "Krea2 C33 TeacherFix EMA5000 preset needs the exact TeacherFix LoRA in "
        "ComfyUI/models/loras. Install it there (for example as "
        f"{aliases}). Expected SHA256: {TEACHERFIX_SHA256}"
    )


def _load_teacherfix_lora():
    global _TEACHERFIX_CACHE
    if _TEACHERFIX_CACHE is not None:
        return _TEACHERFIX_CACHE

    relative_name, path = _find_teacherfix_file()
    comfy_utils = getattr(base.comfy, "utils", None)
    if comfy_utils is None:
        comfy_utils = importlib.import_module("comfy.utils")
    state = comfy_utils.load_torch_file(path, safe_load=True)
    if not isinstance(state, dict):
        raise RuntimeError("Krea2 TeacherFix preset did not load as a state dict")

    target_bases = {
        key[: -len(".lora_down.weight")]
        for key in state
        if isinstance(key, str) and key.endswith(".lora_down.weight")
    }
    if len(target_bases) != TEACHERFIX_TARGET_COUNT:
        raise RuntimeError(
            "Krea2 TeacherFix preset has an unexpected target count: "
            f"{len(target_bases)} != {TEACHERFIX_TARGET_COUNT}"
        )

    _TEACHERFIX_CACHE = (state, relative_name)
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
    patches = comfy_lora.load_lora(state, key_map, log_missing=False)
    if len(patches) != TEACHERFIX_TARGET_COUNT:
        raise RuntimeError(
            "Krea2 TeacherFix preset mapped an unexpected number of targets: "
            f"{len(patches)} != {TEACHERFIX_TARGET_COUNT}. "
            "Make sure the input MODEL is Krea 2."
        )

    patched = model.clone()
    loaded = patched.add_patches(patches, strength_patch=strength)
    loaded_count = len(loaded)
    if loaded_count != TEACHERFIX_TARGET_COUNT:
        raise RuntimeError(
            "Krea2 TeacherFix preset could not patch every target: "
            f"{loaded_count}/{TEACHERFIX_TARGET_COUNT} loaded"
        )
    return patched, loaded_count, relative_name


def _rewrite_teacherfix_diagnostics(diagnostics, strength, loaded_count, relative_name):
    diagnostics = str(diagnostics)
    diagnostics = diagnostics.replace(
        f"preset_label={base.PRESET_MANUAL}; preset_is_ui_only=true",
        f"preset_label={PRESET_TEACHERFIX}; preset_is_ui_only=false",
        1,
    )
    diagnostics = diagnostics.replace(
        "external_files_loaded=none",
        (
            f"teacherfix_file={relative_name}; teacherfix_sha256={TEACHERFIX_SHA256}; "
            f"teacherfix_strength={float(strength):g}; teacherfix_targets={loaded_count}"
        ),
        1,
    )
    return diagnostics


class DonutKrea2FusionControl(base.DonutKrea2FusionControl):
    """Existing Krea2 Fusion Control plus TeacherFix and compact UI mode."""

    DESCRIPTION = (
        "Krea 2 text-fusion controls with Simple/Advanced UI modes, the existing "
        "community compatibility presets, and a hash-verified C33 TeacherFix "
        "EMA5000 preset discovered automatically from ComfyUI/models/loras."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        original_required = schema["required"]
        required = {}
        for name, spec in original_required.items():
            if name == "compatibility_preset":
                required[name] = _copy_combo_spec(
                    spec,
                    PRESET_TEACHERFIX,
                    (
                        "Select a preset. Existing compatibility presets copy settings "
                        "into the controls; TeacherFix automatically finds and applies "
                        "the exact hash-verified LoRA from ComfyUI/models/loras."
                    ),
                )
            elif name == "tap_strength":
                settings = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
                settings["tooltip"] = (
                    "Tap strength. In Simple mode this is the single preset-strength "
                    "control; for TeacherFix it is the LoRA strength."
                )
                required[name] = (spec[0], settings)
            else:
                required[name] = spec

        # Append after every legacy widget so existing widgets_values positions
        # remain unchanged when old workflows are loaded.
        required["ui_mode"] = (list(UI_MODES), {
            "default": UI_MODE_ADVANCED,
            "tooltip": (
                "Simple shows only mode, preset and tap_strength. Advanced restores "
                "the current full Text Fusion control surface."
            ),
        })

        return {**schema, "required": required}

    def apply(self, *args, ui_mode=UI_MODE_ADVANCED, **kwargs):
        if ui_mode not in UI_MODES:
            raise ValueError(f"Unknown Krea2 Fusion UI mode: {ui_mode}")

        preset = kwargs.get("compatibility_preset", base.PRESET_MANUAL)
        if preset != PRESET_TEACHERFIX:
            return super().apply(*args, **kwargs)

        strength = float(kwargs.get("tap_strength", 1.0))
        delegated = dict(kwargs)
        # The base node validates its own preset labels. Keep all submitted
        # controls and substitute only the label for that validation call.
        delegated["compatibility_preset"] = base.PRESET_MANUAL
        result = list(super().apply(*args, **delegated))

        patched_model, loaded_count, relative_name = _apply_teacherfix(result[0], strength)
        result[0] = patched_model
        result[-1] = _rewrite_teacherfix_diagnostics(
            result[-1], strength, loaded_count, relative_name
        )
        return tuple(result)


NODE_CLASS_MAPPINGS = {
    "DonutKrea2FusionControl": DonutKrea2FusionControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutKrea2FusionControl": "Donut Krea2 Fusion Control",
}

"""Pure activation-space controls for Krea 2 text fusion.

This node intentionally loads no LoRAs or other external weights.  It exposes
two independent intervention points:

* pre-fusion scaling of the 12 concatenated Qwen hidden-state taps; and
* post-layerwise scaling of the 12 inputs to Krea 2's bias-free 12 -> 1
  projector, which is exactly equivalent to scaling its weight columns.

Both controls support several strength laws and normalization strategies so
that their behaviour can be compared with fixed seeds.
"""

import math

import torch

import comfy.patcher_extension


WRAPPER_KEY = "donut_krea2_fusion_control"
CONFIG_KEY = "donut_krea2_fusion_control"

KREA2_TAP_COUNT = 12
KREA2_TAP_DIM = 2560
KREA2_CONDITIONING_DIM = KREA2_TAP_COUNT * KREA2_TAP_DIM
CONDITIONING_SLOT_COUNT = 4

_PROFILE_OFF = (1.0,) * KREA2_TAP_COUNT
_PROFILE_CLASSIC = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)
_PROFILE_DEEP_2 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0)
_PROFILE_DEEP_3 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0)

PROFILES = {
    "off": _PROFILE_OFF,
    "classic": _PROFILE_CLASSIC,
    "deep_2": _PROFILE_DEEP_2,
    "deep_3": _PROFILE_DEEP_3,
}
STRENGTH_FORMULAS = ("scale_around_1", "geometric_power", "raw_multiply")
NORMALIZATIONS = ("none", "mean_gain", "rms_gain", "tensor_rms")

PRESET_MANUAL = "Custom settings"
PRESET_BYPASS_2 = "COPY settings: Krea2FilterBypass 2vector"
PRESET_BYPASS_3 = "COPY settings: Krea2FilterBypass 3vector"
PRESET_REBALANCE = "COPY settings: nova452 ConditioningKrea2Rebalance profile @ tap strength 1"
PRESET_ENHANCER = "COPY settings: capitan01R Krea2T-Enhancer defaults"
PRESET_REBALANCE_ENHANCER = "HYBRID settings: Rebalance + Krea2T-Enhancer"
PRESET_REBALANCE_BYPASS_2 = "HYBRID settings: Rebalance + Krea2FilterBypass 2vector"
PRESET_REBALANCE_BYPASS_3 = "HYBRID settings: Rebalance + Krea2FilterBypass 3vector"
PRESET_DONUT_BALANCED = "DONUT settings: RMS-balanced classic"
PRESET_DONUT_BALANCED_ENHANCER = "DONUT settings: RMS-balanced classic + Krea2T-Enhancer"
COMPATIBILITY_PRESETS = (
    PRESET_MANUAL,
    PRESET_BYPASS_2,
    PRESET_BYPASS_3,
    PRESET_REBALANCE,
    PRESET_ENHANCER,
    PRESET_REBALANCE_ENHANCER,
    PRESET_REBALANCE_BYPASS_2,
    PRESET_REBALANCE_BYPASS_3,
    PRESET_DONUT_BALANCED,
    PRESET_DONUT_BALANCED_ENHANCER,
)

TAP_METHOD_DONUT = "Donut 12-tap gains"
TAP_METHOD_REBALANCE = "nova452 Rebalance operation"
TAP_METHODS = (TAP_METHOD_DONUT, TAP_METHOD_REBALANCE)
PROJECTOR_METHOD_DONUT = "Donut projector-input gains"
PROJECTOR_METHOD_BYPASS_2 = "Krea2FilterBypass 2vector diff"
PROJECTOR_METHOD_BYPASS_3 = "Krea2FilterBypass 3vector diff"
PROJECTOR_METHODS = (PROJECTOR_METHOD_DONUT, PROJECTOR_METHOD_BYPASS_2, PROJECTOR_METHOD_BYPASS_3)
FUSION_METHOD_STANDARD = "Standard Krea2 fusion"
FUSION_METHOD_ENHANCER = "capitan01R Krea2T-Enhancer operation"
FUSION_METHODS = (FUSION_METHOD_STANDARD, FUSION_METHOD_ENHANCER)

_BYPASS_2_DIFF = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.51171875, -0.890625, 0.0, 0.0)
_BYPASS_3_DIFF = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.51171875, -0.890625, -0.609375, 0.0,
)
_PROJECTOR_WEIGHT_KEY = "diffusion_model.txtfusion.projector.weight"

# Exact compatibility implementation adapted from ComfyUI-Krea2T-Enhancer.
# Copyright (c) 2026 capitan01R, used under the MIT License. See
# THIRD_PARTY_NOTICES.md. The unusual profile concatenation is intentionally
# preserved because a COPY preset must reproduce the upstream implementation,
# not silently correct it.
_ENHANCER_CHUNK_PROFILE = _PROFILE_CLASSIC + _PROFILE_CLASSIC
_ENHANCER_CHUNK_COUNT = 24
_ENHANCER_CHUNK_DIM = 1280
_ENHANCER_GLOBAL_MULTIPLIER = 15.0
_ENHANCER_TOKEN_REL_CAP = 0.75


def _parse_profile(text):
    parts = [part.strip() for chunk in str(text).split(";") for part in chunk.split(",")]
    values = [float(part) for part in parts if part]
    if len(values) != KREA2_TAP_COUNT:
        raise ValueError(f"Expected exactly {KREA2_TAP_COUNT} profile values, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Profile values must all be finite")
    return tuple(values)


def _profile_values(name, custom):
    if name == "off":
        return _PROFILE_OFF
    if name not in PROFILES and name != "custom":
        raise ValueError(f"Unknown Krea 2 profile: {name}")
    return _parse_profile(custom)


def _strength_gains(profile, strength, formula):
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError("Strength must be finite")

    if formula == "scale_around_1":
        gains = tuple(1.0 + strength * (value - 1.0) for value in profile)
    elif formula == "geometric_power":
        if any(value <= 0.0 for value in profile):
            raise ValueError("geometric_power requires every profile value to be greater than zero")
        gains = tuple(value ** strength for value in profile)
    elif formula == "raw_multiply":
        gains = tuple(strength * value for value in profile)
    else:
        raise ValueError(f"Unknown strength formula: {formula}")
    if not all(math.isfinite(value) for value in gains):
        raise ValueError("Strength formula produced a non-finite gain")
    return gains


def _normalize_gains(gains, normalization):
    gains = tuple(float(value) for value in gains)
    if normalization in ("none", "tensor_rms"):
        return gains
    if normalization == "mean_gain":
        denominator = sum(gains) / len(gains)
    elif normalization == "rms_gain":
        denominator = math.sqrt(sum(value * value for value in gains) / len(gains))
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    if abs(denominator) <= 1e-12:
        raise ValueError(f"Cannot apply {normalization}: normalization denominator is zero")
    return tuple(value / denominator for value in gains)


def _resolve_gains(profile_name, custom_profile, strength, formula, normalization):
    if profile_name == "off":
        return _PROFILE_OFF
    profile = _profile_values(profile_name, custom_profile)
    gains = _strength_gains(profile, strength, formula)
    return _normalize_gains(gains, normalization)


def _is_neutral(gains):
    return all(abs(value - 1.0) <= 1e-12 for value in gains)


def _match_batch_rms(reference, value):
    if reference.ndim < 2:
        return value
    dims = tuple(range(1, reference.ndim))
    reference_rms = reference.float().square().mean(dim=dims, keepdim=True).sqrt()
    value_rms = value.float().square().mean(dim=dims, keepdim=True).sqrt()
    ratio = torch.where(value_rms > 1e-12, reference_rms / value_rms, torch.ones_like(value_rms))
    return value * ratio.to(device=value.device, dtype=value.dtype)


def _apply_tensor_gains(value, gains, axis, normalization):
    gain_tensor = torch.tensor(gains, device=value.device, dtype=value.dtype)
    shape = [1] * value.ndim
    shape[axis] = len(gains)
    scaled = value * gain_tensor.reshape(shape)
    if normalization == "tensor_rms":
        scaled = _match_batch_rms(value, scaled)
    return scaled


def _rebalance_conditioning(conditioning, gains, normalization):
    if _is_neutral(gains):
        return conditioning

    output = []
    for cond, metadata in conditioning:
        new_metadata = metadata.copy() if isinstance(metadata, dict) else metadata
        if cond is None:
            output.append([None, new_metadata])
            continue
        if not torch.is_tensor(cond) or not cond.is_floating_point():
            raise TypeError("Krea 2 conditioning must be a floating-point tensor")
        if cond.ndim < 2 or cond.shape[-1] != KREA2_CONDITIONING_DIM:
            shape = tuple(cond.shape)
            raise ValueError(
                f"Expected Krea 2 conditioning with last dimension {KREA2_CONDITIONING_DIM}, got {shape}"
            )

        taps = cond.reshape(*cond.shape[:-1], KREA2_TAP_COUNT, KREA2_TAP_DIM)
        scaled = _apply_tensor_gains(taps, gains, axis=-2, normalization=normalization)
        output.append([scaled.reshape_as(cond), new_metadata])
    return output


def _nova452_rebalance_tensor(value, multiplier, profile):
    """Reproduce nova452's default float32-profile/cast/multiplier order."""
    flat = value.shape[-1]
    if flat % KREA2_TAP_COUNT != 0:
        return value * multiplier
    original_dtype = value.dtype
    taps = value.float().view(*value.shape[:-1], KREA2_TAP_COUNT, flat // KREA2_TAP_COUNT)
    gains = torch.tensor(profile, dtype=taps.dtype, device=taps.device)
    taps = taps * gains.view(*([1] * (taps.dim() - 2)), KREA2_TAP_COUNT, 1)
    return taps.view(*taps.shape[:-2], flat).to(original_dtype) * multiplier


def _nova452_rebalance_structure(structure, multiplier, profile):
    if isinstance(structure, list):
        output = []
        for item in structure:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and torch.is_tensor(item[0])
                and isinstance(item[1], dict)
            ):
                output.append([_nova452_rebalance_tensor(item[0], multiplier, profile), dict(item[1])])
            else:
                output.append(_nova452_rebalance_structure(item, multiplier, profile))
        return output
    if torch.is_tensor(structure):
        return _nova452_rebalance_tensor(structure, multiplier, profile)
    if isinstance(structure, dict):
        return {key: _nova452_rebalance_structure(value, multiplier, profile) for key, value in structure.items()}
    return structure


def _run_txtfusion_parts(txtfusion, value, mask=None, transformer_options=None):
    transformer_options = transformer_options or {}
    batch, sequence, taps, dimension = value.shape
    fused = value.reshape(batch * sequence, taps, dimension)
    for block in txtfusion.layerwise_blocks:
        fused = block(fused.contiguous(), mask=None, transformer_options=transformer_options)
    tap_mix = fused.reshape(batch, sequence, taps, dimension).permute(0, 1, 3, 2).contiguous()
    projected = txtfusion.projector(tap_mix).squeeze(-1)
    output = projected
    for block in txtfusion.refiner_blocks:
        output = block(output, mask=mask, transformer_options=transformer_options)
    return output


def _capitan01r_enhancer_forward(
    txtfusion,
    value,
    mask=None,
    transformer_options=None,
    strength=1.0,
):
    """Exact output path of ComfyUI-Krea2T-Enhancer at the given strength."""
    transformer_options = transformer_options or {}
    batch, sequence, taps, dimension = value.shape
    if taps != KREA2_TAP_COUNT or dimension != KREA2_TAP_DIM:
        raise ValueError(f"Expected Krea 2 text fusion input, got {tuple(value.shape)}")

    reference_output = _run_txtfusion_parts(
        txtfusion,
        value,
        mask=mask,
        transformer_options=transformer_options,
    )
    gains = torch.tensor(_ENHANCER_CHUNK_PROFILE, device=value.device, dtype=torch.float32)
    gains = 1.0 + float(strength) * (gains - 1.0)
    global_multiplier = 1.0 + float(strength) * (_ENHANCER_GLOBAL_MULTIPLIER - 1.0)
    scaled_value = (
        value.reshape(batch, sequence, _ENHANCER_CHUNK_COUNT, _ENHANCER_CHUNK_DIM)
        * gains.to(dtype=value.dtype).view(1, 1, _ENHANCER_CHUNK_COUNT, 1)
        * global_multiplier
    ).reshape_as(value)
    candidate_output = _run_txtfusion_parts(
        txtfusion,
        scaled_value,
        mask=mask,
        transformer_options=transformer_options,
    )

    delta = candidate_output.detach().float() - reference_output.detach().float()
    base_rms = torch.sqrt(torch.mean(reference_output.detach().float() ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    delta_rms = torch.sqrt(torch.mean(delta ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    relative_delta = delta_rms / base_rms
    token_scale = (_ENHANCER_TOKEN_REL_CAP / relative_delta).clamp(max=1.0)
    return (reference_output.detach().float() + delta * token_scale).to(candidate_output.dtype)


def _is_krea2_diffusion_model(diffusion_model):
    return (
        hasattr(diffusion_model, "txtfusion")
        and hasattr(diffusion_model, "txtmlp")
        and hasattr(diffusion_model, "blocks")
        and hasattr(diffusion_model, "_unpack_context")
        and int(getattr(diffusion_model, "txtlayers", 0)) == KREA2_TAP_COUNT
        and int(getattr(diffusion_model, "txtdim", 0)) == KREA2_TAP_DIM
        and hasattr(diffusion_model.txtfusion, "projector")
    )


def krea2_fusion_control_wrapper(
    executor,
    x,
    timesteps,
    context,
    attention_mask,
    transformer_options,
    **kwargs,
):
    transformer_options = transformer_options or {}
    config = transformer_options.get(CONFIG_KEY)
    if not config or config.get("_active"):
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    diffusion_model = executor.class_obj
    if not _is_krea2_diffusion_model(diffusion_model):
        if "projector_gains" not in config:
            return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)
        raise RuntimeError("Donut Krea2 Fusion Control requires a Krea 2 diffusion model")

    txtfusion = diffusion_model.txtfusion
    projector = txtfusion.projector
    original_projector_forward = projector.forward
    original_txtfusion_forward = None

    if "projector_gains" in config:
        gains = config["projector_gains"]
        normalization = config["projector_normalization"]

        def scaled_projector_forward(projector_input, *args, **forward_kwargs):
            if projector_input.shape[-1] != KREA2_TAP_COUNT:
                raise ValueError(
                    f"Expected Krea 2 projector input with {KREA2_TAP_COUNT} taps, "
                    f"got {tuple(projector_input.shape)}"
                )
            scaled_input = _apply_tensor_gains(
                projector_input,
                gains,
                axis=-1,
                normalization=normalization,
            )
            return original_projector_forward(scaled_input, *args, **forward_kwargs)

        projector.forward = scaled_projector_forward

    enhancer_strength = max(0.0, min(2.0, float(config.get("enhancer_strength", 0.0))))
    if enhancer_strength != 0.0:
        original_txtfusion_forward = txtfusion.forward

        def enhanced_forward(value, mask=None, transformer_options=None):
            return _capitan01r_enhancer_forward(
                txtfusion,
                value,
                mask=mask,
                transformer_options=transformer_options or {},
                strength=enhancer_strength,
            )

        txtfusion.forward = enhanced_forward

    config["_active"] = True
    try:
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)
    finally:
        projector.forward = original_projector_forward
        if original_txtfusion_forward is not None:
            txtfusion.forward = original_txtfusion_forward
        config.pop("_active", None)


def _format_gains(gains):
    return ",".join(f"{value:.6g}" for value in gains)


def _format_profile_string(gains):
    return ",".join(str(float(value)) for value in gains)


def _attach_runtime_wrapper(model, config):
    patched = model.clone()
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options[CONFIG_KEY] = config
    if hasattr(patched, "remove_wrappers_with_key"):
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
        )
    wrappers = transformer_options.get("wrappers", {})
    diffusion_wrappers = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
    diffusion_wrappers.pop(WRAPPER_KEY, None)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        krea2_fusion_control_wrapper,
    )
    return patched


def _apply_projector_diff(model, values, strength):
    if float(strength) == 0.0:
        return model
    patched = model.clone()
    diff = torch.tensor([values], dtype=torch.float32)
    loaded = patched.add_patches(
        {_PROJECTOR_WEIGHT_KEY: ("diff", (diff,))},
        strength_patch=float(strength),
    )
    if _PROJECTOR_WEIGHT_KEY not in loaded:
        raise RuntimeError("The loaded model does not expose the Krea 2 text-fusion projector weight")
    return patched


class DonutKrea2FusionControl:
    @classmethod
    def INPUT_TYPES(cls):
        formula_options = list(STRENGTH_FORMULAS)
        normalization_options = list(NORMALIZATIONS)
        profile_options = ["off", "classic", "deep_2", "deep_3", "custom"]
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning_in_1": ("CONDITIONING",),
                "compatibility_preset": (list(COMPATIBILITY_PRESETS), {
                    "default": PRESET_MANUAL,
                    "tooltip": "UI helper: copies values into the visible settings below; it is not a runtime override.",
                }),
                "tap_method": (list(TAP_METHODS), {"default": TAP_METHOD_DONUT}),
                "tap_profile": (profile_options, {"default": "classic"}),
                "per_layer_weights": ("STRING", {
                    "default": _format_profile_string(_PROFILE_CLASSIC), "multiline": False,
                }),
                "tap_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Strength of the pre-fusion 12-tap profile.",
                }),
                "tap_formula": (formula_options, {"default": "scale_around_1"}),
                "tap_normalization": (normalization_options, {"default": "tensor_rms"}),
                "projector_method": (list(PROJECTOR_METHODS), {"default": PROJECTOR_METHOD_DONUT}),
                "projector_profile": (profile_options, {"default": "off"}),
                "projector_layer_weights": ("STRING", {
                    "default": _format_profile_string(_PROFILE_DEEP_2), "multiline": False,
                }),
                "projector_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Strength of the live post-layerwise projector-input profile.",
                }),
                "projector_formula": (formula_options, {"default": "scale_around_1"}),
                "projector_normalization": (normalization_options, {"default": "none"}),
                "fusion_method": (list(FUSION_METHODS), {"default": FUSION_METHOD_STANDARD}),
                "fusion_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Strength used by the selected fusion operation.",
                }),
            },
            "optional": {
                "conditioning_in_2": ("CONDITIONING",),
                "conditioning_in_3": ("CONDITIONING",),
                "conditioning_in_4": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",) + ("CONDITIONING",) * CONDITIONING_SLOT_COUNT + ("STRING",)
    RETURN_NAMES = (
        "model",
        "conditioning_out_1",
        "conditioning_out_2",
        "conditioning_out_3",
        "conditioning_out_4",
        "diagnostics",
    )
    FUNCTION = "apply"
    CATEGORY = "Donut/conditioning"
    DESCRIPTION = (
        "Pure Krea 2 controls with explicitly attributed COPY presets for known community nodes/files. "
        "No external LoRA or safetensors file is loaded."
    )

    def apply(
        self,
        model,
        conditioning_in_1,
        tap_profile,
        tap_strength,
        tap_formula,
        tap_normalization,
        projector_profile,
        projector_strength,
        projector_formula,
        projector_normalization,
        per_layer_weights,
        projector_layer_weights,
        conditioning_in_2=None,
        conditioning_in_3=None,
        conditioning_in_4=None,
        compatibility_preset=PRESET_MANUAL,
        tap_method=TAP_METHOD_DONUT,
        projector_method=PROJECTOR_METHOD_DONUT,
        fusion_method=FUSION_METHOD_STANDARD,
        fusion_strength=1.0,
    ):
        if compatibility_preset not in COMPATIBILITY_PRESETS:
            raise ValueError(f"Unknown compatibility preset label: {compatibility_preset}")
        conditioning_inputs = (
            conditioning_in_1,
            conditioning_in_2,
            conditioning_in_3,
            conditioning_in_4,
        )
        output_model = model
        runtime_config = {}

        if tap_method == TAP_METHOD_DONUT:
            tap_gains = _resolve_gains(
                tap_profile,
                per_layer_weights,
                tap_strength,
                tap_formula,
                tap_normalization,
            )
            output_conditionings = tuple(
                None if conditioning is None else _rebalance_conditioning(conditioning, tap_gains, tap_normalization)
                for conditioning in conditioning_inputs
            )
            tap_details = (
                f"tap[{tap_method}; {tap_profile}; {tap_formula}; {tap_normalization}]="
                f"{_format_gains(tap_gains)}"
            )
        elif tap_method == TAP_METHOD_REBALANCE:
            tap_profile_values = _profile_values(tap_profile, per_layer_weights)
            multiplier = float(tap_strength)
            if not math.isfinite(multiplier):
                raise ValueError("Tap strength must be finite")
            if tap_profile == "off":
                output_conditionings = conditioning_inputs
                tap_details = f"tap[{tap_method}; off]"
            else:
                output_conditionings = tuple(
                    None if conditioning is None else _nova452_rebalance_structure(
                        conditioning,
                        multiplier,
                        tap_profile_values,
                    )
                    for conditioning in conditioning_inputs
                )
                tap_details = (
                    f"tap[{tap_method}; {tap_profile}]={_format_gains(tap_profile_values)}; "
                    f"multiplier={multiplier:g}"
                )
        else:
            raise ValueError(f"Unknown tap method: {tap_method}")

        if projector_method == PROJECTOR_METHOD_DONUT:
            projector_gains = _resolve_gains(
                projector_profile,
                projector_layer_weights,
                projector_strength,
                projector_formula,
                projector_normalization,
            )
            if not _is_neutral(projector_gains):
                runtime_config["projector_gains"] = projector_gains
                runtime_config["projector_normalization"] = projector_normalization
            projector_details = (
                f"projector[{projector_method}; {projector_profile}; {projector_formula}; "
                f"{projector_normalization}]="
                f"{_format_gains(projector_gains)}"
            )
        elif projector_method in (PROJECTOR_METHOD_BYPASS_2, PROJECTOR_METHOD_BYPASS_3):
            projector_strength = float(projector_strength)
            if not math.isfinite(projector_strength):
                raise ValueError("Projector strength must be finite")
            values = _BYPASS_2_DIFF if projector_method == PROJECTOR_METHOD_BYPASS_2 else _BYPASS_3_DIFF
            output_model = _apply_projector_diff(output_model, values, projector_strength)
            projector_details = f"projector[{projector_method}; strength={projector_strength:g}]={_format_gains(values)}"
        else:
            raise ValueError(f"Unknown projector method: {projector_method}")

        fusion_strength = float(fusion_strength)
        if not math.isfinite(fusion_strength):
            raise ValueError("Fusion strength must be finite")
        if fusion_method == FUSION_METHOD_STANDARD:
            fusion_details = f"fusion[{fusion_method}]"
        elif fusion_method == FUSION_METHOD_ENHANCER:
            fusion_strength = max(0.0, min(2.0, fusion_strength))
            if fusion_strength != 0.0:
                runtime_config["enhancer_strength"] = fusion_strength
            fusion_details = f"fusion[{fusion_method}; strength={fusion_strength:g}]"
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        if runtime_config:
            output_model = _attach_runtime_wrapper(output_model, runtime_config)

        diagnostics = (
            f"preset_label={compatibility_preset}; preset_is_ui_only=true\n"
            f"{tap_details}\n"
            f"{projector_details}\n"
            f"{fusion_details}\n"
            f"conditioning_routes={sum(value is not None for value in conditioning_inputs)}/"
            f"{CONDITIONING_SLOT_COUNT}\n"
            "external_files_loaded=none"
        )
        return (output_model, *output_conditionings, diagnostics)


NODE_CLASS_MAPPINGS = {
    "DonutKrea2FusionControl": DonutKrea2FusionControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutKrea2FusionControl": "Donut Krea2 Fusion Control",
}

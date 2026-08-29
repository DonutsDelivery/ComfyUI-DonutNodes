"""Safe Krea2 LoRA stack application.

Overrides DonutApplyLoRAStack with an opt-in per-block RMS energy limiter for
Krea2 LoRAs and optional fusion-aware budgeting for the 12-column text-fusion
projector. Safety remains off by default so old workflows retain their exact
behaviour.
"""

import math
import re

import comfy.sd
import comfy.utils
import folder_paths
import torch

from .donut_lora_nodes import (
    _TEXT_MERGE_VECTOR,
    _lora_has_real_text_encoder,
    _split_fused_text,
)
from .lora_block_weight import LoraLoaderBlockWeight


_KREA_BLOCK_RE = re.compile(r"(?<![a-z_])blocks\.(\d+)")
_KREA_BLOCK_COUNT = 28
_KREA_VECTOR_SIZE = _KREA_BLOCK_COUNT + 1  # non-block bucket + 28 blocks
_SAFE_ENERGY_BUDGET = 1.0
_FUSION_BUDGET_KEY = "donut_krea2_fusion_budget"
_FUSION_AWARE_MODES = ("Off", "Attenuate only", "Use headroom")
_KREA_TEXT_RE = re.compile(r"txt(?:fusion|mlp)|text_(?:fusion|mlp)")
_PROJECTOR_RE = re.compile(r"(?:txtfusion|text_fusion).*projector")
_PROJECTOR_COLUMN_COUNT = 12


def _krea2_block_indices(lora):
    """Return the Krea2 single-stream block indices referenced by a LoRA."""
    block_nums = set()
    for key in lora.keys():
        match = _KREA_BLOCK_RE.search(key)
        if match:
            block_nums.add(int(match.group(1)))
    return block_nums


def _is_krea2_lora(lora):
    """Return True when a LoRA contains Krea2-style single-stream blocks."""
    block_nums = _krea2_block_indices(lora)
    return bool(block_nums) and max(block_nums) < _KREA_BLOCK_COUNT


def _is_krea2_text_lora(lora):
    """Return True for Krea2's diffusion-side text-fusion adapter weights."""
    return any(_KREA_TEXT_RE.search(key.lower()) for key in lora)


def _split_projector_text(lora_text):
    """Separate the 12-column projector adapter from other text-fusion keys."""
    projector, other = {}, {}
    for key, value in lora_text.items():
        (projector if _PROJECTOR_RE.search(key.lower()) else other)[key] = value
    return projector, other


def _read_fusion_budget(model):
    """Read Fusion Control metadata without depending on its implementation."""
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        return None
    transformer_options = model_options.get("transformer_options")
    if not isinstance(transformer_options, dict):
        return None
    metadata = transformer_options.get(_FUSION_BUDGET_KEY)
    if not isinstance(metadata, dict) or int(metadata.get("version", 0)) != 1:
        return None
    gains = metadata.get("projector_gains")
    if not isinstance(gains, (list, tuple)) or len(gains) != _PROJECTOR_COLUMN_COUNT:
        return None
    try:
        gains = tuple(float(value) for value in gains)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in gains):
        return None
    output = dict(metadata)
    output["projector_gains"] = gains
    return output


def _nominal_projector_gains(metadata):
    """Resolve static gains used for scalar-energy accounting.

    tensor_rms adds a prompt-dependent common multiplier at runtime.  Dividing
    by the profile RMS gives a neutral-energy nominal profile for attenuation,
    but its unknown runtime multiplier makes automatic headroom boosts unsafe.
    """
    gains = tuple(float(value) for value in metadata["projector_gains"])
    dynamic = metadata.get("projector_normalization") == "tensor_rms"
    if dynamic:
        rms = math.sqrt(sum(value * value for value in gains) / len(gains))
        if rms > 1e-12:
            gains = tuple(value / rms for value in gains)
    return gains, dynamic


def _projector_column_scales(
    entries,
    gains,
    mode,
    max_boost,
    dynamic=False,
    budget=_SAFE_ENERGY_BUDGET,
):
    """Budget projector LoRA scalar energy after Fusion Control's 12 gains."""
    participants = [
        idx for idx, entry in enumerate(entries)
        if entry["is_krea_text"] and entry["projector_text"] and float(entry["cw"]) != 0.0
    ]
    if not participants:
        return (1.0,) * _PROJECTOR_COLUMN_COUNT, None

    energy = math.sqrt(sum(float(entries[idx]["cw"]) ** 2 for idx in participants))
    allow_boost = mode == "Use headroom" and not dynamic
    max_boost = max(1.0, float(max_boost))
    scales = []
    for gain in gains:
        effective_energy = abs(float(gain)) * energy
        if effective_energy <= 1e-12:
            scale = max_boost if allow_boost else 1.0
        else:
            scale = budget / effective_energy
            scale = min(max_boost if allow_boost else 1.0, scale)
        scales.append(scale)

    return tuple(scales), {
        "raw_energy": energy,
        "dynamic": dynamic,
        "boosted": any(scale > 1.0 + 1e-12 for scale in scales),
        "limited": any(scale < 1.0 - 1e-12 for scale in scales),
    }


def _scale_projector_lora_columns(lora, scales):
    """Scale projector LoRA delta columns without scaling the base projector.

    Standard LoRA/PEFT adapters are adjusted on their input/down matrix. Direct
    ``.diff`` projector patches are adjusted directly. Unsupported adapter
    families are returned unchanged so the caller can use a conservative
    uniform strength fallback.
    """
    if len(scales) != _PROJECTOR_COLUMN_COUNT:
        raise ValueError(f"Expected {_PROJECTOR_COLUMN_COUNT} projector scales")

    adjusted = dict(lora)
    transformed = False
    for key, value in lora.items():
        if not torch.is_tensor(value) or not value.is_floating_point() or value.ndim < 2:
            continue
        lower = key.lower()
        if not _PROJECTOR_RE.search(lower) or value.shape[-1] != _PROJECTOR_COLUMN_COUNT:
            continue

        is_down = lower.endswith("lora_down.weight") or lower.endswith("lora_a.weight")
        is_diff = lower.endswith(".diff")
        is_direct_weight = lower.endswith("projector.weight") and value.shape[-2] == 1
        if not (is_down or is_diff or is_direct_weight):
            continue

        scale = torch.tensor(scales, device=value.device, dtype=value.dtype)
        adjusted[key] = value * scale.reshape(*([1] * (value.ndim - 1)), -1)
        transformed = True

    return adjusted, transformed


def _parse_numeric_vector(vector, required_size=1):
    """Parse a numeric Krea2 vector into a full safety-analysis vector.

    Donut's normal auto-vector path intentionally sizes a Krea2 vector only up
    to the highest block actually present in that LoRA. A LoRA that only has
    blocks 0..15 therefore has a valid 17-value vector (base + 16 blocks), not
    a 29-value vector. Safe Stack must accept that rather than requiring all 28
    architectural blocks.

    Returns ``(padded_values, original_size)``. Missing higher Krea2 blocks are
    zero-filled for the energy calculation because the LoRA has no weights for
    them. The caller later trims back to ``original_size`` before handing the
    vector to the normal Donut block loader, preserving its original shape.
    """
    if not vector:
        values = [1.0] * max(1, min(required_size, _KREA_VECTOR_SIZE))
        return values + [0.0] * (_KREA_VECTOR_SIZE - len(values)), len(values)

    parts = [part.strip() for part in vector.split(",")]
    if len(parts) < required_size or len(parts) > _KREA_VECTOR_SIZE:
        return None

    try:
        values = [float(part) for part in parts]
    except (TypeError, ValueError):
        return None

    original_size = len(values)
    values.extend([0.0] * (_KREA_VECTOR_SIZE - original_size))
    return values, original_size


def _format_vector(values):
    def fmt(value):
        if abs(value) < 1e-12:
            return "0"
        if abs(value - 1.0) < 1e-12:
            return "1"
        return f"{value:.6g}"

    return ",".join(fmt(value) for value in values)


def _normalise_krea_vectors(entries, budget=_SAFE_ENERGY_BUDGET):
    """Cap effective per-block RMS energy while preserving relative strengths.

    Each effective contribution is model_weight * block_weight. If the root
    sum of squares of all Krea2 contributions in a block exceeds ``budget``,
    every LoRA touching that block is attenuated by the same factor. Blocks
    below budget are left equivalent apart from numeric vector formatting.
    """
    vectors = []
    original_sizes = []
    eligible = []

    for entry in entries:
        if not entry["is_krea"]:
            vectors.append(None)
            original_sizes.append(0)
            eligible.append(False)
            continue

        block_indices = entry["krea_blocks"]
        required_size = (max(block_indices) + 2) if block_indices else 1
        parsed = _parse_numeric_vector(entry["vector"], required_size=required_size)
        if parsed is None:
            vectors.append(None)
            original_sizes.append(0)
            eligible.append(False)
            print(
                f"[DonutApplyLoRAStack] Safe Stack: '{entry['name']}' uses a "
                "non-numeric vector or one too short for its populated Krea2 "
                "blocks; leaving it unchanged"
            )
            continue

        values, original_size = parsed
        vectors.append(values)
        original_sizes.append(original_size)
        eligible.append(True)

    scales = [[1.0] * _KREA_VECTOR_SIZE for _ in entries]
    limited_blocks = []

    for block_idx in range(_KREA_VECTOR_SIZE):
        energy_sq = 0.0
        participants = []
        for idx, entry in enumerate(entries):
            if not eligible[idx]:
                continue
            effective = float(entry["mw"]) * vectors[idx][block_idx]
            if effective == 0.0:
                continue
            energy_sq += effective * effective
            participants.append(idx)

        energy = math.sqrt(energy_sq)
        if energy > budget and participants:
            scale = budget / energy
            limited_blocks.append((block_idx, energy, scale))
            for idx in participants:
                scales[idx][block_idx] = scale

    adjusted = []
    for idx, entry in enumerate(entries):
        if not eligible[idx]:
            adjusted.append(entry["vector"])
            continue

        adjusted_full = [
            value * scales[idx][block_idx]
            for block_idx, value in enumerate(vectors[idx])
        ]
        adjusted.append(_format_vector(adjusted_full[:original_sizes[idx]]))

    return adjusted, limited_blocks


def _normalise_fused_text_weights(entries, component=None, budget=_SAFE_ENERGY_BUDGET):
    """RMS-limit a Krea2 fused-text component as one shared scalar bucket."""
    participants = [
        idx for idx, entry in enumerate(entries)
        if (
            entry["is_krea_text"]
            and entry["fused_text"]
            and (component is None or bool(entry[component]))
            and float(entry["cw"]) != 0.0
        )
    ]
    if not participants:
        return [float(entry["cw"]) for entry in entries], None

    energy = math.sqrt(sum(float(entries[idx]["cw"]) ** 2 for idx in participants))
    scale = min(1.0, budget / energy) if energy > 0.0 else 1.0
    weights = [float(entry["cw"]) for entry in entries]
    if scale < 1.0:
        for idx in participants:
            weights[idx] *= scale
        return weights, (energy, scale)
    return weights, None


class DonutApplyLoRAStackSafe:
    """Drop-in replacement for DonutApplyLoRAStack with optional Krea2 safety."""

    class_type = "CUSTOM"
    aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_stack": ("LORA_STACK",),
                "safe_stack": (["Off", "On"], {
                    "default": "Off",
                    "tooltip": (
                        "Krea2 only. RMS-limits overlapping LoRA strength per block "
                        "so stacked LoRAs cannot collectively exceed one full-strength "
                        "LoRA worth of block energy. Off preserves legacy behaviour."
                    ),
                }),
                "fusion_aware": (list(_FUSION_AWARE_MODES), {
                    "default": "Off",
                    "tooltip": (
                        "Requires the model output from Donut Krea2 Fusion Control. "
                        "Budgets projector LoRA columns against the resolved 12-channel "
                        "projector gains. Use headroom can boost quiet columns; dynamic "
                        "tensor_rms profiles are attenuation-only."
                    ),
                }),
                "max_fusion_boost": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Maximum per-column LoRA boost in Use headroom mode.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "show_help")
    FUNCTION = "apply_stack"
    CATEGORY = "Comfyanonymous/LoRA"

    def apply_stack(
        self,
        model,
        clip,
        lora_stack=None,
        safe_stack="Off",
        fusion_aware="Off",
        max_fusion_boost=2.0,
    ):
        help_url = (
            "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/"
            "wiki/LoRA-Nodes#cr-apply-lora-stack"
        )

        if lora_stack is None or len(lora_stack) == 0:
            return (model, clip, help_url)
        if fusion_aware not in _FUSION_AWARE_MODES:
            raise ValueError(f"Unknown fusion-aware safety mode: {fusion_aware}")
        max_fusion_boost = float(max_fusion_boost)
        if not math.isfinite(max_fusion_boost) or max_fusion_boost < 1.0:
            raise ValueError("max_fusion_boost must be finite and at least 1.0")

        # Preserve DonutApplyLoRAStack's duplicate semantics before doing any
        # safety calculation, otherwise duplicate entries would consume budget
        # even though the apply pass would skip them.
        entries = []
        seen = set()
        for name, mw, cw, bv in lora_stack:
            if mw == 0.0 and cw == 0.0:
                continue
            if name in seen:
                print(
                    f"[DonutApplyLoRAStack] Skipping duplicate LoRA '{name}' "
                    "(already applied this run)"
                )
                continue
            seen.add(name)

            path = folder_paths.get_full_path("loras", name)
            lora = comfy.utils.load_torch_file(path, safe_load=True)
            krea_blocks = _krea2_block_indices(lora)

            # Match the original apply node's automatic vector behaviour.
            vector = bv
            if not vector:
                block_nums = set()
                for key in lora.keys():
                    match = _KREA_BLOCK_RE.search(key)
                    if match:
                        block_nums.add(int(match.group(1)))
                    elif "layers." in key:
                        layer_match = re.search(r"layers\.(\d+)", key)
                        if layer_match:
                            block_nums.add(int(layer_match.group(1)))

                if block_nums:
                    vector = ",".join(["1"] * (max(block_nums) + 2))
                else:
                    vector = ",".join(["1"] * 13)

            has_real_te = _lora_has_real_text_encoder(lora)
            lora_main, lora_text = _split_fused_text(lora)
            fused_text = bool(lora_text) and not has_real_te
            projector_text, other_text = _split_projector_text(lora_text)

            entries.append({
                "name": name,
                "mw": float(mw),
                "cw": float(cw),
                "vector": vector,
                "lora": lora,
                "lora_main": lora_main,
                "lora_text": lora_text,
                "projector_text": projector_text,
                "other_text": other_text,
                "has_projector_text": bool(projector_text),
                "has_other_text": bool(other_text),
                "fused_text": fused_text,
                "is_krea": bool(krea_blocks) and max(krea_blocks) < _KREA_BLOCK_COUNT,
                "is_krea_text": _is_krea2_text_lora(lora),
                "krea_blocks": krea_blocks,
            })

        fusion_metadata = _read_fusion_budget(model) if fusion_aware != "Off" else None
        fusion_aware_active = safe_stack == "On" and fusion_metadata is not None and fusion_aware != "Off"
        if fusion_aware != "Off" and safe_stack != "On":
            print("[DonutApplyLoRAStack] Fusion-aware safety requires safe_stack=On; using legacy behavior")
        elif fusion_aware != "Off" and fusion_metadata is None:
            print(
                "[DonutApplyLoRAStack] Fusion-aware safety found no Fusion Control metadata; "
                "connect Donut Krea2 Fusion Control's model output before this node"
            )

        if safe_stack == "On":
            adjusted_vectors, limited_blocks = _normalise_krea_vectors(entries)
            for idx, entry in enumerate(entries):
                entry["vector"] = adjusted_vectors[idx]

            if limited_blocks:
                block_labels = ["base" if idx == 0 else str(idx - 1) for idx, _, _ in limited_blocks]
                print(
                    "[DonutApplyLoRAStack] Safe Stack: limited Krea2 block energy "
                    f"in {len(limited_blocks)} bucket(s): {', '.join(block_labels)}"
                )

            if fusion_aware_active:
                other_weights, other_limit = _normalise_fused_text_weights(
                    entries,
                    component="has_other_text",
                )
                projector_gains, dynamic = _nominal_projector_gains(fusion_metadata)
                projector_scales, projector_report = _projector_column_scales(
                    entries,
                    projector_gains,
                    fusion_aware,
                    max_fusion_boost,
                    dynamic=dynamic,
                )

                unsupported = []
                for idx, entry in enumerate(entries):
                    entry["fusion_split"] = bool(entry["fused_text"] and entry["is_krea_text"])
                    entry["effective_cw"] = entry["cw"]
                    entry["effective_other_cw"] = other_weights[idx]
                    entry["effective_projector_cw"] = entry["cw"]
                    entry["effective_projector_text"] = entry["projector_text"]
                    if not (entry["is_krea_text"] and entry["projector_text"]):
                        continue

                    adjusted_projector, transformed = _scale_projector_lora_columns(
                        entry["projector_text"],
                        projector_scales,
                    )
                    if transformed:
                        entry["effective_projector_text"] = adjusted_projector
                    else:
                        # A scalar fallback must use the most restrictive column
                        # scale to keep every projector column within budget.
                        uniform_scale = min(projector_scales)
                        entry["effective_projector_cw"] = entry["cw"] * uniform_scale
                        unsupported.append(entry["name"])

                if other_limit:
                    energy, scale = other_limit
                    print(
                        "[DonutApplyLoRAStack] Safe Stack: limited non-projector "
                        f"Krea2 fused-text energy {energy:.3f}x -> {_SAFE_ENERGY_BUDGET:.3f}x "
                        f"(scale {scale:.3f})"
                    )
                if projector_report:
                    print(
                        "[DonutApplyLoRAStack] Fusion-aware projector budget: "
                        f"LoRA energy={projector_report['raw_energy']:.3f}x, "
                        f"column scales={min(projector_scales):.3f}..{max(projector_scales):.3f}, "
                        f"mode={fusion_aware}"
                    )
                    if projector_report["dynamic"] and fusion_aware == "Use headroom":
                        print(
                            "[DonutApplyLoRAStack] Fusion-aware projector budget: "
                            "tensor_rms is prompt-dependent, so automatic boosts were disabled"
                        )
                if unsupported:
                    print(
                        "[DonutApplyLoRAStack] Fusion-aware projector budget used a "
                        "conservative scalar fallback for unsupported adapter format(s): "
                        + ", ".join(unsupported)
                    )
            else:
                adjusted_text_weights, text_limit = _normalise_fused_text_weights(entries)
                for idx, entry in enumerate(entries):
                    entry["fusion_split"] = False
                    entry["effective_cw"] = adjusted_text_weights[idx]
                if text_limit:
                    energy, scale = text_limit
                    print(
                        "[DonutApplyLoRAStack] Safe Stack: limited Krea2 fused-text "
                        f"energy {energy:.3f}x -> {_SAFE_ENERGY_BUDGET:.3f}x "
                        f"(scale {scale:.3f})"
                    )
        else:
            for entry in entries:
                entry["fusion_split"] = False
                entry["effective_cw"] = entry["cw"]

        unet, text_enc = model, clip
        loader = LoraLoaderBlockWeight()

        for entry in entries:
            mw = entry["mw"]
            lora = entry["lora"]
            lora_main = entry["lora_main"]
            lora_text = entry["lora_text"]
            fused_text = entry["fused_text"]
            vector = entry["vector"]

            # 1) block-weighted diffusion-model merge.
            merge_main = lora_main if fused_text else lora
            if mw != 0.0 and merge_main:
                unet, _, _ = loader.load_lora_for_models(
                    unet,
                    None,
                    merge_main,
                    strength_model=mw,
                    strength_clip=0.0,
                    inverse=False,
                    seed=0,
                    A=1.0,
                    B=1.0,
                    block_vector=vector,
                )

            # 2) text handling, matching the original DonutApplyLoRAStack.
            if fused_text:
                text_applications = (
                    (
                        (entry["other_text"], entry["effective_other_cw"]),
                        (entry["effective_projector_text"], entry["effective_projector_cw"]),
                    )
                    if entry["fusion_split"]
                    else ((lora_text, entry["effective_cw"]),)
                )
                for text_lora, text_strength in text_applications:
                    if not text_lora or text_strength == 0.0:
                        continue
                    unet, _, _ = loader.load_lora_for_models(
                        unet,
                        None,
                        text_lora,
                        strength_model=text_strength,
                        strength_clip=0.0,
                        inverse=False,
                        seed=0,
                        A=1.0,
                        B=1.0,
                        block_vector=_TEXT_MERGE_VECTOR,
                    )
            elif entry["effective_cw"] != 0.0:
                _, text_enc = comfy.sd.load_lora_for_models(
                    unet,
                    text_enc,
                    lora,
                    0.0,
                    entry["effective_cw"],
                )

        return (unet, text_enc, help_url)


NODE_CLASS_MAPPINGS = {
    "DonutApplyLoRAStack": DonutApplyLoRAStackSafe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutApplyLoRAStack": "Donut Apply LoRA Stack",
}

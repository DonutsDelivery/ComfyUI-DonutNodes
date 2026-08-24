"""Safe Krea2 LoRA stack application.

Overrides DonutApplyLoRAStack with an opt-in per-block RMS energy limiter for
Krea2 LoRAs. Safe mode is deliberately conservative and off by default so old
workflows retain their exact behaviour.
"""

import math
import re

import comfy.sd
import comfy.utils
import folder_paths

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


def _is_krea2_lora(lora):
    """Return True when a LoRA contains Krea2-style single-stream blocks."""
    block_nums = set()
    for key in lora.keys():
        match = _KREA_BLOCK_RE.search(key)
        if match:
            block_nums.add(int(match.group(1)))
    return bool(block_nums) and max(block_nums) < _KREA_BLOCK_COUNT


def _parse_numeric_vector(vector):
    """Parse a Krea2 block vector, returning None for symbolic/random vectors."""
    if not vector:
        return [1.0] * _KREA_VECTOR_SIZE

    parts = [part.strip() for part in vector.split(",")]
    if len(parts) != _KREA_VECTOR_SIZE:
        return None

    values = []
    try:
        for part in parts:
            values.append(float(part))
    except (TypeError, ValueError):
        return None
    return values


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
    below budget are left byte-for-byte equivalent apart from vector formatting.
    """
    vectors = []
    eligible = []

    for entry in entries:
        if not entry["is_krea"]:
            vectors.append(None)
            eligible.append(False)
            continue

        parsed = _parse_numeric_vector(entry["vector"])
        vectors.append(parsed)
        eligible.append(parsed is not None)
        if parsed is None:
            print(
                f"[DonutApplyLoRAStack] Safe Stack: '{entry['name']}' uses a "
                "non-numeric or non-Krea2-sized block vector; leaving it unchanged"
            )

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
        adjusted.append(
            _format_vector([
                value * scales[idx][block_idx]
                for block_idx, value in enumerate(vectors[idx])
            ])
        )

    return adjusted, limited_blocks


def _normalise_fused_text_weights(entries, budget=_SAFE_ENERGY_BUDGET):
    """RMS-limit Krea2 fused-text weights as one shared non-block bucket."""
    participants = [
        idx for idx, entry in enumerate(entries)
        if entry["is_krea"] and entry["fused_text"] and float(entry["cw"]) != 0.0
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
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "show_help")
    FUNCTION = "apply_stack"
    CATEGORY = "Comfyanonymous/LoRA"

    def apply_stack(self, model, clip, lora_stack=None, safe_stack="Off"):
        help_url = (
            "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/"
            "wiki/LoRA-Nodes#cr-apply-lora-stack"
        )

        if lora_stack is None or len(lora_stack) == 0:
            return (model, clip, help_url)

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

            entries.append({
                "name": name,
                "mw": float(mw),
                "cw": float(cw),
                "vector": vector,
                "lora": lora,
                "lora_main": lora_main,
                "lora_text": lora_text,
                "fused_text": fused_text,
                "is_krea": _is_krea2_lora(lora),
            })

        if safe_stack == "On":
            adjusted_vectors, limited_blocks = _normalise_krea_vectors(entries)
            adjusted_text_weights, text_limit = _normalise_fused_text_weights(entries)
            for idx, entry in enumerate(entries):
                entry["vector"] = adjusted_vectors[idx]
                entry["effective_cw"] = adjusted_text_weights[idx]

            if limited_blocks:
                block_labels = ["base" if idx == 0 else str(idx - 1) for idx, _, _ in limited_blocks]
                print(
                    "[DonutApplyLoRAStack] Safe Stack: limited Krea2 block energy "
                    f"in {len(limited_blocks)} bucket(s): {', '.join(block_labels)}"
                )
            if text_limit:
                energy, scale = text_limit
                print(
                    "[DonutApplyLoRAStack] Safe Stack: limited Krea2 fused-text "
                    f"energy {energy:.3f}x -> {_SAFE_ENERGY_BUDGET:.3f}x "
                    f"(scale {scale:.3f})"
                )
        else:
            for entry in entries:
                entry["effective_cw"] = entry["cw"]

        unet, text_enc = model, clip
        loader = LoraLoaderBlockWeight()

        for entry in entries:
            mw = entry["mw"]
            cw = entry["effective_cw"]
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
                if cw != 0.0:
                    unet, _, _ = loader.load_lora_for_models(
                        unet,
                        None,
                        lora_text,
                        strength_model=cw,
                        strength_clip=0.0,
                        inverse=False,
                        seed=0,
                        A=1.0,
                        B=1.0,
                        block_vector=_TEXT_MERGE_VECTOR,
                    )
            elif cw != 0.0:
                _, text_enc = comfy.sd.load_lora_for_models(
                    unet,
                    text_enc,
                    lora,
                    0.0,
                    cw,
                )

        return (unet, text_enc, help_url)


NODE_CLASS_MAPPINGS = {
    "DonutApplyLoRAStack": DonutApplyLoRAStackSafe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutApplyLoRAStack": "Donut Apply LoRA Stack",
}

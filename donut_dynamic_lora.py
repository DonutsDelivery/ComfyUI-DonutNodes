"""Resizable LoRA rows; reuse Donut resolution/metadata and whole-stack safety."""
from __future__ import annotations
from copy import deepcopy
import json
import math
import logging
from pathlib import Path

from . import donut_lora_nodes as _legacy
from .donut_lora_nodes import DonutLoRAStack
from .DonutSafeApplyLoRAStack import DonutApplyLoRAStackSafe


def parse_slots(value):
    if isinstance(value, str):
        if len(value) > 2_000_000:
            raise ValueError("LoRA slot configuration is too large")
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("LoRA slots must be a JSON array")
    result, ids = [], set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"LoRA row {index + 1} must be an object")
        row = dict(id=str(index + 1), enabled=True, lora_name="None", model_weight=1.0,
                   clip_weight=1.0, block_vector="", block_preset="None", lora_hash="", inherit_block_vector=False)
        row.update(item)
        row["id"] = str(row["id"])
        if not row["id"] or row["id"] in ids:
            raise ValueError(f"Duplicate/empty LoRA row ID: {row['id']!r}")
        ids.add(row["id"])
        for flag in ("enabled", "inherit_block_vector"):
            if not isinstance(row[flag], bool):
                raise ValueError(f"Row {index + 1}: {flag} must be true or false")
        for key in ("lora_name", "block_vector", "block_preset", "lora_hash"):
            if not isinstance(row[key], str):
                raise ValueError(f"Row {index + 1}: {key} must be text")
        for key in ("model_weight", "clip_weight"):
            if isinstance(row[key], bool):
                raise ValueError(f"Row {index + 1}: {key} must be a finite number")
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or not -1000 <= row[key] <= 1000:
                raise ValueError(f"Row {index + 1}: {key} must be finite and between -1000 and 1000")
        result.append(row)
    return result


def slot_inputs():
    legacy = DonutLoRAStack.INPUT_TYPES()["required"]
    return {
        "model_type": deepcopy(legacy["model_type"]),
        "slots_json": ("STRING", {"default": "[]", "multiline": True, "dynamicPrompts": False,
                                   "donut_loras": list(legacy["lora_name_1"][0]),
                                   "donut_presets": list(legacy["block_preset_1"][0])}),
        "global_block_vector": ("STRING", {"default": "", "tooltip": "Used only by rows with Inherit global enabled."}),
        "civitai_lookup": deepcopy(legacy["civitai_lookup"]),
    }


def build_dynamic_stack(slots_json, model_type="Auto", global_block_vector="", civitai_lookup="Off", lora_stack=None):
    rows = parse_slots(slots_json)
    stack, metadata = list(lora_stack or ()), []
    builder = DonutLoRAStack()
    # The legacy builder remains a single source of truth for block vectors,
    # hash-based relocation and CivitAI metadata. Applying in chunks would be
    # WRONG: the limiter must see the complete stack, once, after this loop.
    for start in range(0, len(rows), 3):
        chunk = rows[start:start + 3]
        args = dict(model_type=model_type, civitai_lookup=civitai_lookup, lora_stack=stack,
                    unique_id="dynamic-slots", extra_pnginfo={"workflow": {"nodes": [
                        {"id": "dynamic-slots", "properties": {"lora_hashes": [r["lora_hash"] for r in chunk]}}
                    ]}})
        for slot in range(1, 4):
            row = chunk[slot - 1] if slot <= len(chunk) else None
            args.update({f"switch_{slot}": "On" if row and row["enabled"] else "Off",
                         f"lora_name_{slot}": row["lora_name"] if row else "None",
                         f"model_weight_{slot}": row["model_weight"] if row else 1.0,
                         f"clip_weight_{slot}": row["clip_weight"] if row else 1.0,
                         f"block_preset_{slot}": row["block_preset"] if row else "None",
                         f"block_vector_{slot}": (global_block_vector if row["inherit_block_vector"] else row["block_vector"]) if row else ""})
        previous_length = len(stack)
        built = builder.build_stack(**args)
        stack = list(built["result"][0])
        ui = built.get("ui", {})
        texts, images = ui.get("text", []), ui.get("images", [])
        resolved = iter(stack[previous_length:])
        for index, row in enumerate(chunk):
            file_hash = row["lora_hash"]
            resolved_name = row["lora_name"]
            if row["enabled"] and row["lora_name"] != "None":
                resolved_name = next(resolved)[0]
                hash_fn = getattr(_legacy, "get_or_compute_hash", None)
                if hash_fn is not None:
                    import folder_paths
                    path = folder_paths.get_full_path("loras", resolved_name)
                    if path and Path(path).is_file():
                        try:
                            file_hash = hash_fn(path, use_cache=True)
                        except (OSError, ValueError) as exc:
                            logging.getLogger(__name__).warning("LoRA metadata hash unavailable for %s: %s", resolved_name, exc)
            metadata.append({"id": row["id"], "lora_name": row["lora_name"], "lora_hash": file_hash,
                             "resolved_name": resolved_name,
                             "text": texts[index + 3] if index + 3 < len(texts) else row["lora_name"],
                             "image": images[index] if index < len(images) else None})
    return stack, metadata


class DonutDynamicLoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": slot_inputs(), "optional": {"lora_stack": ("LORA_STACK",)}}
    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build"
    CATEGORY = "donut/LoRA"
    OUTPUT_NODE = True

    def build(self, **kwargs):
        stack, metadata = build_dynamic_stack(**kwargs)
        return {"ui": {"donut_loras": metadata}, "result": (stack,)}

    @classmethod
    def IS_CHANGED(cls, slots_json="[]", lora_stack=None, **kwargs):
        import folder_paths
        names = [r["lora_name"] for r in parse_slots(slots_json) if r["enabled"]]
        names += [entry[0] for entry in (lora_stack or ())]
        result = []
        for name in names:
            path = folder_paths.get_full_path("loras", name) if name != "None" else None
            if path and Path(path).is_file():
                stat = Path(path).stat()
                result.append((name, str(path), stat.st_mtime_ns, stat.st_size))
            else:
                result.append((name, None))
        return tuple(result)


class DonutLoRALoader(DonutDynamicLoRAStack):
    @classmethod
    def INPUT_TYPES(cls):
        safety = deepcopy(DonutApplyLoRAStackSafe.INPUT_TYPES())
        required = {"model": ("MODEL",), "clip": ("CLIP",), **slot_inputs()}
        required.update({key: spec for key, spec in safety["required"].items() if key not in ("model", "clip", "lora_stack")})
        # Keep execution_mode after the safety widgets, as in the original node.
        required.update(safety.get("optional", {}))
        return {"required": required, "optional": {"lora_stack": ("LORA_STACK",)}}
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK", "STRING")
    RETURN_NAMES = ("model", "clip", "lora_stack", "show_help")
    FUNCTION = "load"
    OUTPUT_NODE = False

    def load(self, model, clip, slots_json="[]", model_type="Auto", global_block_vector="", civitai_lookup="Off", lora_stack=None, **safety):
        stack, metadata = build_dynamic_stack(slots_json, model_type, global_block_vector, civitai_lookup, lora_stack)
        import folder_paths
        for name, mw, cw, _ in stack:
            if (mw or cw) and not folder_paths.get_full_path("loras", name):
                raise FileNotFoundError(f"Enabled LoRA could not be resolved: {name}")
        model, clip, help_text = DonutApplyLoRAStackSafe().apply_stack(model, clip, stack, **safety)
        return {"ui": {"donut_loras": metadata}, "result": (model, clip, stack, help_text)}


NODE_CLASS_MAPPINGS = {"DonutDynamicLoRAStack": DonutDynamicLoRAStack, "DonutLoRALoader": DonutLoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutDynamicLoRAStack": "Donut Dynamic LoRA Stack", "DonutLoRALoader": "Donut LoRA Loader · Dynamic"}

import comfy.sd
import comfy.utils
import folder_paths
import json
import os
import re
import shutil
from .lora_block_weight import LoraLoaderBlockWeight

# CivitAI integration imports
try:
    from .shared.lora_hash import get_or_compute_hash
    from .shared.civitai_api import get_cache
    from .shared.config import get_civitai_api_key
    from .shared.lora_resolver import resolve_lora, relative_name_for
    HAS_CIVITAI = True
except ImportError:
    HAS_CIVITAI = False
    def get_civitai_api_key(): return ""
    def resolve_lora(name, expected_hash=None, auto_download=True, api_key=None):
        path = folder_paths.get_full_path("loras", name) if name and name != "None" else None
        return (path, "exact" if path and os.path.exists(path) else "missing")
    def relative_name_for(path): return os.path.basename(path) if path else None
    print("[DonutLoRAStack] CivitAI integration not available")

# ------------------------------------------------------------------------
# DonutLoRAStack: build up to 3 LoRAs with independent model & clip strengths + optional per-block vectors
# Now with CivitAI metadata integration!
#
# Block weight vector format: position 0 = non-block components (embedders, refiners, etc.),
# positions 1+ = block/layer weights
#
# Architecture block counts:
#   SDXL: 12 blocks (IN0-8, MID, OUT0-2)
#   SD1.5: 17 blocks (IN0-11, MID, OUT0-3)
#   FLUX: 57 blocks (19 double + 38 single)
#   ZIT: 30 transformer layers (+ x_embedder, refiners, etc. in position 0)

MODEL_TYPES = ["Auto", "SDXL", "SD15", "FLUX", "ZIT", "ZIT-NE"]

# Presets by model type
# All presets use contiguous ranges only (no holes in the middle)
# SDXL: IN blocks (0-8), MID block (9), OUT blocks (10-11) = 12 blocks
SDXL_PRESETS = {
    "ALL": ",".join(["1"] * 13),
    # Enable from start (first N blocks)
    "IN-ONLY": ",".join(["1"] + ["1"]*9 + ["0"]*3),              # IN only (0-8)
    "IN-MID": ",".join(["1"] + ["1"]*10 + ["0"]*2),              # IN + MID (0-9)
    # Enable from end (last N blocks)
    "OUT-ONLY": ",".join(["1"] + ["0"]*10 + ["1"]*2),            # OUT only (10-11)
    "MID-OUT": ",".join(["1"] + ["0"]*9 + ["1"]*3),              # MID + OUT (9-11)
    # Contiguous middle sections (gaps on edges only)
    "MID-ONLY": ",".join(["1"] + ["0"]*9 + ["1"] + ["0"]*2),     # MID only (9)
    "CORE": ",".join(["1"] + ["0"]*4 + ["1"]*4 + ["0"]*4),       # Core (4-7)
    "CORE-WIDE": ",".join(["1"] + ["0"]*3 + ["1"]*6 + ["0"]*3),  # Wider core (3-8)
    # Eased gradients
    "EASE-IN": ",".join(["1"] + [f"{i/11:.2f}" for i in range(12)]),
    "EASE-OUT": ",".join(["1"] + [f"{1-i/11:.2f}" for i in range(12)]),
    "EASE-MID": ",".join(["1"] + [f"{1-abs(i-5.5)/5.5:.2f}" for i in range(12)]),
    "HALF": ",".join(["1"] + ["0.5"]*12),
}

# SD1.5: IN blocks (0-11), MID block (12), OUT blocks (13-16) = 17 blocks
SD15_PRESETS = {
    "ALL": ",".join(["1"] * 18),
    # Enable from start (first N blocks)
    "IN-ONLY": ",".join(["1"] + ["1"]*12 + ["0"]*5),             # IN only (0-11)
    "IN-MID": ",".join(["1"] + ["1"]*13 + ["0"]*4),              # IN + MID (0-12)
    # Enable from end (last N blocks)
    "OUT-ONLY": ",".join(["1"] + ["0"]*13 + ["1"]*4),            # OUT only (13-16)
    "MID-OUT": ",".join(["1"] + ["0"]*12 + ["1"]*5),             # MID + OUT (12-16)
    # Contiguous middle sections (gaps on edges only)
    "MID-ONLY": ",".join(["1"] + ["0"]*12 + ["1"] + ["0"]*4),    # MID only (12)
    "CORE": ",".join(["1"] + ["0"]*6 + ["1"]*5 + ["0"]*6),       # Core (6-10)
    "CORE-WIDE": ",".join(["1"] + ["0"]*4 + ["1"]*9 + ["0"]*4),  # Wider core (4-12)
    # Eased gradients
    "EASE-IN": ",".join(["1"] + [f"{i/16:.2f}" for i in range(17)]),
    "EASE-OUT": ",".join(["1"] + [f"{1-i/16:.2f}" for i in range(17)]),
    "EASE-MID": ",".join(["1"] + [f"{1-abs(i-8)/8:.2f}" for i in range(17)]),
    "HALF": ",".join(["1"] + ["0.5"]*17),
}

# FLUX: Double blocks (0-18), Single blocks (19-56) = 57 blocks
FLUX_PRESETS = {
    "ALL": ",".join(["1"] * 58),
    # Enable from start
    "DOUBLE-ONLY": ",".join(["1"] + ["1"]*19 + ["0"]*38),        # Double only (0-18)
    "DOUBLE-EARLY-SINGLE": ",".join(["1"] + ["1"]*38 + ["0"]*19),# Double + first half single (0-37)
    # Enable from end
    "SINGLE-ONLY": ",".join(["1"] + ["0"]*19 + ["1"]*38),        # Single only (19-56)
    "LATE-DOUBLE-SINGLE": ",".join(["1"] + ["0"]*10 + ["1"]*47), # Late double + all single (10-56)
    # Contiguous middle sections (gaps on edges only)
    "EARLY-DOUBLE": ",".join(["1"] + ["1"]*10 + ["0"]*47),       # First half of double (0-9)
    "LATE-SINGLE": ",".join(["1"] + ["0"]*38 + ["1"]*19),        # Second half of single (38-56)
    "MID-SECTION": ",".join(["1"] + ["0"]*10 + ["1"]*38 + ["0"]*9),  # Middle (10-47)
    "CORE": ",".join(["1"] + ["0"]*19 + ["1"]*19 + ["0"]*19),    # Core center (19-37)
    "CORE-WIDE": ",".join(["1"] + ["0"]*14 + ["1"]*29 + ["0"]*14),   # Wider core (14-42)
    # Eased gradients
    "EASE-IN": ",".join(["1"] + [f"{i/56:.2f}" for i in range(57)]),
    "EASE-OUT": ",".join(["1"] + [f"{1-i/56:.2f}" for i in range(57)]),
    "EASE-MID": ",".join(["1"] + [f"{1-abs(i-28)/28:.2f}" for i in range(57)]),
    "HALF": ",".join(["1"] + ["0.5"]*57),
}

# ZIT (Z-Image Turbo): 30 transformer layers
# Early (0-5), Lower-Mid (6-14), Upper-Mid (15-23), Late (24-29)
ZIT_PRESETS = {
    "ALL": ",".join(["1"] * 31),
    # Enable from start (cumulative)
    "EARLY-ONLY": ",".join(["1"] + ["1"]*6 + ["0"]*24),          # Early only (0-5)
    "EARLY-LOWMID": ",".join(["1"] + ["1"]*15 + ["0"]*15),       # Early + LowMid (0-14)
    "EARLY-MID": ",".join(["1"] + ["1"]*24 + ["0"]*6),           # Early + all Mid (0-23)
    # Enable from end (cumulative)
    "LATE-ONLY": ",".join(["1"] + ["0"]*24 + ["1"]*6),           # Late only (24-29)
    "UPMID-LATE": ",".join(["1"] + ["0"]*15 + ["1"]*15),         # UpMid + Late (15-29)
    "MID-LATE": ",".join(["1"] + ["0"]*6 + ["1"]*24),            # All Mid + Late (6-29)
    # Contiguous middle sections (gaps on edges only)
    "LOWMID-ONLY": ",".join(["1"] + ["0"]*6 + ["1"]*9 + ["0"]*15),   # LowMid only (6-14)
    "UPMID-ONLY": ",".join(["1"] + ["0"]*15 + ["1"]*9 + ["0"]*6),    # UpMid only (15-23)
    "MID-ONLY": ",".join(["1"] + ["0"]*6 + ["1"]*18 + ["0"]*6),      # All Mid (6-23)
    "CORE": ",".join(["1"] + ["0"]*10 + ["1"]*10 + ["0"]*10),        # Core center (10-19)
    "CORE-WIDE": ",".join(["1"] + ["0"]*8 + ["1"]*14 + ["0"]*8),     # Wider core (8-21)
    "CORE-NARROW": ",".join(["1"] + ["0"]*12 + ["1"]*6 + ["0"]*12),  # Narrow core (12-17)
    # Eased gradients
    "EASE-IN": ",".join(["1"] + [f"{i/29:.2f}" for i in range(30)]),
    "EASE-OUT": ",".join(["1"] + [f"{1-i/29:.2f}" for i in range(30)]),
    "EASE-MID": ",".join(["1"] + [f"{1-abs(i-14.5)/14.5:.2f}" for i in range(30)]),
    "EASE-IN-QUAD": ",".join(["1"] + [f"{(i/29)**2:.2f}" for i in range(30)]),
    "EASE-OUT-QUAD": ",".join(["1"] + [f"{1-(i/29)**2:.2f}" for i in range(30)]),
    # Half strength
    "HALF": ",".join(["1"] + ["0.5"]*30),
}

# ZIT-NE (Z-Image Turbo No Embeds): Same as ZIT but with 0 for non-layer components
# Use this when you don't want LoRA to affect x_embedder, refiners, etc.
ZIT_NE_PRESETS = {
    "ALL": ",".join(["0"] + ["1"]*30),
    # Enable from start (cumulative)
    "EARLY-ONLY": ",".join(["0"] + ["1"]*6 + ["0"]*24),          # Early only (0-5)
    "EARLY-LOWMID": ",".join(["0"] + ["1"]*15 + ["0"]*15),       # Early + LowMid (0-14)
    "EARLY-MID": ",".join(["0"] + ["1"]*24 + ["0"]*6),           # Early + all Mid (0-23)
    # Enable from end (cumulative)
    "LATE-ONLY": ",".join(["0"] + ["0"]*24 + ["1"]*6),           # Late only (24-29)
    "UPMID-LATE": ",".join(["0"] + ["0"]*15 + ["1"]*15),         # UpMid + Late (15-29)
    "MID-LATE": ",".join(["0"] + ["0"]*6 + ["1"]*24),            # All Mid + Late (6-29)
    # Contiguous middle sections (gaps on edges only)
    "LOWMID-ONLY": ",".join(["0"] + ["0"]*6 + ["1"]*9 + ["0"]*15),   # LowMid only (6-14)
    "UPMID-ONLY": ",".join(["0"] + ["0"]*15 + ["1"]*9 + ["0"]*6),    # UpMid only (15-23)
    "MID-ONLY": ",".join(["0"] + ["0"]*6 + ["1"]*18 + ["0"]*6),      # All Mid (6-23)
    "CORE": ",".join(["0"] + ["0"]*10 + ["1"]*10 + ["0"]*10),        # Core center (10-19)
    "CORE-WIDE": ",".join(["0"] + ["0"]*8 + ["1"]*14 + ["0"]*8),     # Wider core (8-21)
    "CORE-NARROW": ",".join(["0"] + ["0"]*12 + ["1"]*6 + ["0"]*12),  # Narrow core (12-17)
    # Eased gradients
    "EASE-IN": ",".join(["0"] + [f"{i/29:.2f}" for i in range(30)]),
    "EASE-OUT": ",".join(["0"] + [f"{1-i/29:.2f}" for i in range(30)]),
    "EASE-MID": ",".join(["0"] + [f"{1-abs(i-14.5)/14.5:.2f}" for i in range(30)]),
    "EASE-IN-QUAD": ",".join(["0"] + [f"{(i/29)**2:.2f}" for i in range(30)]),
    "EASE-OUT-QUAD": ",".join(["0"] + [f"{1-(i/29)**2:.2f}" for i in range(30)]),
    # Half strength
    "HALF": ",".join(["0"] + ["0.5"]*30),
}

# Build lookup dict for preset name -> vector
def build_preset_lookup():
    """Build a lookup dictionary from preset name to vector.
    Supports both formats for backwards compatibility:
    - Short: "SDXL-ALL" -> vector
    - Full:  "SDXL-ALL:1,1,1,..." -> vector
    """
    lookup = {"None": "", "": ""}
    for name, vec in SDXL_PRESETS.items():
        lookup[f"SDXL-{name}"] = vec
        lookup[f"SDXL-{name}:{vec}"] = vec
    for name, vec in SD15_PRESETS.items():
        lookup[f"SD15-{name}"] = vec
        lookup[f"SD15-{name}:{vec}"] = vec
    for name, vec in FLUX_PRESETS.items():
        lookup[f"FLUX-{name}"] = vec
        lookup[f"FLUX-{name}:{vec}"] = vec
    for name, vec in ZIT_PRESETS.items():
        lookup[f"ZIT-{name}"] = vec
        lookup[f"ZIT-{name}:{vec}"] = vec
    for name, vec in ZIT_NE_PRESETS.items():
        lookup[f"ZIT-NE-{name}"] = vec
        lookup[f"ZIT-NE-{name}:{vec}"] = vec
    return lookup

PRESET_LOOKUP = build_preset_lookup()

# Build flat list for ComfyUI dropdown (show vectors in dropdown)
def build_preset_list(model_type=None):
    """Build preset list, optionally filtered by model type."""
    # Include empty string for backwards compatibility with cached workflows
    presets = ["None", ""]

    if model_type is None or model_type == "Auto":
        # Show all presets with prefixes and vectors
        for name, vec in SDXL_PRESETS.items():
            presets.append(f"SDXL-{name}:{vec}")
        for name, vec in SD15_PRESETS.items():
            presets.append(f"SD15-{name}:{vec}")
        for name, vec in FLUX_PRESETS.items():
            presets.append(f"FLUX-{name}:{vec}")
        for name, vec in ZIT_PRESETS.items():
            presets.append(f"ZIT-{name}:{vec}")
        for name, vec in ZIT_NE_PRESETS.items():
            presets.append(f"ZIT-NE-{name}:{vec}")
    elif model_type == "SDXL":
        for name, vec in SDXL_PRESETS.items():
            presets.append(f"SDXL-{name}:{vec}")
    elif model_type == "SD15":
        for name, vec in SD15_PRESETS.items():
            presets.append(f"SD15-{name}:{vec}")
    elif model_type == "FLUX":
        for name, vec in FLUX_PRESETS.items():
            presets.append(f"FLUX-{name}:{vec}")
    elif model_type == "ZIT":
        for name, vec in ZIT_PRESETS.items():
            presets.append(f"ZIT-{name}:{vec}")
    elif model_type == "ZIT-NE":
        for name, vec in ZIT_NE_PRESETS.items():
            presets.append(f"ZIT-NE-{name}:{vec}")

    return presets

# Default preset list (all presets with vectors shown)
BLOCK_PRESETS = build_preset_list()


def _valid_lora_hash(value):
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if re.fullmatch(r"[0-9a-fA-F]{10,128}", value):
        return value.upper()
    return ""


def _merge_lora_hashes(target, source):
    if not isinstance(source, (list, tuple)):
        return
    for i in range(min(3, len(source))):
        h = _valid_lora_hash(source[i])
        if h:
            target[i] = h


def _hashes_from_workflow_node(node):
    hashes = ["", "", ""]
    if not isinstance(node, dict):
        return hashes

    props = node.get("properties") or {}
    _merge_lora_hashes(hashes, props.get("lora_hashes"))
    _merge_lora_hashes(hashes, node.get("lora_hashes"))

    inputs = node.get("inputs") or {}
    if isinstance(inputs, dict):
        direct = [inputs.get("lora_hash_1"), inputs.get("lora_hash_2"), inputs.get("lora_hash_3")]
        _merge_lora_hashes(hashes, direct)

    values = node.get("widgets_values") or []
    widget_hashes = [_valid_lora_hash(v) for v in values]
    widget_hashes = [h for h in widget_hashes if h]
    if widget_hashes:
        _merge_lora_hashes(hashes, widget_hashes[-3:])

    return hashes


def _workflow_nodes_from_extra(extra_pnginfo):
    if not isinstance(extra_pnginfo, dict):
        return []
    workflow = extra_pnginfo.get("workflow")
    if isinstance(workflow, str):
        try:
            workflow = json.loads(workflow)
        except json.JSONDecodeError:
            workflow = None
    if isinstance(workflow, dict):
        nodes = workflow.get("nodes")
        if isinstance(nodes, list):
            return nodes
    nodes = extra_pnginfo.get("nodes")
    if isinstance(nodes, list):
        return nodes
    return []


def _workflow_node_mentions_loras(node, lora_names):
    values = node.get("widgets_values") or []
    wanted = {name for name in lora_names if name and name != "None"}
    if not wanted:
        return False
    return any(isinstance(v, str) and v in wanted for v in values)


def _extract_lora_hashes(extra_pnginfo, unique_id, direct_hashes, lora_names):
    hashes = ["", "", ""]

    nodes = _workflow_nodes_from_extra(extra_pnginfo)
    uid = str(unique_id) if unique_id is not None else ""

    matched_node = None
    for node in nodes:
        if str(node.get("id")) == uid:
            matched_node = node
            break

    if matched_node is None:
        for node in nodes:
            if node.get("type") == "DonutLoRAStack" and _workflow_node_mentions_loras(node, lora_names):
                matched_node = node
                break

    if matched_node is not None:
        _merge_lora_hashes(hashes, _hashes_from_workflow_node(matched_node))

    _merge_lora_hashes(hashes, direct_hashes)

    return hashes


# ------------------------------------------------------------------------
class DonutLoRAStack:
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "model_type": (MODEL_TYPES, {"default": "Auto", "tooltip": "Filter block presets by model architecture. Auto shows all presets."}),

                "switch_1":       (["Off","On"],),
                "lora_name_1":    (loras,),
                "model_weight_1": ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "clip_weight_1":  ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "block_preset_1": (BLOCK_PRESETS, {"default": "None", "tooltip": "Select a preset to populate block_vector_1."}),
                "block_vector_1": ("STRING",{"default":"","placeholder":"SDXL:12, SD1.5:17, ZIT:30 blocks"}),

                "switch_2":       (["Off","On"],),
                "lora_name_2":    (loras,),
                "model_weight_2": ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "clip_weight_2":  ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "block_preset_2": (BLOCK_PRESETS, {"default": "None", "tooltip": "Select a preset to populate block_vector_2."}),
                "block_vector_2": ("STRING",{"default":"","placeholder":"SDXL:12, SD1.5:17, ZIT:30 blocks"}),

                "switch_3":       (["Off","On"],),
                "lora_name_3":    (loras,),
                "model_weight_3": ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "clip_weight_3":  ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "block_preset_3": (BLOCK_PRESETS, {"default": "None", "tooltip": "Select a preset to populate block_vector_3."}),
                "block_vector_3": ("STRING",{"default":"","placeholder":"SDXL:12, SD1.5:17, ZIT:30 blocks"}),

                "civitai_lookup": (["On", "Off"], {"default": "On", "tooltip": "Fetch LoRA info from CivitAI (requires API key in settings)"}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
                "lora_hash_1": ("STRING", {"default": ""}),
                "lora_hash_2": ("STRING", {"default": ""}),
                "lora_hash_3": ("STRING", {"default": ""}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }
        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION     = "build_stack"
    CATEGORY     = "donut/LoRA"
    OUTPUT_NODE  = True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Allow empty strings for float fields - we handle them in build_stack."""
        for key in ['model_weight_1', 'clip_weight_1', 'model_weight_2', 'clip_weight_2', 'model_weight_3', 'clip_weight_3']:
            val = kwargs.get(key)
            if val == "" or val is None:
                continue  # Allow empty, we'll default to 1.0
            try:
                float(val)
            except (ValueError, TypeError):
                return f"Invalid value for {key}: {val}"
        return True

    def build_stack(
        self,
        model_type,
        switch_1, lora_name_1, model_weight_1, clip_weight_1, block_preset_1, block_vector_1,
        switch_2, lora_name_2, model_weight_2, clip_weight_2, block_preset_2, block_vector_2,
        switch_3, lora_name_3, model_weight_3, clip_weight_3, block_preset_3, block_vector_3,
        civitai_lookup="On",
        lora_stack=None,
        lora_hash_1="",
        lora_hash_2="",
        lora_hash_3="",
        extra_pnginfo=None,
        unique_id=None,
    ):
        try:
            lora_hashes = _extract_lora_hashes(
                extra_pnginfo,
                unique_id,
                [lora_hash_1, lora_hash_2, lora_hash_3],
                [lora_name_1, lora_name_2, lora_name_3],
            )
        except Exception as e:
            print(f"[DonutLoRAStack] Could not read lora_hashes from workflow: {e}")
            lora_hashes = [
                _valid_lora_hash(lora_hash_1),
                _valid_lora_hash(lora_hash_2),
                _valid_lora_hash(lora_hash_3),
            ]
        lora_hash_1, lora_hash_2, lora_hash_3 = lora_hashes
        stack = list(lora_stack) if lora_stack else []

        # Handle empty string values from cached workflows (backwards compatibility)
        def safe_float(val, default=1.0):
            if val == "" or val is None:
                return default
            return float(val)

        def safe_preset(val):
            if val == "" or val is None:
                return "None"
            return val

        model_weight_1 = safe_float(model_weight_1, 1.0)
        clip_weight_1 = safe_float(clip_weight_1, 1.0)
        block_preset_1 = safe_preset(block_preset_1)
        model_weight_2 = safe_float(model_weight_2, 1.0)
        clip_weight_2 = safe_float(clip_weight_2, 1.0)
        block_preset_2 = safe_preset(block_preset_2)
        model_weight_3 = safe_float(model_weight_3, 1.0)
        clip_weight_3 = safe_float(clip_weight_3, 1.0)
        block_preset_3 = safe_preset(block_preset_3)

        # CivitAI metadata collection
        lora_info_lines = []
        all_trigger_words = []
        civitai_urls = []
        individual_infos = ["", "", ""]  # Info for each slot
        collage_images = [None, None, None]  # Collage for each slot (indexed by slot)

        # Get API key from settings (optional - public API works without it)
        civitai_api_key = get_civitai_api_key() if HAS_CIVITAI else ""

        # Get cache and temp directory if CivitAI enabled
        cache = None
        hash_cache_dir = None
        temp_dir = folder_paths.get_temp_directory()
        if civitai_lookup == "On" and HAS_CIVITAI:
            cache = get_cache()
            # Use civitai_cache/hashes for hash caching (faster than alongside LoRA files)
            hash_cache_dir = os.path.join(cache.cache_dir, "hashes")

        def _maybe_add(slot_idx, sw, name, mw, cw, bv, expected_hash=""):
            if sw == "On" and name != "None":
                # Resolve LoRA (handles renamed/moved files + auto-download from Civitai)
                resolved_path, source = resolve_lora(
                    name,
                    expected_hash=expected_hash or None,
                    auto_download=True,
                    api_key=civitai_api_key if civitai_api_key else None,
                )

                # If resolver found a different local path, swap to the relative
                # form ComfyUI expects (so the loader can open it)
                resolved_name = name
                if resolved_path and source != "exact":
                    rel = relative_name_for(resolved_path)
                    if rel:
                        resolved_name = rel
                        print(f"[DonutLoRAStack] Resolved '{name}' via {source} → '{rel}'")
                elif not resolved_path:
                    print(f"[DonutLoRAStack] WARNING: could not resolve LoRA '{name}'"
                          + (f" (hash {expected_hash[:10]})" if expected_hash else ""))

                # Always use block_vector - preset only populates the field via JS
                final_bv = bv.strip()
                stack.append((resolved_name, mw, cw, final_bv))

                # Fetch CivitAI metadata
                if cache is not None:
                    lora_path = resolved_path
                    if lora_path and os.path.exists(lora_path):
                        try:
                            file_hash = get_or_compute_hash(lora_path, use_cache=True, cache_dir=hash_cache_dir)
                            info = cache.get_or_fetch_info(
                                file_hash,
                                api_key=civitai_api_key if civitai_api_key else None,
                                download_preview=True
                            )
                            if info:
                                display_name = info.get_display_name()
                                rec_weight = info.recommended_weight
                                weight_hint = f" [rec:{rec_weight}]" if rec_weight != 1.0 else ""
                                lora_info_lines.append(f"{display_name} (w:{mw}{weight_hint})")
                                if info.trained_words:
                                    all_trigger_words.extend(info.trained_words)
                                civitai_urls.append(info.model_url)
                                print(f"[DonutLoRAStack] CivitAI: {name} -> {display_name}")

                                # Build individual info for this slot
                                slot_info = f"{display_name}{weight_hint}"
                                slot_info += f"\n{info.base_model} | ↓{info.download_count}"
                                if info.trained_words:
                                    slot_info += f"\nTriggers: {', '.join(info.trained_words)}"
                                slot_info += f"\n{info.model_url}"
                                # Add description at the end (strip HTML tags)
                                if info.description:
                                    desc = re.sub(r'<[^>]+>', '', info.description).strip()
                                    if desc:
                                        slot_info += f"\n\n{desc}"
                                individual_infos[slot_idx] = slot_info

                                # Create collage for this LoRA
                                collage_path = cache.create_preview_collage(info, max_images=4)
                                if collage_path:
                                    # Copy to temp dir for ComfyUI to serve (skip if already exists)
                                    collage_filename = f"lora_preview_{slot_idx}_{file_hash[:8]}.jpg"
                                    temp_path = os.path.join(temp_dir, collage_filename)
                                    if not os.path.exists(temp_path):
                                        shutil.copy2(collage_path, temp_path)
                                    collage_images[slot_idx] = {
                                        "filename": collage_filename,
                                        "subfolder": "",
                                        "type": "temp"
                                    }
                            else:
                                # Not found on CivitAI
                                search_url = f"https://civitai.com/search/models?query={file_hash}"
                                lora_info_lines.append(f"{name} (w:{mw}) [not on CivitAI]")
                                civitai_urls.append(search_url)
                                individual_infos[slot_idx] = f"{name}\n[not on CivitAI]"
                        except Exception as e:
                            print(f"[DonutLoRAStack] CivitAI error for {name}: {e}")
                            lora_info_lines.append(f"{name} (w:{mw})")
                            individual_infos[slot_idx] = name
                    else:
                        lora_info_lines.append(f"{name} (w:{mw})")
                        individual_infos[slot_idx] = name
                else:
                    lora_info_lines.append(f"{name} (w:{mw})")
                    individual_infos[slot_idx] = name

        _maybe_add(0, switch_1, lora_name_1, model_weight_1, clip_weight_1, block_vector_1, lora_hash_1)
        _maybe_add(1, switch_2, lora_name_2, model_weight_2, clip_weight_2, block_vector_2, lora_hash_2)
        _maybe_add(2, switch_3, lora_name_3, model_weight_3, clip_weight_3, block_vector_3, lora_hash_3)

        # Format outputs
        lora_info = "\n".join(lora_info_lines) if lora_info_lines else "No LoRAs selected"
        trigger_words = ", ".join(sorted(set(all_trigger_words))) if all_trigger_words else ""
        urls_output = "\n".join(civitai_urls) if civitai_urls else ""

        print(f"[DonutLoRAStack] Built stack with {len(stack)} LoRAs")
        if all_trigger_words:
            print(f"[DonutLoRAStack] Trigger words: {trigger_words}")

        # Return with UI data for the JavaScript extension to display in-node
        ui_data = {
            "text": [lora_info, trigger_words, urls_output, individual_infos[0], individual_infos[1], individual_infos[2]]
        }
        # Add collage images if any are available (keep as list with None for empty slots)
        if any(img is not None for img in collage_images):
            ui_data["images"] = collage_images

        return {
            "ui": ui_data,
            "result": (stack,)
        }


# ------------------------------------------------------------------------
# DonutApplyLoRAStack: per-block UNet + uniform CLIP merges, always in that order
# ------------------------------------------------------------------------
class DonutApplyLoRAStack:
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "model":      ("MODEL",),
            "clip":       ("CLIP",),
            "lora_stack": ("LORA_STACK",),
        }}

    RETURN_TYPES = ("MODEL","CLIP","STRING")
    RETURN_NAMES = ("model","clip","show_help")
    FUNCTION     = "apply_stack"
    CATEGORY     = "Comfyanonymous/LoRA"

    def apply_stack(self, model, clip, lora_stack=None):
        help_url = (
            "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/"
            "wiki/LoRA-Nodes#cr-apply-lora-stack"
        )

        if lora_stack is None or len(lora_stack) == 0:
            return (model, clip, help_url)

        unet, text_enc = model, clip
        loader = LoraLoaderBlockWeight()

        seen = set()
        for i, (name, mw, cw, bv) in enumerate(lora_stack):
            if mw == 0.0 and cw == 0.0:
                continue

            if name in seen:
                print(f"[DonutApplyLoRAStack] Skipping duplicate LoRA '{name}' (already applied this run)")
                continue
            seen.add(name)

            path = folder_paths.get_full_path("loras", name)
            lora = comfy.utils.load_torch_file(path, safe_load=True)

            # Auto-detect block count from LoRA keys if no vector provided
            if bv:
                vector = bv
            else:
                # Count unique block numbers to determine architecture
                block_nums = set()
                for k in lora.keys():
                    if "layers." in k:  # Z-Image/Lumina2
                        m = re.search(r'layers\.(\d+)', k)
                        if m:
                            block_nums.add(int(m.group(1)))
                    elif "input_blocks." in k or "output_blocks." in k or "middle_block." in k:
                        # UNet architecture (SD1.5/SDXL)
                        pass
                    elif "double_blocks." in k or "single_blocks." in k:
                        # Flux
                        pass

                if block_nums:  # Z-Image detected
                    num_blocks = max(block_nums) + 1
                    vector = ",".join(["1"] * (num_blocks + 1))  # +1 for base
                else:
                    vector = ",".join(["1"] * 13)  # Default: base + 12 blocks (SDXL)

            # 1) block-weighted UNet merge (pass clip=None to skip CLIP processing)
            if mw != 0.0:
                unet, _, _ = loader.load_lora_for_models(
                    unet, None, lora,  # clip=None skips CLIP processing entirely
                    strength_model=mw,
                    strength_clip=0.0,
                    inverse=False,
                    seed=0,
                    A=1.0,
                    B=1.0,
                    block_vector=vector
                )

            # 2) uniform CLIP merge (no block control)
            if cw != 0.0:
                _, text_enc = comfy.sd.load_lora_for_models(
                    unet, text_enc, lora,
                    0.0,  # no UNet change
                    cw    # clip strength
                )

        return (unet, text_enc, help_url)


# ------------------------------------------------------------------------
# DonutLoraStackCombine: merge two lora stacks into one
# ------------------------------------------------------------------------
class DonutLoraStackCombine:
    """Combines two LORA_STACK inputs into a single output."""
    class_type = "CUSTOM"
    aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "lora_stack_1": ("LORA_STACK",),
                "lora_stack_2": ("LORA_STACK",),
            },
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "combine"
    CATEGORY = "donut/LoRA"

    def combine(self, lora_stack_1=None, lora_stack_2=None):
        def _normalize_stack(stack):
            """Normalize incoming stack shapes to a list of 4-tuples."""
            if stack is None:
                return []

            # Unwrap ComfyUI dict outputs if someone passed the whole result object.
            if isinstance(stack, dict) and "result" in stack:
                stack = stack["result"][0] if stack["result"] else []

            # Unwrap single-item tuple/list that contains the stack.
            if isinstance(stack, (list, tuple)) and len(stack) == 1 and isinstance(stack[0], (list, tuple)):
                inner = stack[0]
                if inner and isinstance(inner[0], (list, tuple)):
                    stack = inner

            # Single LoRA entry passed directly.
            if isinstance(stack, (list, tuple)) and len(stack) == 4 and isinstance(stack[0], str):
                return [tuple(stack)]

            # Standard stack: list/tuple of entries.
            if isinstance(stack, (list, tuple)):
                normalized = []
                for item in stack:
                    if isinstance(item, (list, tuple)) and len(item) == 4:
                        normalized.append(tuple(item))
                    elif isinstance(item, (list, tuple)) and len(item) == 3:
                        # Backward/alternate format: default empty block vector.
                        normalized.append((item[0], item[1], item[2], ""))
                return normalized

            return []

        combined = []
        combined.extend(_normalize_stack(lora_stack_1))
        combined.extend(_normalize_stack(lora_stack_2))
        return (combined,)


NODE_CLASS_MAPPINGS = {
    "DonutLoRAStack":         DonutLoRAStack,
    "DonutApplyLoRAStack":    DonutApplyLoRAStack,
    "DonutLoraStackCombine":  DonutLoraStackCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}

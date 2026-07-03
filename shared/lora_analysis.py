"""
LoRA file composition analysis.

Reads only the safetensors header (keys, no tensor data) and reports which
components a LoRA affects: UNet/diffusion model vs text encoder(s), and which
block/layer indices actually contain weights.
"""

import json
import os
import re
import struct

# Suffixes that distinguish tensors of the same module (lora_down/lora_up/alpha...).
# Stripping them lets us count modules instead of tensors.
_TENSOR_SUFFIX = re.compile(
    r"\.(?:lora_(?:down|up|mid|A|B)\.weight|alpha|dora_scale|diff|diff_b|"
    r"(?:hada_|lokr_)?(?:w[12](?:_[ab])?|t[12])|on_input|scale)$"
)

# Prefix -> (domain, component name). Checked in order; first match wins.
_DOMAIN_PREFIXES = [
    ("lora_unet_", "unet", "UNet"),
    ("lora_transformer_", "unet", "UNet"),
    ("lora_te1_", "te", "TE1"),
    ("lora_te2_", "te", "TE2"),
    ("lora_te3_", "te", "TE3"),
    ("lora_te_", "te", "TE"),
    ("model.diffusion_model.", "unet", "UNet"),
    ("diffusion_model.", "unet", "UNet"),
    ("unet.", "unet", "UNet"),
    ("transformer.", "unet", "UNet"),
    ("text_encoder_2.", "te", "TE2"),
    ("text_encoder.", "te", "TE1"),
    ("clip_l.", "te", "CLIP-L"),
    ("clip_g.", "te", "CLIP-G"),
    ("t5xxl.", "te", "T5-XXL"),
]

# UNet block-group patterns, checked in order (specific before generic).
# Group label is the key stem so the display stays truthful for any arch.
_UNET_GROUP_PATTERNS = [
    ("input_blocks", re.compile(r"input_blocks[._](\d+)")),
    ("middle_block", re.compile(r"middle_block")),
    ("output_blocks", re.compile(r"output_blocks[._](\d+)")),
    ("double_blocks", re.compile(r"double_blocks[._](\d+)")),
    ("single_blocks", re.compile(r"single(?:_transformer)?_blocks[._](\d+)")),
    ("transformer_blocks", re.compile(r"transformer_blocks[._](\d+)")),
    ("txtfusion.layerwise", re.compile(r"txtfusion[._]layerwise(?:[._]blocks)?[._](\d+)")),
    ("txtfusion.refiner", re.compile(r"txtfusion[._]refiner(?:[._]blocks)?[._](\d+)")),
    ("noise_refiner", re.compile(r"noise_refiner[._](\d+)")),
    ("context_refiner", re.compile(r"context_refiner[._](\d+)")),
    ("layers", re.compile(r"(?:^|[._])layers[._](\d+)")),
    ("blocks", re.compile(r"(?:^|[._])blocks[._](\d+)")),
]

# Text-encoder layer patterns (CLIP layers, T5 blocks, LLM layers).
_TE_GROUP_PATTERNS = [
    ("layers", re.compile(r"(?:^|[._])layers[._](\d+)")),
    ("blocks", re.compile(r"(?:^|[._])blocks?[._](\d+)")),
    ("h", re.compile(r"(?:^|[._])h[._](\d+)")),
]

_analysis_cache = {}  # path -> (mtime, size, result)


def read_safetensors_keys(path):
    """Read tensor names from a safetensors file header without loading data."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header length: {header_len}")
        header = json.loads(f.read(header_len))
    return [k for k in header if k != "__metadata__"]


def _classify_domain(key):
    """Return (domain, component_name, remainder) for a tensor base key."""
    for prefix, domain, name in _DOMAIN_PREFIXES:
        if key.startswith(prefix):
            return domain, name, key[len(prefix):]
    # ComfyUI multi-encoder format: text_encoders.<name>.transformer...
    m = re.match(r"text_encoders\.([^.]+)\.(.*)", key)
    if m:
        return "te", f"TE ({m.group(1)})", m.group(2)
    # Bare diffusion-model keys (no prefix at all)
    for _, pat in _UNET_GROUP_PATTERNS:
        if pat.search(key):
            return "unet", "UNet", key
    return "other", key.split(".")[0].split("_lora")[0], key


def analyze_lora_keys(keys):
    """Group LoRA module keys into components (UNet / text encoders) and
    per-component block/layer index sets."""
    # Dedup tensors -> modules
    modules = {_TENSOR_SUFFIX.sub("", k) for k in keys}

    components = {}  # name -> {"type", "groups", "ungrouped_names": [], "modules": int}
    for mod in modules:
        domain, comp_name, rest = _classify_domain(mod)
        comp = components.setdefault(
            comp_name, {"type": domain, "groups": {}, "ungrouped_names": [], "modules": 0}
        )
        comp["modules"] += 1

        patterns = _UNET_GROUP_PATTERNS if domain == "unet" else _TE_GROUP_PATTERNS
        for label, pat in patterns:
            m = pat.search(rest)
            if m:
                idx = int(m.group(1)) if m.groups() else 0
                comp["groups"].setdefault(label, set()).add(idx)
                break
        else:
            # Not inside a numbered block: keep the module's own name so the UI
            # can say what a small/surgical LoRA actually targets.
            comp["ungrouped_names"].append(rest)

    # Order: unet first, then text encoders, then anything else
    order = {"unet": 0, "te": 1, "other": 2}
    result = []
    for name in sorted(components, key=lambda n: (order.get(components[n]["type"], 3), n)):
        comp = components[name]
        ungrouped = sorted(comp["ungrouped_names"])
        result.append({
            "name": name,
            "type": comp["type"],
            "modules": comp["modules"],
            "ungrouped": len(ungrouped),
            "ungrouped_names": ungrouped[:12],  # cap for payload size
            "groups": [
                {"name": label, "indices": sorted(idxs)}
                for label, idxs in sorted(comp["groups"].items())
            ],
        })
    return result


def analyze_lora_file(path):
    """Analyze a LoRA file, returning a JSON-serializable composition report.
    Results are cached by (mtime, size)."""
    try:
        st = os.stat(path)
    except OSError:
        return {"found": False, "error": "file not found"}

    cached = _analysis_cache.get(path)
    if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
        return cached[2]

    if not path.lower().endswith(".safetensors"):
        result = {"found": True, "supported": False,
                  "error": "only .safetensors files can be inspected"}
    else:
        try:
            keys = read_safetensors_keys(path)
            result = {
                "found": True,
                "supported": True,
                "tensor_count": len(keys),
                "components": analyze_lora_keys(keys),
            }
        except Exception as e:
            result = {"found": True, "supported": False, "error": str(e)}

    _analysis_cache[path] = (st.st_mtime, st.st_size, result)
    return result

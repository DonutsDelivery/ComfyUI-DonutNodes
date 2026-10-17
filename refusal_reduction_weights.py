"""Bundled Krea2 TextFusion Refusal-Reduction factors (v1 and v2).

assets/refusal_reduction_v2.bf16 contains raw little-endian bfloat16 data: for each
target below, the up tensor (lora_B) precedes the down tensor (lora_A). All 35 targets
are v2. v1 is exactly its first 32 block targets, bit-identical to the original v1
file, so one asset serves both. The loader checks size and SHA-256 before use. No
executable payload, safetensors header or runtime download is involved.
"""
import hashlib
import math
import sys
from functools import lru_cache
from pathlib import Path

PAYLOAD_SHA256 = "8d2798e11b79f90c28744e6615a40ff1d8d2ee30fa86f47e2dfddb71467480ba"
PAYLOAD_SIZE_BYTES = 30344832
RANK = 64
V1_TARGET_COUNT = 32
V2_TARGET_COUNT = 35
# (canonical model key, up shape, down shape, alpha); alpha equals rank, so scale 1.
TARGETS = (
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.gate.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.wo.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.wv.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.mlp.down.weight', (2560, 64), (64, 6912), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.mlp.gate.weight', (6912, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.mlp.up.weight', (6912, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.gate.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.wk.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.wo.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.wq.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.wv.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.mlp.down.weight', (2560, 64), (64, 6912), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.mlp.gate.weight', (6912, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.mlp.up.weight', (6912, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.projector.weight', (1, 64), (64, 12), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.gate.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.wk.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.wo.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.wv.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.mlp.down.weight', (2560, 64), (64, 6912), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.mlp.gate.weight', (6912, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.mlp.up.weight', (6912, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.gate.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.wk.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.wo.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.wq.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.wv.weight', (2560, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.mlp.down.weight', (2560, 64), (64, 6912), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.mlp.gate.weight', (6912, 64), (64, 2560), 64.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.mlp.up.weight', (6912, 64), (64, 2560), 64.0),
    ('diffusion_model.txtmlp.1.weight', (6144, 64), (64, 2560), 64.0),
    ('diffusion_model.txtmlp.3.weight', (6144, 64), (64, 6144), 64.0),
)

PAYLOAD_PATH = Path(__file__).parent / "assets" / "refusal_reduction_v2.bf16"


def _is_v1_target(key):
    return "_blocks." in key


def _read_payload():
    try:
        with PAYLOAD_PATH.open("rb") as stream:
            raw = stream.read(PAYLOAD_SIZE_BYTES + 1)
    except OSError as exc:
        raise RuntimeError("Bundled Refusal Reduction data is missing or unreadable; reinstall DonutNodes") from exc
    if (len(raw) != PAYLOAD_SIZE_BYTES
            or hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256):
        raise RuntimeError("Bundled Refusal Reduction data failed its size/SHA-256 check")
    return raw


@lru_cache(maxsize=1)
def _all_factors():
    import torch

    raw = bytearray(_read_payload())
    if sys.byteorder != "little":
        # The stored byte order is little-endian on every host.
        raw[0::2], raw[1::2] = raw[1::2], raw[0::2]
    flat = torch.frombuffer(raw, dtype=torch.bfloat16).to(torch.float32)
    factors, offset, seen = [], 0, set()
    for key, up_shape, down_shape, alpha in TARGETS:
        if (key in seen or not key.endswith(".weight") or len(up_shape) != 2 or len(down_shape) != 2
                or up_shape[1] != RANK or down_shape[0] != RANK or alpha != float(RANK)
                or any(n <= 0 for n in (*up_shape, *down_shape))):
            raise RuntimeError("Invalid embedded Refusal Reduction target specification")
        seen.add(key)
        up_count, down_count = math.prod(up_shape), math.prod(down_shape)
        if offset + up_count + down_count > flat.numel():
            raise RuntimeError("Truncated embedded Refusal Reduction tensor data")
        up = flat[offset:offset + up_count].reshape(up_shape)
        offset += up_count
        down = flat[offset:offset + down_count].reshape(down_shape)
        offset += down_count
        if not torch.isfinite(up).all().item() or not torch.isfinite(down).all().item():
            raise RuntimeError("Non-finite embedded Refusal Reduction factors")
        factors.append((key, up, down, alpha))
    if len(factors) != V2_TARGET_COUNT or offset != flat.numel():
        raise RuntimeError("Incomplete embedded Refusal Reduction target set")
    return tuple(factors)


def get_refusal_reduction_v2_factors():
    """Return cached (key, up, down, alpha) tuples; callers must not mutate them."""
    return _all_factors()


def get_refusal_reduction_v1_factors():
    """v1 is the 32 transformer-block targets, without projector or txtmlp."""
    return tuple(item for item in _all_factors() if _is_v1_target(item[0]))

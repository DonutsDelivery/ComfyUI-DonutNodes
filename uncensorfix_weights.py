"""Bundled UncensorFix numerical factors.

assets/uncensorfix.f32 contains raw little-endian float32 data, with shapes and
alpha values listed below. The loader checks its size and SHA-256 before use.
No executable payload, original checkpoint header or runtime download is used.
"""
import array
import hashlib
import math
import sys
from pathlib import Path
from functools import lru_cache

PAYLOAD_SHA256 = "f3c817bd957e6d47883346237b5e067697f0b9e1c9909bd06353da455949aacf"
PAYLOAD_SIZE_BYTES = 3457232
TARGET_COUNT = 33
# (canonical model key, up shape, down shape, alpha)
TARGETS = (
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.gate.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.wo.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.attn.wv.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.mlp.down.weight', (2560, 4), (4, 6912), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.mlp.gate.weight', (6912, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.0.mlp.up.weight', (6912, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.gate.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.wk.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.wo.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.wq.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.attn.wv.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.mlp.down.weight', (2560, 4), (4, 6912), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.mlp.gate.weight', (6912, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.layerwise_blocks.1.mlp.up.weight', (6912, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.projector.weight', (1, 4), (4, 12), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.gate.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.wk.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.wo.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.attn.wv.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.mlp.down.weight', (2560, 4), (4, 6912), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.mlp.gate.weight', (6912, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.0.mlp.up.weight', (6912, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.gate.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.wk.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.wo.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.wq.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.attn.wv.weight', (2560, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.mlp.down.weight', (2560, 4), (4, 6912), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.mlp.gate.weight', (6912, 4), (4, 2560), 4.0),
    ('diffusion_model.txtfusion.refiner_blocks.1.mlp.up.weight', (6912, 4), (4, 2560), 4.0),
)

PAYLOAD_PATH = Path(__file__).parent / "assets" / "uncensorfix.f32"


def _read_payload():
    try:
        with PAYLOAD_PATH.open("rb") as stream:
            raw = stream.read(PAYLOAD_SIZE_BYTES + 1)
    except OSError as exc:
        raise RuntimeError("Bundled UncensorFix data is missing or unreadable; reinstall DonutNodes") from exc
    if (len(raw) != PAYLOAD_SIZE_BYTES
            or hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256):
        raise RuntimeError("Bundled UncensorFix data failed its size/SHA-256 check")
    return raw


@lru_cache(maxsize=1)
def get_uncensorfix_factors():
    """Return cached (key, up, down, alpha) tuples; callers must not mutate them."""
    import torch

    raw = _read_payload()
    # The stored byte order is independent of the runtime host.
    if sys.byteorder != "little":
        words = array.array("I")
        if words.itemsize != 4:
            raise RuntimeError("UncensorFix requires a platform with 32-bit unsigned ints")
        words.frombytes(raw)
        words.byteswap()
        raw = words.tobytes()
    flat = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    factors, offset, seen = [], 0, set()
    for key, up_shape, down_shape, alpha in TARGETS:
        if (key in seen or not key.startswith("diffusion_model.txtfusion.")
                or not key.endswith(".weight") or len(up_shape) != 2 or len(down_shape) != 2
                or up_shape[1] != 4 or down_shape[0] != 4
                or any(n <= 0 for n in (*up_shape, *down_shape)) or alpha != 4.0):
            raise RuntimeError("Invalid embedded UncensorFix target specification")
        seen.add(key)
        up_count, down_count = math.prod(up_shape), math.prod(down_shape)
        if offset + up_count + down_count > flat.numel():
            raise RuntimeError("Truncated embedded UncensorFix tensor data")
        up = flat[offset:offset + up_count].reshape(up_shape)
        offset += up_count
        down = flat[offset:offset + down_count].reshape(down_shape)
        offset += down_count
        if not torch.isfinite(up).all().item() or not torch.isfinite(down).all().item():
            raise RuntimeError("Non-finite embedded UncensorFix factors")
        factors.append((key, up, down, alpha))
    if len(factors) != TARGET_COUNT or offset != flat.numel():
        raise RuntimeError("Incomplete embedded UncensorFix target set")
    return tuple(factors)

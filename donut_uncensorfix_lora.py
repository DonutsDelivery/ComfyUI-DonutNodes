"""Bridge embedded UncensorFix factors to Donut Apply's bypass implementation.

No filesystem lookup or safetensors parsing is involved. Keeping this import
lazy also leaves native patches, Off and strength zero free of loader imports.
"""

import torch


def _canonical_lora_state(patches):
    """Reconstruct ordinary LoRA tensor keys, not a serialized LoRA file."""
    lora = {}
    for key, adapter in patches.items():
        weights = adapter.weights
        if not key.endswith(".weight") or len(weights) != 6:
            raise RuntimeError(f"Unexpected embedded UncensorFix adapter: {key}")
        up, down, alpha, mid, dora, reshape = weights
        if any(value is not None for value in (mid, dora, reshape)):
            raise RuntimeError(f"UncensorFix expects plain linear LoRA factors: {key}")
        stem = key[:-len(".weight")]
        lora[stem + ".lora_up.weight"] = up
        lora[stem + ".lora_down.weight"] = down
        lora[stem + ".alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
    return lora


def apply_embedded_bypass(model, patches, strength):
    """Use the exact helper, text vector and fallbacks used by Donut Apply."""
    from .DonutSafeApplyLoRAStack import _apply_bypass_applications
    from .donut_lora_nodes import _TEXT_MERGE_VECTOR
    from .lora_block_weight import LoraLoaderBlockWeight

    lora = _canonical_lora_state(patches)
    # Donut Apply normally permits partially matching LoRAs. The embedded
    # preset promises all targets, so reject any lost/remapped/muted target
    # before installing adapters. The shared helper performs its own normal
    # load afterwards; no monkey-patching or separate bypass math is used.
    block_weights, muted, _ = LoraLoaderBlockWeight.load_lbw(
        model, None, lora, inverse=False, seed=0, A=1.0, B=1.0,
        block_vector=_TEXT_MERGE_VECTOR,
    )
    if (
        set(block_weights) != set(patches)
        or muted
        or any(float(ratio) != 1.0 for _, ratio in block_weights.values())
    ):
        raise RuntimeError(
            "UncensorFix bypass must preserve every embedded target at unit block weight: "
            f"{len(block_weights)}/{len(patches)} mapped"
        )
    result = _apply_bypass_applications(model, [(lora, strength, _TEXT_MERGE_VECTOR)])

    # The shared helper can use regular patches for unsupported modules or
    # existing injections. Require either appended patches or a NEW bypass
    # injection, rather than reporting success for an unchanged model.
    before = getattr(model, "patches", {})
    after = getattr(result, "patches", {})
    regular = {
        key for key in patches
        if len(after.get(key, ())) > len(before.get(key, ()))
    }
    old_injection = getattr(model, "injections", {}).get("donut_bypass_lora")
    new_injection = getattr(result, "injections", {}).get("donut_bypass_lora")
    if regular != set(patches) and (not new_injection or new_injection is old_injection):
        raise RuntimeError("UncensorFix bypass did not install the expected patches or forward adapters")
    return result

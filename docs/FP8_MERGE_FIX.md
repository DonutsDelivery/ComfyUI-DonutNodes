# FP8 source merge compatibility

Fixes `DonutModelMergeKrea2` raising `AttributeError: 'Linear' object has no
attribute 'weight_scale'` after an FP8 model is connected as **model2**.
Both the regular and Experimental bypass paths use the corrected accessor.
The grouped node inherits the same fix; no sockets, widget positions, model
selections, ratios, LoRA controls, or workflow connections change.

## Why swapping the inputs exposed it

The merge clones model1 but enumerates model2 through `get_key_patches`.
ComfyUI's scaled-FP8 loader stores the scale inside its live `QuantizedTensor`.
Its export state contains `weight_scale` and `comfy_quant`, even though those
are not separate attributes on the live Linear. The core accessor attempts
`getattr` on every exported key and can fail on those metadata entries.
This applies to scaled FP8; it does not imply that the user selected INT8 or
NVFP4. Plain FP8 tensors without export-only metadata keep the native path.

Reference implementation examined: ComfyUI `3216c62e`, specifically
`comfy/model_patcher.py` (`get_key_weight`, `get_key_patches`), `comfy/ops.py`
(`_load_quantized_module`, `_quantized_weight_state_dict`, `convert_weight`),
and the FP8 layouts in `comfy/quant_ops.py`.

## Narrow compatibility fallback

The normal `get_key_patches` result is used unchanged whenever it succeeds.
Only an AttributeError on a verified export-only field activates the fallback.
The fallback checks the actual live weight type and its exported key names,
not just a list of suspicious suffixes. Real parameters, buffers, properties,
and dynamically supplied attributes remain merge candidates, including tensors
whose names do not end in `.weight` or `.bias`.

For actual weights, use core `get_key_weight`, preserve its converter, then
apply the same backup and hook-backup precedence and ordered patch list as core.
Do not substitute packed state-dict storage, invent scales of 1, modify upstream
modules, dequantize the entire model, or globally monkeypatch ComfyUI.
The original scales remain attached to the live weight wrapper and its normal
conversion path. Unrelated exceptions and explicit patches/backups targeting
non-addressable metadata are errors, not silently discarded work.

The existing hard-swap forwarding and partial-blend behavior is unchanged.
No claim is made that the two execution modes are numerically interchangeable.

## Input orientation is unchanged

For each selected component, the materialized weight blend is:

`ratio * model1 + (1 - ratio) * model2`

**1 keeps model1; 0 selects model2.** Grouped body/fusion controls use that
same convention. Swapping the model cables alone changes the selected model
at the endpoints; reversing the ratios as well is the corresponding weight
blend, subject to the destination dtype's rounding and execution mode.

## Validation

```sh
python -m unittest discover -s tests -p test_krea2_fp8_merge.py -v
```

28 CPU regression tests pass with PyTorch 2.10.0+cpu. They use real E4M3/E5M2
FP8 storage and torch layers, with fixture Comfy patcher/layout/injection
interfaces. Coverage includes reproducing the old source-only exception,
both input orders, plain and scaled FP8, non-unit scales, 0/1 endpoints,
partial blends, grouped controls, forward-swap injection/ejection, backups,
existing patches, real scale attributes, and unrelated-error propagation.
These are not full ComfyUI integration tests. The user's checkpoint, GPU
inference, memory usage, and generated-image parity have not been tested.

After merging, pull main in the existing DonutNodes folder and restart the
ComfyUI backend. Keep the current workflow; no node replacement or manual
rewiring is required for this code fix.

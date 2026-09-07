# Krea2 merge: operand direction and serialized scale keys

## Ratio convention

Donut intentionally follows the actual `ModelMergeBlocks.merge` implementation
in ComfyUI (including revision `3216c62e`), inherited by `ModelMergeKrea2`:

```text
result = ratio * model1 + (1 - ratio) * model2
1 = keep model1
0 = use model2
```

The generated documentation at
https://docs.comfy.org/built-in-nodes/ModelMergeKrea2 currently describes the
opposite direction. The implementation calls
`add_patches(source, strength_patch=1-ratio, strength_model=ratio)`.
This fix does not invert ratios or rename model inputs.

For **Fineporn body + Krea2 Turbo txtfusion**, connect Fineporn to `model1` and
Turbo to `model2`. Use `body_ratio=1`, `fusion_ratio=0`, and keep the independent
`tmlp.`, `txtmlp.`, and `tproj.` sliders at 1. The five `txtfusion.*` selectors
(two layerwise blocks, projector, two refiner blocks) are 0; all 33 other
component selectors are 1. `txtmlp.` is not part of `txtfusion.*`.

When the two cables are intentionally reversed, invert every component ratio
as well. This describes the merge output, before downstream LoRA/fusion edits.
Grouped controls retain their existing scope and defaults; their tooltips now
explain the operand direction and the three independent sliders explicitly.

## Failure and compatibility repair

Some ComfyUI versions enumerate serialized state entries in `get_key_patches`
and then resolve every entry as a live module attribute. Mixed-precision
`Linear.state_dict` can emit `weight_scale`, other packed-weight companions,
and `comfy_quant`, even though the live module has no such attributes. Its
actual runtime weight carries the scale/layout, and `convert_weight` performs
the corresponding conversion. An FP8 runtime representation can use this path;
this is not evidence of a particular integer format or a corrupt checkpoint.
A filename or filesystem properties dialog cannot establish tensor dtypes.

The previous merge called this collector before planning experimental swaps,
which explains why changing which model is model2 can expose the exception.

The repair first uses the native collector unchanged. Only after AttributeError
it identifies serialization-only keys proven by an actual Comfy QuantizedTensor's
own state export (plus its owning module's comfy_quant descriptor). Live
attributes, weights, biases and unrelated entries are not filtered. A private
reader clone hides only those virtual entries from the native collector; no
input model, tensor, global Comfy function, or source state_dict is modified.
The reader override is removed/restored in a finally block, including on error.
Native patch order, backups, hook backups and converters are retained. No
replacement unit scale is invented; raw serialized bytes are not substituted
for live packed weights. Unknown missing attributes still raise, and unexpected
patch/backup entries on virtual keys fail explicitly rather than being dropped.

Both regular merging and the experimental compatibility path use this helper.
Hard-swap forwarding and partial-blend arithmetic are otherwise unchanged.
Plain BF16, FP16 and raw FP8 tensors stay on the unmodified native fast path.
There is no file-format conversion, new runtime dependency, or workflow rewiring.

## Validation

```sh
python -m unittest discover -s tests -p test_krea2_merge_state.py -v
```

21 CPU tests use actual torch FP8/BF16 storage and linear arithmetic, with a
small patcher/packed-weight adapter modeling the relevant Comfy contracts.
They reproduce the missing weight_scale error; exercise both operand orders,
regular/experimental paths, nonunit scale preservation, every recipe selector,
partial blends, source patches/backups, genuine errors, and reader cleanup.
The packed-weight adapter is not Comfy Kitchen, and this is not a complete
ComfyUI renderer/server or GPU/model-output parity test. The user's checkpoint
bytes were not available and their exact per-layer dtypes were not inspected.

Two previously delivered workflow files were independently audited offline:
all 38 component selectors match the intended recipe and nested model inputs
resolve by link to their expected loaders. Private files are not committed.
A more recent user-edited workflow cannot be certified without its JSON.

After merging this PR, update the existing DonutNodes installation on main and
restart ComfyUI. Existing workflows and node instances do not need recreation.

# Embedded UncensorFix preset

UncensorFix lives in the existing Donut Krea2 Fusion Control node. Its numerical
factor tensors are embedded in `uncensorfix_weights.py`. The original safetensors
file and its metadata are not included. There is no separate asset, file lookup,
file picker, safetensors parser, LoRA-file loader, or runtime download.

## Use

Select **UncensorFix**, then adjust `tap_strength`. Both Simple and Advanced mode
keep UncensorFix selected during edits. Select **Off** for pass-through, or select another preset to change effects. Strength zero skips the embedded patch and does not decode its data.
The existing short names, node ID and legacy widget order are retained.

Selecting UncensorFix disables extra tap/projector profiles and selects standard
fusion. Leave them that way for a comparison with ordinary LoRA application.
Enabling extra Advanced controls intentionally combines their effects with the
embedded UncensorFix patch. Do not also apply UncensorFix elsewhere in the chain.

## Off preset

**Off** disables this Fusion Control node's processing, independently of any
stored tap, projector, enhancer or UncensorFix settings. The incoming MODEL and
all four conditioning routes are returned as the exact same objects, including
metadata and unused optional slots. No embedded weights are decoded, no model
is cloned or patched, and no fusion wrapper or budget is added.

Off does **not** remove upstream LoRAs, merges, runtime injections or UncensorFix
applied by a different node. It only skips the changes this node would add.
Stored control values are kept, and editing them or strength while Off is
selected does not silently reactivate the node. Select an active preset to
resume, or Custom to use the stored manual settings. Off is available in both
Simple and Advanced modes; the existing Custom default and widget order stay
unchanged. Diagnostics report `preset_label=Off` and `fusion_control=off`.

## Representation and execution

The generated Python module stores only raw little-endian float32 up/down
factor values, losslessly compressed and text-encoded, plus target keys, shapes
and alpha values. This is not the original safetensors file encoded as a string:
there is no original container header or source-checkpoint metadata. The
numerical values are reconstructible; this is a packaging format, not encryption.

The module checks its tensor-data SHA-256 and decodes once, lazily. The node
constructs `comfy.weight_adapter.lora.LoRAAdapter` objects directly, using
`(up, down, alpha, None, None, None)`. It does not call `comfy.lora.load_lora`.
The native ComfyUI weight-patch arithmetic is retained, not approximated with tap
gains or converted to a potentially huge set of dense difference matrices.
Every one of the 33 canonical model targets must exist with the expected shape
and be accepted by `add_patches`. The input model is cloned, and fresh factor
copies isolate the cached embedded tensors from downstream mutation.

This requires the ComfyUI LoRAAdapter API used by current Krea2-capable versions.
Its factors still use ordinary weight-patch arithmetic. It now supports an
upstream **Donut Model Merge Krea2 → Experimental bypass** automatically; it does
not add a separate activation-side execution mode or a new node.

## Experimental model-merge bypass

Connect the merged MODEL (including any existing LoRA stack) to Fusion Control,
select UncensorFix, and send Fusion Control's MODEL output to the sampler. No
extra mode switch is needed in Fusion Control.

Exact model2 swaps in the upstream merge execute retained model2 layers. For
these targets, UncensorFix appends its adapters to a **clone of that source
patcher**. Unswapped and partially blended targets stay on the main patcher.
There is no patch applied to an unused model1 copy of a swapped target, and no
extra copy of UncensorFix is added through a forward hook.

Existing main/source patch stacks and runtime LoRA injections are preserved.
Only the output model references the newly patched source. Source-only changes
also invalidate the outer model's clone cache so a later run does not retain
the old swap forward. The existing save/extract helpers see the same source
patch stack, so the routed changes remain available for serialization.

For a full 33-target text-fusion swap the diagnostics include:

```text
uncensorfix_source=embedded
uncensorfix_bypass_source_targets=33
uncensorfix_model_targets=0
uncensorfix_targets=33
```

Mixed merges report the corresponding counts. Missing source/plan metadata or
wrong target shapes raise an error instead of silently patching inactive
weights. The original embedded data, strengths, and Simple/Advanced UI do not
change. UncensorFix strength zero still performs no decoding or patching.

Apply UncensorFix **after** a model merge to affect the resulting merged model.
A later model merge can intentionally replace weights, including adapters
applied before that merge. This fix does not override those merge semantics.

## Validation

Run from the node-pack root:

```sh
python test_krea2_fusion_preset.py -v
python test_uncensorfix_merge_bypass.py -v
node --test tests/teacherfix_ui.test.mjs
```

The preset Python suite has 26 CPU tests of the actual embedded data, including checksum,
shape, rank/alpha, corruption handling, cloning, strength propagation, target
validation, unchanged UI schema, and operation with safetensors, comfy.lora,
comfy.utils and folder_paths imports blocked. ComfyUI model/base/adapter interfaces
are test doubles. The 30 JavaScript tests execute the actual preset-label mutation
callback and both frontend extensions in both registration orders. They include
legacy-label migration and Simple/Advanced strength edits in an isolated VM, not a full browser.

A separate local comparison against the private original checked all 99 tensor
and alpha values and 66 full CPU weight/linear-output comparisons (33 targets at
strengths 0.75 and 1.0) against the ordinary LoRA reference formula. All were
bitwise equal. This was not a full ComfyUI/GPU image-generation A/B test.

The Off regressions check exact model/conditioning identity, preservation of
upstream patches and bypass forwards, no base fusion processing or embedded-data
access, saved settings, switching modes/presets, and rapid selection while old
strength callbacks are queued. The merge-bypass suite has 22 CPU tests.

## Distribution

Ship `uncensorfix_weights.py` alongside `DonutKrea2FusionPreset.py`, not an
`assets/*.safetensors` file. The complete Embedded Update ZIP contains the module
and needs no original weight file. This replaces the earlier bundled-file ZIP.

## Rename compatibility

The visible preset is **UncensorFix** and its data module is `uncensorfix_weights.py`.
Saved workflows using either previous TeacherFix label are normalized to UncensorFix;
the numerical data is unchanged. The existing documentation and test filenames are
retained to avoid breaking links and test commands.

The merge-bypass regression suite uses small CPU linear layers with the actual
Donut merge-plan resolver and swap-injection implementation. ComfyUI's patcher,
adapter, and injection-manager interfaces are test doubles. It checks forward
outputs and save-state routing, not just accepted patch counts. It is not a
full ComfyUI/GPU or quantized-checkpoint image-generation test.

# Embedded UncensorFix preset

UncensorFix lives in the existing Donut Krea2 Fusion Control node. Its numerical
factors remain in `uncensorfix_weights.py`; no original safetensors file, source
checkpoint metadata, weight-file lookup, file picker or runtime download is
required. The embedded numerical data is unchanged by the execution-mode update.

## Compare with Donut Apply LoRA Stack

Use the same incoming MODEL, conditioning, upstream patches, sampler settings
and seed on both branches. Do not apply this LoRA twice.

| Setting | Fusion Control branch | Donut Apply branch |
| --- | --- | --- |
| Preset | UncensorFix | Fusion Control Off, or omit Fusion Control |
| UncensorFix controls | LoRA only | No additional fusion-control changes |
| Strength | `tap_strength = s` | This LoRA's `text_weight = s` |
| Execution | `execution_mode = Comfy patches` | `execution_mode = Comfy patches` |
| Safety | No automatic attenuation | `safe_stack = Off`, `fusion_aware = Off` |

For an Experimental bypass comparison, select **Experimental bypass on both
branches**, not on just the Donut Apply node. This is the LoRA execution mode,
not Donut Model Merge Krea2's separate Experimental bypass setting.

All 33 targets in this LoRA are diffusion-side `txtfusion` weights. Donut Apply
routes them through `text_weight` (the stack's `clip_weight`/`cw` field), not its
main `model_weight`. This does not patch a separate CLIP encoder. A nonzero main
model weight with text weight zero is therefore not an equivalent comparison.

**Comfy patches and Experimental bypass are not numerically interchangeable.**
Native execution patches model weights. Bypass keeps the base forward and adds
a separately evaluated low-rank contribution, using Donut Apply's own helper.
Rounding and quantization can distinguish these computations even with identical
factors. The shared helper retains Donut Apply's conservative regular-patch
fallbacks for unsupported targets and existing runtime injections; check the
console for those fallback messages. Diagnostics report the requested mode.

## LoRA-only and combined controls

`uncensorfix_controls` defaults to **LoRA only** in both UI modes. The backend
skips this node's tap gains, projector changes, enhancer, wrapper registration
and fusion-budget publication, even when hidden or restored widget values are
non-neutral. All four incoming conditioning routes are returned by identity.
Existing upstream controls and patches are preserved, not cleared.

This backend behavior also covers API workflows: it does not rely on frontend
preset callbacks resetting controls. Strength zero in LoRA-only mode returns
the original MODEL and conditioning without decoding the embedded payload.

Select **LoRA + fusion controls** to intentionally combine UncensorFix with the
Advanced settings. That combination is not equivalent to applying the LoRA
alone. **Migration:** older workflows that intentionally combined Advanced
controls with UncensorFix must select this option to retain that combination.
Stored control values are not erased. Selecting or editing these options does
not require renaming the active preset to Custom.

Both comparison options are appended as optional widgets, leaving legacy
widget indices and the node ID intact. They remain explicit in both Simple and
Advanced mode. Existing TeacherFix preset labels are still accepted.

Diagnostics include, for example:

```text
preset_label=UncensorFix; preset_is_ui_only=false
fusion_control=lora_only; conditioning=unchanged
uncensorfix_source=embedded; uncensorfix_strength=0.75; uncensorfix_targets=33
uncensorfix_controls=LoRA only; uncensorfix_execution_mode=Comfy patches; reference_text_weight=0.75
```

## Off preset

**Off** returns the incoming MODEL and all four conditioning routes as the exact
same objects. No model is cloned, no embedded data is decoded, and no patches,
wrappers or fusion budget are added. Stored settings remain dormant.

Off does **not** remove upstream LoRAs, merges, runtime injections, or UncensorFix
applied by another node. `uncensorfix_targets=0` means this node added no targets;
it does not mean the incoming model has no LoRA patches. Switching this node to
Off does not mutate an earlier output from a run where it was enabled.

## Representation and execution

The data module contains losslessly compressed, text-encoded little-endian
float32 up/down values, canonical target names, shapes and alpha values. It is
not the original safetensors container encoded as a string. Its payload is
checksummed and decoded lazily once. Packaging is not encryption.

The repository's recorded factor-payload SHA-256 is:

```text
f3c817bd957e6d47883346237b5e067697f0b9e1c9909bd06353da455949aacf
```

The reference upload used for this parity investigation has the same checksum
for its 3,457,232-byte, sorted-target, up-then-down float32 factor payload. It
contains 33 rank-4 targets with alpha 4, so alpha/rank is 1. No factor values or
strength convention were changed to make this match.

Native mode constructs `comfy.weight_adapter.lora.LoRAAdapter` objects with
`(up, down, alpha, None, None, None)` and appends them to cloned patch lists.
This is the same adapter layout produced by the ordinary Comfy LoRA loader.
It retains native weight-patch arithmetic and does not form dense deltas.
Native mode still does not import the LoRA-file loader.

Experimental bypass uses `donut_uncensorfix_lora.py` to reconstruct canonical
LoRA tensor keys **in memory**, then calls
`DonutSafeApplyLoRAStack._apply_bypass_applications` with the same
`_TEXT_MERGE_VECTOR` used for Donut Apply's fused-text route. There is no separate
bypass arithmetic implementation. A preflight requires every expected target
to map at unit block weight; missing, renamed or muted targets raise an error.
The bridge rejects a helper result that installs neither new regular patches
nor a new forward injection. Standard loader dependencies are needed for this
mode, but no LoRA file is read.

Both modes retain model shape validation and independent copies of the cached
factor tensors. They append to existing patches rather than replacing them.

## Experimental model-merge bypass

Apply UncensorFix **after** a model merge to affect the resulting model. A later
merge can intentionally replace weights patched earlier in the chain.

Exact model2 swaps execute retained model2 layers. Their UncensorFix targets are
routed to a clone of that source patcher; unswapped and partially blended targets
stay on the main patcher. This routing is retained for both LoRA execution modes.
Do not confuse the merge's bypass mode with the separate LoRA execution choice.

Existing main/source patches, injections and merge plans are preserved. Only the
output references the updated source. A fresh source-identity attachment and
patch UUID invalidate the outer clone cache for source-only changes. Native
source patches remain visible to the existing save/extract helpers. Bypass
injections retain Donut's existing bypass serialization/materialization rules.

A full 33-target source swap reports `uncensorfix_bypass_source_targets=33` and
`uncensorfix_model_targets=0`, as well as the total target count. Mixed merges
report the corresponding counts. Missing plans/injections or incompatible
source shapes fail instead of patching inactive main-model weights.

## Validation

Run from the node-pack root:

```sh
python test_uncensorfix_lora_parity.py -v
python test_krea2_fusion_preset.py -v
python test_uncensorfix_merge_bypass.py -v
node --test tests/teacherfix_ui.test.mjs
```

The new parity suite has 20 CPU contract tests. It executes the modified preset
and bypass bridge with doubles for ComfyUI, the base fusion node and the shared
Donut Apply helper's environment. It covers LoRA-only isolation, explicit
combined controls, conditioning identity, positive/negative/zero strengths,
same-target patch accumulation, Off, legacy positional arguments and labels,
canonical bypass factors, shared-helper dispatch, fallback preservation,
invalid target mapping, silent no-op rejection, and main/source routing.

The original code fails the new stored-controls isolation and full positional
preset regressions; the modified code passes the 20-test suite. This is not a
full ComfyUI/CUDA, quantized-checkpoint or image-generation A/B test. The existing
embedded-data, merge-bypass and frontend suites provide additional coverage and
should also be run in a complete checkout.

## Distribution

Ship `uncensorfix_weights.py`, `DonutKrea2FusionPreset.py` and
`donut_uncensorfix_lora.py` with the rest of DonutNodes. Do not add a separate
`assets/*.safetensors` file. The original data module is unchanged. Existing node
IDs and the older documentation/test filenames are retained.

# Embedded TeacherFix preset

TeacherFix lives in the existing Donut Krea2 Fusion Control node. Its numerical
factor tensors are embedded in `teacherfix_weights.py`. The original safetensors
file and its metadata are not included. There is no separate asset, file lookup,
file picker, safetensors parser, LoRA-file loader, or runtime download.

## Use

Select **TeacherFix**, then adjust `tap_strength`. Both Simple and Advanced mode
keep TeacherFix selected during edits. Select Custom or another preset to turn
it off. Strength zero skips the embedded patch and does not decode its data.
The existing short names, node ID and legacy widget order are retained.

Selecting TeacherFix disables extra tap/projector profiles and selects standard
fusion. Leave them that way for a comparison with ordinary LoRA application.
Enabling extra Advanced controls intentionally combines their effects with the
embedded TeacherFix patch. Do not also apply TeacherFix elsewhere in the chain.

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
It is the ordinary weight-patch path, not Experimental bypass execution.

## Validation

Run from the node-pack root:

```sh
python test_krea2_fusion_preset.py -v
node --test tests/teacherfix_ui.test.mjs
```

The Python suite has 20 CPU tests of the actual embedded data, including checksum,
shape, rank/alpha, corruption handling, cloning, strength propagation, target
validation, unchanged UI schema, and operation with safetensors, comfy.lora,
comfy.utils and folder_paths imports blocked. ComfyUI model/base/adapter interfaces
are test doubles. The 8 JavaScript tests execute the actual preset-label mutation
callback in an isolated VM, not a full browser.

A separate local comparison against the private original checked all 99 tensor
and alpha values and 66 full CPU weight/linear-output comparisons (33 targets at
strengths 0.75 and 1.0) against the ordinary LoRA reference formula. All were
bitwise equal. This was not a full ComfyUI/GPU image-generation A/B test.

## Distribution

Ship `teacherfix_weights.py` alongside `DonutKrea2FusionPreset.py`, not an
`assets/*.safetensors` file. The complete Embedded Update ZIP contains the module
and needs no original weight file. This replaces the earlier bundled-file ZIP.

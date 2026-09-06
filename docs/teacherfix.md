# Self-contained TeacherFix preset

TeacherFix stays in the existing **Donut Krea2 Fusion Control** node, with the
short preset names and the Simple/Advanced selector. Its original 3.31 MiB
safetensors is shipped under `assets/` inside DonutNodes. Users do not need to
install anything in `ComfyUI/models/loras`, choose a file, connect another LoRA
node, or download weights at runtime.

## Use

Select **TeacherFix** and adjust `tap_strength`. Strength 1 applies the original
adapter at strength 1; strength 0 skips the bundled adapter. Simple shows the
mode, preset and strength; Advanced retains the existing conditional controls.
TeacherFix remains selected when strength or other Advanced controls change.
Select **Custom** or another preset explicitly to turn off its bundled patch.

Selecting TeacherFix disables extra tap/projector profiles and uses standard
fusion. Keep those controls off for the ordinary-LoRA comparison; enabling
extra Advanced controls intentionally combines their effects with TeacherFix.
Do not apply the same TeacherFix LoRA a second time elsewhere in the model chain.

## Implementation

The bundle is verified and decoded lazily once. The original factor tensors
and alpha values are passed unchanged through ComfyUI's `load_lora` and added
to a clone with `add_patches(..., strength_patch=tap_strength)`. This preserves
the existing ordinary weight-patch route rather than inventing a gain-profile
approximation or an Experimental bypass implementation. All 33 target keys
must map and be accepted, or execution fails explicitly.

## Tests

Run from the repository root:

```sh
python test_krea2_fusion_preset.py -v
node --test tests/teacherfix_ui.test.mjs
```

The Python tests use the real bundled file but doubles for ComfyUI's base node,
model patcher and adapter parser. They cover its exact checksum, tensor identity,
no external-folder dependency, caching, strength routing, model cloning, invalid
assets, incomplete mapping and legacy labels. The JavaScript tests execute the
actual preset-label callback in an isolated VM, not a full browser frontend.
They check that TeacherFix survives edits and can still be explicitly deselected.
These are unit tests, not a full image-generation A/B test on a real Krea 2 model.

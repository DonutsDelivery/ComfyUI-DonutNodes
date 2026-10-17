# Refusal Reduction presets

`Refusal Reduction v1` and `Refusal Reduction v2` are Krea2 Fusion Control presets, like UncensorFix:
the LoRA factors ship inside DonutNodes (`assets/refusal_reduction_v2.bf16`), so nothing needs to be
downloaded or placed in `models/loras`. They are Capitan01R's Krea2 TextFusion Refusal-Reduction LoRA
(see `THIRD_PARTY_NOTICES.md`).

| Preset | Targets | Notes |
| --- | --- | --- |
| Refusal Reduction v1 | 32 | txtfusion layerwise and refiner blocks 0 and 1 only. |
| Refusal Reduction v2 | 35 | The v1 targets plus `txtfusion.projector` and `txtmlp.1` / `txtmlp.3`. |

Both are rank 64 with alpha 64. Their weights are different from UncensorFix and from the Bypass presets
(checked tensor by tensor); they share some target layers with UncensorFix, so applying both stacks
their effects.

Selecting a preset keeps neutral fusion settings and applies the weights, like UncensorFix. In the
Fusion Control node, **tap strength** (or **UncensorFix weight strength** when the composition is
`Fusion + UncensorFix weights`) scales the LoRA as the normal LoRA strength, and **execution mode**
works as it does for UncensorFix.

v2's projector patch edits the same `txtfusion.projector.weight` as the Krea2FilterBypass presets. Do
not combine v2 with a Bypass preset unless you have tested it.

Verified on CPU only: exact tensors against the original files, target and shape checks, and the patch
plumbing. Image quality on a real Krea 2 model has not been verified.

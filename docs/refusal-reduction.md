# Refusal Reduction presets

`Refusal Reduction v1` and `Refusal Reduction v2` are embedded weights for Krea2 Fusion Control, like
UncensorFix: the LoRA factors ship inside DonutNodes (`assets/refusal_reduction_v2.bf16`), so nothing needs to be
downloaded or placed in `models/loras`. They are Capitan01R's Krea2 TextFusion Refusal-Reduction LoRA
(see `THIRD_PARTY_NOTICES.md`).

| Embedded weights | Targets | Notes |
| --- | --- | --- |
| Refusal Reduction v1 | 32 | txtfusion layerwise and refiner blocks 0 and 1 only. |
| Refusal Reduction v2 | 35 | The v1 targets plus `txtfusion.projector` and `txtmlp.1` / `txtmlp.3`. |

Both are rank 64 with alpha 64. Their weights are different from UncensorFix and from the Bypass presets
(checked tensor by tensor); they share some target layers with UncensorFix, so applying both stacks
their effects.

Weights are no longer presets. The Fusion Control node has an **embedded_weights** selector (None,
UncensorFix, Refusal Reduction v1, Refusal Reduction v2) that works with any fusion preset. Use the
**Neutral** preset to apply only the weights without changing the fusion settings. **UncensorFix weight
strength** (shown as *weights strength*) scales the weights as the normal LoRA strength, and **execution
mode** works as it does for UncensorFix.

Workflows saved with the old `UncensorFix` or `Refusal Reduction` preset load as the Neutral preset with the
matching embedded weights. Only one set of embedded weights is applied at a time.

v2's projector patch edits the same `txtfusion.projector.weight` as the Krea2FilterBypass presets. Do
not combine v2 with a Bypass preset unless you have tested it.

Verified on CPU only: exact tensors against the original files, target and shape checks, and the patch
plumbing. Image quality on a real Krea 2 model has not been verified.

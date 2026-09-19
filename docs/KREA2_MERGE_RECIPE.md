# Verified Krea2 component recipe

The intended base merge is **Krea2 Turbo txtfusion layers, Fineporn everywhere
else**. The FP8 source-metadata compatibility fix is already in PR #50; this
follow-up adds tests/documentation and does not replace that runtime code.

## Input and ratio convention

Each ratio is the model1 coefficient:

`merged = ratio * model1 + (1 - ratio) * model2`

| Setting | Value |
| --- | --- |
| model1 | Fineporn |
| model2 | Krea2 Turbo |
| ratio_mode | Grouped |
| body_ratio | 1 |
| fusion_ratio | 0 |
| tmlp., txtmlp., tproj. | 1 each |

Turbo supplies `txtfusion.layerwise_blocks.0.`,
`txtfusion.layerwise_blocks.1.`, `txtfusion.projector.`,
`txtfusion.refiner_blocks.0.`, and `txtfusion.refiner_blocks.1.`.
Fineporn supplies `first.`, all 28 `blocks.*`, `last.`, `tmlp.`, `txtmlp.`,
`tproj.`, and otherwise unmatched keys through the `first.` fallback.

`txtmlp` is not part of `txtfusion`. Body grouping covers first/last and
blocks 0-27; the three separate MLP/projection controls remain independent.
All 38 component ratios are checked in the supplied workflow audit.

Swapping model1/model2 without changing ratios reverses the selection.
To use reversed inputs, also invert body/fusion AND tmlp/txtmlp/tproj ratios.
This establishes the component source; differing destination dtypes/execution
modes can still affect rounding, so reversed inputs are not a pixel-parity claim.

These statements describe the merge output **before** downstream LoRA and
fusion-control modifications. They do not disable any of those later nodes.
No workflow migration, socket change or UI change is introduced here.

## Regression coverage

`python -m unittest discover -s tests -p test_merge_weight_metadata.py -v`

22 additional CPU tests pass against the exact PR #50 merge implementation.
They use real PyTorch FP8 storage and a tensor-subclass scale-export fixture,
plus strict Comfy patcher/injection interfaces. The full 38-component model
fixture tests both input orientations, grouped controls, regular patches,
actual small-module exact-swap forwards, partial blends, source preservation,
backup/converter behavior, rejection of unrelated errors, and standalone imports.

The separately delivered audit also traces the recovered workflow's subgraph
connections and cross-checks the recorded execution prompt in the user's PNG.
Those private files are not committed. The newest reported node ID is not in
the last supplied workflow; no access to an unsaved live canvas is claimed.

No complete ComfyUI session, user-checkpoint load, CUDA kernel, GPU inference,
or generated-image parity test was performed. The runtime compatibility fix
is already merged in PR #50.

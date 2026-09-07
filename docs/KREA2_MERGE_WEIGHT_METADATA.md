# Krea2 merge: FP8 metadata compatibility and component orientation

## The reported crash

`DonutModelMergeKrea2` calls `model2.get_key_patches("diffusion_model.")`
before planning either regular patches or experimental exact-swap hooks.
In affected ComfyUI builds, this enumerates exported state-dict keys and
looks each up as a live module attribute. A weight tensor can export
`weight_scale` while keeping that scale inside its tensor layout, so the
Linear itself has no such attribute. The same issue can occur with FP8;
a filename is not used to decide which compatibility path to take.

Relevant host contracts inspected at ComfyUI commit `3216c62e`:

- `comfy/model_patcher.py`: `get_key_weight`, `get_key_patches`, backups,
  hook backups, and weight conversion.
- `comfy/ops.py`: `_quantized_weight_state_dict`, MixedPrecisionOps Linear
  state export, and `convert_weight`.
- Upstream report: https://github.com/Comfy-Org/ComfyUI/issues/14382

## Compatibility fix

The ordinary host getter remains the first choice. Only an AttributeError
matching a proven export-only field activates the fallback. A skipped field
must be absent on the live module and exported by the weight's own serializer;
`comfy_quant` is also recognized on that serializer-backed format. There must
be a weight converter. Real attributes, including legacy scale buffers, are
not filtered just because their names contain `scale`.

The fallback keeps the live logical weight, its converter, normal/hook-backup
precedence, and the source patch list. It never substitutes packed state-dict
bytes for a logical weight, creates a made-up scale, changes an input model,
or monkey-patches global ComfyUI functions. Unknown missing parameters still
raise; explicit patches/backups on export-only metadata fail rather than being
discarded. Both regular and experimental paths use the same accessor.

Ratios, source/destination meaning, exact-swap hooks, LoRA composition, and
partial-blend behavior are unchanged. No new inputs or workflow migration.

## Recipe orientation

Every ratio is the **model1 coefficient**:

`merged = ratio * model1 + (1 - ratio) * model2`

For Turbo text fusion and Fineporn everywhere else:

| Setting | Value |
| --- | --- |
| model1 | Fineporn |
| model2 | Krea2 Turbo |
| ratio_mode | Grouped |
| body_ratio | 1 |
| fusion_ratio | 0 |
| tmlp., txtmlp., tproj. | 1 each |

This uses Turbo for the two txtfusion layerwise blocks, projector, and two
refiner blocks. `txtmlp` is NOT part of `txtfusion` and stays Fineporn.
`first.` is the fallback for otherwise unmatched keys and stays Fineporn too.
Body grouping covers first/last and blocks 0-27; the three separately exposed
MLP/projection ratios keep their existing independent semantics.

Swapping model1/model2 without changing ratios reverses the recipe. To use the
opposite input order, invert the grouped ratios AND tmlp/txtmlp/tproj ratios.
These statements concern the merge output before downstream LoRA and fusion
control nodes deliberately change it.

## Validation

Run `python -m unittest discover -s tests -p test_merge_weight_metadata.py -v`.
The 22 CPU tests use real PyTorch modules and FP8 tensors with strict patcher
and injection test doubles. They reproduce the old failure, test both source
orientations and execution modes, preserve backups/converters/source patches,
exercise partial blends, reject unproven missing keys, and check standalone
module imports. They require PyTorch but no new runtime dependency.

No full ComfyUI session, user checkpoint load, CUDA kernel, GPU inference, or
image-output parity is claimed. The separate workflow audit is delivered to
the user, not committed with their private workflow or PNG data.

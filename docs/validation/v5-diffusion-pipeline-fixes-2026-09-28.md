# V5 diffusion pipeline audit fixes

Baseline local main: `309b914`, which includes the reviewed NAG fixes plus the unpushed label and legacy-reference validation fixes. The supplied audit inspected GitHub `e361815`. Its ten strict failures reproduced unchanged on local main before this patch.

## Changes and semantics

1. **Non-edit Face Detailer wildcards:** prepare the freshly encoded positive Fusion taps before replacement or `ConditioningConcat`. This preserves already processed primary tokens and avoids inheriting a processed marker over an unprocessed wildcard tail. Positive processing remains independent of the negative `nag_match_taps` switch.
2. **Force inpaint:** oversized bounding boxes and crops skip only when force is off. Force on allows refinement at the computed target size, including downsampling.
3. **Maximum canvas:** force no longer restores uncapped original crop dimensions. The calculated target remains in use; alignment and maximum-edge enforcement also run after `touch_scaled_size`. The reported 3000×1000 crop now uses 1280×448 at max_resolution 1280. Arbitrary later latent-replacement hooks remain outside these canvas calculations.
4. **Advanced CFG:** explicit start/end slicing selects CFG at the same indices as the executed sigma intervals. Example: eight CFG values `8,7,6,5,4,3,2,1`, start 4/end 8 → `4,3,2,1`. Ending early also ends at the corresponding CFG, rather than regenerating a new curve. The pre-range CFG schedule still spans the configured/effective step count; this does not reinterpret denoise or continue NAG alpha across stages. Diagnostic output now shows only selected steps without adding the start offset twice.
5. **CFG midpoint:** simple and advanced midpoint indices clamp to `[1, steps-2]` for three or more steps. One evaluation uses start CFG; two evaluations use start and end. The same endpoint defect in the multi-model calculation is corrected. Diagnostics show the clamped midpoint. Valid interior midpoints and constant CFG retain their behavior.

No workflow JSON, panel bindings, saved widget ordering, or input defaults changed.

## Verification

**127 focused CPU checks passed:**

| Check | Result |
| --- | --- |
| Supplied audit against production checkout, strict mode | 23 passed; previously 10 failures |
| `python -m unittest test_donut_face_detailer test_nag_fusion_taps` | 32 passed |
| `python -m unittest discover -s tests -p test_cfg_interval_contract.py` | 11 passed |
| `python -m unittest test_donut_grounding_nag test_donut_grounding_schedule` | 61 passed |

The supplied audit command was:

```sh
python /tmp/donut-v5-diffusion-audit/v5_diffusion_audit/run_audit.py \
  --repo /home/user/Programs/ComfyUI/custom_nodes/donutnodes --strict
```

Additional regression coverage includes:

- Force off → on → off for both bbox and crop guidance with a 1088×1088 input: off skips; on samples 1024×1024.
- 3000×1000 crop capped at 1280×448, including a sizing hook requesting triple size.
- Wildcard replacement/concatenation with non-neutral taps, both none/tensor RMS normalization, and negative tap matching on/off. Original positive tensors and wildcard metadata remain unchanged.
- Full, head, middle, tail, and single-interval CFG ranges; constant CFG; disabled noise; terminal zero sigma; effective denoise-tail schedule; empty-range rejection; repeated model evaluations within one interval.
- Midpoints below, inside, and beyond the step range; one/two/three-step runs; all configured advanced curve endpoints; accurate selected-step diagnostics.

`test_donut_sampler_dynamic_cfg.py` also has its existing full-runtime lifecycle expectation corrected from prefix CFG 8 to selected CFG 4. That full module was not executed because ComfyUI core is unavailable; the new isolated CFG suite exercises the production guider and slicing functions with explicit application doubles.

Changed Python files compile and `git diff --check` passes. The supplied audit and isolated CFG suite extract production functions from the checkout rather than running a full application import. Neural prediction, encoding, sampling lifecycle, and application setup use doubles; small conditioning tensors and indexing arithmetic use PyTorch.

## Runtime limits

No ComfyUI/GPU workflow, UI Run, backend restart, or output PNG was available. No residual-grain improvement is claimed. These results establish conditioning, canvas, control-flow, and CFG-indexing contracts under the stated settings. The fixes remain local until pushed/published.

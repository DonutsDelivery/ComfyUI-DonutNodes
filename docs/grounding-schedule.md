# Scheduled grounding for V4 Edit Mode

On **DonutSampler**, set `grounding_schedule` to `linear`, `ease_in`, `ease_out`,
or `ease_in_out`, then choose `grounding_start_px` and `grounding_end_px`.
Existing V4 app-control panels acquire these controls beside Turbo when the graph
is loaded. They only affect the base DonutSampler while Edit Mode is enabled;
Edit Studio, face-detailer and upscale grounding inputs are not reinterpreted.
No workflow rewiring is required.

A starting experiment for freer early edits and more late reference information:

| Control | Value |
| --- | --- |
| Edit Mode | enabled |
| Grounding schedule | ease_in |
| Start grounding px | 512 |
| End grounding px | 1088 |

For an eight-step run this produces **512, 512, 576, 640, 704, 832, 960, 1088** px.
`linear` advances evenly, `ease_in` delays the increase, and `ease_out` increases
sooner. `ease_in_out` eases both ends. Descending schedules also work. The first
and last values are exact; intermediate values are rounded to the existing
64-pixel UI increment. A one-step run uses the end value. A truncated advanced
run spans its executed steps, rather than stopping partway through the curve.

## What changes (and what does not)

This schedules the **semantic reference resolution** given to
`Krea2EditGroundedEncode`, not a numeric LoRA strength or an output-image resize.
Each distinct resolution is encoded once for each prompt polarity before
sampling. The sampler selects the corresponding, fully processed positive and
negative conditioning at the same completed-step index used by Donut's CFG
schedule. Different token lengths are never interpolated, and inactive
resolutions are not averaged together.

The existing Edit Mode path still prepares appearance tokens, applies the edit
LoRA and upstream model patches, handles both references and inpaint masks, and
runs the solver once. Extra resolutions increase text-encoding time and
conditioning memory. They do not add VAE encodes or restart/reseed the solver.
There is no cross-run embedding cache retaining references or GPU tensors.

Higher grounding resolution is intended to supply more identity/reference
information, but this is not a guaranteed monotonic identity-strength control.
The encoder caps the longest reference edge; values above the prepared reference
size do not upsample it or add detail. In the upstream encoder, **0 means native/
unlimited resolution, not no grounding**. Genuinely changing schedules therefore
require positive endpoints; equal zero endpoints retain the existing native path.
Visual quality, identity improvement and optimal endpoint settings still need
real Krea2 image comparisons. Appearance tokens and the identity LoRA remain
active throughout; reducing semantic grounding does not disable those paths.

## Compatibility and initial limits

- `constant` is the default and preserves the existing `grounding_px` path.
  Dynamic fields are ignored outside Edit Mode. Equal endpoints use ordinary
  constant grounding at that resolution, without scheduling wrappers.
- Dynamic schedules support `simple` and `advanced` DonutSampler modes with
  Euler, ER-SDE or DPM++ 2M, including live, verified Bleh presets such as V4's
  usual ER-SDE preset. Preset options and scheduler selection are preserved.
- NAG and `multi_model` are explicitly rejected for genuinely changing
  schedules. NAG has a separate negative-conditioning path, and multi-model
  sampling needs a shared cross-phase schedule; neither is silently claimed
  compatible. Other/multi-evaluation/adaptive solvers, Euler churn and Bleh sigma
  overrides are also rejected. Normal constant grounding retains its existing
  compatibility.

The registered node remains `DonutSampler`; the implementation follows the
pack's existing isolated sampler-override pattern. New widgets are appended
rather than inserted into serialized widget positions. Only a cloned model
receives wrappers; a context-local request and `finally` cleanup prevent state
from leaking between runs or after an interruption.

## Validation

Run the focused CPU/stub tests and frontend graph-binding tests:

```sh
python -m unittest test_donut_grounding_schedule.py -v
node test_donut_grounding_controls.mjs
python -m py_compile donut_grounding_schedule.py
```

Before merging, run the existing sampler/Edit Mode suites in a complete ComfyUI
installation and visually compare constant 512, constant 1088, and 512-to-1088
linear/ease-in/ease-out runs with the same seed and reference. Cover Raw and
Turbo, CFG 1 and CFG > 1, one/two references, an inpaint mask, and the installed
V4 Bleh preset. Confirm constant mode matches the previous implementation and
that a subsequent constant run is unaffected by a scheduled run or cancellation.
The unit tests alone do not establish GPU integration or improved likeness.

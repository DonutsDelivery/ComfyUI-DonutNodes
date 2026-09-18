# Scheduled grounding for V4 Edit Mode

Use **Edit Studio**'s grounding schedule/start/end controls in the current V4
workflow, or set `grounding_schedule`, `grounding_start_px` and
`grounding_end_px` directly on **DonutSampler** when those inputs are not linked.
The existing Edit Studio-to-sampler wiring is unchanged by NAG support.
The schedule affects the base DonutSampler only while Edit Mode is enabled;
face-detailer and upscale grounding settings remain independent.

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
conditioning memory. They do not restart/reseed the solver. With NAG, the
installed combined edit/NAG node prepares one wrapper per distinct resolution.
A bounded, run-local VAE cache shares identical reference-pixel encodes between
these wrappers; it compares actual pixel contents, so equal-size A/B references
remain distinct. There is no cross-run cache retaining references or tensors.

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
- NAG is supported, including zero `nag_phi` or `nag_alpha`, with no requirement
  to disable the NAG toggle. Raw/Turbo, one/two reference inputs, fit/crop mode,
  reference boosts and their mask, and the inpaint target keep their existing
  paths. See the NAG details below.
- `multi_model` still needs a shared cross-phase schedule and remains unsupported
  for genuinely changing grounding. Other/multi-evaluation/adaptive solvers,
  Euler churn and Bleh sigma overrides retain their existing guards. Normal
  constant grounding retains its existing compatibility. This NAG fix does not
  claim that every third-party sampler/patch combination has been validated.

The registered node remains `DonutSampler`; the implementation follows the
pack's existing isolated sampler-override pattern. New widgets are appended
rather than inserted into serialized widget positions. Only a cloned model
receives wrappers; a context-local request and `finally` cleanup prevent state
from leaking between runs or after an interruption.

## NAG and reference guidance

When `nag_enabled` is true, the default grounded edit negative is encoded at the
same resolutions as the positive. Each prediction selects both the positive
conditioning and the corresponding NAG negative wrapper. Fusion Control's NAG
conditioning preparation is applied to every new negative. Turbo's ordinary
sampler negative remains zeroed; the separate negative used by NAG is **not**
zeroed. The existing base sampler continues to use CFG 1 with NAG.

An explicit `nag_negative` connection is an authoritative conditioning override:
it stays exactly as supplied (with the normal Fusion preparation), rather than
being replaced by `edit_negative_prompt`. Zero phi/alpha preserves the installed
combined edit forward, including reference boost, fit geometry and masking; it
does not silently switch those features off. Phi/tau/alpha and sigma windows
are passed to the installed NAG node without changing its attention math.

The previous error at zero NAG came from checking `nag_enabled` alone. That
blanket rejection is removed. NAG is prepared through the installed node's
public patch API; no closure inspection, global patch or shared negative-context
mutation is used. Per-prediction option copies replace only the NAG diffusion
wrapper key, leaving other wrappers and settings intact.

## Validation

Run the focused CPU/stub tests and frontend graph-binding tests:

```sh
python -m unittest test_donut_grounding_schedule.py test_donut_grounding_nag.py -v
node test_donut_grounding_controls.mjs
python -m py_compile donut_grounding_schedule.py donut_grounding_nag.py krea2_nag_integration.py
```

Before merging, run the existing sampler/Edit Mode suites in a complete ComfyUI
installation and visually compare constant 512, constant 1088, and 512-to-1088
linear/ease-in/ease-out runs with the same seed and reference. Cover Raw and
Turbo, CFG 1 and CFG > 1, one/two references, an inpaint mask, and the installed
V4 Bleh preset. Confirm constant mode matches the previous implementation and
that a subsequent constant run is unaffected by a scheduled run or cancellation.
For NAG, repeat with phi 0, alpha 0 and active guidance, with/without an explicit
negative override. Check reference boosts, fit/crop and inpainting, and verify
no repeated VAE encode occurs during a correctly primed sampling run. Measure
runtime/VRAM and compare against constant grounding. The focused tests use CPU
tensors and Comfy/NAG doubles, not Krea2 weights; they do not establish GPU
integration or improved likeness.

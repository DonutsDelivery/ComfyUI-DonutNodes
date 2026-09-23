# Tone Lab general-synth checkpoint + scanner-flag fix — 2026-09-23

Follow-up to the Tone Lab bundle: a second independently trained v4 checkpoint
ships as an alternative selection, and the `$socket4` registry false positive
that held 3.0.29–3.0.33 in Flagged state is removed.

## Second checkpoint: donut-tone-v4-general-synth.json

User-supplied `general_model.json`, minified without changing any value.
Source upload SHA-256 `ff1c1f5c…1699d5` (105,724 bytes); bundled minified
SHA-256 `64a5288e…71d0ef` (69,874 bytes). Revision 1, 169→16→8→3, 2,883
weights, `trained: true`, gain head untrained (`gainPercentMax: 0.0`), fitted
on 20,057 training groups / 2,421 validation groups — real photographs from
the Unsplash Lite Dataset (attribution in the checkpoint card) with randomly
applied gamma offsets (sliderStd 18, sliderMax 55, identityProb 0.18), best
validation slider MAE 12.47957 at epoch 28. No images or rating history are
bundled.

## Scanner-flag fix: donut_txtfusion_guard_sampler.py

`signature(_Base.sample).bind(...)` (line 46 in the published 3.0.31–3.0.33
archives) matched the registry YARA rule `$socket4` (`python_network_operations`,
severity info) and every version since 3.0.29 is `NodeVersionStatusFlagged`;
3.0.28 (Active) remained the newest version Manager serves. The wrapper now
mirrors the parent's explicit positional/keyword signature (the established
3.0.27 pattern, cf. `donut_krea2_sda.py`) and forwards by keyword through
`super()`; no `inspect` import and no `.bind(` text remains anywhere in the
file, including comments. Dispatch semantics are unchanged: guard-off is exact
parent passthrough, guard-on attaches to model/model_2/model_3 when present,
keyword-only values and `**nag_options` are forwarded untouched, and the
deprecated reference filename still warns without loading. The parent-signature
contract test now also pins this wrapper against
`donut_grounding_schedule.DonutSampler.sample`.

Test-double note: `tests/test_txtfusion_guard_sampler.py`'s `ParentSampler`
previously exposed a shortened signature the real parent does not have (the old
wrapper accepted any positional tail). It now mirrors the real interface and
calls pass the full required tail, matching how ComfyUI keyword-invokes node
functions.

## Executed locally

- `python3 tests/test_txtfusion_guard_sampler.py` — **17 passed**.
- `python3 tests/test_wrapper_signatures.py` — **2 passed** (includes the new
  guard-sampler parent-signature pin).
- `python3 tests/test_tone_lab_bundle.py` — **9 passed** (r12 bundle intact).
- New-checkpoint conformance, same harness as the r12 bundle test: engine
  `validate_export` accepts the minified file; 169 features agree between
  Python and the engine extracted from the shipped HTML on **18 synthetic RGBA
  proxies** (1e-11); gamma/gain/noOp/raw/coverage predictions agree to 1e-10.
- Flat-frame prediction probe (new vs r12): gamma 0.89–1.21 vs 1.33–1.45 —
  materially different corrections, as expected for the different training
  distributions.
- `py_compile` on the changed module; `git diff --check` clean; the file
  contains neither `inspect` nor `.bind(` byte patterns (scanner re-check).

## Remaining release checks

Same limits as the r12 bundle record: no CUDA execution, no live ComfyUI
panel → Run → PNG metadata → reload pass, no real-photo aesthetic comparison
of the two checkpoints. The dual-selection dropdown is verified only through
`INPUT_TYPES()` (both filenames listed, default None) and the discovery
priority tests, not in a live panel. Registry review of the published version
must be verified per AGENTS.md before calling it available.

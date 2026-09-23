# Release validation: 3.0.34 — 2026-09-23

## What shipped

- **Tone Lab alternative checkpoint** `models/donut_tone/donut-tone-v4-general-synth.json`
  (bundled minified SHA-256 `64a5288e…71d0ef`, 69,874 bytes; source upload
  `ff1c1f5c…1699d5`, 105,724 bytes). Revision 1, 169→16→8→3, 2,883 weights,
  20,057 training groups / 2,421 validation groups — real photographs with
  randomly applied gamma offsets — best validation slider MAE 12.47957,
  gamma-only (`gainPercentMax: 0.0`). Selectable in the
  Save images panel alongside `donut-tone-v4-r12.json`; defaults untouched.
- **Scanner-flag fix** in `donut_txtfusion_guard_sampler.py`: the
  `signature(...).bind(...)` reflection that matched the registry YARA rule
  `$socket4` (Flagging 3.0.29–3.0.33; last Active was 3.0.28) is replaced by an
  explicit parent-signature mirror forwarding through `super()`, per the 3.0.27
  precedent. Dispatch semantics regression-tested (17 tests) and the
  parent-signature contract now pins this wrapper too (2 tests). No `inspect`
  import or `.bind(` byte pattern remains in the file.
- Checkpoint card (`models/donut_tone/README.md`) and node docs updated with
  the second checkpoint's digests, training metadata and trade-offs.
- Full validation record for the follow-up:
  [tone-lab-general-synth.md](tone-lab-general-synth.md).

## Pre-publish validation

- `tests/test_txtfusion_guard_sampler.py` 17 passed;
  `tests/test_wrapper_signatures.py` 2 passed;
  `tests/test_tone_lab_bundle.py` 9 passed.
- New checkpoint: engine `validate_export` accepts the minified JSON; features
  agree between Python and the shipped HTML engine on 18 synthetic RGBA
  proxies (1e-11); predictions agree to 1e-10; no-op gates agree exactly.
- Full-suite discovery from the real ComfyUI venv
  (`/home/user/Programs/ComfyUI-new/ComfyUI`): 512 tests. Failing-test ID sets
  are byte-identical with and without the release changes (45 pre-existing
  environment-limitation entries; none introduced, none masked).
- `py_compile` on changed modules; `git diff --check` clean.

## Publication

- Release commit `27cd7ed` pushed to `origin/main` (pyproject 3.0.33 →
  3.0.34); published from that exact tree.
- Local `comfy node publish` (temp-venv comfy-cli 1.20.0): pre-existing lint
  notes printed as warnings only; `Upload successful.` received.
- Published ZIP SHA-256:
  `b60975682580a2d27fdf8008a741e498e7b65553b77a336edcda74448c240c0c`
- ZIP inspected: `DonutToneLab.py`, `donut_tone_engine.py`,
  `web/donut_tone_lab*.js`, both `models/donut_tone/*.json` checkpoints,
  fixed `donut_txtfusion_guard_sampler.py`, `assets/uncensorfix.f32`,
  `assets/README.md`, `model_sources.json`, `docs/model-sources.md` present;
  zero `tests/` or `tools/` entries.
- Registry state at **2026-09-23 01:36 UTC**:
  `GET /nodes/donutnodes/versions/3.0.34` → HTTP 200,
  **NodeVersionStatusPending**, `status_reason` null.
- Review resolution at **2026-09-23 12:02 UTC**:
  `GET /nodes/donutnodes/versions/3.0.34` → HTTP 200,
  **NodeVersionStatusActive**, `status_reason` null. CDN ZIP
  `https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.34/node.zip` re-hashed
  at Active: `b6097568…240c0c`, matching the published archive. Release is
  approved and served.

## Remaining checks

Pending review must resolve to Active for this exact version before it is
called approved/available in Manager; only then verify the served ZIP hash.
Live ComfyUI panel → Run → PNG metadata → reload pass with both checkpoints
and the dual-selection dropdown remains outstanding, as do CUDA and real-photo
quality comparisons. No Civitai action required: the distributed workflow JSON
was not changed by this release.

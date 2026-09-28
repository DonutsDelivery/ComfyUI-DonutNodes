# Release validation: 3.0.41 — 2026-09-27

## Changes

- Adds configurable Turbo SDA step counts while preserving the scheduler's
  supported-sampler and partial-denoise guards.
- Shares the dynamic NAG alpha curve and other shared NAG settings across the
  base sampler, both Donut tiled-upscale stages and Face Detailer. Each stage
  schedules against its executed sigma list; per-stage NAG enable toggles stay
  independent.
- Consolidates related controls in Settings / Configuration and adds one
  global VAE correction toggle and strength, mirrored to the decode, upscale
  and face-detail execution stages.
- Updates panel organization and saved-workflow repair paths while preserving
  custom panel placement and saved choices.

## Automated verification

- Dynamic grounding/NAG backend tests: 59 passed.
- Tiled-upscale lifecycle tests: 7 passed.
- SeedVR2 stage schema tests: 17 passed.
- SDA native sampler tests: 22 passed.
- Focused panel, reload, VAE, NAG and SDA JavaScript tests: 60 passed.
- TextFusion guard Python tests: 84 tests total, 82 passed and 2 skipped.
- Grounding controls JavaScript test: 1 passed.
- Python compilation passed for the modified backend modules; JavaScript syntax
  checks passed for the changed panel modules; `git diff --check` passed.

## Runtime verification limits

The CUA inventory exposed no ComfyUI application or browser. I could not queue
these panel and workflow changes through the Run button, save a PNG, compare
workflow metadata with the execution prompt, or reload it. No output PNG path
exists. Automated tests cover the backend schedules and panel bindings but do
not replace that end-to-end check. The txtfusion internal guard also remains
experimental and has no image-quality verification.

No distributed workflow JSON changed, so no replacement JSON needs to be
uploaded to Civitai.

## Publication

- Release commit `caf6780` was pushed atomically to GitHub `main` and
  `feat/hires-vae-damage-correction`.
- The Comfy Registry upload succeeded for **3.0.41**. The CLI configuration
  checks and security checks passed.
- Exact-version API check at `2026-09-27T14:30:59Z` returned
  **NodeVersionStatusPending**, with an empty/null `status_reason`. Upload
  success is confirmed; Registry approval and normal update discovery remain
  unverified. On 2026-09-28, the user declined hourly follow-up checks. No
  recurring monitor is scheduled; approval remains unverified at the last
  recorded check.

## Published package verification

- `comfy node pack` produced the source archive; `tools/prepare_registry.py`
  created the Registry staging tree. The publisher's staging-directory fallback
  was used because the staging tree has no `.git` directory.
- Downloaded ZIP checked at `2026-09-27T14:30:59Z`: **229 files**,
  **15,773,280 bytes**, SHA-256
  `45179bfe08a467ca72dea94b56d3b264fb3bb03a2dc3ed432e0056db203dc809`.
  Its bytes and all 229 file hashes match the staged ZIP exactly.
- The package reports version **3.0.41** and includes the new Settings model,
  Registry manual model-files panel, generated model catalog, and required
  `assets/uncensorfix.f32` (3,457,232 bytes; SHA-256
  `f3c817bd957e6d47883346237b5e067697f0b9e1c9909bd06353da455949aacf`).
  Tests, validation reports, credentials, the automatic downloader backend and
  standalone installers are absent.

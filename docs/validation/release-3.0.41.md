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

To be completed after GitHub push, Registry upload and exact-version review.

## Published package verification

To be completed from the staged and downloaded Registry ZIP.

# Release validation: 3.0.43 — 2026-09-28

## Change

Advanced DonutSampler raises a clear error when denoise and explicit step
slicing leave no denoising intervals. A Turbo four-of-eight tail combined with
`start_at_step=4` previously returned the input latent without refinement.
Ordinary tiled hires and valid advanced ranges retain their sampling behavior.
The NAG alpha curve remains stage-local as documented.

## Verification

- A focused CPU harness exercised the production sampler function: the empty
  double-sliced range raised, while Turbo tail sampling and an explicit
  full-schedule range both executed the same four sigma intervals.
- The Turbo resolver returned `(4, 0.5, 0.5)` for `beta` denoise `0.5` and
  `(4, 0.5, 0.4)` for `bong_tangent` denoise `0.4`.
- Changed Python files compiled and `git diff --check` passed. A regression
  test was added to `test_donut_sampler_dynamic_cfg.py`, but the test module
  could not run here because ComfyUI's Python core is absent.

## Runtime limits

No ComfyUI Run-button generation or output PNG was available. No NAG
image-quality comparison was made. No distributed workflow JSON changed.

## Publication

Package and Registry verification will be recorded after upload.

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

- GitHub `main` includes release commit `dd7f274` (`Release 3.0.43: reject
  empty sampler ranges`), pushed before publication.
- `comfy-cli 1.20.0` packed the node. `tools/prepare_registry.py` created the
  registry staging tree, and `comfy node validate` passed configuration and
  security checks.
- Comfy Registry upload of **3.0.43** succeeded. The publisher used its
  staging-directory fallback because the staging tree has no `.git` directory.
- Exact-version API check at `2026-09-28T16:34:37Z` returned
  **NodeVersionStatusPending** with an empty `status_reason`. Upload success is
  confirmed; Registry approval and normal update discovery remain unverified.
  No recurring status monitor is scheduled without the user's opt-in.

## Published package verification

- Downloaded ZIP:
  `https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.43/node.zip`.
- Checked ZIP: **229 files**, **15,775,507 bytes**, SHA-256
  `8a4ee2cfac7087a16657a4c8e0c58ab5b8ed9bda0373be7d21c8b90a5917f1c1`.
- The ZIP reports version **3.0.43** and contains the empty-range guard. All
  229 downloaded file hashes match the validated staging tree. Required
  runtime assets are present; tests, validation reports, credentials, the
  automatic downloader backend, and standalone installers are absent.

# Release validation: 3.0.42 — 2026-09-28

## Changes

- Reapply Fusion tap processing to newly encoded positive edit and grounding
  conditioning, independently of the negative-only NAG tap-match setting.
- Preserve an explicit NAG negative when dynamic alpha and grounding
  resolutions are both active; generated negatives continue to follow the
  grounding schedule when the negative is implicit.
- Keep later diffusion wrappers in the executor chain during upstream T2I and
  edit NAG forwards, including either ordering of Donut Fusion Control and NAG.
- Fail clearly when active NAG encounters ComfyUI attention hooks that its
  dual-text forward cannot safely preserve.
- Correct temporal latent flattening and restoration for five-dimensional
  `[B,C,F,H,W]` inputs in Donut's experimental NAG forward and upstream NAG
  wrapper adapter.

## Automated verification

- `test_nag_fusion_taps`, `test_donut_grounding_schedule` and
  `test_donut_grounding_nag`: 70 tests passed.
- `test_donut_nag_txtfusion`: 22 tests passed with a minimal Comfy wrapper
  executor stub.
- Changed Python modules and test files compiled; `git diff --check` passed.

## Runtime verification limits

No full ComfyUI Run-button generation was performed. This checkout does not
include ComfyUI's Python core or the separate `ComfyUI-Krea2-NAG` pack, so the
wrapper adapter was not exercised against those installed runtime classes and
no image-quality comparison was made.

## Publication

- GitHub `main` includes release commit `536b7d4` (`Release 3.0.42: fix NAG
  conditioning edge cases`). The code and version bump were pushed before
  publishing.
- `comfy-cli 1.20.0` packed the node. `tools/prepare_registry.py` created the
  registry staging tree, and `comfy node validate` passed all configuration
  and security checks.
- Upload of **3.0.42** succeeded. The CLI used its staging-directory fallback
  because the staging tree has no `.git` directory.
- Exact-version API check at `2026-09-28T15:53:58Z` returned
  **NodeVersionStatusPending** with a null `status_reason`. Upload success is
  confirmed; Registry approval and normal update discovery are unverified.
  The user has been asked whether they want hourly follow-up checks; no monitor
  is scheduled unless they opt in.

## Published package verification

- Downloaded ZIP:
  `https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.42/node.zip`.
- Checked ZIP: **229 files**, **15,775,361 bytes**, SHA-256
  `3d932392f114843c016ec587972c89f5fcac5db8eb3ec5057f5af8ce1c4da351`.
- Its version is **3.0.42**. All 229 downloaded file hashes match the
  validated Registry staging tree exactly. The required runtime asset is
  present; tests, validation reports, credentials, automatic downloader
  backend, and standalone installers are absent.

# Release validation: 3.1.0 — 2026-09-28

## Included fixes

- `78b7618`: clarify the Differential Diffusion strength label without moving controls or changing their behavior.
- `309b914`: accept unavailable filenames in the unused legacy txtfusion reference field while preserving validation of other inputs.
- `e8e793c`: restore Fusion taps to non-edit detailer wildcards before replacement/concatenation; honor force-inpaint on oversized faces; retain the canvas cap during forced refinement; align partial advanced CFG with sigma indices; reserve the final evaluation for end CFG when clamping the midpoint.

Includes the earlier NAG edit/refinement fixes from 3.0.45.

## Validation and limits

- [V5 diffusion fixes](v5-diffusion-pipeline-fixes-2026-09-28.md): 127 focused CPU checks passed, including all 23 supplied audit checks in strict mode.
- [Legacy filename validation](legacy-txtfusion-reference-validation-2026-09-28.md): 19 tests and 13 upstream validation checks passed.
- [Caption correction](differential-diffusion-label-2026-09-28.md): 39 tests passed; only the generic caption is updated.
- Full ComfyUI/GPU generation, residual-grain improvement, and live browser verification remain unavailable. No new image-quality claim is made.
- Partial advanced CFG follows indices in the schedule before explicit start/end slicing. NAG alpha remains stage-local. One-step CFG uses start; two-step CFG uses start/end; longer schedules clamp the midpoint inside the range.

Existing workflows receive the caption correction from the frontend and require no replacement JSON. The bundled `workflows/v5/DonutWF_v5.json` also has the corrected caption; manually upload that file to Civitai to refresh its downloadable copy.

Registry packaging uses the documented restricted distribution, replacing the automatic downloader with the manual model-files panel. Exact publication status and downloaded-package verification will be recorded below.

## Publication verification

- GitHub `main` release commit `b8cf7db` was pushed before publication and includes all three previously unpushed fixes.
- `comfy-cli 1.20.0` packed the source; `tools/prepare_registry.py` built a fresh staging directory. Configuration and security validation passed. The staged ZIP was inspected before upload.
- Comfy Registry **3.1.0** upload succeeded using the documented non-Git staging-directory fallback.
- At **2026-09-28T21:04:40Z**, the version listing with status reasons and the exact-version endpoint both reported **NodeVersionStatusPending**. `status_reason` was empty. Upload success is confirmed; Registry approval and normal update discovery remain unverified.
- The user declined hourly review checks for 3.1.0. No recurring monitor is scheduled. Registry approval remains unverified.

## Downloaded package verification

- URL: `https://cdn.comfy.org/donutsdelivery/donutnodes/3.1.0/node.zip`
- **228 files**, **15,774,560 bytes**.
- SHA-256: `d64960db62663aa3c15b1bee63bba661242017902d0c45f6285d4c5ff6a37c4e`.
- Published ZIP is byte-identical to the inspected staged ZIP; all extracted files match the staging tree. It declares version 3.1.0. Changed runtime files, workflow caption, and weight asset also match the source checkout.
- Required assets and manual model catalog are present. Credentials, caches, development tests/tools, validation reports, automatic downloader backend, and standalone installers are absent.

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

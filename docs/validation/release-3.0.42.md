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

Registry package inspection, upload, and exact-version status verification are
recorded after publication.

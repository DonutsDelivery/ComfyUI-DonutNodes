# Release validation: 3.0.40 — 2026-09-27

## Changes

- Fixes shared NAG panel updates by finding each workflow family's stage nodes
  through their retained NAG enable controls. Auto phi, guidance scale and the
  other shared fields now target generation, hires and face stages when edited.
  The existing auto-phi formula, separately wired alpha and per-stage enable
  choices are preserved. Saved settings are not rewritten on page load.
- Bounds inference intermediates for linear LoKr and supported LoKr/LoRA stacks
  to 1024 tokens per batch, retaining the existing factor and strength math.
- Removes a reference cycle in the LoKr coverage callback that could retain
  retired adapter GPU weights until cyclic garbage collection.
- Makes FaceDetailer apply the existing Krea2 MLP/normalization memory options
  to the actual model passed to sampling. Existing option values are respected.

## Review scope

Source, existing execution metadata, startup and packaging inspection only.
The user declined verification generations. No implementation tests, browser
interaction, parity checks, GPU benchmark or new generation was run.
The LoKr cleanup follow-up was loaded by a successful local backend restart;
OOM avoidance and the corrected panel interaction remain unverified.

See the [NAG audit](nag-auto-phi-2026-09-27.md) and
[FaceDetailer investigation](facedetailer-vram-2026-09-27.md).
Raw execution history, prompts and generated images are not distributed.

No workflow JSON or model catalog changed, so no replacement workflow or Civitai
upload is required. The separate model installer is unchanged. Existing users
should restart ComfyUI for the Python changes and hard-refresh the frontend.
Reapplying the shared Auto Phi/scale settings synchronizes existing stage values.

## Publication

Publication and package verification are in progress. Upload success and exact
Registry review status will be recorded separately after publication.

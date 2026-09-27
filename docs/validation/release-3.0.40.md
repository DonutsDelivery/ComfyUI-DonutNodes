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

- Release commit: `bc364b1`, pushed to GitHub `main` and
  `feat/hires-vae-damage-correction`.
- Comfy Registry upload succeeded for **3.0.40**. Registry creation time:
  `2026-09-27T00:50:27.965070Z`.
- Exact-version review check at `2026-09-27T00:53:18.465563Z` returned
  **Pending** (`NodeVersionStatusPending`), with an empty status reason.
  Registry approval and normal update discovery are therefore unverified.
- The user declined the optional hourly review checks offered under `AGENTS.md`.
  No recurring monitor is scheduled. Registry approval remains unverified;
  Pending is the last observed status recorded above.

## Published package verification

The Registry distribution was prepared in a separate staging directory using
the existing release helper. The CLI's configuration/security validation passed.
The uploaded CDN ZIP was downloaded and inspected at
`2026-09-27T00:53:19.090776Z`:

- **228 files**, **15,763,482 bytes**.
- ZIP SHA-256:
  `94e8907ae695aca59621e8f870264d594fa3e015bc73cf82532b5c724ac05084`.
- The archive hash and every file hash match the inspected staging package.
- Version metadata is **3.0.40**; changed runtime files match the release source.
- Required `assets/uncensorfix.f32` is present: **3,457,232 bytes**, SHA-256
  `f3c817bd957e6d47883346237b5e067697f0b9e1c9909bd06353da455949aacf`.
- The Registry manual model interface and generated catalog are included.
  Credentials, development/private files, the automatic downloader backend and
  the separate installer/distribution directory are excluded.

These checks establish package integrity, not Registry approval or runtime
correctness. Implementation verification limits are recorded above.

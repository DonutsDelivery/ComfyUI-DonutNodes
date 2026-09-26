# Release validation: 3.0.37 — 2026-09-25

- Adds opt-in one-pass VAE damage subtraction to the V5 base decode, Donut hires
  upscales, and face detail. Defaults remain Off; each stage uses its connected
  VAE and has an independent 0–4 strength control.
- No distributed workflow JSON changed. The V5 frontend upgrades only the
  directly connected stock base decoder in memory when the backend registers
  `DonutVAEDecode`; the user's JSON file does not need a manual upload.
- `git diff --check` and Python bytecode compilation passed for the four
  changed backend modules. The existing E702 warning in
  `donut_txtfusion_guard.py` was cleared by splitting adjacent digest updates
  onto separate lines; the computed digest is unchanged.
- Source was pushed to GitHub branch `feat/hires-vae-damage-correction` at
  commit `de5b58c`.
- Packed and prepared with comfy-cli 1.20.0. The local Registry validator passed
  with no warnings after the E702 cleanup. The validated packed archive was
  15,756,704 bytes, SHA-256
  `a668d4a6de28b753e614762549da3ccc66bb867010cbe629f9f7f956d0d471b5`.
  The Registry staging check confirmed the new helper and UI, required model
  assets, generated link catalog, and manual model-files panel; no test or
  development directories, credentials, downloader backend, or installers
  were included.
- Registry upload succeeded. The published CDN ZIP was downloaded and checked
  at `2026-09-25T15:47:37Z`: 15,758,273 bytes, SHA-256
  `25d40ffe0632dbe3ae1964d9b733aee99a58c0c7776ca14e64561f52df489b62`.
  Version 3.0.37, the VAE helper and panel files, model assets, manual panel,
  and installer link were present; forbidden development and installer files
  were absent.
- Exact Registry API check at `2026-09-25T15:47:14Z` returned
  `NodeVersionStatusPending` for 3.0.37 and an empty `status_reason`.
  Upload success is confirmed; Registry approval is unverified until the exact
  version becomes Active. At `2026-09-25T16:00:47Z`, the user declined hourly
  follow-up checks; no recurring monitor was scheduled.
- During preparation of 3.0.38, a Registry check at
  `2026-09-26T09:12:47Z` confirmed **NodeVersionStatusActive** for 3.0.37,
  with `status_reason: Passed automated checks`. Registry approval of 3.0.37
  is now confirmed; this was a release-preparation check, not a recurring
  monitor.
- UI end-to-end validation is blocked in this environment: CUA reports no
  available browser, so the changed controls could not be set in the panels or
  queued with ComfyUI's Run button. No audit PNG, metadata/prompt comparison,
  save/reload transition check, or GPU output comparison was produced. The
  running backend's `/object_info` confirms registration of the new decoder and
  updated upscale node, but does not verify panel bindings or generation.
- No distributed workflow JSON changed, so no Civitai workflow JSON upload is
  required.

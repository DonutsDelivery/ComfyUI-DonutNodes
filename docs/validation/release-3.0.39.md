# Release validation: 3.0.39 — 2026-09-27

## Changes

- Caches textarea measurements by content, width and typography so unchanged
  prompts no longer collapse and expand on every 500 ms panel refresh.
- Uses cached panel heights during canvas widget sizing. Resize observation
  updates those heights when the panel dimensions actually change.
- Replaces prompt-variant refresh callbacks when variants are rebuilt, releasing
  references to the removed prompt DOM instead of continuing to poll it.
- Caches standalone reference-crop overlays and resizes their canvas buffers
  only when needed. Context restoration invalidates the drawing cache.
- Skips unchanged status text, field values and slider bounds during refresh.
- Updates all shared layout imports to `?v=17`, including the Registry manual
  model-files panel, and refreshes the app controls' weight-control import.

## Review scope

Source and packaging inspection only. No implementation tests, browser
reproduction, performance recording, generation or crash-free run was performed.
These changes repair identified panel defects; they are not a confirmed fix for
the native Firefox crash. See the
[sanitized investigation](firefox-canvas-crashes-2026-09-26.md).

No browser preferences, GPU acceleration settings, workflow JSON or backend
Python changed. Existing workflows do not need replacement, and no Civitai
workflow upload is required. A hard refresh loads the frontend changes in the
local ComfyUI installation. The model catalog and separate installer are
unchanged from 3.0.38.

## Publication

- Release commit `ab3dc13` was pushed to both `origin/main` and
  `origin/feat/hires-vae-damage-correction` as atomic fast-forward updates.
- Packed the committed source and prepared a new Registry staging directory
  using `tools/prepare_registry.py`. Publishing used the existing isolated
  temporary environment with comfy-cli 1.20.0. The CLI's publication security
  validator passed with no warnings. This was package validation, not an
  implementation or generation test.
- Inspected the staged archive at `2026-09-26T22:20:46Z`: version 3.0.39;
  228 files; changed frontend files match the source; the manual model-files
  panel matches `distribution/registry/donut_model_downloads.js`; and the model
  catalog, generated link catalog and required runtime asset are present.
  The numerical asset matches its documented 3,457,232-byte size and SHA-256
  `f3c817bd957e6d47883346237b5e067697f0b9e1c9909bd06353da455949aacf`.
  Credentials, caches, tests, development tools/reports, the automatic
  downloader backend and standalone installers are excluded.
- Registry upload succeeded. The published CDN ZIP was downloaded at
  `2026-09-26T22:21:32Z`: 15,763,092 bytes, SHA-256
  `29446d7819a93f3ab922d9bbf7004470ec15719f664d5bcf6725922d310528bd`.
  It matches the inspected staging archive byte for byte. All 228 individual
  file hashes match, with no missing, extra or changed files.
- Exact Registry version check at `2026-09-26T22:21:32Z` returned
  `NodeVersionStatusPending` for 3.0.39, with an empty `status_reason`.
  Upload success is confirmed; approval remains unverified until that exact
  version becomes Active. Hourly follow-up checks were offered per `AGENTS.md`;
  no recurring monitor has been scheduled without an explicit yes.

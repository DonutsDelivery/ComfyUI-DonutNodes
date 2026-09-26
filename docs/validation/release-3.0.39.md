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

Publication and package verification are in progress. Upload success and exact
Registry review status will be recorded separately after publication.

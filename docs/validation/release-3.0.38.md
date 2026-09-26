# Release validation: 3.0.38 — 2026-09-26

## Changes

- Adds the Spacepxl Wan2.1/Qwen 2× VAE to the model catalog and installer.
  The existing Models panel VAE selector supplies the encoder and decoder.
  `DonutVAELoader` applies the upstream ComfyUI-VAE-Utils adapter when that
  checkpoint is selected. Its internal 2× decode is filtered immediately to
  the configured image dimensions before downstream stages consume it.
- Base decode, hires and face detail use the selected VAE consistently.
  Existing independent VAE correction toggles and strength controls remain.
  Tagged V5 workflows migrate the Models panel's loader while preserving its
  selected checkpoint, ID and connections; no workflow JSON replacement is
  required. Retired separate decoder controls are removed.
- Excludes oversized full-frame candidates from the tiled upscale solver's
  roughly 1 MP tile budget.
- Reduces redundant reference-preview drawing, canvas-buffer resets, panel
  refreshes and graph-layout redraws. GPU acceleration stays enabled.
  Both GitHub and Registry panels import the same layout module version.
- Rebuilds the separate manual model installer with the updated catalog.
  The Registry package keeps its manual model-files panel and excludes the
  optional automatic downloader and standalone installer.

## Review scope

Source and packaging inspection only. No implementation tests, browser
reproduction, generation, PNG metadata comparison, save/reload exercise or
GPU benchmark was run for this release. The running ComfyUI and Firefox
processes were not restarted. Runtime VAE behavior, panel bindings, image
quality and Firefox crash avoidance remain unverified.

See [the VAE source audit](finetuned-vae-decoder.md) and
[the sanitized Firefox investigation](firefox-canvas-crashes-2026-09-26.md).
Raw crash files and local browser identifiers are not distributed.
No distributed workflow JSON changed, so no Civitai workflow upload is needed.

## Publication

- Release commit `5160767` was pushed to both `origin/main` and
  `origin/feat/hires-vae-damage-correction`. Both updates were fast-forwards.
- The separate installer ZIP contains the current source catalog. Its public
  GitHub `main` download was fetched at `2026-09-26T09:17:19Z` and matched the
  rebuilt artifact: 7,125 bytes, SHA-256
  `abbb985638c8a6e5a3d56101df4fc827e5a0239040619b126cc6d540dae84c44`.
- Packed the committed source and prepared a fresh Registry staging directory
  using `tools/prepare_registry.py`. Publishing used an isolated temporary
  environment with comfy-cli 1.20.0 because the system launcher referenced a
  Python installation without that module. The local CLI's publishing security
  validator passed with no warnings. This was package validation, not an
  implementation or generation test.
- Inspected the staged archive: version 3.0.38; 228 files; required VAE helper,
  frontend changes, model catalog and generated link catalog present. The
  Registry manual panel matches `distribution/registry/donut_model_downloads.js`.
  Its layout import matches the other panels. The numerical asset's documented
  3,457,232-byte size and SHA-256 match. Credentials, caches, tests, development
  tools/reports, automatic downloader backend and standalone installers are
  excluded.
- Registry upload succeeded. The published CDN ZIP was downloaded at
  `2026-09-26T09:18:18Z`: 15,762,354 bytes, SHA-256
  `ed0054cae5b2b44bd7f2b407f08a096673edbc1a9435dad2580482d4b933ce23`.
  This matches the inspected staging archive byte for byte. All 228 individual
  file hashes match, with no missing, extra or changed files.
- Exact Registry version check at `2026-09-26T09:18:18Z` returned
  `NodeVersionStatusPending` for 3.0.38, with an empty `status_reason`.
  Upload success is confirmed; approval remains unverified until that version
  becomes Active. Hourly follow-up checks were offered per `AGENTS.md`; no
  recurring monitor has been scheduled without an explicit yes.

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

Release preparation is in progress. GitHub commits, package checksums, Registry
upload outcome and exact review status will be recorded after publication.

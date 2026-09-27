# Settings panel and global VAE correction validation

## Changes covered

- Rename the existing Seed & guidance card to **Settings / Configuration**.
- Put shared NAG strength beside the dynamic NAG alpha schedule; move the base
  sampler's NAG controls and TextFusion RMS guard into Settings.
- Replace per-stage VAE correction controls with one global toggle and strength
  slider, mirrored to base decode, both Donut hires stages, and face detail.
- Move Settings beside Generation setup only when the saved panel layout still
  matches the canonical prior V2 layout. Keep custom panel placement.

## Regression checks

The focused automated checks cover:

- NAG static strength and dynamic schedule paths appear together in Settings;
  schedule bindings still resolve to the base sampler and repair stale paths.
- One VAE control pair points to the connected base decoder, with mirror paths
  for both hires stages and face detail. Per-stage panel copies are removed.
- Legacy VAE widgets migrate from the saved base decoder toggle and strength
  once. A current saved shared setting skips migration; an incomplete target
  set waits without marking migration complete.
- Canonical V2 layout moves Settings beside Generation and upgrades to V3;
  a manually rearranged layout remains in its saved positions.
- Model-wide TextFusion guard appears in Settings and is removed from Models.

Expected and observed in unit fixtures: all assertions above pass after running
`node --test tests/panel_categories.test.cjs tests/txtfusion_model_guard_controls.test.cjs tests/vae_global_controls.test.cjs tests/app_controls_preset_refresh.test.mjs` (56 passing checks).

## UI generation check

The CUA inventory exposed no browser or application surfaces in this session.
The Run button, generation panels, save/reload path, execution prompt, and PNG
workflow metadata could not be exercised here. No output PNG was produced.
The automated tests inspect panel descriptors and widget propagation; they do
not replace a queued ComfyUI generation. Refresh the browser after the backend
is available, inspect the Settings card, set a distinctive correction strength,
queue an image, then save/reload and confirm the shared values and metadata.

No distributed workflow JSON was edited. Existing V5 JSON files do not need a
manual replacement upload.

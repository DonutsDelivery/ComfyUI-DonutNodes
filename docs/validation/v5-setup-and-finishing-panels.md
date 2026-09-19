# V5 setup and finishing panels — 2026-09-19

Starting commit: `c9a2c37` (main). Local changes; no Registry publication.

## Cause and correction

ComfyUI's installed frontend (`settingStore-Bqt2KcA8.js`) adds a
`control_after_generate` widget after an INT named `seed`. The shipped
`DonutSeedVR2Upscale` array had ten entries and omitted that widget. Model/VAE
and sampling values therefore loaded into the wrong fields, also corrupting
model discovery. The corrected array has eleven entries. The narrow import
repair handles the old ten-entry shape using complete valid named metadata.
Modern arrays remain authoritative, including subsequent user edits.

Generate & finish is split into Generate, First upscale, Face detail, Second
upscale, and SeedVR2 panels. The existing control paths and generation wiring
are retained, with new layout columns and App Mode entries. Tagged older layouts
receive the same split when imported. The authoring helper persists it in JSON.

The fresh-install workflow selects public Krea2 in Single model mode. The
previous primary model's Civitai URL returned HTTP 401 without credentials.
Existing saved workflows keep their model choices. Download errors now identify
the relevant credential configuration and display a failure count.

## Live verification

- Range-checked all 14 catalog links without downloading full model bodies:
  13 returned HTTP 206 with the expected byte totals; the previous primary
  model returned HTTP 401. Redirects remained on supported upstream hosts.
- Imported the corrected workflow in the running ComfyUI frontend. Both
  SeedVR2 selectors, color correction `none`, denoise `1`, steps `1`, and
  VAE tile size `512` loaded correctly and survived reload.
- Clicked **Download missing** in the actual interface. Job
  `967cf07d89bc418d8df31106b4066baf` finished with **11/11 ready, 2 downloaded**.
  The downloaded files were `seedvr2_3b_int8_convrot.safetensors` under
  `/home/user/Programs/ComfyUI/models/diffusion_models/` and
  `seedvr2_ema_vae_fp16.safetensors` under
  `/home/user/Programs/ComfyUI/models/vae/`. The backend checks exact size and
  SHA-256 before publishing files. The other nine files were verified/reused.
- Imported the final public-Krea2 defaults and ran the workflow. Prompt
  `0e289940-beee-44e4-8e3c-fc4e6818a43e` completed successfully, including final
  preview and save nodes (`914`, `64`). Start 2026-09-19 16:09:52 UTC;
  finish 16:12:11 UTC, about 139 seconds. The captured API prompt confirms
  Single model mode, both loaders selecting public Krea2, and correctly typed
  SeedVR2 fields. SeedVR2 itself was disabled as shipped; this is not a GPU
  acceptance test of the enabled SeedVR2 engine.
- One browser import tool call returned unusually slowly, but the imported
  workflow and subsequent execution succeeded. No root cause for that tool
  delay was established.

## Automated checks

- 56 focused JS tests passed: panel categories, model requirements, native model
  discovery/download UI, SeedVR2 controls. New regressions cover the frontend
  seed-control slot, preservation of modern values, all moved controls, App Mode
  entries, idempotence, and fresh-install versus personal model defaults.
- 57 Python tests passed: 13 downloader tests, 22 native-download tests, and 22
  SeedVR2 engine/post-node contract tests. Download fixtures cover integrity,
  destination roots, redirects, credentials, cancellation and atomic writes.
- Broader JS run: 211 passed, 7 failed, 2 skipped (220 total before the final
  default-selection test was added). All seven failures concern the previous
  native LoRA combo-picker harness. Running that test file against untouched
  HEAD reproduced the same seven failures (31 passed, 7 failed, 1 skipped).
  The full suite is therefore not green; these failures predate this change.
- Python syntax and `git diff --check` passed.

This was tested on an existing installation, not an empty machine. Existing
node packs and nine model files were reused. Enabled SeedVR2 GPU inference,
7B downloads/inference, and a Registry package were not tested or published.

## Follow-up: iteration-first working area and native LoRA search

Reordered the panels into setup, finishing/save, then editing/references, prompts,
seed/guidance and result. The result sits below seed/guidance next to prompts.
The saved graph layout and App Mode ordering are updated; the ordering migration
runs once. Checked unchanged settings, control paths and execution wiring.

Replaced both datalist LoRA pickers with ComfyUI's native combo/menu. The panel
opens the complete installed catalog with a blank native Filter list field,
and reads refreshed choices on every open. The node again uses a normal combo
widget. Original row callbacks continue to handle saved selection and hashes.
The earlier seven LoRA tests now pass. Full JS suite: **220 passed, 2 skipped,
0 failed (222 total)**. An additional menu-adapter regression checks full
catalog forwarding, valid selection, refreshed files and stale row protection.

Live follow-up verification: opened the full installed LoRA list in the native
menu, typed `identity`, and observed only the matching identity LoRA. Searched
for and selected the original aesthetics LoRA to verify commits and leave the
original selection active. DOM position checks confirmed prompts beside
seed/guidance and the result, following the setup and finishing columns.
Native menu filtering requires the active LiteGraph canvas; panel clicks now
initialize that context before opening the menu.

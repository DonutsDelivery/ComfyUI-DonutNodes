# SeedVR2 and Auto subject: model setup

The new features use Donut's existing model-file setup module. They do not
install packages or download weights while a sampler or mask job is running.

## Git checkout / git pull

Update DonutNodes, restart ComfyUI and refresh the browser. Select **SeedVR2** in
the desired upscale stage and choose 3B or 7B. For automatic masking, choose
**Auto subject** under Edit Studio's **Smart subject mask · B**. Then click the
existing **Download missing** button. Wait until the selected files are ready.
Enable the relevant stage/Editing controls and run, or use **Auto select subject**
for a mask-only job. Save the workflow to retain the choices.

Discovery reads the chosen feature configuration, including nested modules,
even before the stage/Editing toggle is enabled. It does not run the workflow to
determine dependencies. A muted or bypassed node/subgraph is skipped. The normal
**Donut** engine does not request SeedVR2 weights; **Off**, **Saved mask** and
**External mask** do not request BiRefNet. To prepare the Auto select button
before first use, select **Auto subject** before clicking Download missing.

The default selections require the following catalog files:

| Selection | Default destination | Exact bytes |
| --- | --- | ---: |
| SeedVR2 3B | `models/diffusion_models/seedvr2_3b_int8_convrot.safetensors` | 3,458,259,704 |
| SeedVR2 7B, only when selected | `models/diffusion_models/seedvr2_7b_int8_convrot.safetensors` | 8,334,897,976 |
| SeedVR2 VAE, shared by both | `models/vae/seedvr2_ema_vae_fp16.safetensors` | 501,324,814 |
| BiRefNet Auto subject | `models/background_removal/birefnet.safetensors` | 444,473,596 |

Only the chosen SeedVR2 variant is requested, unless different stages select
both. Duplicate file requests are deduplicated by the existing downloader.
Registered custom model roots are honored, including `extra_model_paths.yaml`.
Matching renamed files are reused by their verified hash and rebound to the
original selectors; a newer user selection is not overwritten when a download
finishes. Different existing files are preserved, not silently replaced.

These catalog entries use pinned Comfy-Org revisions with exact sizes and
SHA-256 values from the upstream Git LFS pointers. Existing download-host
restrictions, authorization handling, cancellation, integrity verification and
atomic publication remain in effect. URLs supplied by a workflow are not used.
Custom native-compatible model filenames can still be selected, but unknown
files are not downloaded from invented sources.

**Core compatibility is separate from model installation.** SeedVR2 requires
`SeedVR2Preprocess`, `SeedVR2Conditioning` and `SeedVR2PostProcessing`. Auto subject
requires `LoadBackgroundRemovalModel` and `RemoveBackground`. Missing native
node IDs are reported before downloading the affected catalog models. Update
ComfyUI and restart, then retry. This button does not update ComfyUI, Python
packages, GPU drivers or external Grounding/SAM/YOLO node packs. Having all model
files also does not guarantee that a particular output size fits in VRAM.

## Comfy Registry package

The staged Registry package continues to exclude `donut_model_downloads.py` and
substitutes the manual **Model files** panel. Select the feature configuration
as above, then choose **List selected models**. The panel shows an upstream link,
exact default save path, byte size and SHA-256 for each catalogued file. Download
and place the files yourself, then refresh ComfyUI. A configured custom model
root can replace the displayed default `models/<folder>/` root.

The Registry panel has no calls to the downloader routes, no network fetches of
model metadata and no model-file writes. Links are ordinary user-clicked HTTPS
anchors. Unknown filenames remain listed with their destination and an explicit
no-catalog-link message.

`tools/prepare_registry.py` generates `web/donut_registry_catalog.js` from the
**packed archive's** `model_sources.json`, then selects the manual panel. This
keeps the links matched to the exact release without maintaining a second
catalogue or introducing a Registry download service. Always use the documented
Registry staging process; publishing approval remains a separate Registry check.

## Sources and validation

The new entries were checked on 2026-09-19 against these upstream revisions:

- [SeedVR2 3B and 7B INT8 files](https://huggingface.co/Comfy-Org/SeedVR2/commit/10f035adc869a5b3ffc466360b869641511c0610).
- [Native SeedVR2 VAE](https://huggingface.co/Comfy-Org/SeedVR2/commit/0bb1f83c716d1cad6dfa730b643a4f603bc2b70b).
- [Native BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet/commit/35767b272f2846752a3aee1259abdd4586f735c8).

See [validation notes](validation/native-model-downloads.md) for the exact tests
run and the remaining real-install acceptance checks. Metadata verification and
offline tests are not a claim that the multi-gigabyte downloads or GPU inference
were run in the test environment.

# Donut Workflow V5

A Krea2 workflow built around DonutNodes, with a redesigned interface for model setup, LoRAs, editing, prompts, generation and saving. DonutNodes supplies the controls and processing nodes; this workflow brings them together.

**Workflow file:** `DonutWF_v5.json` · **[DonutNodes on GitHub](https://github.com/DonutsDelivery/ComfyUI-DonutNodes)**

## Quick start

1. Install or update **DonutNodes to the code accompanying V5**. Restart ComfyUI and refresh the browser after updating. An older registry build that lacks `DonutImageSave` or `DonutEditStudio` cannot run this workflow.
2. Download `DonutWF_v5.json` from this workflow’s download files. Drag it into ComfyUI or use **Open**. If GitHub shows a file preview, use **Download raw file**.
3. Choose **Install Missing Nodes → Install All**, accept the default pack versions, and **Apply Changes/restart**. No WAS package is required.
4. Select your models in **01 · Models**, then use **Download missing** for files supported by Donut's model catalog. Install any reported uncatalogued files separately, or select models you already have.
5. Review the enabled LoRAs, prompts, seed, resolution and output settings. Leave **Editing** off for a first text-to-image run, then click **Run**.

The V5 workflow must be distributed with the updated node code. The local installation test used the development checkout, not an older registry release. Workflow **V5** and the DonutNodes package version are separate version numbers.

### Companion packs

Manager detects these from the workflow, including the unconnected nodes inside **Required node packs**. Keep that subgraph: it makes internally used dependencies visible to the missing-node installer.

- **ComfyUI-bleh:** Sampler preset
- **ComfyUI Impact Pack:** Detection, SAM and face-detail processing
- **ComfyUI Impact Subpack:** Ultralytics detector provider
- **Derfuu ComfyUI ModdedNodes:** Text boxes
- **ComfyUI Krea2 Edit:** Identity-edit integration
- **Krea2 NAG:** Negative attention guidance
- **Krea Seed Variance Enhancer:** Seed-dependent conditioning variance

WAS and rgthree's label nodes are no longer needed by this workflow. Existing installations may keep those packs for other workflows; updating Donut does not uninstall them.

### Models and downloads

The supplied configuration selects the following files. These are the saved choices, not a promise that every file is needed in every mode or included in this repository.

- **Primary diffusion model:** `krea2_turbo_bf16.safetensors` (public upstream; Single model mode)
- **Secondary diffusion model:** `krea2_turbo_bf16.safetensors` (inactive until Merge two models is selected)
- **Text encoder:** `qwen3vl_4b_fp8_scaled.safetensors`
- **VAE:** `qwen-image/qwen_image_vae.safetensors`
- **Upscaler:** `4x_NickelbackFS_72000_G.pth`
- **Face detector:** `bbox/face_yolov8m.pt`
- **SAM:** `sam_vit_b_01ec64.pth`
- **Enabled generation LoRA:** `krea2/Krea2_NSFW_Aesthetics_V1.safetensors`
- **Edit LoRA:** `krea2/krea2_identity_edit_v1_2.safetensors`

Use ComfyUI's corresponding model folders. Single-model mode does not load the secondary model. Editing uses the identity-edit LoRA; editing off requires no reference images. Review the saved model/LoRA choices before distributing your own preset.

**Download missing** only downloads after a click. It uses the repository's reviewed model catalog (`model_sources.json`), checks file size and SHA-256, and can reuse matching renamed files. It reports missing catalog entries rather than guessing download links. Some hosts require your own access credentials: Civitai uses Donut's local API-key setting; Hugging Face can use `HF_TOKEN`.

The shipped SeedVR2 post-pass selects `seedvr2_3b_int8_convrot.safetensors` in `models/diffusion_models/` and `seedvr2_ema_vae_fp16.safetensors` in `models/vae/`. **Download missing** prepares these files even while that post-pass is off. Selecting 7B requests that variant instead. The fresh-install primary model uses public Krea2; existing saved workflows keep their own model choices. The previous primary remains available to select, but its Civitai source requires access credentials.

## What’s in the workflow

Panels run from setup, through finishing and saving, to the controls you iterate on most: editing/references, prompts, and seed/guidance beside the result. The ordering migration runs once when the workflow is opened.

Numbered cards expose everyday controls, with additional controls under **Advanced**. The source loaders and generation wiring live inside inspectable subgraphs. Graph and App Mode use the same underlying settings.

- **Models:** choose a single model or two-model merge, encoder, VAE and upscaler.
- **LoRAs & block weights:** search installed filenames using ComfyUI’s native dropdown filter; add, remove, reorder and enable LoRAs; edit strengths and block weights through sliders or numeric fields.
- **Image setup & editing:** manage editing references, independent crops and selected-area editing.
- **Prompts:** edit the general/face, scene and negative text with autosizing
  editors and a shared wildcard tool. The connected Prompt card is prompt 1;
  add blank variants or duplicate it, then set **Active prompt** to a fixed
  1-based set or choose **increment** to advance after each generation.
- **Seed & guidance:** control the shared seed, NAG and seed variance.
- **03 · Generation setup:** output size, batch and base sampling.
- **04 · First upscale** and **05 · Second upscale:** separate hires controls and advanced settings.
- **06 · Face detail:** face refinement, detection and masks.
- **07 · SeedVR2 upscale:** optional final native upscale, model/VAE selection and advanced sampling.
- **Latest result:** choose a stage preview or follow the latest output; inspect the final expanded prompt and stage progress.
- **Save images:** choose the destination, format, quality and filename behavior.

## Editing and reference guidance

For editing, enable **Editing**, upload/paste/drop image **A** as the base scene, and optionally use image **B** for the subject/identity. Enter the edit instruction and adjust the crop and output sizing. With B connected, face identity comes from B. Image A alone remains supported.

**Reference guidance** is a separate optional path for borrowing visual elements through Krea2's native image conditioning. Describe what to borrow in the prompt. It does not use the edit LoRA and pauses while Editing is enabled.

Save after changing references or crops. Reference images are stored separately under `ComfyUI/user/donut/edit_references/`; copy that folder too when moving a personal workflow to another installation. The distributed beta has empty reference slots.

**Select a specific object in B:** under **Smart subject mask · B**, choose
**Prompt selection**, enter a **Mask prompt** such as `hat` or `shirt`, and click
**Select from prompt**. Review the preview and refine it with the mask editor.
Download missing includes the native SAM3.1 checkpoint. Auto subject remains
whole-foreground removal. See [prompt masking](../../docs/prompt-subject-mask.md).

### Inpainting · edit a selected area

Use the updated workflow JSON together with the accompanying node code, then
restart ComfyUI and refresh the browser. The workflow connects Edit Studio's
selection to the sampler and every finishing stage. Older workflow JSONs show
a reminder to load the updated workflow instead of offering an unconnected mask.

1. Add your base image to **A**, then click **Paint area…**.
2. Paint over what should change, or use **Rectangle** and drag between two
   corners. A circle follows the brush/eraser pointer to show its size before
   drawing. Green is the selection; the red frame is the output crop. Use
   **Erase**, **Undo**, or **Clear** as needed.
   **Invert selection** protects the painted area and edits everything else.
   Adjust **Seam width** in the painter; the amber band previews the inward
   blend in output pixels. Zero gives a hard edge.
3. Click **Use selection**, then describe the change in **Prompts** and run.
   This automatically enables **Editing** and **Edit selected area**.

**Edge softness** on the main card is the same setting as **Seam width** in the
painter. Applying saves both the selection and seam width; Cancel keeps their
previous values. The full cropped
image still supplies context, and optional **B** supplies subject identity.
The selection is saved inside the workflow and stays aligned when output size
or crop changes. Replacing A resets the selection; B can change independently.
Turn **Edit selected area** off to return to whole-image editing without losing
the saved mask. An empty selection cannot run as an accidental whole-image edit.
To remove the saved selection completely, open **Paint / outpaint…**, choose
**Clear selection**, then **Clear & turn off**. Switching the main control on
without a saved selection opens the editor so it cannot enter an invalid state.

The sampler denoises through the mask using the encoded base image. After base
decode, each upscale, and Face Detailer, the workflow restores A outside the
selection. At the same size these are the original cropped pixels; at a larger
output size they are resized pixels from A. The unselected region does not gain
new generative upscale detail. Use PNG or lossless saving for exact pixel
preservation in the saved file. This mode uses the existing identity-edit LoRA
and sampler; it adds no model downloads or node-pack dependencies.

## Wildcards

Use the **Wildcard library** card to create or edit one-choice-per-line text files. Insert `haircolor*` to use `user/wildcards/haircolor.txt`; nested names such as `clothes/shirt*` and `__haircolor__` syntax also work. The shared seed drives expansion, so a fixed seed keeps choices repeatable. Copy your wildcard files when moving installations; they are not embedded in the workflow JSON.

## Saving images

**Donut Image Save** replaces WAS's save node. The supplied settings preserve `output/Final`, the seed-based filename, WebP quality 100 and numbered saves with overwrite disabled. The secondary resized output remains a core ComfyUI save.

The Donut saver supports 8-bit PNG, JPEG, WebP, TIFF, GIF and BMP; output/temp locations; delimiter and number placement; lossless WebP; and optional PNG/WebP workflow metadata. It does not include WAS's history browser, color-profile socket or high-bit-depth/EXR modes. Those features were unused in the migrated configuration.

## Moving from V3

Open V5 as a separate workflow. Keep your older JSON and copy your own model choices, LoRAs, prompts and settings through the visible controls. Do not copy raw widget arrays between versions: the layout and inputs changed. Re-select references in Edit Studio and copy any personal wildcard/reference files.

V3 already supported editing, face detail, model merging, LoRA stacking and upscaling. V5 reorganizes and extends that workflow; it does not introduce face detailing or promise better image quality simply from the version change. See the accompanying `CHANGELOG.md` for the actual differences.

## Beta validation and reporting

A fresh Linux/Python 3.12.7 venv with the current DonutNodes source passed default **Install All → restart → full generation**, without WAS or a manual package repair. Base generation, first upscale, two face refinements and WebP saving completed on an RTX 4070. The saved 1728 × 1344 image matched the previous WAS save byte-for-byte with the same configuration. Six saver tests and 34 registration/dependency-isolation tests passed.

That acceptance run used editing off and the second upscale disabled, and reused model files. It does not certify every configuration, model downloads, other GPUs, or Windows/macOS installation. Companion requirements still install two OpenCV variants in the tested environment; a warning remains, but imports and inference passed. No fixed minimum VRAM requirement has been established.

When reporting a problem, include your workflow file with private content removed, DonutNodes/ComfyUI versions, OS, GPU/VRAM, active mode and full error traceback. **Donut Dependency Check** can provide dependency diagnostics.

Detailed test notes accompany the node source in `docs/validation/no-was-fresh-install-2026-09-09.md`.

### Outpainting

In Edit Studio, load image A and open **Paint / outpaint…**. Enable **Outpaint**,
then resize or drag A, or use Left/Right/Top/Bottom to align it. The full original
A is fitted without cropping. Output shape and pixel count come from the existing
output controls: outpainting does not add pixels beyond that canvas.

Use **Add right**, **Add left**, **Add below**, or **Add above** to build a
side-by-side or stacked canvas automatically. Each preset derives the aspect
ratio from A, chooses the closest pixel-grid resolution within the current
megapixel budget, places A on the opposite side at maximum size, and selects the
new half. **Keep current** returns to the canvas dimensions that were active
when the editor opened. Applying a preset changes Edit Studio to Custom sizing
with the displayed dimensions.

Green areas are generated automatically. **Overlap** allows editing a strip inside
A to join the new surroundings; set it to zero to protect all of placed A.
**Seam width** softens the selected boundary. Brush, rectangle, erase and invert
still work on A; uncovered canvas always remains selected. Click **Use selection**
and describe the extension in Prompts. Placement persists with the workflow.

A is resized to fit its placement, so protection applies to those placed pixels.
Enabled finishing upscales still run afterward. Turn selected-area editing off to
return to normal full-image editing. Restart ComfyUI and refresh the browser after
installing this update to load both the backend and editor changes.

The **Latest result** selector includes Base generation, First upscale, Second
upscale, Face Detailer, and SeedVR2 / final image. Disabled processing stages
pass through their input, so their previews can match the previous stage.
Refresh the browser and reopen a tagged V4 or V5 workflow to add missing preview
branches. Run it again to populate the new stage images.

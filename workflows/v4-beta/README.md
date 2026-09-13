# Donut Workflow V4 Beta

A Krea2 workflow built around DonutNodes, with a redesigned interface for model setup, LoRAs, editing, prompts, generation and saving. DonutNodes supplies the controls and processing nodes; this workflow brings them together.

**Workflow file:** `DonutWF_v4_beta.json` · **[DonutNodes on GitHub](https://github.com/DonutsDelivery/ComfyUI-DonutNodes)**

## Quick start

1. Install or update **DonutNodes to the code accompanying this beta**. Restart ComfyUI and refresh the browser after updating. An older registry build that lacks `DonutImageSave` or `DonutEditStudio` cannot run this workflow.
2. Download `DonutWF_v4_beta.json` from this workflow’s download files. Drag it into ComfyUI or use **Open**. If GitHub shows a file preview, use **Download raw file**.
3. Choose **Install Missing Nodes → Install All**, accept the default pack versions, and **Apply Changes/restart**. No WAS package is required.
4. Select your models in **01 · Models**, then use **Download missing** for files supported by Donut's model catalog. Install any reported uncatalogued files separately, or select models you already have.
5. Review the enabled LoRAs, prompts, seed, resolution and output settings. Leave **Editing** off for a first text-to-image run, then click **Run**.

The beta workflow must be distributed with the updated node code. The local installation test used the development checkout, not the older registry release. Workflow **V4 Beta** and the DonutNodes package version are separate version numbers.

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

- **Primary diffusion model:** `finepornV4INT8NVFP4BF16_v4.safetensors`
- **Secondary diffusion model:** `krea2_turbo_bf16.safetensors`
- **Text encoder:** `qwen3vl_4b_fp8_scaled.safetensors`
- **VAE:** `qwen-image/qwen_image_vae.safetensors`
- **Upscaler:** `4x_NickelbackFS_72000_G.pth`
- **Face detector:** `bbox/face_yolov8m.pt`
- **SAM:** `sam_vit_b_01ec64.pth`
- **Enabled generation LoRA:** `krea2/Krea2_NSFW_Aesthetics_V1.safetensors`
- **Edit LoRA:** `krea2/krea2_identity_edit_v1_2.safetensors`

Use ComfyUI's corresponding model folders. Single-model mode does not load the secondary model. Editing uses the identity-edit LoRA; editing off requires no reference images. Review the saved model/LoRA choices before distributing your own preset.

**Download missing** only downloads after a click. It uses the repository's reviewed model catalog (`model_sources.json`), checks file size and SHA-256, and can reuse matching renamed files. It reports missing catalog entries rather than guessing download links. Some hosts require your own access credentials: Civitai uses Donut's local API-key setting; Hugging Face can use `HF_TOKEN`.

## What’s in the workflow

Numbered cards expose everyday controls, with additional controls under **Advanced**. The source loaders and generation wiring live inside inspectable subgraphs. Graph and App Mode use the same underlying settings.

- **Models:** choose a single model or two-model merge, encoder, VAE and upscaler.
- **LoRAs & block weights:** add, remove, reorder and enable LoRAs; edit strengths and block weights through sliders or numeric fields.
- **Image setup & editing:** set the output size and optional editing references.
- **Prompts:** edit the general/face, scene and negative text with autosizing
  editors and a shared wildcard tool. The connected Prompt card is prompt 1;
  add blank variants or duplicate it, then set **Active prompt** to a fixed
  1-based set or choose **increment** to advance after each generation.
- **Seed & guidance:** control the shared seed, NAG and seed variance.
- **Generate & finish:** adjust sampling, first/second upscale and face detail.
- **Latest result:** choose a stage preview or follow the latest output; inspect the final expanded prompt and stage progress.
- **Save images:** choose the destination, format, quality and filename behavior.

## Editing and reference guidance

For editing, enable **Editing**, upload/paste/drop image **A** as the base scene, and optionally use image **B** for the subject/identity. Enter the edit instruction and adjust the crop and output sizing. With B connected, face identity comes from B. Image A alone remains supported.

**Reference guidance** is a separate optional path for borrowing visual elements through Krea2's native image conditioning. Describe what to borrow in the prompt. It does not use the edit LoRA and pauses while Editing is enabled.

Save after changing references or crops. Reference images are stored separately under `ComfyUI/user/donut/edit_references/`; copy that folder too when moving a personal workflow to another installation. The distributed beta has empty reference slots.

## Wildcards

Use the **Wildcard library** card to create or edit one-choice-per-line text files. Insert `haircolor*` to use `user/wildcards/haircolor.txt`; nested names such as `clothes/shirt*` and `__haircolor__` syntax also work. The shared seed drives expansion, so a fixed seed keeps choices repeatable. Copy your wildcard files when moving installations; they are not embedded in the workflow JSON.

## Saving images

**Donut Image Save** replaces WAS's save node. The supplied settings preserve `output/Final`, the seed-based filename, WebP quality 100 and numbered saves with overwrite disabled. The secondary resized output remains a core ComfyUI save.

The Donut saver supports 8-bit PNG, JPEG, WebP, TIFF, GIF and BMP; output/temp locations; delimiter and number placement; lossless WebP; and optional PNG/WebP workflow metadata. It does not include WAS's history browser, color-profile socket or high-bit-depth/EXR modes. Those features were unused in the migrated configuration.

## Moving from V3

Open V4 Beta as a separate workflow. Keep your V3 JSON and copy your own model choices, LoRAs, prompts and settings through the visible controls. Do not copy raw widget arrays between versions: the layout and inputs changed. Re-select references in Edit Studio and copy any personal wildcard/reference files.

V3 already supported editing, face detail, model merging, LoRA stacking and upscaling. V4 Beta reorganizes and extends that workflow; it does not introduce face detailing or promise better image quality simply from the version change. See the accompanying `CHANGELOG.md` for the actual differences.

## Beta validation and reporting

A fresh Linux/Python 3.12.7 venv with the current DonutNodes source passed default **Install All → restart → full generation**, without WAS or a manual package repair. Base generation, first upscale, two face refinements and WebP saving completed on an RTX 4070. The saved 1728 × 1344 image matched the previous WAS save byte-for-byte with the same configuration. Six saver tests and 34 registration/dependency-isolation tests passed.

That acceptance run used editing off and the second upscale disabled, and reused model files. It does not certify every configuration, model downloads, other GPUs, or Windows/macOS installation. Companion requirements still install two OpenCV variants in the tested environment; a warning remains, but imports and inference passed. No fixed minimum VRAM requirement has been established.

When reporting a problem, include your workflow file with private content removed, DonutNodes/ComfyUI versions, OS, GPU/VRAM, active mode and full error traceback. **Donut Dependency Check** can provide dependency diagnostics.

Detailed test notes accompany the node source in `docs/validation/no-was-fresh-install-2026-09-09.md`.

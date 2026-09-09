# DonutNodes node guide

[Back to DonutNodes](../README.md) · [Donut Workflow V4 Beta](../workflows/v4-beta/README.md)

## Nodes

| Node | Description |
|------|-------------|
| DonutImageSave | Image saving with numbered filenames and optional PNG/WebP metadata |
| DonutEditStudio | Base/identity references, crops and edit settings |
| DonutLoRAStack | Block-weighted LoRA stacking with presets |
| DonutApplyLoRAStack | Apply stacked LoRAs to model/CLIP |
| DonutLoraStackCombine | Merge two LoRA stacks |
| DonutFaceDetailer | Face detection and enhancement |
| DonutUniversalDetailer | Auto-detect object enhancement |
| DonutDetailerZIT | ZIT-based detail enhancement |
| DonutSDXLTeaCache | TeaCache acceleration for SDXL |
| DonutTiledUpscale | Tiled img2img upscaling |
| DonutKSamplerCFG | CFG sampling with curve control |
| DonutSpectralNoiseSharpener | Reference-based spectral sharpening |
| ModelMergeZIT | ZIT model merging |
| DonutModelMergeKrea2 | Krea2 component merging with optional hybrid hard-swap bypass |
| DonutModelSave | Save merged models |

### Experimental quantized LoRA bypass

`DonutApplyLoRAStack` has an optional `execution_mode` named
`Experimental bypass`. It computes the quantized base layer and low-rank LoRA
path separately instead of repeatedly rebuilding patched quantized weights.
This can substantially reduce warm inference time for Krea2 and similar models
under Dynamic VRAM while preserving the LoRA's model strength, block vector,
Safe Stack attenuation, and fusion-aware processing.

Multiple plain linear LoRA, matrix-factor LoHa, and direct or matrix-decomposed
linear LoKr adapters can be stacked. LoKr supports different ranks for its two
decompositions, with scaling matched to ComfyUI's regular patch path. LoHa
rebuilds a dense delta each forward, so bypass may be slower and use more memory.
Plain convolutional LoRA/LoCon supports Conv1d/2d/3d, including flattened factors,
stride, dilation, and standard zero padding; grouped convolutions and other
padding modes retain regular patches.

Overlapping compatible components are composed into one forward hook per model
layer while retaining each LoRA's own strength and block vector. Hooks bind to
the actual sampling model (including model copies/delegates) and are removed
without restoring stale merge forwards when the model is unloaded. If any component
on a target is unsupported—such as DoRA, reshaped or Tucker adapters, output
transforms, convolutional LoHa/LoKr, direct diffs, or unknown adapter classes—the
complete ordered patch sequence for that target stays on ComfyUI's regular patch
path. Models with pre-existing runtime injections and
LoRAs without supported forward adapters use the regular compatibility path
instead of failing. Existing workflows default to `Comfy patches` and retain
their previous behavior.

When a Krea2 merge bypass swaps a layer from model2, downstream LoRA adapters
for that layer follow the selected execution mode on the retained model2 source.
Experimental bypass attaches its hooks to that source, since hooks on the outer
layer would be skipped by the swap.

### Experimental Krea2 model-merge bypass

`DonutModelMergeKrea2` mirrors the component controls and ratio direction of
ComfyUI's built-in `ModelMergeKrea2`: `1.0` keeps `model1`, while `0.0` uses
`model2`. Its optional `execution_mode` also defaults to `Comfy patches`.

`Experimental bypass` uses a hybrid strategy optimized for inference:

- `1.0` keeps the `model1` component unchanged.
- `0.0` uses a runtime hard swap to the compatible `model2` linear layer, keeping that layer's original weight, bias, and quantization metadata intact.
- Partial ratios such as `0.25`, `0.5`, and `0.75` use ComfyUI's normal materialized merge patches so inference executes one merged linear forward instead of two full model forwards.

This avoids the main performance problem of the original experimental version,
which evaluated both full linear layers for every partial blend. Unsupported
hard-swap targets still use ComfyUI's regular patch path. Model2 is retained as
an additional runtime model only when at least one compatible exact `0.0` swap
is active.

Runtime hard swaps are inference-time behavior and are not materialized by
checkpoint saving. Select `Comfy patches` before `DonutModelSave` or another
checkpoint save node when you need fully saved merged weights. Inputs that
share the same underlying model, use different load devices, or already contain
runtime injections automatically use the regular compatibility path instead of
failing.

### Fusion-aware Krea2 LoRA safety

`DonutApplyLoRAStack` can budget Krea2 projector
LoRAs against the 12 resolved projector-input gains from
`DonutKrea2FusionControl`.

Connect the model path in this order:

`Checkpoint -> Donut Krea2 Fusion Control -> Donut Apply LoRA Stack -> Sampler`

Then set `safe_stack` to `On` and choose a `fusion_aware` mode:

- `Attenuate only` reduces LoRA projector columns amplified by Fusion Control.
- `Use headroom` may also boost quieter columns, capped by `max_fusion_boost`.
- `tensor_rms` projector normalization is prompt-dependent, so it never
  receives automatic headroom boosts.

The 12 fusion channels are not mapped onto Krea2's 28 DiT blocks. Existing
per-block Safe Stack behavior remains independent and unchanged. Fusion-aware
column scaling supports standard LoRA/PEFT projector adapters and direct
projector `.diff` patches; other adapter formats use a conservative scalar
fallback.

## Edit Studio

`Donut Edit Studio` combines two image slots, an edit instruction, output sizing,
grounding, and the identity-edit LoRA. Click a slot and press Ctrl+V, drop an image,
or use Upload. A is the base/scene; optional B is the subject/identity. Drag the
red crop frame to choose the retained area, or press Center to reset it.

The modular workflow uses the same DOM cards in Graph and App Mode, ordered from
model setup through LoRAs, images, prompts, generation, and saving. `Donut Section
Controls` keeps advanced controls collapsed inside each card. Cards grow to fit
their contents, with high-contrast headers and no internal scrolling. The modular
graph moves later cards and columns when a section expands; slider banks widen
their cards to show every weight. Long prompt fields grow with their text.
Both views edit the same native widgets, including controls inside subgraphs;
LoRA additions, removals, strengths, and ordering stay synchronized. Control groups
are stored in each card’s `donut_app_controls` property and saved with the workflow.
These cards are frontend controls and do not add processing steps.

`Download missing` checks the models selected in the workflow, including nested
subgraphs and enabled Donut LoRA rows. Its single button verifies local SHA-256
hashes and downloads missing files into ComfyUI's configured model folders.
Matching files moved or renamed within those folders are reused and selected in
the loaders. Existing files with different hashes are preserved and reported.
Downloads remain temporary until their size and hash match; the button becomes
Cancel while working. This module starts downloads only when its button is clicked.

Sources are limited to the repository's `model_sources.json`: each entry specifies
`folder` (ComfyUI loader category), `filename` (relative path), `url`, `sha256`, and
`size` (bytes). Add reviewed upstream entries there to support more models;
workflow-supplied URLs are never used. Pin the exact upstream file/revision.
Civitai uses the existing local Donut API-key setting; Hugging Face can use
`HF_TOKEN`. Hash results are cached under `user/donutnodes` and invalidated when
the file changes. Missing files without a catalog entry are reported, not guessed.

The supplied modular workflow keeps original main-graph parameters visible and
puts internal subgraph parameters under Advanced. Cards are color-coded by
purpose; numeric weight vectors use a horizontal bank of vertical sliders with
precise number entry and an ALL control. Model setup selects either a single primary model or a two-model
merge; single-model mode does not load the secondary model.

Long prompt fields include a wildcard picker and an expanded-text preview.
Insert or type `haircolor*` to select a line from `user/wildcards/haircolor.txt`.
The wildcard library creates and edits persistent files, one choice per line;
nested names such as `clothes/shirt*` and existing `__name__` syntax both work.
Prompt templates retain their tokens; expansion uses the shared seed at execution,
including edit instructions. Fixed seeds keep choices repeatable. The modular
workflow replaces automatic prompt-addition dropdowns with explicit tokens.

`Donut Reference Guidance` supplies independent full-image references to Krea2's
native Qwen3-VL positive conditioning. Describe which elements to borrow in the
generation prompt. It uses no edit LoRA or crop and pauses while Editing is on.
Connect its image and enabled outputs to the corresponding `native_reference_*`
inputs on `Donut Prompt Conditioning`. With guidance off, normal text encoding
is unchanged. Negative conditioning remains text-only.

References are stored under ComfyUI's `user/donut/edit_references/`, separate from
temporary and pasted inputs. Save the workflow after selecting images or changing
crops. Workflow JSONs contain reference IDs; copy that reference folder as well
when moving a workflow to another installation. Editing off requires no images.

Sizing supports presets, custom width/height, reference aspect at a megapixel
budget, and crop-only alignment to 32 or 64 pixels. Crop-only trims to the next
smaller grid dimensions without scaling, except for images smaller than one grid
cell. The preview and backend use the same crop geometry.

`DonutSampler.source_image_b` and `DonutTiledUpscale.edit_source_image_b` feed both
images to Krea2's grounded encoder and appearance-token patch. With
`DonutFaceDetailer.face_reference_b` connected, the detailer extracts face identity
from B instead of A. Single-reference workflows remain supported. Editing requires
`comfyui-krea2edit` and an identity-edit LoRA; restart ComfyUI after installing or
updating the nodes.

### Integrated Krea2 NAG

Enable `nag_enabled` on DonutSampler, DonutTiledUpscale, or DonutFaceDetailer
(requires the installed `krea2-nag` pack). Edit mode uses the combined edit/NAG
patch with the internally prepared references; non-edit mode uses regular NAG.
The legacy sampler nodes also support regular NAG. Multi-model mode patches each
active model. NAG is disabled by default and uses CFG 1 when enabled.

`nag_negative` accepts an unzeroed negative conditioning override. Without it,
NAG uses the encoded `edit_negative_prompt` in edit mode or the sampler's
`negative` input otherwise. With `turbo_mode` on, the sampler negative is zeroed
**after** NAG captures its conditioning, even when NAG is disabled. If your
workflow already zeroes the negative upstream, connect the original conditioning
to `nag_negative`.

The `nag_phi`, `nag_tau`, `nag_alpha`, and sigma controls match the standalone
nodes. Edit NAG also exposes reference boosts, a boost mask, and fit mode. Reference
images, VAE, and target latents are supplied internally. Restart ComfyUI and
refresh the browser to load the new inputs.

### Shared Krea2 seed variance

`Donut Prompt Conditioning` has one optional `variance_enabled` control set for
both general and face positives, using the installed `krea-seed-variance-enhancer`
implementation. Connect `variance_seed` to the sampling seed to vary each run;
face conditioning uses seed + 1 with 64-bit wraparound. The remaining variance
controls match the standalone enhancer. Negative conditioning is unaffected.

The variance recipe travels with positive conditioning. DonutSampler, the
upscaler, and the face detailer reapply it to freshly encoded grounded positives
in edit mode. Ordinary sampling uses the already enhanced positives, so it does
not apply the noise twice. Existing workflows default to variance disabled.

## Third-party attribution

See [third-party notices](../THIRD_PARTY_NOTICES.md) for incorporated code and its licenses.

# ComfyUI-DonutNodes

[![Support on Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Development-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/donutsdelivery)

Custom nodes for ComfyUI focused on LoRA management, model merging, and image enhancement.

## Features

- **Block-weighted LoRA stacking** with per-block strength control, CivitAI integration, and an experimental quantized-model bypass mode
- **Krea2 component model merging** with regular patches or an experimental hybrid hard-swap bypass
- **Donut Detailers** for per-block model tuning and face/object enhancement
- **TeaCache acceleration** for faster SDXL inference
- **Tiled upscaling** with seamless blending
- **CFG sampling** with 18 curve types
- **Spectral noise sharpening** for reference-based detail enhancement

## Installation

### ComfyUI Manager
Search for "DonutNodes" in ComfyUI Manager and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DonutsDelivery/ComfyUI-DonutNodes.git donutnodes
cd donutnodes
python -m pip install -r requirements.txt
```

Run the install command with the same Python interpreter that launches ComfyUI.

## Optional companion packages

- [ComfyUI-DonutLocalAutomation](https://github.com/DonutsDelivery/ComfyUI-DonutLocalAutomation) provides the local-only Prompt Receiver and Image Reporter nodes.
- [ComfyUI-DonutCivitaiLocal](https://github.com/DonutsDelivery/ComfyUI-DonutCivitaiLocal) provides optional local CivitAI library and workflow-recovery tools.

Install `ComfyUI-DonutLocalAutomation` alongside this package to keep the original
`DonutPromptReceiver` and `DonutImageReporter` node IDs in existing workflows.

## Nodes

| Node | Description |
|------|-------------|
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

Multiple plain linear LoRA and direct-factor linear LoKr adapters can be
stacked. Overlapping compatible components are composed into one forward hook
per model layer while retaining each LoRA's own strength and block vector. If
any component on a target is unsupported—such as DoRA, reshaped or decomposed
adapters, output transforms, convolutional targets, direct diffs, or unknown
adapter classes—the complete ordered patch sequence for that target stays on
ComfyUI's regular patch path. Models with pre-existing runtime injections and
LoRAs without supported forward adapters use the regular compatibility path
instead of failing. Existing workflows default to `Comfy patches` and retain
their previous behavior.

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

## License

See [LICENSE](LICENSE) file.
# ComfyUI-DonutNodes

[![Support on Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Development-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/donutsdelivery)

Custom nodes for ComfyUI focused on LoRA management, model merging, and image enhancement.

## Features

- **Block-weighted LoRA stacking** with per-block strength control, CivitAI integration, and an experimental quantized-model bypass mode
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
| DonutModelSave | Save merged models |

### Experimental quantized LoRA bypass

`DonutApplyLoRAStack` has an optional `execution_mode` named
`Experimental bypass`. It computes the quantized base layer and low-rank LoRA
path separately instead of repeatedly rebuilding patched quantized weights.
This can substantially reduce warm inference time for Krea2 and similar models
under Dynamic VRAM while preserving the LoRA's model strength, block vector,
Safe Stack attenuation, and fusion-aware processing.

The mode currently supports one active diffusion-model LoRA. Unsupported direct
patches retain ComfyUI's regular patch path, and LoRAs without supported forward
adapters produce an explicit error. Existing workflows default to
`Comfy patches` and retain their previous behavior.

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

# ComfyUI-DonutNodes

[![Support on Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Development-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/donutsdelivery)

Custom nodes for ComfyUI focused on LoRA management, model merging, and image enhancement.

## Features

- **Block-weighted LoRA stacking** with per-block strength control and CivitAI integration
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
```

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

## License

See [LICENSE](LICENSE) file.

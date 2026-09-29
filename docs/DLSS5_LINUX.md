# Experimental DLSS 5 Neural Rendering on Linux

This branch adds two DonutNodes:

- **Donut DLSS 5 Linux Runtime Status (Experimental)** checks the local runtime layout, launcher, NVIDIA visibility, and Vulkan visibility.
- **Donut DLSS 5 Neural Upscale — Linux (Experimental)** streams ComfyUI images to a user-supplied DLSS 5 feature-18 worker through Proton, Wine, or a future native ELF worker.

The node pack does not include NVIDIA DLLs, ReShade, RenoDX, Proton, Wine, or the native worker. Those files remain under their own licenses and must be obtained legally.

## What was ported

The ComfyUI side is Linux-native Python. The renderer is not recompiled: the currently available feature-18 worker is a Windows PE/D3D12 program, so Linux runs it through Proton or Wine with vkd3d-proton and NVIDIA Vulkan/NVAPI support. A true native port still requires a Linux Vulkan/NGX feature-18 worker and a Linux DLSSNR runtime that is not present in the referenced node repositories.

The bridge speaks the version-4 worker protocol used by current Merserk DLSS 5 Visual Enhancer builds. Every batch item is treated as an independent still image and resets temporal history.

## Required local runtime layout

Point `runtime_dir` at either the portable application root or directly at `bin/runtime`:

```text
bin/runtime/
├── host/
│   ├── nvngx.dll                 # D3D12 worker executable, despite the extension
│   ├── dxgi.dll                  # ReShade full add-on build
│   ├── renodx-dlss5.addon64
│   └── nvngx_dlssnr.dll
└── dlss/
    └── nvngx_dlss.dll
```

The tested layout is the one documented by `Merserk/dlss5-visual-enhancer` v5.0. Do not commit this directory to DonutNodes.

## Linux prerequisites

1. An NVIDIA RTX GPU using the proprietary NVIDIA Linux driver.
2. `nvidia-smi` must list the GPU.
3. Vulkan must list that same NVIDIA GPU. Install your distribution's `vulkan-tools` package to make the status node report it.
4. Proton Experimental, a recent GE-Proton build, or Wine with vkd3d-proton and dxvk-nvapi installed in its prefix.
5. A legally obtained, internally compatible DLSS 5 runtime bundle.

Proton is the default recommendation because it already packages a matching Wine/vkd3d environment. Plain Wine only works when its prefix has the required D3D12 and NVAPI translation components.

## Node setup

For Proton:

- `backend`: `Proton`
- `launcher`: full path to Proton's `proton` script, for example a file below `~/.local/share/Steam/compatibilitytools.d/`
- `prefix`: a writable compat-data directory, such as `~/.local/share/donutnodes/dlss5-proton`
- `runtime_dir`: the extracted portable root or its `bin/runtime` directory

For Wine:

- `backend`: `Wine`
- `launcher`: optional full path to `wine64`
- `prefix`: a Wine prefix containing vkd3d-proton and dxvk-nvapi

`Auto` prefers an ELF worker, then Proton, then Wine.

Equivalent environment variables are:

```bash
export DONUT_DLSS5_RUNTIME=/path/to/DLSS.5.Visual.Enhancer.v5.0
export DONUT_DLSS5_PROTON=/path/to/GE-Proton/proton
export DONUT_DLSS5_PREFIX="$HOME/.local/share/donutnodes/dlss5-proton"
```

Optional timeout variables:

```bash
export DONUT_DLSS5_SETUP_TIMEOUT=90
export DONUT_DLSS5_FRAME_TIMEOUT=600
export DONUT_DLSS5_CLOSE_TIMEOUT=60
```

## Verification policy

A returned image is not enough to prove DLSS ran. The upscale node fails unless the current worker/ReShade output contains all three classes of evidence:

1. The signed DLSSNR D3D12 runtime initialized.
2. NGX feature 18 was created.
3. Feature 18 evaluation succeeded.

This prevents a conventional resize or runtime fallback from being reported as DLSS 5. Successful output includes the matching evidence lines in `runtime_report`.

## First hardware test

1. Run the Status node and confirm the runtime files, NVIDIA GPU, and Vulkan GPU are all present.
2. Start with one 512×512 or 768×768 image.
3. Use `2.0x (Performance)`, default NR preset/style, and model preset `M`.
4. Queue once and inspect both the output image and `runtime_report`.
5. If it fails, inspect `bin/runtime/host/ReShade.log`, the ComfyUI console, and any Proton log generated for the prefix.

## Known constraints

- This is an experimental compatibility route, not a claim that NVIDIA officially supports the supplied feature-18 DLL on Linux.
- Success depends on the exact NVIDIA driver, GPU generation, Proton/Wine build, vkd3d-proton, dxvk-nvapi, RenoDX add-on, worker, and DLSS runtimes.
- RTX 30-series behavior remains especially experimental.
- No depth or motion guide is supplied for independent still images; the worker receives zero motion and resets history for each item.
- The node supports output through the worker's fixed quality modes, not arbitrary scale factors.
- Static CI validates the Python protocol and failure policy. Only a real RTX Linux run can validate the proprietary runtime.

## Sources and licensing

- `vizart-vj/ComfyUI-AetherScale` and `HECer/ComfyUI-DLSS5` establish the current Windows-only ComfyUI approaches.
- `Merserk/dlss5-visual-enhancer` documents the portable runtime layout and version-4 worker protocol under the MIT License.
- `NVIDIA/DLSS`, `NVIDIA-RTX/Streamline`, ReShade, RenoDX, Wine, Proton, vkd3d-proton, and dxvk-nvapi remain separate upstream projects under their own terms.

No upstream runtime binary is redistributed by this branch.

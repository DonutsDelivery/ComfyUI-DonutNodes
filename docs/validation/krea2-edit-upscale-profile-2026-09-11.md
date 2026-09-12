# Krea2 edit/upscale timing and memory evidence

This report records the exact graph runs and the Krea2 NAG operator capture used to investigate the slow edit/upscale path. It is evidence for a review, not a claim that the pipeline has been optimized. The parent commit `e36355e` contains the earlier global LoRA execution-mode work; this branch adds only the report and capture artifacts.

Private prompts, source/reference images, and generated images are omitted. The workflow fingerprints below are SHA-256 hashes of the complete local prompt payload, so a reviewer can distinguish the runs without publishing those inputs.

## What was measured

The two completed edit-enabled runs took 207.592 s and 198.499 s end to end. The comparable historical run with Edit Studio disabled took 83.555 s. These are separate executions with different seeds, so they show the observed cost of the enabled graph, not a controlled speed A/B test.

The instrumented graph produced one OOM and one success. The OOM occurred in `DonutTiledUpscale` after 155.648 s at `krea2_nag.py:_nag_edit_block` line 173, in `block.postnorm(positive)`. The succeeding run took 129.800 s, but ComfyUI marked upstream nodes as `execution_cached`; it must not be treated as a clean cold-run baseline.

| Run | Prompt ID | Start (UTC) | End (UTC) | Elapsed | Result |
| --- | --- | --- | --- | ---: | --- |
| Edit Studio off | `5ac2a4c5-955b-43bf-aea6-00f968ed1977` | 22:18:16.772 | 22:19:40.327 | 83.555 s | success |
| Edit Studio on | `f5f7114d-cc07-4532-a0b7-b2fc4a733b4c` | 22:22:07.134 | 22:25:34.726 | 207.592 s | success |
| Edit Studio on | `e22c77c6-96f5-47df-b4ff-c2fecf0f9b6c` | 22:28:11.600 | 22:31:30.099 | 198.499 s | success |
| Instrumented exact graph | `44963eab-4df9-4216-925d-a14c776f903c` | 23:05:07.343 | 23:07:42.991 | 155.648 s | OOM in `DonutTiledUpscale` |
| Instrumented exact graph | `60d9980a-8dcb-4e58-8d23-18bdb33f97a9` | 23:08:56.672 | 23:11:06.472 | 129.800 s | success; upstream cache reported |

The machine-local profiler capture was written at `2026-09-12T01:37:14+02:00`. It profiled a full-frame Krea2 NAG forward with input shape `(1, 16, 1, 216, 168)`:

- allocated before: 2,820,154,502 bytes
- allocated after: 2,821,315,718 bytes
- peak allocated: 6,713,547,142 bytes
- reserved before: 2,952,790,016 bytes
- peak/after reserved: 7,952,400,384 bytes
- self CUDA time: 32.931 s
- self CPU time: 25.818 s

The top reported CUDA totals were `aten::linear` 18.412 s, Flash attention 9.238 s, pinned host-to-device copies 7.522 s, `aten::matmul` 7.928 s, `aten::mm` 7.672 s, and `aten::copy_` 10.091 s. These profiler totals are inclusive and can overlap; they are not an additive elapsed-time breakdown. The trace already shows Flash attention active. The linear total does not identify how much belongs to any proposed redundant projection until those projections are isolated and an A/B run completes.

## Trace attribution and measured candidate

The compressed Chrome trace was parsed by correlating each host-to-device copy with the next dequantization and linear operation on the same execution lane. Of the 2,034 transfers (7.522320 s, 178.6368 GB), 1,680 are 100,664,320-byte copies consuming 7.122655 s (94.7% of transfer time). Those copies feed `comfy_kitchen::dequantize_fp8` for `(6144, 16384)` weights and then `aten::linear` with `(1, 1024, 16384)` inputs (1,512 copies), `(1, 613, 16384)` inputs (84), and `(1, 43, 16384)` inputs (84). This is repeated FP8 feed-forward weight staging under dynamic offload, rather than image-input movement. The 37,749,760-byte and 9,438,208-byte groups similarly map to `(6144, 6144)` attention projections. The trace establishes the caller and transfer cost; it does not establish that eliminating the copies is numerically safe.

The outermost `aten::linear` hierarchy contains 10,676 calls and 12.969 s inclusive CPU time. The largest attributed group is 1,008 `(1, 1024, 16384) × (6144, 16384)` calls: 6.503 s inclusive CPU, 2.915 s descendant kernel time, and 5.930 s in dequantization. Descendant times overlap across the hierarchy. The trace also reports 24.305 s of `Command Buffer Full` self CPU time across 16,645 events (94.14% of self CPU time), which means the host was blocked submitting more CUDA work; it is not 24.305 s of Python computation or an automatically recoverable amount.

The measured candidate was ComfyUI's `--disable-async-offload`, chosen because the attributed cost is dynamic weight staging and the option changes its scheduling directly. The exact saved edit workflow and seed `681039430899612` were used. The candidate completed in 258.06 s, with log boundaries of 88.32 s for the base eight-step sampler, 109.77 s for the three-step 1.5× upscale sampler, and 33.72 s for the three-step face-detailer sampler. A restored baseline launch (`--vram-headroom 2`, async offload enabled) was run with the same payload twice during this check and both failed at the full-frame NAG RMSNorm allocation after 128.17 s and 130.71 s. The candidate therefore demonstrates a completion/reliability difference under the current GPU state, not a speedup; its end-to-end time is slower than the earlier historical successes. Peak allocator values were not emitted by these unprofiled successful runs, and the candidate output was not retained after the server restart, so no quality or pixel comparison is claimed.

| A/B check | Launch change | Prompt/seed | Result | Elapsed |
| --- | --- | --- | --- | ---: |
| Restored baseline | async offload enabled | exact edit payload / 681039430899612 | OOM in full-frame NAG RMSNorm | 128.17 s |
| Restored baseline | async offload enabled | exact edit payload / 681039430899612 | OOM in full-frame NAG RMSNorm | 130.71 s |
| Candidate | `--disable-async-offload` | exact edit payload / 681039430899612 | success; output not retained for comparison | 258.06 s |

Because the restored baseline did not complete, the candidate is not promoted as a performance change and no source patch is kept for it. The existing historical completed runs (207.592 s and 198.499 s) remain the only completed edit timings with async offload, but they used different seeds and were not a matched A/B pair.

## Failed experiments

Two experiments did not produce a demonstrated optimization:

- Limiting NAG normalization to 256-token chunks still failed at the same RMSNorm allocation site after 140.07 s and provided no reported memory relief.
- Running ComfyUI with `--vram-headroom 0.5` still failed at the same full-frame NAG RMSNorm site after 100.50 s.

These results are recorded so neither change is repeated as if it were a fix. A workload that completes after a memory change would be a reliability result; a speed improvement requires completed, matched before/after timings.

## Capture configuration and dependencies

The profiler trace contains CPU operation, CUDA runtime, kernel, memcpy, and memset events. Its recorded configuration fields are `record_shapes=1` and `profile_memory=1`; the trace metadata reports CUDA runtime `12080`, driver API `13030`, and CUPTI `26`. The raw text table is in [`artifacts/krea2-nag-profile.txt`](artifacts/krea2-nag-profile.txt), and the Chrome trace is compressed in [`artifacts/krea2-nag-profile.trace.json.gz`](artifacts/krea2-nag-profile.trace.json.gz).

The capture host was an NVIDIA GeForce RTX 4070 (12,282 MiB reported by `nvidia-smi`) with driver `610.57.04`, Python `3.12.7`, PyTorch `2.9.1+cu128`, `comfy-kitchen 0.2.31`, `sageattention 2.2.0`, `spandrel 0.4.1`, NumPy `2.5.2`, `safetensors 0.7.0`, and Transformers `4.57.3`.

| Component | Revision/version |
| --- | --- |
| ComfyUI | 0.34.0, git `3216c62e9962c3babd28a4dfea6e5aef50b8fe16` |
| DonutNodes | git `e36355ed863564cb54a50a2d487b0ea8f4247173` |
| Krea2 NAG | `1.0.2` |
| comfyui-krea2edit | `1.2.5`, git `86f886dac23013d88996e3a2e99093ba44d322fb` |
| ComfyUI-bleh | git `5af35d2366bb3e4ed413daa52867000f3d1ac7ad` |
| krea-seed-variance-enhancer | `1.2.0` |
| comfyui-impact-pack | `8.28.3` |
| comfyui-impact-subpack | `1.3.5` |
| ComfyUI frontend | `1.51.9` |

ComfyUI was launched with:

```text
/home/user/Programs/ComfyUI-new/ComfyUI/main.py --port 8188 --listen --vram-headroom 2
```

The ComfyUI checkout contained local model-directory setup changes and untracked setup files at capture time. The commit above pins the source revision, but a clean checkout is required for a fully controlled rerun.

## Reproduction settings

Use the same source/reference image and model files, then queue the graph with the following settings. The source image and prompt are intentionally not included in this public branch.

- `DonutEditStudio`: enabled; 1.0 MP; 1152×896; multiple 64; grounding 1088; identity-edit LoRA enabled at strength 1.0.
- Base `DonutSampler`: 8 steps; denoise 1.0; Turbo on; NAG on; CFG 1.0; `bleh_preset_0` with `beta` scheduler.
- First `DonutTiledUpscale`: enabled; 8 steps; denoise 0.33; rescale factor 1.5; bilinear resampling; tiled diffusion off; tiled VAE off; Turbo on; NAG on.
- `DonutFaceDetailer`: 8 steps; denoise 0.36; Turbo on; NAG on; maximum five faces.
- Second upscale: disabled.
- Pixel upscaler: `4x_NickelbackFS_72000_G.pth`.
- Merge/LoRA execution mode: `Experimental bypass`; grouped merge with fusion ratio 0 and body ratio 1.

For a controlled performance comparison, run at least three completed cold runs for each variant with the same seed, source image, prompt, model files, resolution, step count, guidance, and finishing stages. Record end-to-end time, sampler time, peak memory, and completion status. Restore the baseline and rerun it after each candidate change. Operator traces should explain a measured result, not substitute for it.

The machine-readable timestamps, workflow fingerprints, memory values, trace ID, and artifact hash are in [`artifacts/krea2-edit-upscale-measurements.json`](artifacts/krea2-edit-upscale-measurements.json).

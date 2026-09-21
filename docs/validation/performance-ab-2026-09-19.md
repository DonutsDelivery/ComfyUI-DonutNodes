# DonutNodes 3.0.15 versus 3.0.24 performance investigation

User reports V4 formerly taking 50–100 seconds versus approximately 153 seconds now, with all three new chunking options disabled. Supplied workflow: `/home/user/krea2TurboWorkflowUpscale_v40Revamped.json`.

## Executed comparison

Fresh isolated ComfyUI processes, same current core, Python environment, companion nodes and existing model files. No models downloaded. DonutNodes extracted from commits `a02b10b` (3.0.15) and `aaf3c99` (3.0.24).

Both processes used `--listen 127.0.0.1 --port 8197 --disable-auto-launch --preview-method none`. The user's server on 8188 was left running; its queue was checked empty and its cached models unloaded before testing.

The test used the actual 152.92-second run's API inputs, fixed seeds, full BF16 Krea2, Experimental bypass, NAG, 1152×896 generation, 1.5× full-frame upscale, and FaceDetailer. Removed inactive V5 composite/SeedVR2 passthrough stages and unsupported inputs to make an identical compute graph executable on 3.0.15. A standard SaveImage saved the FaceDetailer result. Current version additionally received the three required chunking booleans, all false. This is a version-isolation test, not an execution of the entire supplied V4 JSON.

| Version | Result | Total time |
| --- | --- | --- |
| 3.0.15 | Successful full generation, upscale, FaceDetailer and save | 157.56 s |
| 3.0.24 | Successful full generation, upscale, FaceDetailer and save | 157.08 s |

Both logs confirmed 224 active Experimental bypass hooks. Each version was tested once, cold. The 0.48-second difference is not evidence of a meaningful performance change.

## Interpretation and limits

The current slow job is reproducible with 3.0.15 code under today's runtime. This comparison does not reproduce the historical 50–100-second result and does not identify its cause. It does not rule out changes elsewhere in ComfyUI, companion nodes, model routing, or historical settings. No speculative performance patch was made.

The supplied V4 file's promoted sampler controls match the slow job for steps, sampler, scheduler, upscale denoise/factor and FaceDetailer denoise. Do not compare its embedded node widget defaults without applying promoted input overrides. Its model definitions include BF16 Krea2 plus a quantized secondary model with two-model merge routing; the 152.92-second captured API graph used single-model BF16 routing. The user also ran a two-model job at 139.01 seconds, so that distinction alone has not explained the full reported slowdown.

Artifacts retained under `/tmp/donut-perf-ab/`: logs, submitted API prompts, histories, and generated output images. Temporary server code and model symlinks are removed after shutdown. No changes to the user's installation or release were made.

## Follow-up: actual V4 execution and matched V5 graph

The user's later V4 run `8643ece6-778e-4dfb-8997-598b5472235a` finished in **102.45 seconds**. One face was detected and one 832×1216 detail crop processed. An earlier V4 run at 170.30 seconds detected ten faces and processed five crops; that run cannot be compared to the single-face timings.

Compared the captured V4 API inputs against the original slow V5 execution. The V4 run uses quantized finepornV4 plus BF16 Krea2 with two-model merge routing, Fusion Off, and the prompt `single woman`. The original 152.92-second run instead uses single-model BF16 routing, Rebalance, and a different prompt/seed. Thus those two timings are not a controlled workflow comparison.

Replayed the original V5 API graph on the user's normal 8188 server with actual V4 compute inputs copied across, preserving V5 preview stages and its disabled SeedVR2 final stage. Redirected saves to `DonutPerformanceAB/`. The initial submission reused cached images (0.41 s), which is excluded from performance results. For the real repeat, provided SeedPlan's identical numeric text seed directly rather than through SeedNode, changing the cache dependency signature without changing resolved seed values.

Matched V5 job `a662b3b0-a950-4287-862d-977051a16d78`: **92.96 seconds**, success. Execution history confirms DonutSampler, enabled DonutTiledUpscale and DonutFaceDetailer were all uncached. One face, same 804×1141 source crop resized to 832×1216, full-frame 1728×1344 upscale (one tile). All added preview stages retained. Models remained warm in the existing process; this is not a cold-start comparison and a single pair is not evidence that V5 is intrinsically faster.

Result: V5 can run within the reported earlier timing range with the V4 compute settings. This does not isolate the individual contributions of model routing, Fusion choice, cold/warm state or the memory-retention warnings in previous runs. No performance code change was needed for this result.

Evidence: `/tmp/donut-v5-matched-prompt-uncached.json`, `/tmp/donut-v5-matched-history.json`, `/tmp/donut-perf-latest-history.json`. Generated full output: `/home/user/Programs/ComfyUI-new/ComfyUI/output/DonutPerformanceAB/v5_matched_final2.webp`.

## Reverse comparison requested by user — 2026-09-20

Executed the actual captured V4 API graph with the slow V5 run's compute inputs, then the actual captured V5 graph with those same inputs. This reverses the preceding comparison instead of giving V5 the faster V4 configuration. Both used the existing current backend on port 8188, not a separate 3.0.15 backend.

Matched inputs: BF16 Krea2, Single model, Rebalance/Fusion only, identical prompt and seed 908969286088091, NAG, 1152×896 base, 1.5× full-frame hires, FaceDetailer enabled, all chunking false. Retained each workflow's stage/preview wiring and original workflow metadata; redirected save names to `DonutPerformanceAB/`. Both captured graphs contain five PreviewImage nodes (the user's V4 graph had already acquired intermediate previews). The V5 graph also retains its disabled SeedVR2 stage. Common computation-node inputs were confirmed identical.

Before each submission, checked the user's queue empty and called `/free` with `unload_models=true, free_memory=true`. Both histories report an empty cached-node list. All stages actually executed. OS file caches and GPU thermal state were not reset.

| Workflow with slow V5 settings | Job | Result | Time |
| --- | --- | --- | --- |
| V4 | e57297b2-84dd-42e2-8dbe-1fbaacb7dabd | Success | 140.89 s |
| V5 | bbdbcfa8-37ce-4cad-9e92-af87c9f19230 | Success | 137.69 s |

Both logs confirm bypass active, one 1728×1344 full-frame upscale tile, one detected face, and the same 652×939 face crop resized to 832×1216. No extra FaceDetailer passes. V4 was 3.20 seconds slower in this single pair, not meaningfully faster. The result does not establish an intrinsic V5 speed advantage; it shows no V4 speed improvement with these matched settings.

Submitted prompts, histories and summary retained in `/tmp/donut-reverse-ab/`. Full outputs: `output/DonutPerformanceAB/reverse_v4_final1.webp` and `reverse_v5_final1.webp` in the user's ComfyUI installation. No new server, model download, package change, performance code patch, or publication was needed.

## One-factor investigation — 2026-09-20

Starting from the slow V5 API graph and fixed original seed, changed model routing, then Fusion, individually. Used the same 8188 process; execution/model caches cleared before each cold run. Collected actual WebSocket execution timings and full histories. All completed tests below processed exactly one detected face and one face crop. All generation stages executed uncached.

| Change relative to preceding cold case | Total | Base sampler | First upscale | FaceDetailer |
| --- | ---: | ---: | ---: | ---: |
| Only use the V4 quantized-body/BF16-fusion model routing; Rebalance retained | 117.36 s | 43.79 s | 47.66 s | 20.28 s |
| Additionally switch Fusion to Off | 117.29 s | 44.57 s | 48.35 s | 18.78 s |
| Additionally shorten prompt (completed before user's request to skip prompt/seed testing) | 118.01 s | 43.80 s | 47.93 s | 20.58 s |
| Repeat first model-routing-only case, fresh load | 118.36 s | 44.55 s | 47.50 s | 20.66 s |
| Identical preceding job with models retained | 114.17 s | 43.16 s | 47.91 s | 20.25 s |

The user asked to skip further prompt/seed testing. Interrupted the outstanding seed test `2485a04b-f4e3-44c6-bb30-e16b0819c740`; it is excluded from results. No further prompt/seed tests were started.

The warm comparison used the same resolved seed, providing SeedPlan's text seed directly rather than through a link to invalidate generation cache signatures. History confirms only static/model-loader nodes were cached, not DonutSampler, enabled upscale, or FaceDetailer.

Measured model-routing effect versus the recent matched 137.69-second BF16 baseline: approximately 20 seconds. Fusion effect was below run variation. Retaining models saved 4.19 seconds in the measured pair; do not attribute another 20–30 seconds to warm caches without evidence. Historical 152.92/102.45 timings still include an unexplained residual and cannot be perfectly decomposed from these tests alone.

After benchmarks completed, verified ComfyUI's queue empty and sampled per-process GPU usage with `nvidia-smi pmon -c 5 -s um`. Firefox showed 65–98% SM activity and roughly 1.1–1.4 GB framebuffer use in the first five samples, while ComfyUI's process showed no active SM utilization. Xwayland and DonutStudio also had intermittent activity. This is evidence of competing GPU work, not yet a quantified cause of the earlier timing gap. GPU was around 65°C at 2805 MHz during the final warm run, with no demonstrated thermal throttling. Asked the user to pause/hide the active Firefox content for a matched repeat; no browser/application was terminated or paused by the agent.

Evidence retained in `/tmp/donut-perf-factors/`: per-case prompts, histories, event timelines, node timings, logs and summaries, plus `idle-gpu-processes.txt`. Initial 117.36-second model case archived under `model_initial`; fresh-load repeat under `model`.

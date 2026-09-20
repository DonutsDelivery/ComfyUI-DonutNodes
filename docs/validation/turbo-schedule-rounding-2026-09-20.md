# Turbo schedule reconstruction and V5 first-pass audit

Base: `9fc507bae34d613551623dab1171fc80e05d761d` (main).

## Scope: the reported image problem is not reproduced

The report concerns intermittent noisy images immediately after the first
DonutSampler/VAE decode in V5, with NAG alpha 0.26, UncensorFix enabled and
Rebalance selected. No failing image, executed prompt metadata, installed
ComfyUI/companion versions or target GPU session was available for this audit.

This patch fixes a separate, reproducible Turbo schedule-length rounding bug.
It does **not** change the normal eight-step/full-denoise first pass and must
not be described as a verified fix for that image-quality report.

## Confirmed defect

`resolve_turbo_sampling` selects an effective step count and passes
`effective_steps / supported_steps` as ComfyUI's execution denoise. In
[KSampler.set_steps](https://github.com/Comfy-Org/ComfyUI/blob/0f74f7fb9f83a78bf46188fd4fd53e6bc44c1ae8/comfy/samplers.py),
partial denoise reconstructs the complete schedule using
`new_steps = int(steps / denoise)` before selecting its final `steps + 1` sigmas.

Floating-point round trips can fall just below the required integer:

```python
int(7 / (7 / 50)) == 49   # should reconstruct 50 supported steps
int(9 / (9 / 14)) == 13   # should reconstruct 14 supported steps
```

The number of executed intervals can still be correct, but their sigma values
come from the wrong full schedule. This is not a missing terminal denoise step.
It depends on the effective/supported step counts, not on the image seed, NAG
strength or Rebalance settings.

The fix moves only the execution denoise to the next representable float toward
zero when the original ratio would reconstruct too few steps. Selected effective
steps, the user-facing matched denoise, nearest-point/tie behavior, and correctly
round-tripping ratios are unchanged. No fixed epsilon or global sampler patch is
used. Full denoise remains exactly 1.0.

## First-pass trace: source inspection, not GPU validation

The checked-in V5 wiring and [SDA documentation](../krea2-sda.md) identify sampler
993 -> VAE Decode 991. The effective selection is outer node 1014's
`bleh_preset_0` / `beta`, not the nested sampler's stored
`er_sde` / `bong_tangent`. The preset is ER-SDE in ODE mode, stage 3, without a
sigma override; its ODE builder resolves stochastic noise strength to zero.
The base sampler's stored mode is simple, eight steps, denoise 1.0.

The shared Donut sampler passes the resulting sigma schedule to one guider run.
The inspected upstream
[ER-SDE implementation](https://github.com/Comfy-Org/ComfyUI/blob/0f74f7fb9f83a78bf46188fd4fd53e6bc44c1ae8/comfy/k_diffusion/sampling.py)
uses `x = denoised` when the next sigma is zero and does not add stochastic noise
in that terminal branch. No skipped terminal step was identified in this path.
A terminal zero sigma does not guarantee that the model's predicted image is
artifact-free.

V5 also stores Seed Variance as enabled, with beginning-step noise and a 25%
switchover. The inspected
[companion implementation](https://github.com/harukimix/KreaSeedVarianceEnhancer/blob/1515d23a5b399a44ccd97482e1105a43937268ae/krea_seed_variance_enhancer.py)
implements this using conditioning `start_percent` / `end_percent`, not a count
of the beta schedule's actual intervals. This is a possible A/B-test variable,
not evidence that it caused the reported noise. Its behavior is not changed.
NAG, UncensorFix, Rebalance, solver options and finishing stages are not retuned.

## Validation performed

Only the standalone Turbo module and its tests were executed locally. The
fetched baseline files were reconstructed locally and their Git blob hashes
verified before editing:

- `turbo_sampling.py`: `4dda46b0a4f542b04fa37f26a51663d81c5a1260`
- `test_turbo_sampling.py`: `f01799baf3baec8bd1c572b418ca4e925504e230`

Python 3.13.5:

```text
Original six tests: PASS
New 14/50-step regressions against original code: FAIL (13 != 14; 49 != 50)
python -m unittest -v test_turbo_sampling: 13 tests PASS after fix
python -m py_compile turbo_sampling.py test_turbo_sampling.py: PASS
```

Tests cover every selected row for supported counts 1-64 across simple, beta
and bong_tangent (6,240 round trips), selected large counts through the node's
10,000-step limit, the near-full-denoise branch boundary, exact eight-step
argument preservation, full-denoise identity and the one-float correction.
The reconstruction test uses ComfyUI's documented-in-source arithmetic contract;
it is explicitly **not** a full installed-ComfyUI integration test or pixel
parity measurement.

Not run: the repository-wide suite, Krea2 inference, VAE decode, quantized GPU
execution, or frontend panel -> Run -> PNG metadata -> workflow reload checks.
No image-quality improvement is claimed. No workflow JSON, UI, dependency,
package version or release artifact is changed; no Civitai workflow re-upload
is required by this patch.

## Remaining image investigation

Use a failing seed with the original model/LoRAs, prompt and settings, and inspect
the Base generation preview before upscaling. First establish whether that exact
run repeats the failure. Then compare one change at a time: Seed Variance off,
NAG off, and UncensorFix off, restoring the original settings between comparisons.
Keep Rebalance and the sampler/scheduler fixed while isolating each toggle.
A failing PNG with executed prompt/workflow metadata is needed to trace the
actual resolved inputs rather than assuming they match the distributed defaults.

# Hires partial denoise with NAG — 2026-09-28

## Code trace

- Ordinary `DonutTiledUpscale` with Turbo, eight supported steps and `beta`
  denoise `0.5` resolves to four effective steps and ComfyUI denoise `0.5`.
  The tile sampler passes no additional start/end slice. `bong_tangent`
  denoise `0.4` resolves to the same execution arguments.
- ComfyUI reconstructs the eight-step schedule and retains its four-interval
  tail. Its sampler calls the model's `noise_scaling` with the first retained
  sigma. Native Krea2's default `CONST` scaling mixes noise and encoded hires
  latent using that sigma. NAG changes the later model predictions, not this
  initial noise mixture.
- The auxiliary NAG alpha curve spans the executed hires steps. A linear
  `0.45 -> 0` curve on a four-step hires pass is `0.45, 0.30, 0.15, 0`.
  It does not continue the last four values of the base eight-step curve.
  This is the documented stage-local policy; with auto phi enabled, positive
  alpha values also change phi to preserve their unclipped alpha/phi product.
- Advanced mode with Turbo `steps=8`, `denoise=0.5`, `start_at_step=4` and
  `end_at_step=8` previously returned the input latent without sampling: Turbo
  had already selected four intervals and the second slice removed all of
  them. It now raises a `ValueError` describing the effective range. An
  explicit full-schedule range uses `steps=8`, `denoise=1`, start `4`, end `8`.

## Verification

- A focused CPU harness extracted the production sampler function and checked
  three cases: the double-sliced range raises; the Turbo tail executes four
  intervals; the explicit full-schedule range executes the same four sigmas.
- The Turbo resolver returned `(4, 0.5, 0.5)` for `beta` denoise `0.5` and
  `(4, 0.5, 0.4)` for `bong_tangent` denoise `0.4`.
- A regression test was added to `test_donut_sampler_dynamic_cfg.py` for the
  empty advanced range. That module cannot run in this checkout because
  ComfyUI's Python core is absent. The direct CPU harness exercised the changed
  production function. Python compilation and `git diff --check` passed.

## Runtime limit

No ComfyUI Run-button generation was available here. No output PNG was saved,
and no NAG image-quality comparison was made. This change touches the backend
advanced sampler only; no distributed workflow JSON changed.

ComfyUI source used for the noise-scaling trace:
[sampler](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/samplers.py),
[model sampling](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_sampling.py).

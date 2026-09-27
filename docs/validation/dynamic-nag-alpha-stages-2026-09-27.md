# Shared dynamic NAG alpha across sampling stages — 2026-09-27

## Change

The Settings / Configuration NAG curve now reaches the base sampler, both
Donut tiled-upscale stages, and Face Detailer. Each auxiliary stage schedules
the same curve over its own actual sigma list, so partial denoise and Turbo
step adjustments affect the curve length correctly. The per-stage NAG enable
switch remains independent. The SeedVR2 replacement engine does not run Krea2
NAG and does not use this schedule.

The saved base sampler value is the source for global NAG widgets. On panel
load, stale serialized stage copies of shared settings are synchronized from
that saved value; editing the global curve or shared phi options mirrors them
to the stage nodes. Schedule widgets were appended to auxiliary node schemas,
after existing stage and VAE-correction widgets.

## Automated verification

- `python -m unittest test_donut_grounding_schedule test_donut_grounding_nag` —
  59 passed.
- `python test_donut_tiled_upscale_lifecycle.py` — 7 passed, including forwarding
  the shared schedule triple through the edit-upscale NAG bridge.
- `python tests/test_seedvr2_stage.py` — 17 passed, including verification that
  the schedule widgets are appended after VAE correction.
- `node --test tests/app_controls_preset_refresh.test.mjs tests/panel_categories.test.cjs`
  — 43 passed.
- Dynamic-stage regression: a three-step sigma schedule produced alpha values
  `0.1, 0.3, 0.5`; auto-phi produced `15, 5, 3`. A second tile reused the
  prepared wrappers. Global-panel migration copied the saved `ease_out`,
  `0.12`, `0.72`, and auto-phi-on values to hires and face nodes; editing the
  curve to `linear` and auto phi to off updated both stage widgets.
- `python -m py_compile` passed for the changed Python modules; `git diff --check`
  passed.

## UI generation verification

Not completed. The CUA environment reported no available apps or browsers, so I
could not load the changed frontend in ComfyUI, queue it with Run, save a PNG,
compare PNG workflow metadata with the execution prompt, or reload that PNG.
There is no output PNG path. These tests verify the settings bindings and
backend selection logic, not a live Krea2 image generation.

## Limits

Auxiliary dynamic scheduling retains the verified sampler support used by the
base dynamic schedule: Euler, ER-SDE, DPM++ 2M, and inspected compatible Bleh
presets. Other samplers fail explicitly rather than running with static alpha.

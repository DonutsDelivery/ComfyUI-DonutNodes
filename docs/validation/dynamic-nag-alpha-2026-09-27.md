# Dynamic NAG alpha validation — 2026-09-27

## Implementation exercised

Dynamic NAG alpha is appended to the final DonutSampler schema after all
existing widgets, including the sampler-local TextFusion compatibility fields.
`constant` retains the old `nag_alpha` path. Dynamic curves select a prepared
NAG wrapper at each executed step; auto phi is recomputed as
`nag_phi_scale / scheduled_alpha`. The existing grounding selector can choose
the matching conditioning and NAG wrapper in the same prediction. The panel
migration binds the three controls to the nested base sampler and does not copy
them to later upscale/detailer stages.

## Positional workflow regression follow-up

The supplied ComfyUI error screenshot showed `nag_alpha_schedule` receiving
`false` and `nag_alpha_start` receiving a checkpoint filename. The final node
schema is composed by wrappers: the TextFusion wrapper adds two legacy widgets
after the grounding wrapper. Inserting the new schedule controls in the
grounding wrapper had shifted those saved values. The controls now append after
the TextFusion fields, and a regression test verifies that an old positional
widget list still maps `false` and the filename to their TextFusion controls.
No distributed workflow JSON was changed.

An additional load-time repair handles workflows saved during the earlier
misordered schema: an invalid schedule value resets the stale schedule triple
to `constant` and the existing static `nag_alpha`; with a valid curve, only
invalid endpoints are repaired. Valid dynamic choices and endpoint values
remain unchanged. The repair traverses nested sampler nodes and is idempotent.

The regression test and focused Python/frontend checks pass. A live ComfyUI
session is unavailable, so queue-time panel values and PNG metadata remain
unverified. Refresh the browser and reopen the workflow; the one-time repair
will normalize stale invalid NAG values on load.

## Focused checks

CPU and Comfy/NAG double tests exercised:

| Transition | Expected | Observed |
| --- | --- | --- |
| Three-step `linear`, alpha 0.1 → 0.5, auto phi scale 1.5 | Alpha 0.1/0.3/0.5; phi 15/5/3 | Same values reached the selected per-step wrappers |
| Grounding 512 → 1088 with that alpha schedule | 512/832/1088 conditioning and matching NAG negatives | Aligned pairs selected at all three steps |
| Advanced steps 5–8 of a 20-step run | Three scheduled values across the executed range | Three alpha values prepared |
| NAG disabled | Stored alpha curve has no effect | Parent sampler path received no active NAG schedule |
| Corrupted saved NAG values in a nested sampler | Invalid curve resets to Constant and endpoints use static alpha; valid manual curve is preserved | Frontend graph test passed; live ComfyUI reload unavailable |

Commands and results:

- `python -m unittest test_donut_grounding_schedule.py test_donut_grounding_nag.py -v` — 58 passed.
- `python -m unittest discover -s tests -p 'test_txtfusion_guard_sampler.py' -v` — 17 passed.
- `node test_donut_grounding_controls.mjs` — passed, including stale-value repair and valid-value preservation.
- `node --test tests/panel_categories.test.cjs` — 33 passed.
- `python -m py_compile donut_grounding_schedule.py donut_grounding_nag.py donut_krea2_sda.py` — passed.
- `test_krea2_nag_integration.py` could not import because the environment has
  no ComfyUI `comfy` package on its Python path.

## UI generation and metadata

The CUA inventory reported no desktop apps or browsers; opening its in-app
browser at the local ComfyUI address was unavailable. No audit workflow was
queued, no PNG was produced, and no execution prompt or embedded metadata was
available to compare. Actual panel interaction, Run-button generation, backend
restart, saved-workflow reload, and image-quality behavior therefore remain
unverified. The tests above verify graph bindings and backend wrapper selection
with doubles; they do not establish live ComfyUI or GPU behavior.

Output PNG paths: none.

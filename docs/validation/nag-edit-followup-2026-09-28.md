# NAG edit follow-up — 2026-09-28

Baseline: local main `9b9503f` (3.0.43). The follow-up review correctly identified three incomplete fixes. This change repairs those paths locally; it is not a Registry publication.

## Changes

- Hires edit and Face Detailer edit restore positive Fusion taps after fresh encoding and seed variance, matching the main sampler's ordering. Positive processing remains independent of `nag_match_taps`, which controls only negative processing.
- The composable adapter runs the original upstream wrapper at the terminal of the remaining wrapper chain. Inactive edit retains its reference-aware forward; inactive T2I delegates to the native forward once. Active attention-hook conflicts still raise an error. Temporal flattening also covers inactive edit and keyword `x` calls.
- Standalone T2I/edit callbacks bind their own node ID when installed, so each selects its own composition key.

No panel bindings, defaults, workflow JSON, or serialization changed.

## Focused regression results

All 123 tests passed:

| Command | Tests |
| --- | ---: |
| `python -m unittest test_donut_nag_txtfusion test_nag_fusion_taps` | 35 |
| `python -m unittest test_donut_face_detailer` | 19 |
| `python -m unittest test_donut_tiled_upscale_lifecycle` | 8 |
| `python -m unittest test_donut_grounding_nag test_donut_grounding_schedule` | 61 |

New numerical tests exercise the actual hires/detailer dispatch with tensor conditioning and scoped encoder/sampler doubles. Distinctive tap gains `(1,1,1,1,1,1,1,2.5,5,1.1,4,1)` are applied after mocked variance changes ones to twos. Expected and sampled positives match. Negative tap matching on produces the same values; off preserves the raw negative while positive taps remain active. Source metadata stays unchanged.

Routing transitions cover active → alpha zero → phi zero → below/above sigma window → active, with reference edit retained in every inactive edit case and downstream wrappers called once. T2I inactive delegates to native once, including when a downstream wrapper changes activation. Keyword edit input preserves frame order. Standalone callbacks select distinct keys for both positional and keyword calls, including repeated installation.

An additional isolated check executed AST-extracted, unmodified upstream `_nag_is_active`, T2I wrapper, and nested edit wrapper through the actual downloaded ComfyUI `WrapperExecutor`. All 36 routing cases passed, covering active/inactive conditions, 4D/5D inputs, and positional/keyword edit calls. Reference identity was asserted. Forward math was stubbed. The NAG test suite also passed all 25 tests using that real executor.

Sources used for the isolated check (SHA-256):

- `Comfy-Org/ComfyUI`, `master/comfy/patcher_extension.py`: `1c7b3b855278d0b95d1c1a705fa0b3c91a265f320e2f97bdeb9d15469d6d4ac7`
- `iljung1106/ComfyUI-Krea2-NAG`, `main/krea2_nag.py`: `48396e5a12944e83ee4538a4a6d41cc0f1e06d580b9ac37500ad07f83115df17`
- Same NAG repository, `main/nodes.py`: `4368b9509e994debc34b7d2a7d8349882ea781e3c574f9de15169405ba4de306`

Changed Python files compile; `git diff --check` passes.

## Limits and outstanding behavior

Full ComfyUI core, installed upstream NAG, and a GPU generation session are unavailable in this environment. No UI Run, backend restart, image-quality comparison, or output PNG was produced. These checks establish conditioning and routing contracts only; temporal round trips do not establish full multi-frame conditioning support.

Alpha remains stage-local. Turbo denoise zero still snaps to a nonzero step. Attention hooks are rejected during active NAG rather than implemented. Those behaviors were not changed by this follow-up. Existing empty-range rejection and explicit-negative preservation remain covered by their existing regression suites.

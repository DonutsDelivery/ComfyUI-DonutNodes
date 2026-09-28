# Ignored txtfusion reference filename validation

## Failure and fix

`txtfusion_reference_checkpoint` remains a model-list combo to preserve saved widget positions and UI compatibility. Execution no longer reads that file, but ComfyUI's default combo membership check rejected saved filenames absent from the current machine, even when `txtfusion_internal_guard` was disabled.

The final registered DonutSampler now defines a narrow `VALIDATE_INPUTS` method naming only `txtfusion_reference_checkpoint`. It accepts string values without requiring a local file and rejects non-string values. All other input validation remains owned by ComfyUI. There is no catch-all argument that would bypass validation of unrelated inputs. The filename remains ignored with the legacy switch enabled as well: references come from the effective model, not this file.

No frontend, binding, serialization, workflow JSON, or widget order changes are needed for this fix. Existing selections remain intact. Updating Python requires a backend restart before the new validator is used.

## Checks

- All 19 tests in `tests/test_txtfusion_guard_sampler.py` passed, including missing filename acceptance, off → on → off dispatch, and narrow validator signature/type checks. Parent sampler/model guard operations use scoped doubles.
- An isolated harness executed the actual upstream ComfyUI `validate_inputs`, `get_input_data`, `_async_map_node_over_list`, `resolve_map_node_over_list_results`, and `get_input_info` bodies. All 13 cases passed: the old class failed for a missing filename with either toggle value; the fixed class accepted it; unrelated invalid combo and numeric inputs still failed; omitted/default/installed/empty string values passed; a numeric filename failed custom validation.
- ComfyUI source SHA-256 values: `execution.py` = `c9ef8ea11b8cb7aa80d05f670ca457211869d27623991b66d0014fb9b33fb246`; `comfy_execution/graph.py` = `3c2f8f7c5a543f432b63e9e27e480a8fd58098d8b254e25d91781b99a3dfbde1`. Sources came from `Comfy-Org/ComfyUI` master. Application setup, parent schema, and execution context were explicit doubles; no neural execution was involved.

Full ComfyUI/UI Run verification remains unavailable here: no browser/application is exposed and ComfyUI's Python runtime is absent. No generation, output PNG, or running-backend restart is claimed.

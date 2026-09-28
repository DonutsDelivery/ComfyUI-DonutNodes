# Differential Diffusion label correction

The generic `Strength` caption in the face panel is now `Differential diffusion strength`. Its existing group, position, target path, widget name, node mode, and value are unchanged. `Vary seed per face` remains a separate control.

The presentation update resolves the control's path and changes only the generic caption targeting a `DifferentialDiffusion` node. Custom captions and unrelated strength controls are preserved. The distributed `workflows/v5/DonutWF_v5.json` contains the same caption correction; that updated JSON needs manual upload to Civitai.

Validation: all 39 `tests/panel_categories.test.cjs` tests passed. The new regression compares the entire graph before/after with only the expected caption changed, including a distinctive strength of 0.37, bypass mode, control order, unrelated strength, and custom caption. A serialization round trip followed by repeated organization preserves the same result. `git diff --check` passes.

Visual and live interaction verification are blocked: computer-use inventory exposes no browser or application, and the local browser harness lacks Playwright/Chromium. No ComfyUI Run or PNG was produced. This is a label-only change and does not repair or alter the existing bypassed Differential Diffusion node's behavior.

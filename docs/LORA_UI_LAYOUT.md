# LoRA row controls and information layout

This update is for DonutLoRALoader and DonutDynamicLoRAStack. Keep the existing
workflow and node instances: input/output sockets, slots_json, backend loading,
whole-stack safety and the PNG/subgraph reload guard are unchanged.

## Row controls

Each LoRA has a separate header with visible **up**, **down** and **Remove**
buttons above its native filename picker. Removal no longer requires finding
an item in a Row actions dropdown. These buttons remain visible for disabled
rows and when information is collapsed. Remove deletes the slot from the stack,
not the installed file. The final row can be removed; Add LoRA works on an empty
stack. Boundary reorder buttons are disabled.

Operations resolve the current row identity, not a cached index or filename.
Two slots selecting the same file remain independently removable. Add, remove,
reorder and explicit suggested-weight application are bracketed by graph
beforeChange/afterChange hooks for the host's undo tracking.

Disabled rows keep their picker and enable toggle, while retaining all strengths,
vectors, hashes and other saved fields internally. Re-enabling restores those
controls without resetting values. UI-only toolbars never enter API inputs or
positional workflow widget arrays.

## Information layout

The normal view is a bordered, compact card, not a long unformatted text dump:

- **Detected weights** is a collapsed summary; expand it for all reported module
  and block groups. Consecutive indices use lossless ranges such as 0–27. Gaps
  are preserved, and exact reported indices remain available in tooltips.
- **CivitAI** starts expanded, with a small thumbnail alongside the model/version,
  base model, suggested weight and a distinct Open on CivitAI link. Trigger words
  remain below the overview. Click the thumbnail for the full preview.
- **More details** contains the full hash on its own wrapping line, author,
  description, current strengths, Retry metadata and the explicit suggested
  model-weight action. Suggestions do not automatically change either strength.

Long details scroll within a maximum 360px panel. The natural content height is
measured using ResizeObserver, so short panels shrink instead of clipping the
thumbnail or leaving a large blank area. Remove/reorder controls are outside
that scroll area. Section expansion survives asynchronous metadata updates and
catalog refreshes during the session. Lookup errors automatically expose retry;
no network credentials or additional runtime dependencies are introduced.

Native ComfyUI filename/preset dropdowns and strength widgets are retained,
including their constructor option-list guard. CivitAI lookup, local analysis,
full-hash preservation and stale-result checks continue using the existing code.

## Tests and installation

```sh
node --test tests/test_streamlining_frontend.cjs
python -m unittest discover -s tests -p test_lora_cleanup_browser.py -v
```

The frontend suite has 39 tests: 38 run without private workflow fixtures; the
existing original-vs-corrected workflow check runs when DONUT_BROKEN_WORKFLOW and
DONUT_FIXED_WORKFLOW are provided. All 39 passed with those fixtures. Eight
additional offline Chromium tests passed, including real toolbar clicks,
remove-after-reorder, last-row removal/addition, save/reload, disabled-state
preservation, metadata expansion and non-clipped layouts at 320/420px widths.

Chromium runs the shipped DOM and editor logic with a small Comfy node adapter
and mock server responses. This is not a complete ComfyUI renderer, live CivitAI
or GPU test. Playwright/Chromium are development-only test requirements.

Update DonutNodes on main, restart ComfyUI and hard-refresh the browser. Do not
replace or rewire the workflow for this UI-only change.

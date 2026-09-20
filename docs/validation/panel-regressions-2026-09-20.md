# Panel regression audit — 2026-09-20

## Additional findings

- Selecting an experimental Fusion preset after UncensorFix retained the
  `Fusion + UncensorFix weights` composition. Regular presets reset this field.
  Experimental selections now reset it to `Fusion only` too. Users can still
  explicitly enable the weights afterward.
- Restoring an experimental workflow reapplied its recipe, overwriting saved
  strength and other adjustments. Restore now preserves an already-configured
  inner recipe; the legacy outer-only recipe repair remains available.
- Rendering an older workflow's seed control changed its saved `fixed` policy
  to `randomize` when a migration marker was absent. Removed that mutation;
  rendering preserves saved values.

## Actual UI generation evidence

All runs were queued by clicking ComfyUI's Run button. No API prompt submission
was used. Values below were extracted from PNG `prompt` metadata; PNGs also
contain the complete `workflow` metadata.

Before fix:
`/home/user/Programs/ComfyUI-new/ComfyUI/output/Final/5478487336201311.png`

- Preset: `Experiment · NAG-friendly power 0.60`
- Composition: `Fusion + UncensorFix weights` (left behind by previous preset)

After fix, from a fresh browser session:
`/home/user/Programs/ComfyUI-new/ComfyUI/output/Final/panel-regression-audit/4242421.png`

- Preset: `Experiment · NAG-friendly power 0.60`
- Composition: `Fusion only`
- User-adjusted tap strength: `0.42`
- Shared seed: `424242`, with Fixed selected in the panel
- Base sampler, first upscale and Face Detailer NAG alpha: `0.2`
- First upscale: enabled, scale `1.5`, denoise approximately `0.33`
- Face Detailer: present and executed, denoise approximately `0.36`
- Second upscale and SeedVR2: disabled
- Final image: 1248 × 960, matching 832 × 640 base at 1.5×

Reloaded the audit tab afterward and verified that strength 0.42, seed 424242,
Fixed, NAG 0.2, Fusion only, first upscale and Face Detailer remained selected.

## Automated checks

Passing: `test_krea2_fusion_preset.py` (27 cases) and Node test suites for
app-controls preset refresh, experimental Fusion recipes, TeacherFix UI,
panel categories, preview stages, SeedVR2 controls, workflow reload and result
prompt. Regression assertions cover composition reset, saved experimental
adjustments, legacy outer recipe repair and saved Fixed seed policy.

## Limits

This is a focused audit of preset transitions, reload behavior and representative
generation controls. It does not prove every edit/mask/SeedVR2 combination.
The generation run verifies frontend serialization; the earlier Python Off
guard remains covered by unit tests, not a restarted-backend runtime trace.
No release was published during this audit.

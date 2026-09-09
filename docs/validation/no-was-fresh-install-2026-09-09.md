# Workflow installation without WAS — 2026-09-09

**Passed with the current development DonutNodes source:** load the saved
workflow → Install All missing packs with default versions → Apply Changes →
Run. No WAS installation, version selection or dependency repair override.

The existing test checkout and venv were removed before recreating
`/tmp/donut-fresh-install`. Official ComfyUI and Manager requirements were
installed in a new Python 3.12.7 venv. Models were reused from
`/tmp/donut-test-models`; only one test environment remains. As previously
authorized, DonutNodes used the current local checkout and its declared
requirements. This does not test the older published DonutNodes package;
users need a release containing DonutImageSave and the other workflow changes.

Manager detected and installed these seven packs through the workflow Errors
panel's Install All button, without changing versions:

- ComfyUI-bleh (nightly)
- Impact Pack 8.28.3
- Impact Subpack 1.3.5
- Derfuu ModdedNodes 1.0.1
- Krea2 Edit 1.2.3
- Krea2 NAG 1.0.2
- Krea Seed Variance Enhancer 1.2.0

After restart, all required backend nodes registered and pip check passed.
WAS is absent and no pip_auto_fix.list exists. Full GPU generation, the first
upscale, two face refinements and DonutImageSave completed in 111.82 seconds.
The saved 1728 × 1344 WebP decodes successfully and is byte-identical to the
previous WAS v3 output for the same workflow settings and seed.

Six real image-save tests pass: numbering across batches/runs, overwrite,
PNG/WebP metadata, all supported formats, output/temp previews, path containment
and surfaced write errors. The 34 registration/dependency-isolation tests also
pass after updating their stale bootstrap fixtures for the current modules.
Exact package records, workflow/output hashes, runtime messages and save inputs
are in the [JSON receipt](no-was-fresh-install-2026-09-09.json).

DonutImageSave preserves the active save settings. It uses ComfyUI's Pillow and
NumPy dependencies and retains MIT attribution for adapted WAS numbering code.
Unused WAS history, color-profile and high-bit-depth/EXR controls were removed
from the migrated workflow. The pre-migration workflow backup is
`/tmp/DonutWF_before_donut_save.json`.

This verifies Linux/CUDA with the saved editing-off configuration; editing,
the disabled second upscale and model downloads were not rerun. The existing
multiple-OpenCV-variants warning remains from companion requirements; current
imports and inference work. No manual package alteration concealed the warning.
Logs: `/tmp/donut-no-was-{base,donut,server,final-runtime}.log`.

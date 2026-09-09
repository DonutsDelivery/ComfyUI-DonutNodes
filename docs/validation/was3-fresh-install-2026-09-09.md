# WAS v3 fresh installation — 2026-09-09

Result: installation and full generation passed after explicitly selecting WAS
3.0.2 in Manager. The default Install All path still chooses WAS 1.0.1 despite
the workflow's 3.0.2 metadata, so an entirely automatic default installation is
not certified.

## Clean setup

The previous `/tmp/donut-fresh-install` checkout, virtual environment, custom
packs, and Manager state were removed before recreating that same path. Only
model files were retained in `/tmp/donut-test-models` and linked into the new
checkout. One ComfyUI test environment remains. The working installation and
its Python environment were not modified.

A fresh official ComfyUI checkout at `672ba9e5e388bd6bfac5ceef61f89ffdd9467200`
was installed with Python 3.12.7's normal venv and the official requirements and
Manager requirements. The current development DonutNodes source was symlinked
as previously authorized, and its declared requirements installed. This does
not certify the older published DonutNodes registry release.

The exact saved `DonutWF_v3_censored.json` was opened in the browser. Manager
found all eight missing companion packs, including the three Krea dependencies
inside the collapsed subgraph. WAS 3.0.2 was selected through Manager's normal
version selector; the other seven packs used Install All. All tasks completed,
then Apply Changes restarted ComfyUI. No pip repair, constraint override,
NumPy downgrade, or custom pack source patch was applied to this environment.
The obsolete Donut OpenCV installer hook and its dedicated tests were removed
from the development source. The new Manager state has no pip_auto_fix.list.

## Verification

- Base and final `pip check`: passed.
- All required backend workflow node types registered; no missing types.
- Real OpenCV color conversion and Torch/NumPy round trip: passed; CUDA available.
- Base generation, first upscale, two detected-face refinements and WAS WebP save: passed.
- Output: 1728 × 1344; completed in 106.298 seconds; decoded and visually inspected.
- Exact run ID, package versions, workflow/output hashes and execution messages:
  [machine-readable receipt](was3-fresh-install-2026-09-09.json).

The install still includes both OpenCV distributions because Donut and another
pack request different variants. Both resolve to 5.0.0.93 and current imports
work, but the shared-namespace warning remains. No manual uninstall was used
to conceal that outcome. Optional Florence-2/LayerStyle is absent; it is not
used by this workflow. Editing and the disabled second upscale were not
executed. Models were reused, so this run does not retest model downloads.

Installation logs and final runtime log are under `/tmp/donut-reinstall-*.log`
and `/tmp/donut-reinstall-evidence/`.

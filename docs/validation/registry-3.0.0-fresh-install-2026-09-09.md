# DonutNodes 3.0.0 — fresh registry installation

Passed on Linux with Python 3.12.7 and an RTX 4070.

A new ComfyUI checkout and virtual environment were created at
`/tmp/donut-registry-fresh`. Official ComfyUI and Manager requirements were
installed, then ComfyUI Manager 4.2.2's own registry installer downloaded
DonutNodes from `https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.0/node.zip`.
The registry's default install endpoint selected 3.0.0, despite its Pending
status. This used Manager's Python installation implementation, not its GUI.

There are no local-source symlinks in the installed DonutNodes directory.
Manager created its `.tracking` manifest. All 156 installed package files
matched the uploaded archive byte-for-byte. User cache and credentials were
absent. The bundled V4 Beta workflow and third-party notices were present.

The seven companion packs were also installed through Manager, using current
registry versions and bleh nightly. No WAS pack, manual dependency repair,
custom pip override, or pip_auto_fix.list was used. `pip check` passed.
Existing model weights were reused through an extra-model-paths configuration;
node code and Python dependencies were freshly installed.

The server registered every backend class in the previous successful test
prompt and exposed all 24 Donut browser modules. The same 39-node API prompt
was submitted to the new server on port 8190. It completed in 126.50 seconds:
base generation, first upscale, three face refinement crops, core PNG saving,
and Donut WebP saving. The final WebP decoded successfully at 1728 × 1344.
The saved image is `/tmp/donut-registry-fresh/ComfyUI/output/Final/4539680068827511.webp`.

This establishes registry-package installation and execution for that workload.
It does not establish that every mode is issue-free. Editing/reference guidance
were off and second upscale was disabled. Donut model downloading and full
interactive browser acceptance were not tested. Other operating systems and
GPUs were not tested.

Remaining warnings: companion requirements install both opencv-python and
opencv-python-headless; optional Florence-2 functionality needs
comfyui_layerstyle; Manager's optional matrix-sharing feature lacks matrix-nio.
None prevented this workload from completing. The missing project LICENSE
file remains a separate release metadata issue, not a runtime failure.

Test setup encountered temporary-storage quota exhaustion and an initial
Manager harness initialization error. The old test venv and incomplete new
venv were removed, the new venv was recreated, and Manager's normal prestartup
initialization was loaded before rerunning installation. No installed package
source was patched to obtain the passing result.

See [the receipt](registry-3.0.0-fresh-install-2026-09-09.json) for installation
results, exact provenance, versions, output hash, runtime status, and log paths.

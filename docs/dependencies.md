# Dependency compatibility and the built-in checker

DonutNodes does not downgrade NumPy, reinstall PyTorch, or run pip when ComfyUI
starts or a node executes. Those packages are shared with ComfyUI and other
custom node packs. Repairing one pack must not silently change everyone else's
runtime.

## What is handled natively

Each node module registers independently. A catchable import failure in an
image-analysis or other component no longer discards every Donut node. The
console names the failed component and retains the complete traceback. Nodes
that actually require the broken dependency remain unavailable; the failure is
not replaced by an identity operation or hidden behind a successful output.
The settings/CivitAI routes are isolated too; a route import failure is recorded
and those routes remain unavailable until repaired.

The Safe LoRA and Krea2 Fusion preset overrides are mandatory for their existing
node IDs. If an override fails, DonutNodes does not silently register its older
base implementation with different behavior. Other nodes still load.

OpenCV is no longer imported by the shared LoRA/block-weight utilities during
startup. It is loaded only when mask dilation/erosion is executed, with an
explicit dependency error on failure. The mask algorithm, LoRA arithmetic,
model patches, fusion presets and saved input positions are unchanged.

The binary dependencies in `requirements.txt` have minimum versions:

| Dependency | Minimum |
| --- | --- |
| opencv-python-headless | 4.10.0.84 |
| scipy | 1.13.1 |
| matplotlib | 3.9.2 |

These floors prevent an old pre-NumPy-2 wheel generation from satisfying a new
DonutNodes installation merely because the package name is already installed.
They are **not** a guarantee that every custom build or every future NumPy/Python
combination is compatible. Pip still resolves versions for the active Python.
No global NumPy upper bound, forced NumPy major version or PyTorch requirement
is added. Dependency installation may still change transitive dependencies as
part of pip's normal resolution; review the environment manager's proposed
changes. Updating code alone does not repair previously installed wheels.

## In ComfyUI

Open a blank workflow, add **Donut Dependency Check** from **Donut/diagnostics**,
and queue it. Use a blank workflow when the original workflow contains missing
nodes, since ComfyUI cannot validate such a workflow just to run its checker.

The default check is fast and read-only: it lists installed versions, the actual
Python executable, failed DonutNodes startup components and their tracebacks.
It also warns when multiple OpenCV distributions share the `cv2` namespace.
The report appears as selectable text on the node, in its STRING output and in
the console. The report widget is not serialized into the workflow.

Enable `probe_imports` to test NumPy, the PyTorch/NumPy conversion bridge,
OpenCV, SciPy FFT/ndimage/optimization, and Matplotlib's noninteractive renderer.
Each test uses a separate child process with the same Python and import paths,
a 20-second timeout, and a temporary Matplotlib cache. A crashed extension or a
stalled probe is reported without crashing the **checker**. No model is loaded,
no GPU is used, and no package-install command is executed. Normal startup
imports are not subprocess-isolated; a true native crash during startup cannot
be recovered by Python's exception handling.

## When ComfyUI itself cannot start

Run `donut_dependencies.py` as a script using the Python executable that launches
ComfyUI. This standalone mode probes imports automatically; it cannot know a
previous ComfyUI process's startup errors. Examples from the portable root or
with an already activated virtual environment:

```powershell
# Windows portable, from the folder containing python_embeded and ComfyUI:
.\python_embeded\python.exe -s ComfyUI\custom_nodes\ComfyUI-DonutNodes\donut_dependencies.py
```

```sh
# Activated ComfyUI virtual environment; adapt the node-pack folder if renamed:
python ComfyUI/custom_nodes/ComfyUI-DonutNodes/donut_dependencies.py
```

## Repair the identified package, not a guessed NumPy version

For `compiled using NumPy 1.x`, `_ARRAY_API not found`, `multiarray failed to
import` or `numpy.dtype size changed`, find the first traceback frame outside
NumPy. Repair the indicated binary package using ComfyUI's interpreter or its
environment manager, then **fully restart ComfyUI**. The checker does not infer
a particular offending package from the generic NumPy warning alone.

For DonutNodes' declared dependencies, update the pack and rerun its dependency
installation (ComfyUI Manager's repair/reinstall route, or `-m pip install -r
requirements.txt` using that interpreter). Review the OpenCV warning first.
`pip check` can detect declared version conflicts but cannot establish that
compiled extensions actually import; that is why the checker has import probes.

A range such as `numpy<2.5,>=2.0` still installs **NumPy 2**. It does not resolve a
NumPy-1-only extension's ABI mismatch by itself. An old PyTorch build may import
with warnings yet fail on `.numpy()`/`torch.from_numpy()`; use the PyTorch/NumPy
probe and repair the ComfyUI environment instead of replacing CUDA packages from
inside a custom node.

### OpenCV variants

`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`, and
`opencv-contrib-python-headless` all supply `cv2`. Use only one appropriate
variant. Donut's standard requirements select headless; requirements files
cannot express "any one of these four installed distributions". If another
node requires standard/contrib OpenCV, manage one compatible variant explicitly
rather than blindly adding headless. The checker detects conflicts but does
not uninstall other packs' dependencies. Even two identically versioned
OpenCV distributions can overwrite the same files.

## Scope and validation

This protects DonutNodes registration and supplies diagnosis for the listed
binary packages. It does not repair unrelated custom nodes, identify a package
absent from the supplied traceback, fix a broken ComfyUI installation, or make
an incompatible native binary safe to execute.

```sh
python test_dependency_isolation.py -v
node --test tests/dependency_check_ui.test.mjs
python donut_dependencies.py
```

The regression suite runs the real initializer with stand-ins for individual
node modules and simulates import errors, failed required overrides, timeouts,
and native crashes. The utility tests use real Torch/NumPy/Pillow/OpenCV when
available; those tests are skipped when their optional test dependencies are
missing. The frontend tests run the actual extension in an isolated JavaScript
VM, not a full browser. Full ComfyUI/GPU/Windows-portable testing is separate.

### Upstream references

- NumPy's downstream import-error guidance:
  https://numpy.org/devdocs/user/troubleshooting-importerror.html
- SciPy 1.13 NumPy 2 support:
  https://docs.scipy.org/doc/scipy/release/1.13.0-notes.html
- OpenCV package releases and NumPy 2 wheels:
  https://github.com/opencv/opencv-python/releases
- OpenCV's one-variant installation guidance:
  https://pypi.org/project/opencv-python/

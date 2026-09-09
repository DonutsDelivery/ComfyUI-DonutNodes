"""Dependency isolation and read-only diagnostics; uses only the standard library.

Never runs pip, changes NumPy, suppresses ABI errors, or imports optional binary
packages just to query their versions. Reports use installed metadata and
previously recorded startup errors only.
"""

import importlib
from importlib import metadata
import logging
import os
import shlex
import sys
import traceback


LOGGER = logging.getLogger("DonutNodes.dependencies")
IMPORT_FAILURES = {}
OPENCV_DISTRIBUTIONS = (
    "opencv-python", "opencv-python-headless",
    "opencv-contrib-python", "opencv-contrib-python-headless",
)
DISTRIBUTIONS = ("numpy", "torch", "scipy", "matplotlib") + OPENCV_DISTRIBUTIONS
ABI_MARKERS = (
    "compiled using numpy 1.x", "_array_api not found",
    "numpy.core.multiarray failed to import", "numpy.core.umath failed to import",
    "numpy.dtype size changed", "numpy.ufunc size changed",
)


def _failure_record(component, error):
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return {
        "component": component,
        "error": f"{type(error).__name__}: {error}",
        "numpy_abi_error": any(marker in trace.lower() for marker in ABI_MARKERS),
        "traceback": trace,
    }


def import_component(package, module_name, classes=None, display_names=None, *, overrides=()):
    """Import one component without disabling unrelated nodes on failure.

    Required overrides fail closed: never silently fall back to an older node
    implementation with different options, outputs or weight arithmetic.
    Programming errors retain their complete traceback and are NOT labelled as
    NumPy errors unless a recognized ABI error actually appears in the chain.
    """
    try:
        module = importlib.import_module(f".{module_name}", package)
        if classes is not None:
            mappings = module.NODE_CLASS_MAPPINGS
            names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
            if not isinstance(mappings, dict) or not isinstance(names, dict):
                raise TypeError(f"{module_name} must export dictionary node mappings")
            missing = set(overrides) - mappings.keys()
            if missing:
                raise ValueError(f"{module_name} is missing required overrides: {sorted(missing)}")
            # Validate everything before updating either public mapping.
            classes.update(mappings)
            if display_names is not None:
                display_names.update(names)
    except MemoryError:
        raise
    except Exception as error:
        if classes is not None:
            for key in overrides:
                classes.pop(key, None)
                if display_names is not None:
                    display_names.pop(key, None)
        record = _failure_record(module_name, error)
        IMPORT_FAILURES[module_name] = record
        kind = "NumPy binary incompatibility" if record["numpy_abi_error"] else "import failure"
        LOGGER.error(
            "[DonutNodes] %s: %s. Unrelated nodes will continue loading. "
            "Use Donut Dependency Check for versions and diagnostics.\n%s",
            module_name, kind, record["traceback"],
        )
        return None
    IMPORT_FAILURES.pop(module_name, None)
    return module


def require_cv2(feature):
    """OpenCV is only needed when executing image/mask operations, not LoRAs."""
    try:
        import cv2
        return cv2
    except (ImportError, AttributeError, ValueError, OSError, RuntimeError) as error:
        raise RuntimeError(
            f"[DonutNodes] {feature} requires a working OpenCV (cv2) installation. "
            "Run Donut Dependency Check in a blank workflow "
            "to see the failing binary and installed OpenCV variants. Repair that "
            "package in ComfyUI's Python environment and restart ComfyUI; do not "
            "blindly change NumPy. Original error: "
            f"{type(error).__name__}: {error}"
        ) from error


def installed_versions():
    """Read distribution metadata without importing potentially broken wheels."""
    result = {}
    for name in DISTRIBUTIONS:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def opencv_conflict(versions):
    present = [name for name in OPENCV_DISTRIBUTIONS if versions.get(name)]
    if len(present) < 2:
        return None
    return (
        "Multiple OpenCV distributions are installed: " + ", ".join(present) + ". "
        "They share the cv2 namespace. Keep only one variant appropriate for your "
        "other node packs; do not install headless alongside an existing variant. "
        "Nothing has been uninstalled or changed automatically."
    )


def python_command(*arguments):
    """Display a command for the actual interpreter, including portable -s."""
    parts = [sys.executable]
    if sys.flags.no_user_site:
        parts.append("-s")
    parts.extend(arguments)
    if os.name == "nt":
        # PowerShell needs & before a quoted executable; quote arguments too.
        return "& " + " ".join("'" + part.replace("'", "''") + "'" for part in parts)
    return shlex.join(parts)


def dependency_report():
    versions = installed_versions()
    lines = [
        "DonutNodes dependency report (read-only)",
        f"Python: {sys.version.split()[0]}",
        f"Interpreter: {sys.executable}",
        "No packages were installed, removed, upgraded or downgraded.",
        "\nInstalled distributions (metadata only, not proof of ABI compatibility):",
    ]
    lines.extend(f"  {name}: {value or 'not installed'}" for name, value in versions.items())
    conflict = opencv_conflict(versions)
    if conflict:
        lines.extend(["", "WARNING: " + conflict])
    lines.append(f"\nDonutNodes startup failures: {len(IMPORT_FAILURES)}")
    for record in IMPORT_FAILURES.values():
        lines.extend([f"\n[{record['component']}] {record['error']}", record["traceback"]])
    lines.extend([
        "\nNext steps for a failing check:",
        "Repair/update the package identified by the first non-NumPy traceback frame, "
        "using the interpreter above (or your environment manager), then restart ComfyUI.",
        "For DonutNodes dependencies, rerun its installation requirements after updating "
        "the node pack. Review OpenCV variant conflicts before doing so.",
        "Do not blindly force-reinstall NumPy: a NumPy 2.x version range cannot make "
        "a NumPy-1-only extension compatible. PyTorch/CUDA repairs belong to the "
        "ComfyUI environment, not a custom node's automatic installer.",
        "Package metadata check (does not test binary compatibility):",
        python_command("-m", "pip", "check"),
        "Only recorded DonutNodes startup failures are covered; failures in "
        "other node packs require their own traceback.",
    ])
    return "\n".join(lines)


class DonutDependencyCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check"
    CATEGORY = "Donut/diagnostics"
    OUTPUT_NODE = True
    DESCRIPTION = "Inspect dependency versions and startup failures without changing your Python environment."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check(self, **kwargs):
        # Ignore obsolete inputs from workflows saved with the old checker.
        report = dependency_report()
        print(report)
        return {"ui": {"text": [report]}, "result": (report,)}


if __name__ == "__main__":
    print(dependency_report())

"""Prepare a registry staging directory from a Comfy CLI packed node.zip.

Usage: python tools/prepare_registry.py node.zip /tmp/donut-registry-release
Then run comfy node publish from that staging directory. GitHub source retains
its full downloader UI; only the staged registry copy uses the manual panel.
"""
import argparse
from pathlib import Path, PurePosixPath
import zipfile


def prepare(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Use a new staging directory to avoid stale release files")
    replacement = Path(__file__).resolve().parents[1] / "distribution/registry/donut_model_downloads.js"
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"Unsafe archive path: {name}")
            if name in ("donut_model_downloads.py", "docs/publishing.md", "AGENTS.md") or name.startswith(("tools/", "tests/", "distribution/", "docs/validation/", ".git/")):
                raise ValueError(f"Non-registry file in packed archive: {name}; check .comfyignore")
        required = {"assets/uncensorfix.f32", "uncensorfix_weights.py", "model_sources.json", "pyproject.toml"}
        if not required.issubset(source.namelist()):
            raise ValueError("Packed archive is missing required runtime assets")
        source.extractall(destination)
    (destination / "web/donut_model_downloads.js").write_bytes(replacement.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.destination)

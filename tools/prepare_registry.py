"""Prepare a registry staging directory from a Comfy CLI packed node.zip.

Usage: python tools/prepare_registry.py node.zip /tmp/donut-registry-release
Then run comfy node publish from that staging directory. GitHub source retains
its full downloader UI; only the staged registry copy uses the manual panel.
"""
import argparse
import json
import re
from urllib.parse import urlsplit
from pathlib import Path, PurePosixPath
import zipfile


def registry_catalog(data):
    """Build a static, inert link catalog; no downloader imports or HTTP calls.

    Use the archive's catalog, not the working tree, to keep release links and
    hashes matched to the package. Do not embed workflow-supplied URLs.
    """
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("models"), list):
        raise ValueError("Invalid model catalog in packed archive")
    entries, seen = [], set()
    for entry in data["models"]:
        folder, filename = entry["folder"], entry["filename"]
        if not isinstance(folder, str) or not re.fullmatch(r"[A-Za-z0-9_]+", folder):
            raise ValueError("Invalid model folder in packed catalog")
        if (not isinstance(filename, str) or not filename or "\\" in filename
                or ":" in filename or any(part in ("", ".", "..") for part in filename.split("/"))):
            raise ValueError("Invalid model filename in packed catalog")
        key = (folder, filename)
        if key in seen:
            raise ValueError("Duplicate model in packed catalog")
        seen.add(key)
        url = urlsplit(entry["url"])
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.port not in (None, 443)):
            raise ValueError("Invalid upstream URL in packed catalog")
        if (not re.fullmatch(r"[a-fA-F0-9]{64}", entry["sha256"])
                or type(entry["size"]) is not int or entry["size"] <= 0):
            raise ValueError("Invalid checksum or size in packed catalog")
        required = entry.get("requires_nodes", [])
        if (not isinstance(required, list) or len(required) > 32
                or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name)
                       for name in required)):
            raise ValueError("Invalid native node requirements in packed catalog")
        entries.append({key: entry[key] for key in ("folder", "filename", "url", "sha256", "size")})
        if required:
            entries[-1]["requires_nodes"] = required
    return "// Generated from this package's model_sources.json. Data only.\nexport const MODEL_CATALOG = " + json.dumps(entries, indent=2, ensure_ascii=True) + ";\n"


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
        required = {"assets/uncensorfix.f32", "uncensorfix_weights.py", "model_sources.json", "pyproject.toml",
                    "web/donut_model_requirements.js", "web/donut_model_downloads.js", "web/donut_layout.js"}
        if not required.issubset(source.namelist()):
            raise ValueError("Packed archive is missing required runtime assets")
        if "export function manualModelFiles(" not in source.read("web/donut_model_requirements.js").decode("utf-8"):
            raise ValueError("Packed model requirements are outdated; pack the updated source again")
        link_catalog = registry_catalog(json.loads(source.read("model_sources.json")))
        source.extractall(destination)
    (destination / "web/donut_registry_catalog.js").write_text(link_catalog, encoding="utf-8")
    (destination / "web/donut_model_downloads.js").write_bytes(replacement.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.destination)

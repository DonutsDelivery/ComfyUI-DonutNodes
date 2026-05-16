"""
LoRA file resolver with fallbacks for cross-machine workflows.

Pipeline (in order):
  1. Exact path via folder_paths.get_full_path
  2. Basename scan across all loras roots
  3. Hash-cache lookup: any cached entry with matching SHA256
  4. Civitai lookup + download by hash

Lets a workflow saved on one machine load on another even when the LoRA
sits in a different subfolder, has been renamed, or is missing entirely.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

from .civitai_api import get_cache, fetch_model_by_hash
from .civitai_download import (
    get_downloader,
    get_download_path,
    invalidate_folder_cache,
)


RESOLVER_EVENT = "donut-lora-resolver"


def _notify(payload: dict):
    """Push a progress event to the UI; no-op if server unavailable."""
    if not HAS_SERVER:
        return
    try:
        PromptServer.instance.send_sync(RESOLVER_EVENT, payload)
    except Exception:
        pass


def _loras_roots() -> list:
    """Get every directory ComfyUI searches for loras."""
    if not HAS_FOLDER_PATHS:
        return []
    try:
        return list(folder_paths.get_folder_paths("loras"))
    except Exception:
        return []


def _basename_search(name: str) -> Optional[str]:
    """Walk all loras roots for a file with the same basename. Prefer the
    one whose path overlaps the requested relative path the most."""
    target = Path(name).name
    matches = []
    for root in _loras_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if target in files:
                matches.append(os.path.join(dirpath, target))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    wanted = set(Path(name).parts)
    matches.sort(key=lambda m: -sum(1 for p in Path(m).parts if p in wanted))
    return matches[0]


def _hash_cache_lookup(target_hash: str) -> Optional[str]:
    """Scan civitai_cache/hashes/*.hash for a matching SHA256; return its
    file_path if the file still exists."""
    target = target_hash.upper()
    hashes_dir = get_cache().cache_dir / "hashes"
    if not hashes_dir.exists():
        return None
    for hf in hashes_dir.glob("*.hash"):
        try:
            with open(hf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        sha = (data.get("hashes") or {}).get("SHA256", "").upper()
        if sha == target:
            fp = data.get("file_path")
            if fp and os.path.exists(fp):
                return fp
    return None


def _civitai_download(target_hash: str, requested_name: str,
                      api_key: Optional[str] = None) -> Optional[str]:
    """Look up hash on Civitai, download, return local path. Blocks."""
    short_name = Path(requested_name).name

    _notify({"status": "looking_up",
             "name": short_name,
             "message": f"Looking up {short_name} on Civitai"})

    info = fetch_model_by_hash(target_hash, api_key=api_key)
    if not info or not info.download_url:
        _notify({"status": "not_found",
                 "name": short_name,
                 "message": f"Not found on Civitai: {short_name}"})
        return None

    filename = info.file_name or short_name
    save_path = get_download_path(info.model_type or "LORA",
                                  info.base_model, filename)

    if os.path.exists(save_path):
        invalidate_folder_cache("loras")
        return save_path

    _notify({"status": "downloading",
             "name": short_name,
             "filename": filename,
             "size_kb": info.file_size_kb,
             "message": f"Downloading {filename}"})

    last_pct = [-1]

    def on_progress(s):
        if s.total_size <= 0:
            return
        pct = int(100 * s.downloaded_size / s.total_size)
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        _notify({"status": "progress",
                 "name": short_name,
                 "filename": filename,
                 "percent": pct,
                 "downloaded": s.downloaded_size,
                 "total": s.total_size})

    downloader = get_downloader()
    download_id = downloader.start_download(
        download_url=info.download_url,
        save_path=save_path,
        api_key=api_key,
        on_progress=on_progress,
        model_type=info.model_type or "LORA",
        sha256=target_hash,
    )

    while True:
        status = downloader.get_status(download_id)
        if not status:
            return None
        if status.status == "completed":
            invalidate_folder_cache("loras")
            _notify({"status": "complete",
                     "name": short_name,
                     "filename": filename,
                     "message": f"Downloaded {filename}"})
            return save_path
        if status.status in ("error", "cancelled"):
            _notify({"status": "error",
                     "name": short_name,
                     "filename": filename,
                     "message": f"Download failed: {status.error or status.status}"})
            return None
        time.sleep(0.5)


def resolve_lora(name: str, expected_hash: Optional[str] = None,
                 auto_download: bool = True,
                 api_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Find a LoRA file by name, with fallbacks.

    Returns (resolved_path, source) — source is one of:
      "exact"      — found at workflow's stated path
      "basename"   — found by basename anywhere under loras roots
      "hash_cache" — found via SHA256 hit in the local hash cache
      "civitai"    — downloaded from Civitai by hash
      "missing"    — could not be resolved
    """
    if not name or name == "None" or not HAS_FOLDER_PATHS:
        return (None, "missing")

    path = folder_paths.get_full_path("loras", name)
    if path and os.path.exists(path):
        return (path, "exact")

    path = _basename_search(name)
    if path:
        _notify({"status": "found_local",
                 "name": Path(name).name,
                 "message": f"Located {Path(name).name} at {path}"})
        return (path, "basename")

    if expected_hash:
        path = _hash_cache_lookup(expected_hash)
        if path:
            _notify({"status": "found_local",
                     "name": Path(name).name,
                     "message": f"Located {Path(name).name} by hash"})
            return (path, "hash_cache")

    if auto_download and expected_hash:
        path = _civitai_download(expected_hash, name, api_key=api_key)
        if path:
            return (path, "civitai")

    _notify({"status": "missing",
             "name": Path(name).name,
             "message": f"Missing LoRA: {name}" +
                        (f" (hash {expected_hash[:10]})" if expected_hash else " (no hash)")})
    return (None, "missing")


def relative_name_for(path: str) -> Optional[str]:
    """Given an absolute path, return the relative-to-loras-root form that
    folder_paths uses (e.g., 'subdir/foo.safetensors'). Falls back to the
    basename if no root contains it."""
    if not path or not HAS_FOLDER_PATHS:
        return None
    abs_path = os.path.abspath(path)
    for root in _loras_roots():
        root_abs = os.path.abspath(root)
        if abs_path.startswith(root_abs + os.sep):
            return os.path.relpath(abs_path, root_abs).replace(os.sep, "/")
    return os.path.basename(path)

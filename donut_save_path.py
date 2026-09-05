"""Safe model-save path resolution with support for intentional output symlinks.

ComfyUI's image-save helper resolves symlinks before checking containment. That is
correct for browser-facing image writes, but it also rejects a common local model
workflow where e.g. ``output/diffusion_models`` is a user-created symlink to a
larger/faster SSD.

Donut first delegates to ComfyUI unchanged. Only when that call fails do we allow
a narrow fallback: the requested path must be lexically inside ``output_dir`` and
an already-existing symlink component inside ``output_dir`` must explain why the
resolved path leaves it. Absolute paths and ``..`` traversal are never accepted.
"""

from __future__ import annotations

import os
import re


def _lexically_within(base_dir: str, target: str) -> bool:
    base_abs = os.path.abspath(base_dir)
    target_abs = os.path.abspath(target)
    try:
        return os.path.commonpath([base_abs, target_abs]) == base_abs
    except ValueError:
        return False


def _first_existing_symlink(base_dir: str, target_dir: str):
    base_abs = os.path.abspath(base_dir)
    target_abs = os.path.abspath(target_dir)
    if not _lexically_within(base_abs, target_abs):
        return None

    relative = os.path.relpath(target_abs, base_abs)
    current = base_abs
    if relative in ("", "."):
        return None

    for part in relative.split(os.sep):
        if part in ("", "."):
            continue
        current = os.path.join(current, part)
        if os.path.lexists(current) and os.path.islink(current):
            return current
    return None


def _next_counter(folder: str, filename: str) -> int:
    pattern = re.compile(rf"^{re.escape(filename)}_(\d+)_\.safetensors$")
    highest = 0
    try:
        entries = os.listdir(folder)
    except FileNotFoundError:
        entries = []

    for entry in entries:
        match = pattern.match(entry)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def get_model_save_path(folder_paths_module, filename_prefix: str, output_dir: str):
    """Resolve a model-save path, permitting only intentional output symlinks."""
    try:
        return folder_paths_module.get_save_image_path(filename_prefix, output_dir)
    except Exception as original_error:
        if not isinstance(filename_prefix, str) or not filename_prefix:
            raise
        if "\x00" in filename_prefix or os.path.isabs(filename_prefix):
            raise
        if "%" in filename_prefix:
            raise

        normalized = os.path.normpath(filename_prefix)
        subfolder = os.path.dirname(normalized)
        filename = os.path.basename(normalized)
        if filename in ("", ".", ".."):
            raise

        output_abs = os.path.abspath(output_dir)
        full_output_folder = os.path.abspath(os.path.join(output_abs, subfolder))
        if not _lexically_within(output_abs, full_output_folder):
            raise

        symlink = _first_existing_symlink(output_abs, full_output_folder)
        if symlink is None or not os.path.isdir(symlink):
            raise original_error

        os.makedirs(full_output_folder, exist_ok=True)
        counter = _next_counter(full_output_folder, filename)
        print(
            "[DonutSave] ComfyUI rejected an output symlink; allowing explicit "
            f"local symlink {symlink} -> {os.path.realpath(symlink)}"
        )
        return full_output_folder, filename, counter, subfolder, filename_prefix

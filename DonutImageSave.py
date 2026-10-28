"""Image saving without an external node pack.

Numbering adapted from WAS Node Suite v3 modules/io/naming.py (MIT).
Copyright (c) 2023 Jordan Thompson (WASasquatch). See THIRD_PARTY_NOTICES.md.

When nothing consumes the ``files`` output, the slow part of a save (listing
the destination folder, creating and writing the file) runs on a background
writer so the next job can start while a busy disk catches up. Each encoded
image is first staged under the ComfyUI user directory with a small job file,
so a restart (ComfyUI-Manager uses ``os.execv``) or crash cannot lose it:
staged jobs left behind are delivered when the node pack loads again.
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import json
import logging
import os
import re
import time
import uuid

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args

try:
    from .donut_prompt import consumed_outputs
except ImportError:
    from donut_prompt import consumed_outputs

STAGING_NAME = "donut_pending_saves"
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="donut_image_save")


def _staging_directory():
    return Path(folder_paths.get_user_directory()) / STAGING_NAME


def _expand_prefix(prefix, width, height):
    """ComfyUI's %width%/%date% filename variables, without listing the folder."""
    now = time.localtime()
    for key, value in (("width", width), ("height", height), ("year", now.tm_year),
                       ("month", f"{now.tm_mon:02}"), ("day", f"{now.tm_mday:02}"),
                       ("hour", f"{now.tm_hour:02}"), ("minute", f"{now.tm_min:02}"),
                       ("second", f"{now.tm_sec:02}")):
        prefix = prefix.replace(f"%{key}%", str(value))
    return prefix


def _encode(pil, extension, options):
    buffer = BytesIO()
    pil.save(buffer, format={"jpg": "JPEG"}.get(extension, extension.upper()), **options)
    return buffer.getvalue()


def _write_numbered(data, job):
    """Create the next free numbered file in the job's folder and write data."""
    base, destination = Path(job["base"]), Path(job["destination"])
    prefix, delimiter, extension = job["prefix"], job["delimiter"], job["extension"]
    destination.mkdir(parents=True, exist_ok=True)
    parts = (r"(\d+)", re.escape(prefix)) if job["number_start"] else (re.escape(prefix), r"(\d+)")
    pattern = re.compile(parts[0] + re.escape(delimiter) + parts[1] + r"\.[^.]+$")
    counter = max((int(m.group(1)) for p in destination.iterdir()
                   if (m := pattern.fullmatch(p.name))), default=0) + 1
    # Exclusive creation also protects numbering against concurrent runs.
    while True:
        number = str(counter).zfill(max(1, int(job["padding"])))
        stem = number + delimiter + prefix if job["number_start"] else prefix + delimiter + number
        name = f"{prefix if job['overwrite'] else stem}.{extension}"
        path = destination / name
        if not path.resolve().is_relative_to(base):
            raise ValueError("Save file must stay inside the selected save location")
        try:
            handle = path.open("wb" if job["overwrite"] else "xb")
            break
        except FileExistsError:
            counter += 1
    try:
        with handle:
            handle.write(data)
    except Exception:
        if not job["overwrite"]:
            path.unlink(missing_ok=True)
        raise
    return path


def _deliver(job_file):
    """Write one staged image to its destination, then remove the staging files."""
    data_file = job_file.with_suffix(".data")
    try:
        job = json.loads(job_file.read_text())
        path = _write_numbered(data_file.read_bytes(), job)
    except Exception:
        logging.exception("[DonutImageSave] Could not save staged image %s; it is kept for the next start", data_file)
        return None
    data_file.unlink(missing_ok=True)
    job_file.unlink(missing_ok=True)
    return path


def _resume_staged_saves():
    try:
        jobs = sorted(_staging_directory().glob("*.json"))
    except Exception:
        return
    for job_file in jobs:
        _writer.submit(_deliver, job_file)


class DonutImageSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "root": (["output", "temp"],),
            "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            "filename_delimiter": ("STRING", {"default": "_"}),
            "filename_number_padding": ("INT", {"default": 4, "min": 1, "max": 9}),
            "filename_number_start": ("BOOLEAN", {"default": False}),
            "extension": (["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp"],),
            "dpi": ("INT", {"default": 300, "min": 1, "max": 2400}),
            "quality": ("INT", {"default": 100, "min": 1, "max": 100}),
            "optimize_image": ("BOOLEAN", {"default": True}),
            "lossless_webp": ("BOOLEAN", {"default": False}),
            "overwrite_mode": ("BOOLEAN", {"default": False}),
            "embed_workflow": ("BOOLEAN", {"default": True}),
            "show_previews": ("BOOLEAN", {"default": True}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "files")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Donut/image"

    def save_images(self, images, root="output", filename_prefix="ComfyUI",
                    filename_delimiter="_", filename_number_padding=4,
                    filename_number_start=False, extension="png", dpi=300,
                    quality=100, optimize_image=True, lossless_webp=False,
                    overwrite_mode=False, embed_workflow=True, show_previews=True,
                    prompt=None, extra_pnginfo=None, unique_id=None):
        if root not in ("output", "temp"):
            raise ValueError("Save location must be output or temp")
        if extension not in self.INPUT_TYPES()["required"]["extension"][0]:
            raise ValueError("Unsupported image format")
        if any(c in filename_delimiter for c in ("/", "\\", "\0")):
            raise ValueError("Filename delimiter cannot contain path separators")
        base = Path(folder_paths.get_output_directory() if root == "output"
                    else folder_paths.get_temp_directory()).resolve()
        wanted = filename_prefix.replace("\\", "/")
        # Validate resolved paths too: an output subfolder can be a symlink.
        if not (base / wanted).resolve().is_relative_to(base):
            raise ValueError("Filename prefix must stay inside the selected save location")
        # Only the folder and prefix are needed here; the folder listing for
        # numbering happens when the file is written.
        expanded = os.path.normpath(_expand_prefix(wanted or "_", images[0].shape[1], images[0].shape[0]))
        destination = (base / os.path.dirname(expanded)).resolve()
        if not destination.is_relative_to(base):
            raise ValueError("Save folder must stay inside the selected save location")
        prefix = os.path.basename(expanded) if wanted else ""
        metadata = {}
        if embed_workflow and not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            metadata.update({k: json.dumps(v) for k, v in (extra_pnginfo or {}).items()})
        # Write later only when no node needs the saved paths during this run.
        consumed = consumed_outputs(prompt, unique_id)
        background = consumed is not None and 1 not in consumed
        job = dict(base=str(base), destination=str(destination), prefix=prefix,
                   delimiter=filename_delimiter, padding=int(filename_number_padding),
                   number_start=bool(filename_number_start), overwrite=bool(overwrite_mode),
                   extension=extension)
        subfolder = "" if destination == base else destination.relative_to(base).as_posix()
        if not background:
            destination.mkdir(parents=True, exist_ok=True)
        results, files = [], []
        for frame in images:
            pil = Image.fromarray(np.clip(frame.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
            options = {}
            if extension == "png":
                chunks = PngInfo()
                for key, value in metadata.items():
                    chunks.add_text(key, value)
                options.update(pnginfo=chunks, optimize=optimize_image, dpi=(dpi, dpi))
            elif extension == "webp":
                exif = Image.Exif()
                if "prompt" in metadata:
                    exif[0x0110] = "prompt:" + metadata["prompt"]
                tag = 0x010F
                for key, value in metadata.items():
                    if key != "prompt":
                        exif[tag] = key + ":" + value
                        tag -= 1
                options.update(quality=quality, lossless=lossless_webp, exif=exif)
            elif extension in ("jpg", "jpeg"):
                pil = pil.convert("RGB")
                options.update(quality=quality, optimize=optimize_image, dpi=(dpi, dpi))
            elif extension in ("gif", "tiff"):
                options.update(optimize=optimize_image)
            data = _encode(pil, extension, options)
            if not background:
                path = _write_numbered(data, job)
                files.append(str(path))
                results.append({"filename": path.name, "subfolder": subfolder, "type": root})
                continue
            staging = _staging_directory()
            staging.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid1().hex
            (staging / f"{token}.data").write_bytes(data)
            job_file = staging / f"{token}.json"
            job_file.write_text(json.dumps(job))
            _writer.submit(_deliver, job_file)
            if show_previews:
                # The final file may not exist yet, so preview a fast temp copy.
                preview_dir = Path(folder_paths.get_temp_directory())
                preview_dir.mkdir(parents=True, exist_ok=True)
                preview = f"donut_save_{token}.png"
                pil.save(preview_dir / preview, compress_level=1)
                results.append({"filename": preview, "subfolder": "", "type": "temp"})
        return {"ui": {"images": results if show_previews else []}, "result": (images, files)}


_resume_staged_saves()

NODE_CLASS_MAPPINGS = {"DonutImageSave": DonutImageSave}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutImageSave": "Donut Image Save"}

"""Image saving without an external node pack.

Numbering adapted from WAS Node Suite v3 modules/io/naming.py (MIT).
Copyright (c) 2023 Jordan Thompson (WASasquatch). See THIRD_PARTY_NOTICES.md.
"""
from pathlib import Path
import json
import re

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args


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
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

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
                    prompt=None, extra_pnginfo=None):
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
        directory, prefix, _, _, _ = folder_paths.get_save_image_path(
            wanted or "_", str(base), images[0].shape[1], images[0].shape[0])
        destination = Path(directory).resolve()
        if not destination.is_relative_to(base):
            raise ValueError("Save folder must stay inside the selected save location")
        destination.mkdir(parents=True, exist_ok=True)
        if not wanted:
            prefix = ""
        parts = (r"(\d+)", re.escape(prefix)) if filename_number_start else (re.escape(prefix), r"(\d+)")
        pattern = re.compile(parts[0] + re.escape(filename_delimiter) + parts[1] + r"\.[^.]+$")
        counter = max((int(m.group(1)) for p in destination.iterdir()
                       if (m := pattern.fullmatch(p.name))), default=0) + 1
        metadata = {}
        if embed_workflow and not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            metadata.update({k: json.dumps(v) for k, v in (extra_pnginfo or {}).items()})
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
            # Exclusive creation also protects numbering against concurrent runs.
            while True:
                number = str(counter).zfill(max(1, int(filename_number_padding)))
                stem = (number + filename_delimiter + prefix if filename_number_start
                        else prefix + filename_delimiter + number)
                name = f"{prefix if overwrite_mode else stem}.{extension}"
                path = destination / name
                if not path.resolve().is_relative_to(base):
                    raise ValueError("Save file must stay inside the selected save location")
                try:
                    handle = path.open("wb" if overwrite_mode else "xb")
                    break
                except FileExistsError:
                    counter += 1
            try:
                with handle:
                    pil.save(handle, format={"jpg": "JPEG"}.get(extension, extension.upper()), **options)
            except Exception:
                if not overwrite_mode:
                    path.unlink(missing_ok=True)
                raise
            counter += 1
            files.append(str(path))
            results.append({"filename": name, "subfolder": "" if destination == base else destination.relative_to(base).as_posix(), "type": root})
        return {"ui": {"images": results if show_previews else []}, "result": (images, files)}


NODE_CLASS_MAPPINGS = {"DonutImageSave": DonutImageSave}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutImageSave": "Donut Image Save"}

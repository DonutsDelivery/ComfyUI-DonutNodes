"""Persistent reference images and crop controls for the Krea2 editing workflow."""

import hashlib
import io
import math
import os
from pathlib import Path
import re
import tempfile

from aiohttp import web
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

import folder_paths
import nodes
from server import PromptServer
try:
    from .donut_prompt import expand_text, directory_fingerprint
except ImportError:
    from donut_prompt import expand_text, directory_fingerprint


ASPECT_RATIOS = {
    "1:1 Square": (1, 1),
    "2:3 Portrait": (2, 3),
    "3:2 Photo": (3, 2),
    "3:4 Portrait": (3, 4),
    "4:3 Standard": (4, 3),
    "9:16 Portrait": (9, 16),
    "16:9 Wide": (16, 9),
    "21:9 Ultrawide": (21, 9),
}
RESOLUTION_MODES = ["Preset", "Reference A · megapixels", "Reference A · crop only", "Custom"]


def _reference_root():
    return Path(folder_paths.get_user_directory()) / "donut" / "edit_references"


def _reference_path(name):
    if name.startswith("donutref:"):
        reference_id = name.removeprefix("donutref:")
        if not re.fullmatch(r"[a-f0-9]{64}", reference_id):
            raise ValueError("Invalid saved reference ID.")
        root = _reference_root().resolve()
        path = (root / (reference_id + ".png")).resolve()
        if path.parent != root:
            raise ValueError("Invalid saved reference path.")
        return path
    return Path(folder_paths.get_annotated_filepath(name))


def store_reference(file):
    with Image.open(file) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    data = encoded.getvalue()
    name = "donutref:" + hashlib.sha256(data).hexdigest()
    path = _reference_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as output:
                output.write(data)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {"reference": name, "width": image.width, "height": image.height}


@PromptServer.instance.routes.post("/donut/edit-studio/reference")
async def upload_reference(request):
    post = await request.post()
    file = post.get("image")
    if not isinstance(file, web.FileField):
        raise web.HTTPBadRequest(text="Choose an image file.")
    try:
        return web.json_response(store_reference(file.file))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise web.HTTPBadRequest(text="The uploaded file could not be read as an image.") from None


@PromptServer.instance.routes.get("/donut/edit-studio/reference/{reference_id}")
async def view_reference(request):
    try:
        path = _reference_path("donutref:" + request.match_info["reference_id"])
    except ValueError:
        raise web.HTTPNotFound() from None
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Content-Type": "image/png", "X-Content-Type-Options": "nosniff"})


def _round(value):
    return math.floor(value + 0.5)


def target_dimensions(mode, aspect_ratio, megapixels, width, height, multiple, source_size=None, source_b_size=None):
    multiple = int(multiple)
    if mode == "Reference A · crop only" and source_size is not None:
        return tuple(max(multiple, int(edge) // multiple * multiple) for edge in source_size)
    if mode == "Custom":
        return tuple(max(multiple, _round(edge / multiple) * multiple) for edge in (width, height))
    if aspect_ratio.startswith("Auto"):
        source = source_b_size if aspect_ratio.endswith("B") else source_size
        aspect_ratio = min(ASPECT_RATIOS, key=lambda key: abs(math.log((ASPECT_RATIOS[key][0] / ASPECT_RATIOS[key][1]) / (source[0] / source[1])))) if source else "4:3 Standard"
    ratio = source_size if mode.startswith("Reference A") and source_size is not None else ASPECT_RATIOS[aspect_ratio]
    scale = math.sqrt(float(megapixels) * 1024 * 1024 / (ratio[0] * ratio[1]))
    return tuple(max(multiple, _round(edge * scale / multiple) * multiple) for edge in ratio)


def crop_box(source_width, source_height, target_width, target_height, x=0.5, y=0.5, crop_only=False):
    if crop_only:
        crop_width, crop_height = min(source_width, target_width), min(source_height, target_height)
    elif source_width * target_height > source_height * target_width:
        crop_width, crop_height = max(1, _round(source_height * target_width / target_height)), source_height
    else:
        crop_width, crop_height = source_width, max(1, _round(source_width * target_height / target_width))
    left = _round((source_width - crop_width) * min(1.0, max(0.0, x)))
    top = _round((source_height - crop_height) * min(1.0, max(0.0, y)))
    return left, top, left + crop_width, top + crop_height


def _open_reference(name):
    path = _reference_path(name)
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _crop_reference(image, size, x, y, crop_only=False):
    box = crop_box(*image.size, *size, x, y, crop_only)
    cropped = image.crop(box)
    # Match the exact target ratio here so downstream edit preparation cannot
    # silently crop another pixel off a rounded aspect-ratio crop.
    if cropped.size != size:
        cropped = cropped.resize(size, Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(cropped).astype(np.float32) / 255.0).unsqueeze(0)


def _load_edit_lora(model, lora_name, strength):
    mode = getattr(model, "model_options", {}).get("donut_lora_execution_mode", "Comfy patches")
    if mode == "Experimental bypass":
        import comfy.utils
        from .DonutSafeApplyLoRAStack import _apply_bypass_applications
        path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        return _apply_bypass_applications(model, [(lora, strength, ",".join(["1"] * 29))])
    return nodes.LoraLoaderModelOnly().load_lora_model_only(model, lora_name, strength)[0]


class DonutEditStudio:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        identity = next((name for name in loras if name.replace("\\", "/").endswith("krea2_identity_edit_v1_2.safetensors")), "None")
        return {"required": {
            "enabled": ("BOOLEAN", {"default": False}),
            "image_a": ("STRING", {"default": ""}),
            "image_b": ("STRING", {"default": ""}),
            "use_reference_b": ("BOOLEAN", {"default": False}),
            "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            "resolution_mode": (RESOLUTION_MODES, {"default": "Preset"}),
            "aspect_ratio": (["Auto · Reference A", "Auto · Reference B", *ASPECT_RATIOS], {"default": "4:3 Standard"}),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
            "width": ("INT", {"default": 1152, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "height": ("INT", {"default": 896, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "multiple": (["16", "32", "64"], {"default": "64"}),
            "grounding_px": ("INT", {"default": 1088, "min": 0, "max": 4096, "step": 64}),
            "lora_name": ("STRING", {"default": identity, "donut_loras": ["None", *loras]}),
            "lora_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.05}),
            "crop_a_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            "crop_a_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            "crop_b_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            "crop_b_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
        }, "optional": {
            "model": ("MODEL", {"lazy": True, "tooltip": "Base model; the edit LoRA is applied only when editing is enabled."}),
            "text_seed": ("INT", {"default": 0, "forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOOLEAN", "INT", "INT", "INT", "MODEL", "STRING")
    RETURN_NAMES = ("reference_a", "reference_b", "edit_mode", "width", "height", "grounding_px", "edit_model", "edit_prompt")
    FUNCTION = "prepare"
    CATEGORY = "donut/editing"
    DESCRIPTION = "Two persistent reference slots with clipboard upload, draggable crop previews, output sizing, and Krea2 edit controls. Blank slots are allowed while editing is off."

    def check_lazy_status(self, enabled=False, model=None, **kwargs):
        return ["model"] if enabled and model is None else []

    @classmethod
    def VALIDATE_INPUTS(cls, enabled=False, image_a="", image_b="", use_reference_b=False, **kwargs):
        if not enabled:
            return True
        references = [("A", image_a)]
        if use_reference_b:
            references.append(("B", image_b))
        for label, name in references:
            try:
                exists = bool(name) and _reference_path(name).is_file()
            except ValueError:
                exists = False
            if not exists:
                return f"Add reference {label} in Edit Studio, or turn editing off."
        return True

    @classmethod
    def IS_CHANGED(cls, enabled=False, image_a="", image_b="", use_reference_b=False, **kwargs):
        if not enabled:
            return "disabled"
        digest = hashlib.sha256()
        for name in [image_a, *([image_b] if use_reference_b or kwargs.get("aspect_ratio") == "Auto · Reference B" else [])]:
            if name and _reference_path(name).is_file():
                with _reference_path(name).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest(), directory_fingerprint()

    def prepare(self, enabled, image_a, image_b, use_reference_b, prompt, resolution_mode,
                aspect_ratio, megapixels, width, height, multiple, grounding_px,
                lora_name, lora_strength, crop_a_x=0.5, crop_a_y=0.5,
                crop_b_x=0.5, crop_b_y=0.5, model=None, text_seed=0):
        source_a = _open_reference(image_a) if enabled else None
        source_b = _open_reference(image_b) if enabled and image_b and (use_reference_b or aspect_ratio == "Auto · Reference B") else None
        size = target_dimensions(resolution_mode, aspect_ratio, megapixels, width, height, multiple,
                                 source_a.size if source_a is not None else None,
                                 source_b.size if source_b is not None else None)
        reference_a = reference_b = edit_model = None
        if enabled:
            prompt = expand_text(prompt, text_seed)
            if model is None:
                raise ValueError("Connect a Krea2 model to Edit Studio.")
            reference_a = _crop_reference(source_a, size, crop_a_x, crop_a_y,
                                          resolution_mode == "Reference A · crop only")
            if use_reference_b:
                reference_b = _crop_reference(source_b, size, crop_b_x, crop_b_y)
            edit_model = model
            if lora_name and lora_name != "None" and lora_strength != 0:
                edit_model = _load_edit_lora(model, lora_name, lora_strength)
        return (reference_a, reference_b, bool(enabled), *size, int(grounding_px), edit_model, prompt)


class DonutReferenceStudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enabled": ("BOOLEAN", {"default": False}),
            "image_a": ("STRING", {"default": ""}),
            "image_b": ("STRING", {"default": ""}),
            "use_reference_b": ("BOOLEAN", {"default": False}),
        }, "optional": {"edit_active": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("reference_a", "reference_b", "enabled")
    FUNCTION = "prepare"
    CATEGORY = "donut/conditioning"
    DESCRIPTION = "Independent full-image references for native Krea2 visual conditioning. Pauses while Edit Studio is active."

    @classmethod
    def IS_CHANGED(cls, enabled=False, image_a="", image_b="", use_reference_b=False, edit_active=False):
        return DonutEditStudio.IS_CHANGED(enabled=enabled and not edit_active, image_a=image_a,
                                          image_b=image_b, use_reference_b=use_reference_b)

    def prepare(self, enabled=False, image_a="", image_b="", use_reference_b=False, edit_active=False):
        if not enabled or edit_active:
            return None, None, False
        if not image_a:
            raise ValueError("Add reference A in Reference Guidance, or turn reference guidance off.")
        if use_reference_b and not image_b:
            raise ValueError("Add reference B in Reference Guidance, or turn the second reference off.")
        def load(name):
            return torch.from_numpy(np.asarray(_open_reference(name)).astype(np.float32) / 255.0).unsqueeze(0)
        return load(image_a), load(image_b) if use_reference_b else None, True


NODE_CLASS_MAPPINGS = {"DonutEditStudio": DonutEditStudio, "DonutReferenceStudio": DonutReferenceStudio}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutEditStudio": "Donut Edit Studio", "DonutReferenceStudio": "Donut Reference Guidance"}

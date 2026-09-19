"""Subject isolation owned by Edit Studio's Reference B, not A's inpaint mask.

Native BiRefNet runs only in ComfyUI's execution queue. HTTP routes below only
store/serve masks; they never load a model or run GPU inference. The original B
and its source-coordinate mask are persisted separately. Both Krea edit paths
receive the same neutral-composited B through the existing reference_b output.
"""
from copy import deepcopy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import re
import tempfile

from aiohttp import web
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
import torch

import folder_paths
import nodes
from server import PromptServer
from .DonutEditStudio import DonutEditStudio as _Base, _open_reference, _crop_reference


MODEL_NAME = "birefnet.safetensors"
PROMPT_MODEL = "sam3.1_multiplex_fp16.safetensors"
MODES = ["Off", "Auto subject", "Saved mask", "External mask", "Prompt selection"]
BACKGROUNDS = {"Neutral gray": 0.5, "White": 1.0, "Black": 0.0}
MAX_PIXELS = 32_000_000
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
NATIVE_NODES = ("LoadBackgroundRemovalModel", "RemoveBackground")


def _root():
    return Path(folder_paths.get_user_directory()) / "donut" / "edit_subject_masks"


def _mask_path(token):
    if not isinstance(token, str) or not re.fullmatch(r"donutmask:[a-f0-9]{64}", token):
        raise ValueError("Invalid saved subject-mask ID.")
    root = _root().resolve()
    path = (root / (token[10:] + ".png")).resolve()
    if path.parent != root:
        raise ValueError("Invalid subject-mask path.")
    return path


def _source(name):
    image = _open_reference(name)
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Reference B is too large for Smart Mask; use an image under 32 megapixels.")
    return image


def fingerprint(image):
    return hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode() + image.convert("RGB").tobytes()).hexdigest()


def validate_mask(mask, size):
    if not torch.is_tensor(mask):
        raise ValueError("A subject mask must be a MASK tensor; white keeps the subject.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape) != (1, size[1], size[0]):
        raise ValueError("The subject mask must be one mask at Reference B's ORIGINAL width and height, before its crop.")
    mask = mask.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()) or bool((mask > 1).any()):
        raise ValueError("Subject mask values must be finite and between zero and one.")
    if not bool((mask > 0).any()):
        raise ValueError("The subject mask is empty. Select a subject or turn masking off.")
    return mask


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def store_mask(name, source, mask):
    mask = validate_mask(mask, source.size)
    pixels = (mask[0].numpy() * 255).round().astype(np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")
    data = encoded.getvalue()
    token = "donutmask:" + hashlib.sha256(data).hexdigest()
    path = _mask_path(token)
    if not path.exists():
        _atomic_write(path, data)
    return {"version": 1, "image": name, "source": fingerprint(source), "mask": token,
            "width": source.width, "height": source.height}


def load_mask(record, name, source):
    if (not isinstance(record, dict) or record.get("version") != 1 or record.get("image") != name
            or record.get("source") != fingerprint(source)
            or (record.get("width"), record.get("height")) != source.size):
        raise ValueError("This subject mask belongs to another version of B. Run Auto select or paint a new mask.")
    path = _mask_path(record.get("mask"))
    if not path.is_file():
        raise FileNotFoundError("Saved subject mask is missing. Copy user/donut/edit_subject_masks/ or run Auto select again.")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["mask"][10:]:
        raise ValueError("Saved subject mask was modified; select the subject again.")
    with Image.open(io.BytesIO(data)) as image:
        if image.size != source.size:
            raise ValueError("Saved subject mask dimensions do not match B.")
        mask = torch.from_numpy(np.asarray(image.convert("L")).astype(np.float32) / 255).unsqueeze(0)
    return validate_mask(mask, source.size)


def _model_signature(name):
    try:
        path = folder_paths.get_full_path("background_removal", name)
    except KeyError:
        path = None
    if not path or not Path(path).is_file():
        raise FileNotFoundError(
            f"Smart Mask: install {name!r} in models/background_removal/ using the native "
            "Comfy-Org BiRefNet package. Saved/manual/external masks need no model. "
            "No model is downloaded automatically."
        )
    stat = Path(path).stat()
    return str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns


def _model_available(name):
    try:
        path = folder_paths.get_full_path("checkpoints" if name == PROMPT_MODEL else "background_removal", name)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(path and Path(path).is_file())


def _native(node_id, **kwargs):
    cls = nodes.NODE_CLASS_MAPPINGS[node_id]
    result = getattr(cls(), getattr(cls, "FUNCTION", "execute"))(**kwargs)
    if isinstance(result, dict):
        result = result["result"]
    elif hasattr(result, "result"):
        result = result.result
    return result[0]


def prompt_mask_inputs():
    return {
        "mask_b_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
        "mask_b_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
    }


def prompt_mask(name, source, prompt, threshold=0.5, force=False):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a mask prompt, such as 'hat' or 'jacket'.")
    if len(prompt) > 2048 or not 0.01 <= float(threshold) <= 1:
        raise ValueError("Invalid mask prompt or detection threshold.")
    if "SAM3_Detect" not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("Prompt selection requires native SAM3 support. Update ComfyUI and restart.")
    path = folder_paths.get_full_path("checkpoints", PROMPT_MODEL)
    if not path or not Path(path).is_file():
        raise FileNotFoundError("Prompt selection needs " + PROMPT_MODEL + ". Run Download missing or the standalone installer.")
    stat = Path(path).stat()
    key = ["sam31-prompt-v1", fingerprint(source), prompt.strip(), float(threshold), str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns]
    cache = _root() / "cache" / (hashlib.sha256(json.dumps(key).encode()).hexdigest() + ".json")
    if not force and cache.is_file():
        try:
            record = json.loads(cache.read_text())
            record["image"] = name
            return record, load_mask(record, name, source)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    pixels = torch.from_numpy(np.asarray(source.convert("RGB")).astype(np.float32) / 255).unsqueeze(0)
    with torch.inference_mode():
        model, clip, _ = nodes.NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]().load_checkpoint(PROMPT_MODEL)
        conditioning = _native("CLIPTextEncode", clip=clip, text=prompt.strip())
        mask = _native("SAM3_Detect", model=model, image=pixels, conditioning=conditioning,
                       threshold=float(threshold), refine_iterations=2, individual_masks=False)
    if not bool((mask > 0).any()):
        raise ValueError("No matching subject found. Try a simpler mask prompt or lower the detection threshold.")
    record = store_mask(name, source, mask)
    record.update(prompt=prompt.strip(), threshold=float(threshold))
    _atomic_write(cache, json.dumps(record).encode())
    return record, load_mask(record, name, source)


def auto_mask(name, source, model_name=MODEL_NAME, force=False):
    missing = [node_id for node_id in NATIVE_NODES if node_id not in nodes.NODE_CLASS_MAPPINGS]
    if missing:
        raise RuntimeError("Smart Mask needs native BiRefNet support. Update ComfyUI and restart, "
                           "or use a saved/manual/external mask. Missing: " + ", ".join(missing))
    signature = _model_signature(model_name)
    cache_key = hashlib.sha256(json.dumps(["native-birefnet-v1", fingerprint(source), signature]).encode()).hexdigest()
    cache_path = _root() / "cache" / (cache_key + ".json")
    if not force and cache_path.is_file():
        try:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            # The same pixels may have a new reference ID; mask/source stay valid.
            record["image"] = name
            return record, load_mask(record, name, source)
        except (OSError, ValueError, TypeError, KeyError):
            pass  # A missing/corrupt cache is regenerated, never used as full B.
    pixels = torch.from_numpy(np.asarray(source.convert("RGB")).astype(np.float32) / 255).unsqueeze(0)
    with torch.inference_mode():
        model = _native("LoadBackgroundRemovalModel", bg_removal_name=model_name)
        mask = _native("RemoveBackground", bg_removal_model=model, image=pixels)
    record = store_mask(name, source, mask)
    _atomic_write(cache_path, json.dumps(record).encode())
    return record, load_mask(record, name, source)


def composite(source, mask, grow=0, feather=0, background="Neutral gray"):
    if isinstance(grow, bool) or not isinstance(grow, int) or not -64 <= grow <= 64:
        raise ValueError("Subject mask grow must be an integer between -64 and 64 source pixels.")
    if isinstance(feather, bool) or not isinstance(feather, int) or not 0 <= feather <= 64:
        raise ValueError("Subject mask feather must be an integer between 0 and 64 source pixels.")
    if background not in BACKGROUNDS:
        raise ValueError("Unknown subject-mask background.")
    alpha = validate_mask(mask, source.size)[0].numpy()
    if grow or feather:
        matte = Image.fromarray((alpha * 255).round().astype(np.uint8))
        if grow:
            matte = matte.filter((ImageFilter.MaxFilter if grow > 0 else ImageFilter.MinFilter)(abs(grow) * 2 + 1))
        if feather:
            matte = matte.filter(ImageFilter.GaussianBlur(feather))
        alpha = np.asarray(matte).astype(np.float32) / 255
    if not np.any(alpha > 0):
        raise ValueError("Grow/shrink removed the entire subject mask; reduce the shrink amount.")
    rgb = np.asarray(source.convert("RGB")).astype(np.float32)
    result = rgb * alpha[..., None] + (255 * BACKGROUNDS[background]) * (1 - alpha[..., None])
    return Image.fromarray(result.clip(0, 255).round().astype(np.uint8))


class DonutSubjectMaskStudio(_Base):
    @classmethod
    def INPUT_TYPES(cls):
        result = deepcopy(_Base.INPUT_TYPES())
        try:
            files = folder_paths.get_filename_list("background_removal")
        except KeyError:
            files = []
        # Append only. Preserve all existing positional widgets, output slots,
        # grounding controls and inpaint wiring in old V4 and API workflows.
        result.setdefault("optional", {}).update({
            "mask_b_mode": (MODES, {"default": "Off"}),
            "mask_b_model": (list(dict.fromkeys([MODEL_NAME, *files])), {"default": MODEL_NAME}),
            "mask_b_data": ("STRING", {"default": "", "dynamicPrompts": False}),
            "mask_b_grow": ("INT", {"default": 0, "min": -64, "max": 64}),
            "mask_b_feather": ("INT", {"default": 0, "min": 0, "max": 64}),
            "mask_b_background": (list(BACKGROUNDS), {"default": "Neutral gray"}),
            "mask_b": ("MASK", {"lazy": True, "tooltip": "External mask in ORIGINAL B coordinates. White keeps the subject. Use External mask mode. Invert Load Image's alpha MASK if necessary."}),
        })
        return result

    def check_lazy_status(self, enabled=False, model=None, use_reference_b=False, mask_b_mode="Off", **kwargs):
        needed = super().check_lazy_status(enabled=enabled, model=model, **kwargs)
        if enabled and use_reference_b and mask_b_mode == "External mask" and "mask_b" in kwargs and kwargs["mask_b"] is None:
            needed.append("mask_b")
        return needed

    @classmethod
    def IS_CHANGED(cls, enabled=False, image_a="", image_b="", use_reference_b=False,
                   mask_b_mode="Off", mask_b_data="", mask_b_model=MODEL_NAME, **kwargs):
        original = _Base.IS_CHANGED(enabled=enabled, image_a=image_a, image_b=image_b, use_reference_b=use_reference_b, **kwargs)
        if not enabled or not use_reference_b or mask_b_mode == "Off":
            return original
        if mask_b_mode == "Prompt selection":
            try:
                path = folder_paths.get_full_path("checkpoints", PROMPT_MODEL)
                stat = Path(path).stat()
                return original, path, stat.st_size, stat.st_mtime_ns
            except (ValueError, TypeError, OSError):
                return original, "missing prompt-selection model"
        if mask_b_mode == "Auto subject":
            try:
                return original, _model_signature(mask_b_model)
            except (ValueError, OSError):
                return original, "missing model"
        if mask_b_mode == "Saved mask":
            try:
                path = _mask_path(json.loads(mask_b_data)["mask"])
                return original, hashlib.sha256(path.read_bytes()).hexdigest()
            except (ValueError, KeyError, TypeError, OSError):
                return original, "missing mask"
        return original

    def prepare(self, *args, mask_b_mode="Off", mask_b_model=MODEL_NAME, mask_b_data="",
                mask_b_grow=0, mask_b_feather=0, mask_b_background="Neutral gray", mask_b=None,
                mask_b_prompt="", mask_b_threshold=0.5, **kwargs):
        result = super().prepare(*args, **kwargs)
        if result[1] is None or mask_b_mode == "Off":
            return result
        if mask_b_mode not in MODES:
            raise ValueError("Unknown Reference B mask mode.")
        bound = inspect.signature(_Base.prepare).bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        name = values["image_b"]
        source = _source(name)
        if mask_b_mode == "Auto subject":
            record, mask = auto_mask(name, source, mask_b_model)
        elif mask_b_mode == "Prompt selection":
            record, mask = prompt_mask(name, source, mask_b_prompt, mask_b_threshold)
        elif mask_b_mode == "Saved mask":
            try:
                record = json.loads(mask_b_data)
            except (ValueError, TypeError):
                raise ValueError("No valid saved B mask. Run Auto select or paint a mask first.") from None
            mask = load_mask(record, name, source)
        else:
            if mask_b is None:
                raise ValueError("Connect a MASK to Edit Studio's mask_b input, or choose Auto subject.")
            record = store_mask(name, source, mask_b)
            mask = load_mask(record, name, source)
        isolated = composite(source, mask, mask_b_grow, mask_b_feather, mask_b_background)
        # Neutralize BEFORE crop/resampling as well as before BOTH Krea encoders.
        # Reuse precisely the same crop geometry as the existing B path.
        masked_b = _crop_reference(isolated, tuple(result[3:5]), values["crop_b_x"], values["crop_b_y"])
        return {"result": (result[0], masked_b, *result[2:]), "ui": {"donut_subject_mask": [record]}}


class DonutSubjectMaskPreview:
    """Output-only preview job queued by Edit Studio; does not run generation."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_b": ("STRING", {"default": ""}),
            "model_name": ("STRING", {"default": MODEL_NAME}),
            "request_id": ("STRING", {"default": ""}),
        }, "optional": prompt_mask_inputs()}
    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "donut/editing"
    DESCRIPTION = "Edit Studio's queued subject-mask preview. No Krea model or full generation is executed."

    def preview(self, image_b, model_name, request_id, mask_b_prompt="", mask_b_threshold=0.5):
        if model_name == PROMPT_MODEL:
            record, _ = prompt_mask(image_b, _source(image_b), mask_b_prompt, mask_b_threshold, force=True)
        else:
            record, _ = auto_mask(image_b, _source(image_b), model_name, force=True)
        return {"result": (), "ui": {"donut_subject_mask": [record]}}


@PromptServer.instance.routes.get("/donut/edit-studio/subject-mask/{mask_id}")
async def view_subject_mask(request):
    try:
        path = _mask_path("donutmask:" + request.match_info["mask_id"])
    except ValueError:
        raise web.HTTPNotFound() from None
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Content-Type": "image/png", "X-Content-Type-Options": "nosniff"})


@PromptServer.instance.routes.get("/donut/edit-studio/subject-mask-model")
async def subject_mask_model_status(request):
    name = request.query.get("name", MODEL_NAME)
    if not isinstance(name, str) or not name or len(name) > 1024:
        raise web.HTTPBadRequest(text="Invalid background-removal model name.")
    return web.json_response({"name": name, "installed": _model_available(name)})


@PromptServer.instance.routes.post("/donut/edit-studio/subject-mask")
async def upload_subject_mask(request):
    post = await request.post()
    name, file = post.get("reference"), post.get("mask")
    # This new HTTP endpoint accepts only Edit Studio's content-addressed images,
    # not arbitrary input paths. Legacy references can first be uploaded to B.
    if not isinstance(name, str) or not re.fullmatch(r"donutref:[a-f0-9]{64}", name) or not isinstance(file, web.FileField):
        raise web.HTTPBadRequest(text="Upload Reference B in Edit Studio and provide a mask image.")
    try:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Mask upload is too large.")
        source = _source(name)
        with Image.open(io.BytesIO(data)) as image:
            if image.size != source.size:
                raise ValueError("The mask must match the original Reference B dimensions.")
            mask = torch.from_numpy(np.asarray(image.convert("L")).astype(np.float32) / 255).unsqueeze(0)
        return web.json_response(store_mask(name, source, mask))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise web.HTTPBadRequest(text=str(error)) from None


NODE_CLASS_MAPPINGS = {"DonutEditStudio": DonutSubjectMaskStudio, "DonutSubjectMaskPreview": DonutSubjectMaskPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutEditStudio": "Donut Edit Studio", "DonutSubjectMaskPreview": "Donut Subject Mask Preview"}

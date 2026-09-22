"""Final-image inference for a replaceable Donut Tone Lab v4 model-only JSON."""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import folder_paths

from .donut_tone_engine import SCHEMA, ALGORITHM, analyze_rgba, validate_export, predict

MODEL_FOLDER = "donut_tone"
NO_MODEL = "None"
MAX_MODEL_BYTES = 1_048_576
BUNDLED_MODEL_DIR = Path(__file__).resolve().parent / "models" / MODEL_FOLDER


def _register_folder():
    # Preserve configured extra_model_paths and add only this model category.
    if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(MODEL_FOLDER, str(Path(folder_paths.models_dir) / MODEL_FOLDER))
    paths, extensions = folder_paths.folder_names_and_paths[MODEL_FOLDER]
    paths = list(paths)
    # Bundled weights are a last-priority, read-only search root. Do not copy
    # over user checkpoints or replace saved selections, including None.
    bundled = str(BUNDLED_MODEL_DIR)
    if BUNDLED_MODEL_DIR.is_dir() and bundled not in paths:
        paths.append(bundled)
    folder_paths.folder_names_and_paths[MODEL_FOLDER] = (paths, set(extensions) | {".json"})


def _model_path(name):
    _register_folder()
    if not isinstance(name, str) or name == NO_MODEL or not name:
        raise ValueError("Tone Lab is enabled: select the bundled checkpoint or a v4 'Export model only' JSON in ComfyUI/models/donut_tone")
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name or rel.suffix.lower() != ".json":
        raise ValueError("Tone Lab model must be a relative JSON filename inside models/donut_tone")
    for root in folder_paths.get_folder_paths(MODEL_FOLDER):
        root = Path(root).resolve()
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Tone Lab model cannot escape its configured model directory")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Tone Lab model not found: {name}. Copy the exported JSON to models/donut_tone and refresh ComfyUI.")


def _read_model(name):
    path = _model_path(name)
    with path.open("rb") as f:
        raw = f.read(MAX_MODEL_BYTES + 1)
    if len(raw) > MAX_MODEL_BYTES:
        raise ValueError("Tone Lab model exceeds 1 MiB; use 'Export model only', not a session")
    return raw


@lru_cache(maxsize=8)
def _decode_model(raw):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    try:
        return validate_export(json.loads(raw.decode("utf-8"), parse_constant=reject_constant))
    except (UnicodeError, json.JSONDecodeError, RecursionError, TypeError, KeyError) as e:
        raise ValueError("Invalid Tone Lab model-only JSON") from e


def analysis_proxy(frame):
    """Deterministic sRGB proxy approximating v4's Chromium image/canvas path.

    The common grid-aligned sizes match the tested CPU PNG -> canvas path:
    integer 2x2 mip levels, then four-fractional-bit bilinear interpolation.
    Browser rasterization is not part of the v4 export schema; odd mip sizes,
    colour-profile conversion and other browsers require real-image comparison.
    Alpha is premultiplied for resampling and preserved in the output image.
    """
    h, w, channels = frame.shape
    scale = min(1.0, 256 / max(h, w))
    dh, dw = max(1, math.floor(h * scale + .5)), max(1, math.floor(w * scale + .5))
    pixels = (frame.detach().float().clamp(0, 1) * 255).round().to(device="cpu", dtype=torch.uint8).numpy()
    if channels == 3:
        pixels = np.concatenate((pixels, np.full((h, w, 1), 255, dtype=np.uint8)), axis=2)
    else:
        pixels = pixels.copy()
        pixels[..., :3] = (pixels[..., :3].astype(np.uint16) * pixels[..., 3:] + 127) // 255
    levels = max(0, math.floor(math.log2(max(w / dw, h / dh))))
    for _ in range(levels):
        ih, iw = pixels.shape[:2]
        yy = np.arange(max(1, ih // 2)) * 2
        xx = np.arange(max(1, iw // 2)) * 2
        reduced = np.zeros((len(yy), len(xx), 4), dtype=np.uint16)
        for oy in (0, 1):
            for ox in (0, 1):
                reduced += pixels[np.minimum(yy + oy, ih - 1)[:, None], np.minimum(xx + ox, iw - 1)[None, :]]
        pixels = (reduced // 4).astype(np.uint8)
    ih, iw = pixels.shape[:2]
    sy = (np.arange(dh) + .5) * ih / dh - .5
    sx = (np.arange(dw) + .5) * iw / dw - .5
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    fy = np.floor((sy - y0) * 16).astype(np.int64)
    fx = np.floor((sx - x0) * 16).astype(np.int64)
    result = np.zeros((dh, dw, 4), dtype=np.int64)
    for oy, wy in ((0, 16 - fy), (1, fy)):
        for ox, wx in ((0, 16 - fx), (1, fx)):
            sample = pixels[np.clip(y0 + oy, 0, ih - 1)[:, None], np.clip(x0 + ox, 0, iw - 1)[None, :]]
            result += sample.astype(np.int64) * wy[:, None, None] * wx[None, :, None]
    result //= 256
    if channels == 4:
        alpha = result[..., 3:]
        result[..., :3] = np.rint(np.divide(result[..., :3] * 255, alpha,
                                           out=np.zeros((dh, dw, 3)), where=alpha > 0)).astype(np.int64)
    return np.clip(result, 0, 255).astype(np.uint8)


class DonutToneLab:
    @classmethod
    def INPUT_TYPES(cls):
        _register_folder()
        names = [NO_MODEL] + sorted(n for n in folder_paths.get_filename_list(MODEL_FOLDER) if n.lower().endswith(".json"))
        return {"required": {
            "image": ("IMAGE",),
            "enabled": ("BOOLEAN", {"default": False, "tooltip": "Off is exact passthrough; no model is loaded."}),
            "model_name": (names, {"default": NO_MODEL, "tooltip": "Select bundled donut-tone-v4-r12.json or your Tone Lab v4 Export model only JSON."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": .05,
                                     "tooltip": "1 uses the trained prediction. 0 is exact passthrough. Intermediate values scale log gamma/gain."}),
            "apply_to_edits": ("BOOLEAN", {"default": False,
                                             "tooltip": "Off skips correction when V5 Editing is on. On globally grades preserved surroundings too."}),
        }, "optional": {"edit_mode": ("BOOLEAN", {"default": False, "forceInput": True})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "apply"
    CATEGORY = "donut/image"
    DESCRIPTION = "Apply one shared Tone Lab v4 feature model after all generation/compositing. No training or image histories."

    @classmethod
    def IS_CHANGED(cls, image=None, enabled=False, model_name=NO_MODEL, strength=1.0, apply_to_edits=False, edit_mode=False):
        if not enabled or strength == 0 or (edit_mode and not apply_to_edits):
            return "passthrough"
        # Hash CONTENT, so replacing weights at the same filename invalidates cache.
        return hashlib.sha256(_read_model(model_name)).hexdigest()

    def apply(self, image, enabled=False, model_name=NO_MODEL, strength=1.0, apply_to_edits=False, edit_mode=False):
        def finish(output, report):
            text = json.dumps(report, ensure_ascii=False, allow_nan=False)
            return {"ui": {"text": [text]}, "result": (output, text)}

        if not enabled or strength == 0 or (edit_mode and not apply_to_edits):
            reason = "disabled" if not enabled else "strength_zero" if strength == 0 else "editing_protected"
            return finish(image, {"applied": False, "reason": reason})
        if not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Tone Lab strength must be finite and between 0 and 1")
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] not in (3, 4) or not image.is_floating_point():
            raise ValueError("Tone Lab expects a floating point BHWC RGB/RGBA IMAGE tensor")
        if min(image.shape) < 1 or not torch.isfinite(image).all().item():
            raise ValueError("Tone Lab received an empty image or non-finite pixels")
        raw = _read_model(model_name)
        model = _decode_model(raw)
        rows, output = [], None
        with torch.no_grad():
            for i, frame in enumerate(image):
                stats = analyze_rgba(analysis_proxy(frame))
                p = predict(stats["features"], model) if stats["sampleCount"] >= 16 else {
                    "gamma": 1.0, "gain": 1.0, "noop": True, "reason": "fewer_than_16_opaque_samples"}
                gamma, gain = p["gamma"] ** strength, p["gain"] ** strength
                applied = gamma != 1 or gain != 1
                if applied:
                    if output is None:
                        output = image.clone()
                    # Do not quantize the full image. Keep alpha and dtype/device.
                    rgb = frame[..., :3].float() if frame.dtype in (torch.float16, torch.bfloat16) else frame[..., :3]
                    output[i, ..., :3] = (rgb.clamp(0, 1).pow(gamma) * gain).clamp(0, 1).to(image.dtype)
                rows.append({"index": i, **p, "applied": applied, "effectiveGamma": gamma,
                             "effectiveGain": gain, "gammaSlider": -100 * math.log(gamma) / math.log(3),
                             "gainPercent": (gain - 1) * 100, "samples": stats["sampleCount"]})
        return finish(image if output is None else output, {
            "applied": output is not None, "schema": SCHEMA, "algorithm": ALGORITHM,
            "model": model_name, "sha256": hashlib.sha256(raw).hexdigest(),
            "revision": model["revision"], "strength": strength, "frames": rows,
            "note": "noOpScore is uncalibrated; coverage is the fraction of features outside 3 training standard deviations.",
        })


NODE_CLASS_MAPPINGS = {"DonutToneLab": DonutToneLab}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutToneLab": "Donut Tone Lab · Learned Auto Tone"}

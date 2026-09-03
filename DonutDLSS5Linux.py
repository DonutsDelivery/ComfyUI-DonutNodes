"""Linux-oriented DLSS/NGX bridge for ComfyUI-DonutNodes.

This module deliberately does not bundle NVIDIA binaries. It wraps an external
renderer supplied by the user and supports native Linux, Wine, and Proton
launch modes. A successful subprocess exit alone is not treated as proof that
DLSS initialized: callers can require a log marker and an optional SHA-256 pin.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


_DEFAULT_MARKER = "NGX"


def _tensor_to_png(image: torch.Tensor, path: Path) -> None:
    if image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError("Expected ComfyUI IMAGE tensor shaped [B,H,W,C]")
    if image.shape[0] != 1:
        raise ValueError("DLSS bridge currently accepts a batch size of 1")
    arr = image[0].detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _png_to_tensor(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise RuntimeError(f"Renderer did not create output image: {path}")
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_command(mode: str, renderer: Path, template: str, input_path: Path,
                   output_path: Path, scale: float, proton_path: str) -> list[str]:
    rendered = template.format(
        renderer=str(renderer),
        input=str(input_path),
        output=str(output_path),
        scale=f"{scale:g}",
    )
    args = shlex.split(rendered)
    if not args:
        raise ValueError("Argument template produced an empty command")

    if mode == "native":
        return args
    if mode == "wine":
        return ["wine", *args]
    if mode == "proton":
        proton = Path(os.path.expanduser(proton_path)).resolve()
        if not proton.is_file():
            raise FileNotFoundError(f"Proton launcher not found: {proton}")
        return [str(proton), "run", *args]
    raise ValueError(f"Unknown execution mode: {mode}")


class DonutDLSS5Linux:
    """Run an external DLSS-capable renderer and return its output as IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "renderer_path": ("STRING", {"default": ""}),
                "mode": (["native", "wine", "proton"], {"default": "wine"}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "arg_template": (
                    "STRING",
                    {"default": '"{renderer}" --input "{input}" --output "{output}" --scale {scale}'},
                ),
                "required_log_marker": ("STRING", {"default": _DEFAULT_MARKER}),
            },
            "optional": {
                "renderer_sha256": ("STRING", {"default": ""}),
                "proton_path": ("STRING", {"default": ""}),
                "timeout_seconds": ("INT", {"default": 120, "min": 1, "max": 3600}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "renderer_log")
    FUNCTION = "run"
    CATEGORY = "Donut/experimental"

    def run(
        self,
        image,
        renderer_path,
        mode,
        scale,
        arg_template,
        required_log_marker,
        renderer_sha256="",
        proton_path="",
        timeout_seconds=120,
    ):
        renderer = Path(os.path.expanduser(renderer_path)).resolve()
        if not renderer.is_file():
            raise FileNotFoundError(f"Renderer not found: {renderer}")

        if renderer_sha256.strip():
            actual = _sha256(renderer)
            expected = renderer_sha256.strip().lower()
            if actual.lower() != expected:
                raise RuntimeError(
                    "Renderer SHA-256 mismatch. "
                    f"Expected {expected}, got {actual}."
                )

        input_h, input_w = int(image.shape[1]), int(image.shape[2])
        expected_w = round(input_w * float(scale))
        expected_h = round(input_h * float(scale))

        with tempfile.TemporaryDirectory(prefix="donut_dlss5_") as td:
            td_path = Path(td)
            input_path = td_path / "input.png"
            output_path = td_path / "output.png"
            _tensor_to_png(image, input_path)

            command = _build_command(
                mode,
                renderer,
                arg_template,
                input_path,
                output_path,
                float(scale),
                proton_path,
            )
            proc = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(timeout_seconds),
                check=False,
                shell=False,
            )
            log = proc.stdout or ""

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Renderer exited with status {proc.returncode}.\n{log[-8000:]}"
                )

            marker = required_log_marker.strip()
            if marker and marker.lower() not in log.lower():
                raise RuntimeError(
                    "Renderer completed, but required DLSS/NGX log marker was not found: "
                    f"{marker!r}. Refusing to accept an unverified fallback.\n{log[-8000:]}"
                )

            result = _png_to_tensor(output_path)
            out_h, out_w = int(result.shape[1]), int(result.shape[2])
            if (out_w, out_h) != (expected_w, expected_h):
                raise RuntimeError(
                    "Renderer output dimensions do not match requested scale: "
                    f"expected {expected_w}x{expected_h}, got {out_w}x{out_h}."
                )

            return (result, log[-16000:])


NODE_CLASS_MAPPINGS = {
    "DonutDLSS5Linux": DonutDLSS5Linux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutDLSS5Linux": "Donut DLSS5 Linux Bridge (Experimental)",
}

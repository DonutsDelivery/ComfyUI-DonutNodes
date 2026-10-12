"""Experimental DLSS 5 neural-rendering nodes for Linux via Proton/Wine."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from .donut_dlss5_linux import (
    MODEL_PRESETS,
    NR_PRESETS,
    NR_STYLES,
    UPSCALE_MODES,
    Dlss5LinuxError,
    Dlss5Session,
    RuntimeLayout,
    build_native_settings,
    format_status,
    probe_host,
    resolve_launch_config,
    resolve_output_size,
)


def _check_interrupted() -> None:
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        return


def _resize_rgba(frame: torch.Tensor, width: int, height: int) -> np.ndarray:
    rgb = frame[..., :3].detach().float().permute(2, 0, 1).unsqueeze(0)
    try:
        resized = F.interpolate(
            rgb,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        resized = F.interpolate(
            rgb,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
    rgb8 = (
        resized.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate((rgb8, alpha), axis=2))


class DonutDLSS5LinuxStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime_dir": (
                    "STRING",
                    {
                        "default": os.environ.get("DONUT_DLSS5_RUNTIME", ""),
                        "multiline": False,
                        "tooltip": "Merserk-compatible runtime root or its bin/runtime directory.",
                    },
                ),
                "backend": (["Auto", "Proton", "Wine", "Native"],),
                "launcher": (
                    "STRING",
                    {
                        "default": os.environ.get("DONUT_DLSS5_PROTON", ""),
                        "multiline": False,
                        "tooltip": "Optional Proton script or wine64 executable.",
                    },
                ),
                "prefix": (
                    "STRING",
                    {
                        "default": os.environ.get("DONUT_DLSS5_PREFIX", ""),
                        "multiline": False,
                        "tooltip": "Proton compat-data directory or Wine prefix.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("runtime_report",)
    FUNCTION = "inspect"
    CATEGORY = "image/Donut/DLSS 5 (Linux experimental)"
    DESCRIPTION = (
        "Checks runtime files, launcher discovery, NVIDIA visibility and Vulkan visibility. "
        "It does not claim DLSS works until an upscale node verifies feature 18."
    )

    def inspect(self, runtime_dir: str, backend: str, launcher: str, prefix: str):
        layout = RuntimeLayout.discover(runtime_dir)
        launch = resolve_launch_config(backend, launcher, prefix, layout.worker)
        return (format_status(layout, launch, probe_host()),)


class DonutDLSS5LinuxUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "runtime_dir": (
                    "STRING",
                    {
                        "default": os.environ.get("DONUT_DLSS5_RUNTIME", ""),
                        "multiline": False,
                    },
                ),
                "backend": (["Auto", "Proton", "Wine", "Native"],),
                "launcher": (
                    "STRING",
                    {
                        "default": os.environ.get("DONUT_DLSS5_PROTON", ""),
                        "multiline": False,
                    },
                ),
                "prefix": (
                    "STRING",
                    {
                        "default": os.environ.get("DONUT_DLSS5_PREFIX", ""),
                        "multiline": False,
                    },
                ),
                "upscaling_mode": (list(UPSCALE_MODES),),
                "nr_preset": (list(NR_PRESETS),),
                "nr_style": (list(NR_STYLES),),
                "model_preset": (list(MODEL_PRESETS), {"default": "M"}),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "local_tone_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "local_structure_strength": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "skin_structure_strength": (
                    "FLOAT",
                    {"default": 2.0, "min": -1.0, "max": 2.0, "step": 0.05},
                ),
                "automatic_mask": ("BOOLEAN", {"default": True}),
                "warmup_frames": ("INT", {"default": 0, "min": 0, "max": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "runtime_report")
    FUNCTION = "upscale"
    CATEGORY = "image/Donut/DLSS 5 (Linux experimental)"
    DESCRIPTION = (
        "Runs a user-supplied DLSS 5 feature-18 worker through Proton/Wine. "
        "The node fails unless ReShade logs prove signed DLSSNR initialization, feature-18 "
        "creation and successful feature-18 evaluation. Batch items are independent stills."
    )

    def upscale(
        self,
        image: torch.Tensor,
        runtime_dir: str,
        backend: str,
        launcher: str,
        prefix: str,
        upscaling_mode: str,
        nr_preset: str,
        nr_style: str,
        model_preset: str,
        nr_intensity: float,
        local_tone_strength: float,
        local_structure_strength: float,
        skin_structure_strength: float,
        automatic_mask: bool,
        warmup_frames: int,
    ):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise Dlss5LinuxError(
                f"Expected ComfyUI IMAGE [batch,height,width,channels], got {tuple(image.shape)}."
            )
        batch, input_height, input_width, _channels = image.shape
        if input_width < 64 or input_height < 64:
            raise Dlss5LinuxError("DLSS requires both input dimensions to be at least 64 pixels.")

        layout = RuntimeLayout.discover(runtime_dir)
        launch = resolve_launch_config(backend, launcher, prefix, layout.worker)
        output_width, output_height, factor, perf_quality, mode_name = resolve_output_size(
            int(input_width), int(input_height), upscaling_mode
        )
        native = build_native_settings(
            nr_preset,
            nr_style,
            model_preset,
            nr_intensity,
            local_tone_strength,
            local_structure_strength,
            skin_structure_strength,
            automatic_mask,
        )
        setup_timeout = float(os.environ.get("DONUT_DLSS5_SETUP_TIMEOUT", "90"))
        frame_timeout = float(os.environ.get("DONUT_DLSS5_FRAME_TIMEOUT", "600"))
        close_timeout = float(os.environ.get("DONUT_DLSS5_CLOSE_TIMEOUT", "60"))

        session = Dlss5Session(
            layout,
            launch,
            input_width=int(input_width),
            input_height=int(input_height),
            output_width=output_width,
            output_height=output_height,
            frame_count=int(batch),
            warmup_frames=warmup_frames,
            perf_quality=perf_quality,
            native_settings=native,
            setup_timeout=setup_timeout,
            frame_timeout=frame_timeout,
            close_timeout=close_timeout,
        )
        outputs: list[torch.Tensor] = []
        try:
            for index in range(int(batch)):
                _check_interrupted()
                rgba = _resize_rgba(
                    image[index], session.render_width, session.render_height
                )
                rendered = session.submit(index, rgba, reset=True)
                outputs.append(torch.from_numpy(rendered[..., :3].copy()).float().div_(255.0))
            combined_log, evidence = session.close()
        except BaseException:
            session.abort()
            raise

        result = torch.stack(outputs, dim=0).to(device=image.device, dtype=image.dtype)
        gpu = probe_host().get("nvidia", "unknown")
        report_lines = [
            "VERIFIED: DLSS 5 neural rendering feature 18 executed successfully.",
            f"Backend: {launch.backend}",
            f"GPU: {gpu}",
            f"Input: {input_width}x{input_height} x {batch} frame(s)",
            f"Render size: {session.render_width}x{session.render_height}",
            f"Output: {output_width}x{output_height} ({factor:g}x, {mode_name})",
            f"Model preset: requested {model_preset}, applied {session.applied_model_preset}",
            f"Worker: {layout.worker}",
            "Verification evidence:",
            *(evidence[-12:] or ["Required feature-18 markers were present."]),
        ]
        del combined_log
        return result, "\n".join(report_lines)


NODE_CLASS_MAPPINGS = {
    "DonutDLSS5LinuxStatus": DonutDLSS5LinuxStatus,
    "DonutDLSS5LinuxUpscale": DonutDLSS5LinuxUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutDLSS5LinuxStatus": "Donut DLSS 5 Linux Runtime Status (Experimental)",
    "DonutDLSS5LinuxUpscale": "Donut DLSS 5 Neural Upscale — Linux (Experimental)",
}

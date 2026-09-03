"""Linux launcher and binary protocol for the experimental DLSS 5 worker.

No NVIDIA, ReShade, RenoDX, Wine, Proton, or worker binary is distributed here.
The module drives a user-supplied Merserk-compatible version-4 worker. Windows
PE workers are launched through Proton or Wine; a future ELF worker can use the
same protocol through the Native backend.
"""

from __future__ import annotations

import os
import platform
import re
import select
import shutil
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np

VIDEO_MAGIC = 0x34563544
SETUP_MAGIC = 0x34505553
FRAME_MAGIC = 0x314D5246
OUT_MAGIC = 0x3154554F
VIDEO_HEADER = struct.Struct("<14I4f")
SETUP_RESPONSE = struct.Struct("<12I")
FRAME_HEADER = struct.Struct("<4Iq")
RESULT_HEADER = struct.Struct("<5Iq")

UPSCALE_MODES: dict[str, tuple[float, int, str]] = {
    "1.0x (DLAA / native)": (1.0, 5, "DLAA"),
    "1.5x (Quality)": (1.5, 2, "Quality"),
    "1.724x (Balanced)": (1.724, 1, "Balanced"),
    "2.0x (Performance)": (2.0, 0, "Performance"),
    "3.0x (Ultra Performance)": (3.0, 3, "Ultra Performance"),
}
NR_PRESETS = {"Default": 0, "Preset #1": 1, "Preset #2": 2, "Preset #3": 3}
NR_STYLES = {"Default": 0, "Natural": 1, "Cinematic": 2}
MODEL_PRESETS = {"Default": 0, "J": 10, "K": 11, "L": 12, "M": 13}

_REQUIRED_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "signed DLSSNR D3D12 runtime initialization",
        re.compile(
            r"signed\s+DLSSNR(?:\s+[\w.\-]+)?\s+D3D12\s+runtime\s+initialized",
            re.I,
        ),
    ),
    ("feature 18 creation", re.compile(r"feature\s+18\s+created", re.I)),
    (
        "feature 18 evaluation",
        re.compile(r"feature\s+18.*(?:evaluation|evaluate).*succeed", re.I),
    ),
)


class Dlss5LinuxError(RuntimeError):
    """Actionable launcher, protocol, or verification failure."""


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    root: Path
    host: Path
    dlss: Path
    worker: Path
    dxgi: Path
    addon: Path
    neural_runtime: Path
    super_resolution_runtime: Path
    reshade_log: Path

    @classmethod
    def discover(cls, raw_path: str | os.PathLike[str] | None) -> "RuntimeLayout":
        value = str(raw_path or os.environ.get("DONUT_DLSS5_RUNTIME", "")).strip()
        supplied = Path(
            value
            or Path.home()
            / ".local"
            / "share"
            / "donutnodes"
            / "dlss5-runtime"
        ).expanduser().resolve()
        candidates = [supplied, supplied / "bin" / "runtime", supplied / "runtime"]
        if supplied.name.lower() == "host":
            candidates.insert(0, supplied.parent)

        root = next(
            (candidate for candidate in candidates if (candidate / "host" / "nvngx.dll").is_file()),
            None,
        )
        if root is None:
            checked = "\n  ".join(
                str(candidate / "host" / "nvngx.dll") for candidate in candidates
            )
            raise Dlss5LinuxError(
                "Could not find a Merserk-compatible DLSS 5 worker. Checked:\n  "
                + checked
            )

        host = root / "host"
        layout = cls(
            root=root,
            host=host,
            dlss=root / "dlss",
            worker=host / "nvngx.dll",
            dxgi=host / "dxgi.dll",
            addon=host / "renodx-dlss5.addon64",
            neural_runtime=host / "nvngx_dlssnr.dll",
            super_resolution_runtime=root / "dlss" / "nvngx_dlss.dll",
            reshade_log=host / "ReShade.log",
        )
        layout.validate()
        return layout

    def validate(self) -> None:
        required = {
            "D3D12 worker": self.worker,
            "ReShade full add-on build": self.dxgi,
            "RenoDX DLSS 5 add-on": self.addon,
            "DLSS neural-rendering runtime": self.neural_runtime,
            "DLSS Super Resolution runtime": self.super_resolution_runtime,
        }
        missing = [
            f"{label}: {path}"
            for label, path in required.items()
            if not path.is_file()
        ]
        if missing:
            raise Dlss5LinuxError(
                "The DLSS 5 runtime bundle is incomplete:\n" + "\n".join(missing)
            )


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    backend: str
    command_prefix: tuple[str, ...]
    env: dict[str, str]
    prefix: Path | None
    worker_kind: str

    def command(self, worker: Path) -> list[str]:
        return [*self.command_prefix, str(worker), "--video"]


def _binary_kind(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            header = stream.read(4)
    except OSError as exc:
        raise Dlss5LinuxError(f"Cannot read worker {path}: {exc}") from exc
    if header.startswith(b"MZ"):
        return "pe"
    if header == b"\x7fELF":
        return "elf"
    return "unknown"


def _find_proton() -> Path | None:
    configured = os.environ.get("DONUT_DLSS5_PROTON", "").strip()
    if configured and Path(configured).expanduser().is_file():
        return Path(configured).expanduser().resolve()
    direct = shutil.which("proton")
    if direct:
        return Path(direct).resolve()

    matches: list[Path] = []
    for root in (
        Path.home() / ".steam" / "root",
        Path.home() / ".local" / "share" / "Steam",
    ):
        matches.extend((root / "compatibilitytools.d").glob("*/proton"))
        matches.extend((root / "steamapps" / "common").glob("Proton*/proton"))
    usable = sorted(
        (path.resolve() for path in matches if path.is_file()), reverse=True
    )
    return usable[0] if usable else None


def _find_wine() -> Path | None:
    configured = os.environ.get("DONUT_DLSS5_WINE", "").strip()
    if configured and Path(configured).expanduser().is_file():
        return Path(configured).expanduser().resolve()
    for name in ("wine64", "wine"):
        executable = shutil.which(name)
        if executable:
            return Path(executable).resolve()
    return None


def _append_dll_overrides(existing: str) -> str:
    values = [item.strip() for item in existing.split(";") if item.strip()]
    names = {item.split("=", 1)[0].lower() for item in values}
    if "dxgi" not in names:
        values.append("dxgi=n,b")
    if "nvapi64" not in names:
        values.append("nvapi64=n,b")
    return ";".join(values)


def _classify_launcher(path: Path) -> str | None:
    name = path.name.lower()
    if "proton" in name:
        return "proton"
    if "wine" in name:
        return "wine"
    return None


def resolve_launch_config(
    backend: str,
    launcher: str,
    prefix: str,
    worker: Path,
) -> LaunchConfig:
    requested = (
        backend or os.environ.get("DONUT_DLSS5_BACKEND", "Auto")
    ).strip().lower()
    if requested not in {"auto", "proton", "wine", "native"}:
        raise Dlss5LinuxError(
            f"Unknown backend {backend!r}; choose Auto, Proton, Wine, or Native."
        )

    worker_kind = _binary_kind(worker)
    explicit = Path(launcher).expanduser().resolve() if launcher.strip() else None
    if explicit is not None and not explicit.is_file():
        raise Dlss5LinuxError(f"The selected launcher does not exist: {explicit}")

    proton = _find_proton()
    wine = _find_wine()
    chosen = requested
    if chosen == "auto":
        inferred = _classify_launcher(explicit) if explicit else None
        if worker_kind == "elf":
            chosen = "native"
        elif inferred:
            chosen = inferred
        elif platform.system() == "Linux" and proton:
            chosen = "proton"
        elif platform.system() == "Linux" and wine:
            chosen = "wine"
        else:
            raise Dlss5LinuxError(
                "The supplied worker is a Windows PE image, but Proton/Wine was not "
                "found. Set launcher or DONUT_DLSS5_PROTON/DONUT_DLSS5_WINE."
            )

    env = os.environ.copy()
    env["DXVK_ENABLE_NVAPI"] = "1"
    env["PROTON_FORCE_NVAPI"] = "1"
    env.pop("PROTON_HIDE_NVIDIA_GPU", None)
    env["WINEDLLOVERRIDES"] = _append_dll_overrides(
        env.get("WINEDLLOVERRIDES", "")
    )
    prefix_value = prefix.strip() or os.environ.get("DONUT_DLSS5_PREFIX", "").strip()
    prefix_path: Path | None = None

    if chosen == "native":
        if worker_kind != "elf":
            raise Dlss5LinuxError(
                f"Native mode requires an ELF worker; {worker} is {worker_kind}."
            )
        if not os.access(worker, os.X_OK):
            raise Dlss5LinuxError(f"Native worker is not executable: {worker}")
        command_prefix: tuple[str, ...] = ()
    elif chosen == "proton":
        executable = explicit or proton
        if executable is None:
            raise Dlss5LinuxError("Proton mode selected, but no Proton script was found.")
        prefix_path = Path(prefix_value).expanduser().resolve() if prefix_value else (
            Path.home() / ".local" / "share" / "donutnodes" / "dlss5-proton"
        )
        prefix_path.mkdir(parents=True, exist_ok=True)
        env["STEAM_COMPAT_DATA_PATH"] = str(prefix_path)
        if not env.get("STEAM_COMPAT_CLIENT_INSTALL_PATH"):
            for candidate in (
                Path.home() / ".steam" / "root",
                Path.home() / ".local" / "share" / "Steam",
            ):
                if candidate.is_dir():
                    env["STEAM_COMPAT_CLIENT_INSTALL_PATH"] = str(candidate.resolve())
                    break
        command_prefix = (str(executable), "run")
    else:
        executable = explicit or wine
        if executable is None:
            raise Dlss5LinuxError("Wine mode selected, but wine64/wine was not found.")
        prefix_path = Path(prefix_value).expanduser().resolve() if prefix_value else (
            Path.home() / ".local" / "share" / "donutnodes" / "dlss5-wineprefix"
        )
        prefix_path.mkdir(parents=True, exist_ok=True)
        env["WINEPREFIX"] = str(prefix_path)
        command_prefix = (str(executable),)

    return LaunchConfig(
        backend=chosen.capitalize(),
        command_prefix=command_prefix,
        env=env,
        prefix=prefix_path,
        worker_kind=worker_kind,
    )


def resolve_output_size(
    width: int, height: int, mode_label: str
) -> tuple[int, int, float, int, str]:
    try:
        factor, quality, mode_name = UPSCALE_MODES[mode_label]
    except KeyError as exc:
        raise Dlss5LinuxError(f"Unknown upscaling mode: {mode_label!r}") from exc

    def even(value: float) -> int:
        return max(2, int(value / 2.0 + 0.5) * 2)

    output_width, output_height = even(width * factor), even(height * factor)
    if max(output_width, output_height) > 7680 or min(
        output_width, output_height
    ) > 4320:
        raise Dlss5LinuxError(
            f"Requested output {output_width}x{output_height} exceeds 7680x4320."
        )
    return output_width, output_height, factor, quality, mode_name


def build_native_settings(
    nr_preset: str,
    nr_style: str,
    model_preset: str,
    intensity: float,
    local_tone: float,
    local_structure: float,
    skin_structure: float,
    automatic_mask: bool,
) -> dict[str, int | float]:
    for label, value, table in (
        ("NR preset", nr_preset, NR_PRESETS),
        ("NR style", nr_style, NR_STYLES),
        ("model preset", model_preset, MODEL_PRESETS),
    ):
        if value not in table:
            raise Dlss5LinuxError(f"Unknown {label}: {value!r}")
    controls = {
        "intensity": (float(intensity), 0.0, 2.0),
        "local_tone": (float(local_tone), 0.0, 2.0),
        "local_structure": (float(local_structure), 0.0, 2.0),
        "skin_structure": (float(skin_structure), -1.0, 2.0),
    }
    for label, (value, low, high) in controls.items():
        if not low <= value <= high:
            raise Dlss5LinuxError(f"{label} must be between {low:g} and {high:g}.")
    return {
        "profile": 0,
        "preset": NR_PRESETS[nr_preset],
        "style": NR_STYLES[nr_style],
        "auto_mask": int(bool(automatic_mask)),
        "ui_correction": 0,
        "intensity": controls["intensity"][0],
        "local_tone": controls["local_tone"][0],
        "local_structure": controls["local_structure"][0],
        "skin_structure": controls["skin_structure"][0],
        "dlss_model_preset": MODEL_PRESETS[model_preset],
    }


def verify_feature_18(text: str) -> list[str]:
    missing = [
        label for label, pattern in _REQUIRED_MARKERS if pattern.search(text) is None
    ]
    if missing:
        evidence = "\n".join(
            line
            for line in text.splitlines()
            if any(
                token in line.lower()
                for token in (
                    "dlssnr",
                    "feature 18",
                    "neural rendering",
                    "error",
                    "failed",
                )
            )
        )
        raise Dlss5LinuxError(
            "Pixels were returned, but real DLSS 5 feature-18 execution was not "
            "verified. Missing: "
            + ", ".join(missing)
            + ("\nRelevant runtime output:\n" + evidence[-8000:] if evidence else "")
        )
    return [
        line
        for line in text.splitlines()
        if "feature 18" in line.lower() or "signed dlssnr" in line.lower()
    ]


def probe_host() -> dict[str, str]:
    report = {"platform": platform.platform(), "nvidia": "not checked", "vulkan": "not checked"}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            report["nvidia"] = (
                result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["nvidia"] = str(exc)
    else:
        report["nvidia"] = "nvidia-smi not found"

    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo:
        try:
            result = subprocess.run(
                [vulkaninfo, "--summary"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            devices = [
                line.strip()
                for line in result.stdout.splitlines()
                if "deviceName" in line
            ]
            report["vulkan"] = "; ".join(devices) if devices else (
                "available" if result.returncode == 0 else result.stderr.strip()
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["vulkan"] = str(exc)
    else:
        report["vulkan"] = "vulkaninfo not found; install vulkan-tools"
    return report


class _BufferedPipeReader:
    """Timeout-aware pipe reader that retains bytes read past a protocol header."""

    def __init__(self, stream: BinaryIO) -> None:
        self.fd = stream.fileno()
        self._buffer = bytearray()
        self._discarded_prefix = bytearray()

    def _read_more(self, deadline: float, label: str, minimum: int = 1) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Dlss5LinuxError(f"Timed out while reading {label}.")
        readable, _, _ = select.select([self.fd], [], [], remaining)
        if not readable:
            raise Dlss5LinuxError(f"Timed out while reading {label}.")
        block = os.read(self.fd, max(4096, minimum))
        if not block:
            raise EOFError(f"Worker stopped while reading {label}.")
        self._buffer.extend(block)

    def read_exact(self, size: int, timeout: float, label: str) -> bytes:
        deadline = time.monotonic() + timeout
        while len(self._buffer) < size:
            self._read_more(deadline, label, size - len(self._buffer))
        payload = bytes(self._buffer[:size])
        del self._buffer[:size]
        return payload

    def read_struct_with_magic(
        self,
        structure: struct.Struct,
        magic: int,
        timeout: float,
        label: str,
        max_prefix: int = 65536,
    ) -> bytes:
        marker = struct.pack("<I", magic)
        deadline = time.monotonic() + timeout
        while True:
            location = self._buffer.find(marker)
            if location >= 0 and len(self._buffer) >= location + structure.size:
                if location:
                    self._discarded_prefix.extend(self._buffer[:location])
                    del self._buffer[:location]
                payload = bytes(self._buffer[: structure.size])
                del self._buffer[: structure.size]
                return payload
            if len(self._buffer) > max_prefix and location < 0:
                preview = bytes(self._buffer[:200]).decode("utf-8", errors="replace")
                raise Dlss5LinuxError(
                    f"Could not find {label} magic after launcher output: {preview!r}"
                )
            self._read_more(deadline, label, structure.size)

    def discarded_text(self) -> str:
        return bytes(self._discarded_prefix).decode("utf-8", errors="replace").strip()


class _LogCollector:
    def __init__(self, stream: BinaryIO | None, limit: int = 800) -> None:
        self.stream = stream
        self.limit = limit
        self.lines: list[str] = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        if self.stream is None:
            return
        try:
            for raw in iter(self.stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    with self.lock:
                        self.lines.append(line)
                        del self.lines[: max(0, len(self.lines) - self.limit)]
        except (OSError, ValueError):
            pass

    def join(self) -> None:
        if self.thread.ident is not None:
            self.thread.join(timeout=2)

    def text(self) -> str:
        with self.lock:
            return "\n".join(self.lines)


class Dlss5Session:
    """One version-4 DLSS neural-rendering worker stream."""

    def __init__(
        self,
        layout: RuntimeLayout,
        launch: LaunchConfig,
        *,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        frame_count: int,
        warmup_frames: int,
        perf_quality: int,
        native_settings: dict[str, int | float],
        setup_timeout: float = 90.0,
        frame_timeout: float = 600.0,
        close_timeout: float = 60.0,
    ) -> None:
        self.layout, self.launch = layout, launch
        self.output_width, self.output_height = output_width, output_height
        self.frame_timeout, self.close_timeout = frame_timeout, close_timeout
        self.closed = False
        self.started = time.time() - 2.0
        try:
            self.process = subprocess.Popen(
                launch.command(layout.worker),
                cwd=str(layout.host),
                env=launch.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise Dlss5LinuxError(
                f"Could not start {launch.backend} worker: {exc}"
            ) from exc
        self.logs = _LogCollector(self.process.stderr)
        assert self.process.stdin is not None and self.process.stdout is not None
        self.reader = _BufferedPipeReader(self.process.stdout)

        native = native_settings
        header = VIDEO_HEADER.pack(
            VIDEO_MAGIC,
            input_width,
            input_height,
            output_width,
            output_height,
            max(0, warmup_frames),
            max(1, frame_count),
            perf_quality,
            int(native["dlss_model_preset"]),
            int(native["profile"]),
            int(native["preset"]),
            int(native["style"]),
            int(native["auto_mask"]),
            int(native["ui_correction"]),
            float(native["intensity"]),
            float(native["local_tone"]),
            float(native["local_structure"]),
            float(native["skin_structure"]),
        )
        try:
            self.process.stdin.write(header)
            self.process.stdin.flush()
            response = SETUP_RESPONSE.unpack(
                self.reader.read_struct_with_magic(
                    SETUP_RESPONSE,
                    SETUP_MAGIC,
                    setup_timeout,
                    "setup response",
                )
            )
            (
                _magic,
                ok,
                result,
                self.render_width,
                self.render_height,
                negotiated_width,
                negotiated_height,
                self.minimum_width,
                self.minimum_height,
                self.maximum_width,
                self.maximum_height,
                self.applied_model_preset,
            ) = response
            if not ok:
                raise Dlss5LinuxError(
                    f"DLSS setup failed with NGX result 0x{result:08X}.\n{self.logs.text()}"
                )
            if (negotiated_width, negotiated_height) != (
                output_width,
                output_height,
            ):
                raise Dlss5LinuxError(
                    f"Worker negotiated {negotiated_width}x{negotiated_height}; "
                    f"expected {output_width}x{output_height}."
                )
            if min(self.render_width, self.render_height) < 64:
                raise Dlss5LinuxError(
                    f"Invalid render size {self.render_width}x{self.render_height}."
                )
            requested_model = int(native["dlss_model_preset"])
            if self.applied_model_preset != requested_model:
                raise Dlss5LinuxError(
                    f"Worker applied model {self.applied_model_preset}; "
                    f"requested {requested_model}."
                )
            self.zero_motion = np.zeros(
                (self.render_height, self.render_width, 2), dtype=np.float16
            ).tobytes()
        except BaseException:
            self.abort()
            raise

    def submit(
        self,
        index: int,
        rgba: np.ndarray,
        *,
        reset: bool = True,
        pts: int | None = None,
    ) -> np.ndarray:
        if self.closed:
            raise Dlss5LinuxError("DLSS session is closed.")
        expected_shape = (self.render_height, self.render_width, 4)
        if rgba.dtype != np.uint8 or rgba.shape != expected_shape:
            raise Dlss5LinuxError(
                f"Expected uint8 RGBA {expected_shape}; got {rgba.dtype} {rgba.shape}."
            )
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(
                FRAME_HEADER.pack(
                    FRAME_MAGIC,
                    index,
                    int(reset),
                    0,
                    index if pts is None else pts,
                )
            )
            self.process.stdin.write(memoryview(np.ascontiguousarray(rgba)).cast("B"))
            self.process.stdin.write(self.zero_motion)
            self.process.stdin.flush()
            response = RESULT_HEADER.unpack(
                self.reader.read_struct_with_magic(
                    RESULT_HEADER,
                    OUT_MAGIC,
                    self.frame_timeout,
                    f"frame {index} response",
                    max_prefix=8192,
                )
            )
            magic, out_index, ok, byte_count, ngx_result, _out_pts = response
            expected_bytes = self.output_width * self.output_height * 4
            if (
                magic != OUT_MAGIC
                or not ok
                or out_index != index
                or byte_count != expected_bytes
            ):
                raise Dlss5LinuxError(f"Malformed worker response for frame {index}.")
            if ngx_result != 1:
                raise Dlss5LinuxError(
                    f"Feature-18 evaluation failed: 0x{ngx_result:08X}."
                )
            pixels = self.reader.read_exact(
                expected_bytes,
                self.frame_timeout,
                f"frame {index} pixels",
            )
            return np.frombuffer(pixels, dtype=np.uint8).reshape(
                self.output_height, self.output_width, 4
            ).copy()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise Dlss5LinuxError(
                f"The {self.launch.backend} worker stopped on frame {index}: {exc}\n"
                f"{self.logs.text()}"
            ) from exc

    def _combined_log(self) -> str:
        parts = [self.logs.text(), self.reader.discarded_text()]
        try:
            if (
                self.layout.reshade_log.is_file()
                and self.layout.reshade_log.stat().st_mtime >= self.started
            ):
                parts.append(
                    self.layout.reshade_log.read_text(
                        encoding="utf-8", errors="replace"
                    )
                )
        except OSError:
            pass
        return "\n".join(part for part in parts if part)

    def close(self) -> tuple[str, list[str]]:
        if self.closed:
            return self._combined_log(), []
        self.closed = True
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=self.close_timeout)
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            self.process.wait(timeout=10)
            self.logs.join()
            raise Dlss5LinuxError("DLSS worker did not exit and was killed.") from exc
        self.logs.join()
        if code:
            raise Dlss5LinuxError(
                f"DLSS worker exited with code {code}.\n{self._combined_log()}"
            )
        combined = self._combined_log()
        return combined, verify_feature_18(combined)

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                except OSError:
                    pass
        self.logs.join()


def format_status(
    layout: RuntimeLayout,
    launch: LaunchConfig,
    host: dict[str, str],
) -> str:
    fields: Iterable[tuple[str, object]] = (
        ("Backend", launch.backend),
        ("Worker kind", launch.worker_kind),
        ("Runtime root", layout.root),
        ("Worker", layout.worker),
        ("RenoDX add-on", layout.addon),
        ("DLSSNR runtime", layout.neural_runtime),
        ("DLSS SR runtime", layout.super_resolution_runtime),
        ("Launcher", " ".join(launch.command_prefix) or "native"),
        ("Prefix", launch.prefix or "not used"),
        ("NVIDIA", host.get("nvidia", "unknown")),
        ("Vulkan", host.get("vulkan", "unknown")),
    )
    return "\n".join(f"{name}: {value}" for name, value in fields)

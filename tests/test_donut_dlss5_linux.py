from pathlib import Path

import numpy as np
import pytest

import donut_dlss5_linux as dlss


def _make_runtime(root: Path) -> Path:
    runtime = root / "bin" / "runtime"
    host = runtime / "host"
    sr = runtime / "dlss"
    host.mkdir(parents=True)
    sr.mkdir(parents=True)
    (host / "nvngx.dll").write_bytes(b"MZ\x00\x00")
    for name in ("dxgi.dll", "renodx-dlss5.addon64", "nvngx_dlssnr.dll"):
        (host / name).write_bytes(b"test")
    (sr / "nvngx_dlss.dll").write_bytes(b"test")
    return root


def test_runtime_discovery_accepts_portable_root(tmp_path):
    root = _make_runtime(tmp_path / "portable")
    layout = dlss.RuntimeLayout.discover(root)
    assert layout.root == root / "bin" / "runtime"
    assert layout.worker.name == "nvngx.dll"


def test_runtime_discovery_rejects_incomplete_bundle(tmp_path):
    root = _make_runtime(tmp_path / "portable")
    (root / "bin" / "runtime" / "host" / "nvngx_dlssnr.dll").unlink()
    with pytest.raises(dlss.Dlss5LinuxError, match="incomplete"):
        dlss.RuntimeLayout.discover(root)


def test_output_sizes_are_even_and_bounded():
    width, height, factor, quality, name = dlss.resolve_output_size(
        513, 257, "2.0x (Performance)"
    )
    assert (width, height, factor, quality, name) == (
        1026,
        514,
        2.0,
        0,
        "Performance",
    )


def test_feature_18_verification_is_version_tolerant():
    evidence = dlss.verify_feature_18(
        "signed DLSSNR 310.8.0 D3D12 runtime initialized\n"
        "feature 18 created via the signed snippet\n"
        "inline feature 18 evaluation succeeded\n"
    )
    assert len(evidence) == 3


def test_feature_18_verification_rejects_pixel_only_fallback():
    with pytest.raises(dlss.Dlss5LinuxError, match="not verified"):
        dlss.verify_feature_18("ordinary resize completed")


def test_dll_overrides_preserve_existing_values():
    value = dlss._append_dll_overrides("foo=n;dxgi=b")
    assert "foo=n" in value
    assert "dxgi=b" in value
    assert "nvapi64=n,b" in value
    assert value.count("dxgi=") == 1


def test_buffered_reader_preserves_pixels_read_with_header():
    import os

    read_fd, write_fd = os.pipe()
    payload = b"launcher chatter\n" + dlss.RESULT_HEADER.pack(
        dlss.OUT_MAGIC, 7, 1, 6, 1, 7
    ) + b"pixels"
    try:
        os.write(write_fd, payload)
        os.close(write_fd)
        write_fd = -1
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            reader = dlss._BufferedPipeReader(stream)
            header = reader.read_struct_with_magic(
                dlss.RESULT_HEADER, dlss.OUT_MAGIC, 1.0, "test header"
            )
            assert dlss.RESULT_HEADER.unpack(header)[1] == 7
            assert reader.read_exact(6, 1.0, "test pixels") == b"pixels"
            assert reader.discarded_text() == "launcher chatter"
    finally:
        if write_fd >= 0:
            os.close(write_fd)


def test_session_round_trip_and_feature_verification(tmp_path):
    import os
    import sys

    root = _make_runtime(tmp_path / "portable")
    layout = dlss.RuntimeLayout.discover(root)
    worker_script = tmp_path / "fake_worker.py"
    worker_script.write_text(
        """
import os
import struct
import sys
from pathlib import Path

VIDEO = struct.Struct('<14I4f')
SETUP = struct.Struct('<12I')
FRAME = struct.Struct('<4Iq')
RESULT = struct.Struct('<5Iq')
header = sys.stdin.buffer.read(VIDEO.size)
fields = VIDEO.unpack(header)
in_w, in_h, out_w, out_h = fields[1:5]
model = fields[8]
os.write(sys.stdout.fileno(), b'fake launcher\\n' + SETUP.pack(
    0x34505553, 1, 1, in_w, in_h, out_w, out_h, 64, 64, 7680, 4320, model
))
frame = FRAME.unpack(sys.stdin.buffer.read(FRAME.size))
sys.stdin.buffer.read(in_w * in_h * 4)
sys.stdin.buffer.read(in_w * in_h * 2 * 2)
pixels = bytes([64, 128, 192, 255]) * (out_w * out_h)
os.write(sys.stdout.fileno(), RESULT.pack(0x3154554F, frame[1], 1, len(pixels), 1, frame[4]) + pixels)
Path('ReShade.log').write_text(
    'signed DLSSNR 310.8.0 D3D12 runtime initialized\\n'
    'feature 18 created via the signed snippet\\n'
    'inline feature 18 evaluation succeeded\\n'
)
""".strip()
        + "\n"
    )
    launch = dlss.LaunchConfig(
        backend="Test",
        command_prefix=(sys.executable, str(worker_script)),
        env=os.environ.copy(),
        prefix=None,
        worker_kind="test",
    )
    settings = dlss.build_native_settings(
        "Default", "Default", "M", 1.0, 1.0, 1.5, 2.0, True
    )
    session = dlss.Dlss5Session(
        layout,
        launch,
        input_width=64,
        input_height=64,
        output_width=64,
        output_height=64,
        frame_count=1,
        warmup_frames=0,
        perf_quality=5,
        native_settings=settings,
        setup_timeout=5,
        frame_timeout=5,
        close_timeout=5,
    )
    output = session.submit(0, np.zeros((64, 64, 4), dtype=np.uint8))
    assert output.shape == (64, 64, 4)
    assert output[0, 0].tolist() == [64, 128, 192, 255]
    _combined, evidence = session.close()
    assert len(evidence) == 3

from pathlib import Path

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
    assert (width, height, factor, quality, name) == (1026, 514, 2.0, 0, "Performance")


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

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
from PIL import Image


events = []


class FakeProgressBar:
    def __init__(self, total):
        self.total = total

    def update(self, amount):
        events.append(("progress", amount))


class FakeVAEEncode:
    def encode(self, vae, image):
        events.append(("encode", tuple(image.shape)))
        return ({"samples": torch.zeros(1, 1, 1, 1)},)


class FakeVAEDecode:
    def decode(self, vae, latent, *args):
        events.append(("decode", tuple(latent["samples"].shape)))
        return (torch.zeros(1, 4, 4, 3),)


fake_nodes = types.ModuleType("nodes")
fake_nodes.VAEEncode = FakeVAEEncode
fake_nodes.VAEDecode = FakeVAEDecode
fake_nodes.VAEDecodeTiled = FakeVAEDecode

fake_management = types.ModuleType("comfy.model_management")
fake_management.OOM_EXCEPTION = RuntimeError
fake_management.get_torch_device = lambda: torch.device("cpu")
fake_management.module_size = lambda model: 17
fake_management.free_memory = lambda required, device: events.append(
    ("free_memory", required, device)
)
fake_management.load_models_gpu = lambda models, memory_required=0: events.append(
    ("load_models_gpu", models, memory_required)
)
fake_management.intermediate_device = lambda: torch.device("cpu")
fake_management.intermediate_dtype = lambda: torch.float32
fake_management.raise_non_oom = lambda error: None

fake_utils = types.ModuleType("comfy.utils")
fake_utils.ProgressBar = FakeProgressBar
fake_utils.get_tiled_scale_steps = lambda *args, **kwargs: 1


def fake_tiled_scale(image, function, **kwargs):
    events.append(("tiled_scale", kwargs.get("output_device")))
    return function(image)


fake_utils.tiled_scale = fake_tiled_scale

fake_sample = types.ModuleType("comfy.sample")
fake_sample.fix_empty_latent_channels = lambda model, latent: latent
fake_sample.prepare_noise = lambda latent, seed: torch.zeros_like(latent)


def fake_sample_call(model, noise, steps, cfg, sampler_name, scheduler,
                     positive, negative, latent_image, **kwargs):
    events.append(("sample", kwargs["seed"]))
    return latent_image.clone()


fake_sample.sample = fake_sample_call

fake_samplers = types.ModuleType("comfy.samplers")
fake_samplers.KSampler = types.SimpleNamespace(
    SAMPLERS=["euler"],
    SCHEDULERS=["simple"],
)

fake_comfy = types.ModuleType("comfy")
fake_comfy.sample = fake_sample
fake_comfy.samplers = fake_samplers
fake_comfy.utils = fake_utils
fake_comfy.model_management = fake_management

fake_preview = types.ModuleType("latent_preview")
fake_preview.prepare_callback = lambda model, steps: None

fake_krea = types.ModuleType("krea2_edit_integration")
fake_krea.prepare_krea2_edit = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("regular-mode test must not prepare Krea2 edit")
)

_fake_modules = {
    "nodes": fake_nodes,
    "comfy": fake_comfy,
    "comfy.sample": fake_sample,
    "comfy.samplers": fake_samplers,
    "comfy.utils": fake_utils,
    "comfy.model_management": fake_management,
    "latent_preview": fake_preview,
    "krea2_edit_integration": fake_krea,
}
_missing = object()
_previous_modules = {name: sys.modules.get(name, _missing) for name in _fake_modules}
sys.modules.update(_fake_modules)
try:
    spec = importlib.util.spec_from_file_location(
        "donut_tiled_upscale_tested",
        Path(__file__).with_name("DonutTiledUpscale.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    for name, previous in _previous_modules.items():
        if previous is _missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class PatcherUpscaleModel:
    scale = 1.0

    def __init__(self):
        self.patcher = types.SimpleNamespace(load_device=torch.device("cpu"))
        self.model = object()

    def __call__(self, image):
        self.call_dtype = image.dtype
        return image

    def to(self, device):
        raise AssertionError("patcher-backed upscale models must not be moved directly")


class LegacyUpscaleModel:
    scale = 1.0

    def __init__(self):
        self.model = object()
        self.moves = []

    def __call__(self, image):
        return image

    def to(self, device):
        self.moves.append(device)
        return self


class DonutTiledUpscaleLifecycleTests(unittest.TestCase):
    def setUp(self):
        events.clear()

    def test_patcher_backed_upscaler_uses_current_comfy_lifecycle(self):
        upscale_model = PatcherUpscaleModel()
        image = torch.zeros(1, 4, 4, 3, dtype=torch.float16)

        output = module.upscale_with_model(upscale_model, image)

        load_events = [event for event in events if event[0] == "load_models_gpu"]
        self.assertEqual(len(load_events), 1)
        self.assertEqual(load_events[0][1], [upscale_model.patcher])
        self.assertFalse(any(event[0] == "free_memory" for event in events))
        self.assertIn(("tiled_scale", torch.device("cpu")), events)
        self.assertEqual(upscale_model.call_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)

    def test_legacy_upscaler_keeps_manual_fallback(self):
        upscale_model = LegacyUpscaleModel()
        image = torch.zeros(1, 4, 4, 3)

        module.upscale_with_model(upscale_model, image)

        self.assertTrue(any(event[0] == "free_memory" for event in events))
        self.assertEqual(upscale_model.moves, [torch.device("cpu"), "cpu"])

    def test_regular_tiles_run_in_encode_sample_decode_phases(self):
        original_upscale = module.upscale_with_model
        original_find = module.find_best_tiling
        original_debug = module.create_debug_image
        module.upscale_with_model = lambda upscale_model, image: image
        module.find_best_tiling = lambda *args, **kwargs: {
            "nx": 1,
            "ny": 3,
            "tile_width": 4,
            "tile_height": 4,
            "overlap_x": 0,
            "overlap_y": 2,
            "output_width": 4,
            "output_height": 8,
            "scale": 1.0,
            "step_x": 4,
            "step_y": 2,
        }
        module.create_debug_image = lambda *args, **kwargs: Image.new("RGB", (4, 8))
        try:
            node = module.DonutTiledUpscale()
            node.upscale(
                image=torch.zeros(1, 4, 4, 3),
                upscale_model=types.SimpleNamespace(scale=1.0),
                model=object(),
                positive=object(),
                negative=object(),
                vae=object(),
                seed=100,
                steps=2,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=0.3,
                rescale_factor=1.0,
                resampling_method="nearest",
                feather=25.0,
                tiled_vae=False,
            )
        finally:
            module.upscale_with_model = original_upscale
            module.find_best_tiling = original_find
            module.create_debug_image = original_debug

        phase_events = [event[0] for event in events
                        if event[0] in ("encode", "sample", "decode")]
        self.assertEqual(
            phase_events,
            ["encode"] * 3 + ["sample"] * 3 + ["decode"] * 3,
        )
        self.assertEqual(
            [event[1] for event in events if event[0] == "sample"],
            [101, 102, 103],
        )


if __name__ == "__main__":
    unittest.main()

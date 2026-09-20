"""CPU contract tests for the standalone SeedVR2 post-upscale node."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PostUpscaleTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(sys.modules)
        self.env.start()
        package = types.ModuleType("_seedvr2_post_test_pack")
        package.__path__ = [str(ROOT)]
        sys.modules[package.__name__] = package
        folders = types.ModuleType("folder_paths")
        folders.get_filename_list = lambda folder: []
        sys.modules["folder_paths"] = folders
        self.engine = load(package.__name__ + ".donut_seedvr2", ROOT / "donut_seedvr2.py")
        self.module = load(package.__name__ + ".donut_seedvr2_post", ROOT / "donut_seedvr2_post.py")
        self.node = self.module.DonutSeedVR2Upscale()

    def tearDown(self):
        self.env.stop()

    def test_disabled_node_passes_image_through(self):
        self.assertEqual(self.node.post_upscale("image", enabled=False), ("image",))

    def test_single_image_output_contract(self):
        with patch.object(self.engine, "upscale", return_value="refined") as run:
            result = self.node.post_upscale("image", seed=7, seedvr2_upscale_factor=2.0,
                                            resampling_method="lanczos", seedvr2_steps=1)
        self.assertEqual(result, ("refined",))
        self.assertEqual(run.call_args.kwargs,
                         dict(seed=7, rescale_factor=2.0, resampling_method="lanczos", seedvr2_steps=1))

    def test_unknown_settings_are_rejected(self):
        with self.assertRaises(TypeError):
            self.node.post_upscale("image", seedvr2_typo=1)

    def test_schema_keeps_required_widgets_visible(self):
        schema = self.module.DonutSeedVR2Upscale.INPUT_TYPES()
        self.assertEqual(list(schema["required"]), ["image", "seedvr2_upscale_factor", "resampling_method"])
        self.assertIn("seedvr2_model_name", schema["optional"])
        self.assertEqual(schema["optional"]["seedvr2_vae_tile_size"][1]["default"], 1024)
        self.assertEqual(schema["optional"]["seedvr2_vae_overlap"][1]["default"], 128)
        self.assertTrue(schema["optional"]["enabled"][1]["default"] is True)

    def test_lazy_status_skips_pass_through_and_requests_model_files(self):
        self.assertEqual(self.node.check_lazy_status("image", enabled=False), [])
        pending = self.node.check_lazy_status("image", enabled=True, seedvr2_model_name=None,
                                              seedvr2_vae_name="installed.safetensors",
                                              seedvr2_steps=None)
        self.assertEqual(pending, ["seedvr2_model_name", "seedvr2_steps"])

    def test_node_id_registered(self):
        self.assertIn("DonutSeedVR2Upscale", self.module.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()

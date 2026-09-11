import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch


fake_paths = types.ModuleType("folder_paths")
fake_paths.get_filename_list = lambda kind: ["krea2/krea2_identity_edit_v1_2.safetensors"]
fake_paths.get_user_directory = lambda: "unused"
fake_paths.get_annotated_filepath = Mock(side_effect=ValueError("Invalid file path"))
fake_nodes = types.ModuleType("nodes")
fake_nodes.MAX_RESOLUTION = 16384
fake_nodes.LoraLoaderModelOnly = Mock()
fake_server = types.ModuleType("server")
fake_server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=types.SimpleNamespace(
    post=lambda path: lambda function: function, get=lambda path: lambda function: function,
)))
with patch.dict(sys.modules, {"folder_paths":fake_paths, "nodes":fake_nodes, "server":fake_server}):
    spec = importlib.util.spec_from_file_location("edit_studio_tested", Path(__file__).with_name("DonutEditStudio.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class EditStudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = patch.object(fake_paths, "get_user_directory", return_value=self.temp.name)
        self.paths.start(); self.addCleanup(self.paths.stop)
        fake_nodes.LoraLoaderModelOnly.reset_mock()
        expansion = patch.object(module, "expand_text", side_effect=lambda text, seed: text)
        expansion.start(); self.addCleanup(expansion.stop)
        fingerprint = patch.object(module, "directory_fingerprint", return_value="test")
        fingerprint.start(); self.addCleanup(fingerprint.stop)

    def test_edit_lora_inherits_donut_execution_mode(self):
        comfy = types.ModuleType("comfy")
        comfy.utils = types.ModuleType("comfy.utils")
        comfy.utils.load_torch_file = Mock(return_value={"weights": 1})
        loader = types.ModuleType("edit_test_package.DonutSafeApplyLoRAStack")
        loader._apply_bypass_applications = Mock(return_value="bypass edit")
        package = types.ModuleType("edit_test_package"); package.__path__ = []
        model = types.SimpleNamespace(model_options={"donut_lora_execution_mode": "Experimental bypass"})
        with patch.object(module, "__package__", "edit_test_package"), patch.dict(sys.modules, {
            "comfy": comfy, "comfy.utils": comfy.utils, "edit_test_package": package,
            loader.__name__: loader,
        }), patch.object(fake_paths, "get_full_path", return_value="edit.safetensors", create=True):
            self.assertEqual(module._load_edit_lora(model, "edit", .7), "bypass edit")
        loader._apply_bypass_applications.assert_called_once_with(model, [({"weights": 1}, .7, ",".join(["1"] * 29))])
        fake_nodes.LoraLoaderModelOnly.assert_not_called()
        model.model_options["donut_lora_execution_mode"] = "Comfy patches"
        fake_nodes.LoraLoaderModelOnly.return_value.load_lora_model_only.return_value = ("regular",)
        module._load_edit_lora(model, "edit", .7)
        fake_nodes.LoraLoaderModelOnly.return_value.load_lora_model_only.assert_called_once_with(model, "edit", .7)

    def settings(self, **overrides):
        settings = {name: spec[1]["default"] for name, spec in module.DonutEditStudio.INPUT_TYPES()["required"].items()}
        settings.update(overrides)
        return settings

    def reference(self, pixels):
        image = Image.fromarray(pixels) if isinstance(pixels, np.ndarray) else Image.new("RGB", pixels, "red")
        data = io.BytesIO(); image.save(data, format="PNG"); data.seek(0)
        return module.store_reference(data)["reference"]

    def test_disabled_editing_needs_no_files_or_model(self):
        settings = self.settings(image_a="missing.png", image_b="missing-b.png", use_reference_b=True)
        self.assertIs(module.DonutEditStudio.VALIDATE_INPUTS(**settings), True)
        self.assertEqual(module.DonutEditStudio().check_lazy_status(**settings), [])
        self.assertEqual(module.DonutEditStudio.IS_CHANGED(**settings), "disabled")
        with patch.object(module, "_open_reference", side_effect=AssertionError("must not load")):
            result = module.DonutEditStudio().prepare(**settings)
        self.assertEqual(result[:6], (None, None, False, 1152, 896, 1088))
        self.assertIsNone(result[6]); fake_nodes.LoraLoaderModelOnly.assert_not_called()

    def test_second_reference_only_validated_when_enabled(self):
        name = self.reference((100, 80))
        self.assertIs(module.DonutEditStudio.VALIDATE_INPUTS(True, name, "missing", False), True)
        self.assertIn("reference B", module.DonutEditStudio.VALIDATE_INPUTS(True, name, "missing", True))
        self.assertIn("reference A", module.DonutEditStudio.VALIDATE_INPUTS(True, "", "", False))

    def test_edit_instruction_expands_with_shared_text_seed(self):
        name = self.reference((80, 60))
        settings = self.settings(enabled=True, image_a=name, prompt="Use haircolor*", lora_name="None")
        with patch.object(module, "expand_text", return_value="Use red hair") as expand:
            result = module.DonutEditStudio().prepare(**settings, model="model", text_seed=123)
        expand.assert_called_once_with("Use haircolor*", 123)
        self.assertEqual(result[-1], "Use red hair")

    def test_storage_survives_fresh_node_and_workflow_reload_without_input_folder(self):
        name = self.reference((101, 79))
        settings = json.loads(json.dumps(self.settings(enabled=True, image_a=name, lora_name="None")))
        path = module._reference_path(name)
        self.assertEqual(path.parent, Path(self.temp.name) / "donut" / "edit_references")
        self.assertIs(module.DonutEditStudio.VALIDATE_INPUTS(**settings), True)
        result = module.DonutEditStudio().prepare(**settings, model="model")
        self.assertEqual(result[0].shape[0], 1)
        self.assertEqual(module.DonutEditStudio.IS_CHANGED(**settings), module.DonutEditStudio.IS_CHANGED(**settings))
        self.assertEqual(name, self.reference((101, 79)))
        self.assertEqual(len(list(path.parent.glob("*.png"))), 1)

    def test_reference_path_rejects_traversal_and_escaping_symlinks(self):
        for name in ["donutref:../../secret", "donutref:" + "g" * 64, "donutref:" + "a" * 64 + "/x"]:
            with self.assertRaises(ValueError): module._reference_path(name)
        root = module._reference_root(); root.mkdir(parents=True)
        link = root / ("a" * 64 + ".png"); link.symlink_to(Path(self.temp.name) / "outside.png")
        with self.assertRaises(ValueError): module._reference_path("donutref:" + "a" * 64)
        with self.assertRaises(ValueError): module._reference_path("../outside.png")
        fake_paths.get_annotated_filepath.assert_called_with("../outside.png")

    def test_crop_only_is_exact_pixels_on_both_grids(self):
        source = np.arange(777 * 1001 * 3, dtype=np.uint32).reshape(777, 1001, 3).astype(np.uint8)
        name = self.reference(source)
        for grid, expected in [(64, (960, 768)), (32, (992, 768)), (16, (992, 768))]:
            settings = self.settings(enabled=True, image_a=name, resolution_mode="Reference A · crop only", multiple=str(grid), lora_name="None", crop_a_x=1, crop_a_y=0)
            result = module.DonutEditStudio().prepare(**settings, model="model")
            width, height = expected
            self.assertEqual(result[3:5], expected)
            actual = result[0][0].mul(255).round().to(torch.uint8).numpy()
            self.assertTrue(np.array_equal(actual, source[:height, 1001-width:]))

    def test_two_images_have_independent_crop_positions_and_lora_is_applied_once(self):
        a_pixels = np.arange(60 * 120 * 3, dtype=np.uint16).reshape(60, 120, 3).astype(np.uint8)
        b_pixels = np.arange(120 * 60 * 3, dtype=np.uint16).reshape(120, 60, 3).astype(np.uint8)
        a, b = self.reference(a_pixels), self.reference(b_pixels)
        settings = self.settings(enabled=True, image_a=a, image_b=b, use_reference_b=True,
                                 resolution_mode="Custom", width=64, height=64, crop_a_x=0, crop_b_y=1)
        fake_nodes.LoraLoaderModelOnly.return_value.load_lora_model_only.return_value = ("edit model",)
        result = module.DonutEditStudio().prepare(**settings, model="base")
        expected_a = np.asarray(Image.fromarray(a_pixels[:, :60]).resize((64, 64), Image.Resampling.LANCZOS))
        expected_b = np.asarray(Image.fromarray(b_pixels[60:]).resize((64, 64), Image.Resampling.LANCZOS))
        self.assertTrue(np.array_equal(result[0][0].mul(255).round().numpy(), expected_a))
        self.assertTrue(np.array_equal(result[1][0].mul(255).round().numpy(), expected_b))
        self.assertEqual(result[6], "edit model")
        fake_nodes.LoraLoaderModelOnly.return_value.load_lora_model_only.assert_called_once_with("base", settings["lora_name"], 1)

    def test_edit_lora_output_records_metadata_for_downstream_model_branches(self):
        name = self.reference((80, 60))
        settings = self.settings(enabled=True, image_a=name)
        source = types.SimpleNamespace(model_options={"donut_lora_execution_mode": "Comfy patches"})
        edited = types.SimpleNamespace(model_options={})
        fake_nodes.LoraLoaderModelOnly.return_value.load_lora_model_only.return_value = (edited,)

        result = module.DonutEditStudio().prepare(**settings, model=source)

        self.assertIs(result[6], edited)
        self.assertEqual(
            edited.model_options[module._EDIT_LORA_METADATA_KEY],
            {
                "name": settings["lora_name"],
                "strength": 1.0,
                "execution_mode": "Comfy patches",
            },
        )
        self.assertEqual(edited.model_options["donut_lora_execution_mode"], "Comfy patches")

    def test_exif_orientation_is_normalized_in_persistent_preview_and_crop(self):
        source = Image.new("RGB", (120, 80), "blue"); exif = Image.Exif(); exif[274] = 6
        data = io.BytesIO(); source.save(data, format="JPEG", exif=exif); data.seek(0)
        saved = module.store_reference(data)
        self.assertEqual((saved["width"], saved["height"]), (80, 120))
        self.assertEqual(module._open_reference(saved["reference"]).size, (80, 120))

    def test_grid_alignment_custom_ratios_and_reference_megapixels(self):
        self.assertEqual(module.target_dimensions("Custom", "4:3 Standard", 1, 1000, 770, 64), (1024, 768))
        self.assertEqual(module.target_dimensions("Reference A · megapixels", "4:3 Standard", 1, 1000, 770, 32, (1000, 2000)), (736, 1440))
        self.assertEqual(module.target_dimensions("Reference A · crop only", "4:3 Standard", 1, 0, 0, 32, (16, 20)), (32, 32))


if __name__ == "__main__":
    unittest.main()

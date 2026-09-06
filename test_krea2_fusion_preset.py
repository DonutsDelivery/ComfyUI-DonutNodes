import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = "db3c2b7612828120e7ef9cc8fe77124c6fd8de2e38f150599e62abd9695f6beb"
EXPECTED_SIZE = 3_470_548
EXPECTED_TARGETS = 33


def _load_module():
    package_name = "donut_krea2_fusion_preset_testpkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package

    base = types.ModuleType(f"{package_name}.DonutKrea2FusionControl")
    base.PRESET_MANUAL = "Custom settings"
    base.PRESET_BYPASS_2 = "COPY settings: Krea2FilterBypass 2vector"
    base.PRESET_BYPASS_3 = "COPY settings: Krea2FilterBypass 3vector"
    base.PRESET_REBALANCE = "COPY settings: nova452 ConditioningKrea2Rebalance profile @ tap strength 1"
    base.PRESET_ENHANCER = "COPY settings: capitan01R Krea2T-Enhancer defaults"
    base.PRESET_REBALANCE_ENHANCER = "HYBRID settings: Rebalance + Krea2T-Enhancer"
    base.PRESET_REBALANCE_BYPASS_2 = "HYBRID settings: Rebalance + Krea2FilterBypass 2vector"
    base.PRESET_REBALANCE_BYPASS_3 = "HYBRID settings: Rebalance + Krea2FilterBypass 3vector"
    base.PRESET_DONUT_BALANCED = "DONUT settings: RMS-balanced classic"
    base.PRESET_DONUT_BALANCED_ENHANCER = "DONUT settings: RMS-balanced classic + Krea2T-Enhancer"

    class BaseNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL",),
                    "conditioning_in_1": ("CONDITIONING",),
                    "compatibility_preset": ([base.PRESET_MANUAL, base.PRESET_DONUT_BALANCED], {
                        "default": base.PRESET_MANUAL,
                    }),
                    "tap_method": (["Donut 12-tap gains"],),
                    "tap_profile": (["off"],),
                    "per_layer_weights": ("STRING", {}),
                    "tap_strength": ("FLOAT", {"default": 1.0}),
                    "tap_formula": (["scale_around_1"],),
                    "tap_normalization": (["none"],),
                    "projector_method": (["Donut projector-input gains"],),
                    "projector_profile": (["off"],),
                    "projector_layer_weights": ("STRING", {}),
                    "projector_strength": ("FLOAT", {"default": 1.0}),
                    "projector_formula": (["scale_around_1"],),
                    "projector_normalization": (["none"],),
                    "fusion_method": (["Standard Krea2 fusion"],),
                    "fusion_strength": ("FLOAT", {"default": 1.0}),
                },
                "optional": {},
            }

        def apply(self, **kwargs):
            preset = kwargs.get("compatibility_preset", base.PRESET_MANUAL)
            return (
                kwargs["model"],
                kwargs["conditioning_in_1"],
                None,
                None,
                None,
                f"preset_label={preset}; preset_is_ui_only=true\n"
                "external_files_loaded=none",
            )

    base.DonutKrea2FusionControl = BaseNode
    state = {
        f"diffusion_model.txtfusion.fake_{index}.lora_down.weight": object()
        for index in range(EXPECTED_TARGETS)
    }

    class FakeUtils:
        @staticmethod
        def load_torch_file(path, safe_load=True):
            return dict(state)

    class FakeLora:
        @staticmethod
        def model_lora_keys_unet(model):
            return {"fake": "fake"}

        @staticmethod
        def load_lora(lora, key_map, log_missing=False):
            return {
                f"diffusion_model.txtfusion.fake_{index}.weight": ("lora", ())
                for index in range(EXPECTED_TARGETS)
            }

    base.comfy = types.SimpleNamespace(utils=FakeUtils, lora=FakeLora)
    sys.modules[base.__name__] = base
    setattr(package, "DonutKrea2FusionControl", base)

    fake_folder_paths = types.ModuleType("folder_paths")
    fake_folder_paths.get_filename_list = lambda category: [
        "other.safetensors",
        "renamed_teacherfix.safetensors",
    ]
    fake_folder_paths.get_full_path = lambda category, name: f"/fake/{name}"

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.DonutKrea2FusionPreset",
        ROOT / "DonutKrea2FusionPreset.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, fake_folder_paths


class TeacherFixPresetTests(unittest.TestCase):
    def test_source_constants_and_ui_schema_are_stable(self):
        source = (ROOT / "DonutKrea2FusionPreset.py").read_text()
        ast.parse(source)
        self.assertIn(f'TEACHERFIX_SHA256 = "{EXPECTED_SHA256}"', source)
        self.assertIn(f"TEACHERFIX_SIZE_BYTES = {EXPECTED_SIZE:_}", source)
        self.assertIn(f"TEACHERFIX_TARGET_COUNT = {EXPECTED_TARGETS}", source)

        module, _ = _load_module()
        schema = module.DonutKrea2FusionControl.INPUT_TYPES()
        required = schema["required"]
        self.assertEqual(list(required)[-1], "ui_mode")
        self.assertEqual(required["ui_mode"][1]["default"], "Advanced")
        self.assertEqual(required["compatibility_preset"][1]["default"], "Custom")
        self.assertEqual(
            required["compatibility_preset"][0],
            [
                "Custom",
                "Bypass 2",
                "Bypass 3",
                "Rebalance",
                "Enhancer",
                "Rebalance + Enhancer",
                "Rebalance + Bypass 2",
                "Rebalance + Bypass 3",
                "Balanced",
                "Balanced + Enhancer",
                "TeacherFix",
            ],
        )

    def test_exact_file_is_discovered_by_size_and_hash_even_when_renamed(self):
        module, fake_folder_paths = _load_module()
        with (
            mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}),
            mock.patch.object(module.os.path, "isfile", return_value=True),
            mock.patch.object(module.os.path, "getsize", side_effect=lambda path: (
                EXPECTED_SIZE if "renamed_teacherfix" in path else 123
            )),
            mock.patch.object(module, "_sha256_file", return_value=EXPECTED_SHA256),
        ):
            name, path = module._find_teacherfix_file()
        self.assertEqual(name, "renamed_teacherfix.safetensors")
        self.assertEqual(path, "/fake/renamed_teacherfix.safetensors")

    def test_simplified_names_delegate_to_existing_presets(self):
        module, _ = _load_module()
        self.assertEqual(
            module.SIMPLE_PRESET_TO_LEGACY["Bypass 2"],
            "COPY settings: Krea2FilterBypass 2vector",
        )
        self.assertEqual(
            module.SIMPLE_PRESET_TO_LEGACY["Balanced + Enhancer"],
            "DONUT settings: RMS-balanced classic + Krea2T-Enhancer",
        )

    def test_teacherfix_strength_patches_all_33_targets(self):
        module, _ = _load_module()
        module._TEACHERFIX_CACHE = (
            {
                f"diffusion_model.txtfusion.fake_{index}.lora_down.weight": object()
                for index in range(EXPECTED_TARGETS)
            },
            "renamed_teacherfix.safetensors",
        )

        class FakeModel:
            def __init__(self):
                self.model = object()
                self.loaded = {}

            def clone(self):
                clone = FakeModel()
                clone.loaded = dict(self.loaded)
                return clone

            def add_patches(self, patches, strength_patch=1.0):
                self.loaded.update({key: strength_patch for key in patches})
                return list(patches)

        conditioning = object()
        result = module.DonutKrea2FusionControl().apply(
            model=FakeModel(),
            conditioning_in_1=conditioning,
            compatibility_preset=module.PRESET_TEACHERFIX,
            tap_method="Donut 12-tap gains",
            tap_profile="off",
            per_layer_weights="1",
            tap_strength=0.75,
            tap_formula="scale_around_1",
            tap_normalization="none",
            projector_method="Donut projector-input gains",
            projector_profile="off",
            projector_layer_weights="1",
            projector_strength=1.0,
            projector_formula="scale_around_1",
            projector_normalization="none",
            fusion_method="Standard Krea2 fusion",
            fusion_strength=1.0,
            ui_mode="Simple",
        )

        self.assertEqual(len(result[0].loaded), EXPECTED_TARGETS)
        self.assertEqual(set(result[0].loaded.values()), {0.75})
        self.assertIs(result[1], conditioning)
        self.assertIn("preset_label=TeacherFix", result[-1])
        self.assertIn("preset_is_ui_only=false", result[-1])
        self.assertIn("renamed_teacherfix.safetensors", result[-1])

    def test_pre_rename_teacherfix_label_is_still_accepted(self):
        module, _ = _load_module()
        self.assertEqual(
            module.LEGACY_PRESET_TO_SIMPLE[module.LEGACY_TEACHERFIX],
            "TeacherFix",
        )

    def test_frontend_has_compact_simple_and_full_advanced_modes(self):
        js = (ROOT / "web" / "donut_krea2_fusion_simple_mode.js").read_text()
        self.assertIn('const SIMPLE = "Simple";', js)
        self.assertIn('const ADVANCED = "Advanced";', js)
        self.assertIn('const CUSTOM = "Custom";', js)
        self.assertIn('const TEACHERFIX = "TeacherFix";', js)
        self.assertIn('"Bypass 2"', js)
        self.assertIn('"Balanced + Enhancer"', js)
        self.assertIn('name === "tap_strength"', js)
        self.assertIn("SIMPLE_PROJECTOR_STRENGTH", js)
        self.assertIn("SIMPLE_FUSION_STRENGTH", js)
        self.assertIn("LEGACY_TO_SIMPLE", js)
        self.assertIn("queueMicrotask", js)


if __name__ == "__main__":
    unittest.main()

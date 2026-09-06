"""Bundled-byte validation plus isolated patch plumbing (no ComfyUI/GPU needed).

These tests load the real bundled tensors. ComfyUI's model and adapter parser
are doubles here; this is not an image-generation or full ModelPatcher test.
"""
import hashlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets/krea2_c33_teacherfix_ema5000.safetensors"
EXPECTED_SHA = "db3c2b7612828120e7ef9cc8fe77124c6fd8de2e38f150599e62abd9695f6beb"
BASE_NAMES = {
    "PRESET_MANUAL": "Custom settings",
    "PRESET_BYPASS_2": "COPY settings: Krea2FilterBypass 2vector",
    "PRESET_BYPASS_3": "COPY settings: Krea2FilterBypass 3vector",
    "PRESET_REBALANCE": "COPY settings: nova452 ConditioningKrea2Rebalance profile @ tap strength 1",
    "PRESET_ENHANCER": "COPY settings: capitan01R Krea2T-Enhancer defaults",
    "PRESET_REBALANCE_ENHANCER": "HYBRID settings: Rebalance + Krea2T-Enhancer",
    "PRESET_REBALANCE_BYPASS_2": "HYBRID settings: Rebalance + Krea2FilterBypass 2vector",
    "PRESET_REBALANCE_BYPASS_3": "HYBRID settings: Rebalance + Krea2FilterBypass 3vector",
    "PRESET_DONUT_BALANCED": "DONUT settings: RMS-balanced classic",
    "PRESET_DONUT_BALANCED_ENHANCER": "DONUT settings: RMS-balanced classic + Krea2T-Enhancer",
}
SHORT_NAMES = ["Custom", "Bypass 2", "Bypass 3", "Rebalance", "Enhancer",
               "Rebalance + Enhancer", "Rebalance + Bypass 2", "Rebalance + Bypass 3",
               "Balanced", "Balanced + Enhancer", "TeacherFix"]
LEGACY_WIDGETS = ["model", "conditioning_in_1", "compatibility_preset", "tap_method",
                  "tap_profile", "per_layer_weights", "tap_strength", "tap_formula",
                  "tap_normalization", "projector_method", "projector_profile",
                  "projector_layer_weights", "projector_strength", "projector_formula",
                  "projector_normalization", "fusion_method", "fusion_strength"]


class FakeModel:
    def __init__(self):
        self.model = object()
        self.patches = {}
        self.strength = None

    def clone(self):
        result = FakeModel()
        result.patches = dict(self.patches)
        return result

    def add_patches(self, patches, strength_patch=1.0):
        self.patches.update(patches)
        self.strength = strength_patch
        return list(patches)


def _load_module():
    name = "_donut_bundled_testpkg"
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT)]
    base = types.ModuleType(name + ".DonutKrea2FusionControl")
    for key, value in BASE_NAMES.items():
        setattr(base, key, value)
    base.calls = []

    class BaseNode:
        @classmethod
        def INPUT_TYPES(cls):
            required = {key: ("STRING", {}) for key in LEGACY_WIDGETS}
            required["compatibility_preset"] = (list(BASE_NAMES.values()), {"default": base.PRESET_MANUAL})
            required["tap_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": .05})
            return {"required": required, "optional": {"conditioning_in_2": ("CONDITIONING",)}}

        def apply(self, **kwargs):
            base.calls.append(kwargs)
            return (kwargs["model"], kwargs["conditioning_in_1"], None, None, None,
                    f"preset_label={kwargs['compatibility_preset']}; preset_is_ui_only=true\n"
                    "external_files_loaded=none")

    class FakeLora:
        @staticmethod
        def model_lora_keys_unet(model):
            return {}

        @staticmethod
        def load_lora(state, key_map, log_missing=False):
            output = {}
            for key, down in state.items():
                if key.endswith(".lora_down.weight"):
                    target = key[:-len(".lora_down.weight")]
                    output[target + ".weight"] = ("lora", (state[target + ".lora_up.weight"],
                                                             down, state[target + ".alpha"]))
            return output

    base.DonutKrea2FusionControl = BaseNode
    base.comfy = types.SimpleNamespace(lora=FakeLora)
    setattr(package, "DonutKrea2FusionControl", base)
    spec = importlib.util.spec_from_file_location(name + ".DonutKrea2FusionPreset", ROOT / "DonutKrea2FusionPreset.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {name: package, base.__name__: base}):
        spec.loader.exec_module(module)
    return module


class BundledTeacherFixTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_original_bytes_are_shipped(self):
        payload = ASSET.read_bytes()
        self.assertEqual(len(payload), 3_470_548)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), EXPECTED_SHA)

    def test_decoded_tensors_equal_standard_safetensors_load(self):
        bundled, name = self.module._load_teacherfix_lora()
        reference = load_file(str(ASSET), device="cpu")
        self.assertEqual(len(bundled), 99)
        self.assertEqual(set(bundled), set(reference))
        self.assertEqual(name, "assets/krea2_c33_teacherfix_ema5000.safetensors")
        for key in reference:
            self.assertEqual(bundled[key].dtype, reference[key].dtype)
            self.assertTrue(torch.equal(bundled[key], reference[key]), key)

    def test_loading_needs_no_folder_paths_or_external_file_loader(self):
        with mock.patch.dict(sys.modules, {"folder_paths": None, "comfy.utils": None}):
            self.assertEqual(len(self.module._load_teacherfix_lora()[0]), 99)
        self.assertEqual(self.module.TEACHERFIX_PATH, ASSET)

    def test_decode_is_cached(self):
        first = self.module._load_teacherfix_lora()
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read twice")):
            self.assertIs(first, self.module._load_teacherfix_lora())

    def test_missing_bundle_fails_without_external_fallback(self):
        with mock.patch.object(Path, "read_bytes", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "complete DonutNodes"):
                self.module._load_teacherfix_lora()

    def test_wrong_size_fails(self):
        with mock.patch.object(Path, "read_bytes", return_value=b"bad"):
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self.module._load_teacherfix_lora()

    def test_corrupt_same_size_asset_fails(self):
        payload = bytearray(ASSET.read_bytes())
        payload[-1] ^= 1
        with mock.patch.object(Path, "read_bytes", return_value=bytes(payload)):
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self.module._load_teacherfix_lora()

    def test_schema_and_short_names_are_preserved(self):
        required = self.module.DonutKrea2FusionControl.INPUT_TYPES()["required"]
        self.assertEqual(list(required), LEGACY_WIDGETS + ["ui_mode"])
        self.assertEqual(required["compatibility_preset"][0], SHORT_NAMES)
        self.assertEqual(required["ui_mode"][1]["default"], "Advanced")
        self.assertEqual(required["tap_strength"][1]["max"], 10.0)
        self.assertEqual(list(self.module.NODE_CLASS_MAPPINGS), ["DonutKrea2FusionControl"])

    def test_zero_strength_is_exact_no_op_without_asset(self):
        original = FakeModel()
        with mock.patch.object(self.module, "_load_teacherfix_lora", side_effect=AssertionError("asset accessed")):
            result, count, _ = self.module._apply_teacherfix(original, 0.0)
        self.assertIs(result, original)
        self.assertEqual(count, 0)

    def test_nonfinite_strength_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module._apply_teacherfix(FakeModel(), value)

    def test_every_pair_and_alpha_reaches_patch_path_unchanged(self):
        state, _ = self.module._load_teacherfix_lora()
        for strength in (-0.5, 0.75, 1.0, 2.0):
            with self.subTest(strength=strength):
                original = FakeModel()
                original.patches["unrelated.weight"] = ("diff", ())
                result, count, _ = self.module._apply_teacherfix(original, strength)
                self.assertIsNot(result, original)
                self.assertEqual(count, 33)
                self.assertEqual(result.strength, strength)
                self.assertEqual(len(original.patches), 1)
                self.assertIn("unrelated.weight", result.patches)
                for key, down in state.items():
                    if not key.endswith(".lora_down.weight"):
                        continue
                    target = key[:-len(".lora_down.weight")]
                    _, (up_patch, down_patch, alpha_patch) = result.patches[target + ".weight"]
                    self.assertIs(down_patch, down)
                    self.assertIs(up_patch, state[target + ".lora_up.weight"])
                    self.assertIs(alpha_patch, state[target + ".alpha"])

    def test_short_and_legacy_teacherfix_work_in_both_modes(self):
        for preset in ("TeacherFix", self.module.LEGACY_TEACHERFIX):
            for mode in ("Simple", "Advanced"):
                with self.subTest(preset=preset, mode=mode):
                    cond = object()
                    result = self.module.DonutKrea2FusionControl().apply(
                        model=FakeModel(), conditioning_in_1=cond, compatibility_preset=preset,
                        tap_strength=0.75, ui_mode=mode)
                    self.assertEqual(len(result[0].patches), 33)
                    self.assertEqual(result[0].strength, .75)
                    self.assertIs(result[1], cond)
                    self.assertIn("teacherfix_source=bundled", result[-1])
                    self.assertIn("teacherfix_targets=33", result[-1])
                    self.assertIn("external_files_loaded=none", result[-1])

    def test_other_presets_never_read_bundled_weights(self):
        with mock.patch.object(self.module, "_load_teacherfix_lora", side_effect=AssertionError("asset accessed")):
            for short, legacy in self.module.SIMPLE_PRESET_TO_LEGACY.items():
                original = FakeModel()
                result = self.module.DonutKrea2FusionControl().apply(
                    model=original, conditioning_in_1=object(), compatibility_preset=short)
                self.assertIs(result[0], original)
                self.assertEqual(self.module.base.calls[-1]["compatibility_preset"], legacy)
                self.assertIn(f"preset_label={short};", result[-1])

    def test_same_count_wrong_patch_keys_are_rejected(self):
        wrong = {f"wrong.{i}": () for i in range(33)}
        with mock.patch.object(self.module.base.comfy.lora, "load_lora", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "map all 33"):
                self.module._apply_teacherfix(FakeModel(), 1.0)

    def test_partial_model_patch_acceptance_is_rejected(self):
        with mock.patch.object(FakeModel, "add_patches", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "0/33"):
                self.module._apply_teacherfix(FakeModel(), 1.0)

    def test_incomplete_pairs_are_rejected(self):
        state = dict(self.module._load_teacherfix_lora()[0])
        state.pop(next(key for key in state if key.endswith(".lora_up.weight")))
        with self.assertRaisesRegex(RuntimeError, "complete"):
            self.module._validate_teacherfix_state(state)

    def test_wrong_alpha_is_rejected(self):
        state = dict(self.module._load_teacherfix_lora()[0])
        state[next(key for key in state if key.endswith(".alpha"))] = torch.tensor(8.0)
        with self.assertRaisesRegex(RuntimeError, "rank/alpha"):
            self.module._validate_teacherfix_state(state)

    def test_nonfinite_tensor_is_rejected(self):
        state = dict(self.module._load_teacherfix_lora()[0])
        key = next(key for key in state if key.endswith(".lora_down.weight"))
        state[key] = state[key].clone()
        state[key][0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self.module._validate_teacherfix_state(state)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

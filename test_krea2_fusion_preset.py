import ast
import hashlib
import unittest
from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets" / "krea2_c33_teacherfix_ema5000.safetensors"
EXPECTED_SHA256 = "db3c2b7612828120e7ef9cc8fe77124c6fd8de2e38f150599e62abd9695f6beb"
EXPECTED_TARGETS = 33


class BundledTeacherFixAssetTests(unittest.TestCase):
    def test_asset_is_exact_teacherfix_export(self):
        self.assertTrue(ASSET.is_file())
        self.assertEqual(hashlib.sha256(ASSET.read_bytes()).hexdigest(), EXPECTED_SHA256)

        with safe_open(str(ASSET), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = list(handle.keys())

        target_bases = {
            key[: -len(".lora_down.weight")]
            for key in keys
            if key.endswith(".lora_down.weight")
        }
        self.assertEqual(len(target_bases), EXPECTED_TARGETS)
        self.assertEqual(metadata.get("target_module_count"), str(EXPECTED_TARGETS))
        self.assertEqual(metadata.get("alpha"), "4.0")
        self.assertEqual(metadata.get("converted_for"), "Comfy Krea LoRA checkpoint sampling")

    def test_override_and_frontend_sources_parse(self):
        source = (ROOT / "DonutKrea2FusionPreset.py").read_text()
        ast.parse(source)
        self.assertIn('PRESET_TEACHERFIX = "DONUT settings: Krea2 C33 TeacherFix EMA5000"', source)
        self.assertIn('UI_MODES = (UI_MODE_SIMPLE, UI_MODE_ADVANCED)', source)

        js = (ROOT / "web" / "donut_krea2_fusion_simple_mode.js").read_text()
        self.assertIn('const SIMPLE = "Simple";', js)
        self.assertIn('const ADVANCED = "Advanced";', js)
        self.assertIn('const TEACHERFIX = "DONUT settings: Krea2 C33 TeacherFix EMA5000";', js)
        self.assertIn('name === "tap_strength"', js)


if __name__ == "__main__":
    unittest.main()

"""Bypass 2/3 projector diffs must hit the live Krea2 merge source.

txtfusion-reset (experimental bypass, fusion ratio 0) swaps in model2's
projector. add_patches on the outer FinePorn patcher is a silent no-op — the
same bug UncensorFix already routes around.
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

from test_model_merge_krea2 import module as merge
from test_uncensorfix_merge_bypass import Patcher, make_root

ROOT = Path(__file__).resolve().parent
PROJECTOR = "diffusion_model.txtfusion.projector.weight"
BODY = "diffusion_model.blocks.0.proj.weight"


def load_fusion_control():
    name = "_donut_bypass_projector_tests"
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT)]
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    extension = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"

    extension.WrappersMP = WrappersMP
    comfy.patcher_extension = extension
    sys.modules[name] = package
    sys.modules["comfy"] = comfy
    sys.modules["comfy.patcher_extension"] = extension

    def load(mod_name, path):
        spec = importlib.util.spec_from_file_location(name + "." + mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    load("donut_krea2_merge_serialization", ROOT / "donut_krea2_merge_serialization.py")
    load("donut_model_patch_routing", ROOT / "donut_model_patch_routing.py")
    return load("DonutKrea2FusionControl", ROOT / "DonutKrea2FusionControl.py")


class BypassProjectorMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fusion = load_fusion_control()

    def merged(self):
        base = Patcher(make_root([BODY, PROJECTOR], .2))
        source = Patcher(make_root([BODY, PROJECTOR], .8))
        result, = merge.DonutModelMergeKrea2().merge(
            base, source, "Experimental bypass", **{"first.": 1., "txtfusion.": 0.},
        )
        return result

    def test_bypass_2_diff_matches_filterbypass_lora_file(self):
        self.assertEqual(
            tuple(self.fusion._BYPASS_2_DIFF),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.51171875, -0.890625, 0.0, 0.0),
        )
        self.assertEqual(
            tuple(self.fusion._BYPASS_3_DIFF),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.51171875, -0.890625, -0.609375, 0.0),
        )

    def test_txtfusion_reset_merge_patches_model2_projector(self):
        merged = self.merged()
        original_source = merged.get_additional_models_with_key(merge._SOURCE_MODELS_KEY)[0]
        patched = self.fusion._apply_projector_diff(
            merged, self.fusion._BYPASS_2_DIFF, 1.0,
        )
        source = patched.get_additional_models_with_key(merge._SOURCE_MODELS_KEY)[0]
        self.assertNotIn(PROJECTOR, patched.patches)
        self.assertIn(PROJECTOR, source.patches)
        self.assertFalse(original_source.patches)
        self.assertIsNot(source, original_source)
        stored = source.patches[PROJECTOR][-1][1]
        self.assertEqual(stored[0], "diff")
        self.assertEqual(tuple(stored[1][0].reshape(-1).tolist()), self.fusion._BYPASS_2_DIFF)

    def test_unmerged_model_still_patches_the_outer_patcher(self):
        model = Patcher(make_root([PROJECTOR], .2))
        patched = self.fusion._apply_projector_diff(model, self.fusion._BYPASS_3_DIFF, 1.0)
        self.assertIn(PROJECTOR, patched.patches)
        self.assertFalse(model.patches)


if __name__ == "__main__":
    unittest.main()

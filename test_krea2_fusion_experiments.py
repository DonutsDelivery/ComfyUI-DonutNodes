import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parent
PACKAGE = "_donut_fusion_experiment_testpkg"

package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
stable = types.ModuleType(PACKAGE + ".DonutKrea2FusionPreset")
stable.PRESET_UNCENSORFIX = "UncensorFix"
stable.PRESET_CUSTOM = "Custom"


def rewrite(diagnostics, old, new):
    return str(diagnostics).replace(f"preset_label={old}", f"preset_label={new}", 1)


stable._rewrite_preset_diagnostics = rewrite


class StableNode:
    calls = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "compatibility_preset": (
                    ["Off", "Custom", "Rebalance", "Balanced", "UncensorFix"],
                    {"default": "Custom", "tooltip": "Stable presets."},
                )
            }
        }

    def apply(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ("model", "c1", None, None, None,
                f"preset_label={kwargs.get('compatibility_preset')}; preset_is_ui_only=true")


stable.DonutKrea2FusionControl = StableNode
sys.modules[PACKAGE] = package
sys.modules[stable.__name__] = stable
spec = importlib.util.spec_from_file_location(
    PACKAGE + ".donut_krea2_fusion_experiments",
    ROOT / "donut_krea2_fusion_experiments.py",
)
experiments = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiments
spec.loader.exec_module(experiments)


class Krea2FusionExperimentTests(unittest.TestCase):
    def test_schema_appends_all_experiments_without_removing_stable_presets(self):
        values = experiments.DonutKrea2FusionControl.INPUT_TYPES()["required"]["compatibility_preset"][0]
        self.assertIn("Balanced", values)
        self.assertIn("Rebalance", values)
        for name in experiments.EXPERIMENTAL_PRESETS:
            self.assertIn(name, values)
            self.assertEqual(values.count(name), 1)
        self.assertLess(values.index(experiments.PRESET_EXP_MEAN), values.index("UncensorFix"))

    def test_recipes_are_deliberately_distinct(self):
        settings = experiments.EXPERIMENTAL_SETTINGS
        mean = settings[experiments.PRESET_EXP_MEAN]
        static_rms = settings[experiments.PRESET_EXP_STATIC_RMS]
        power = settings[experiments.PRESET_EXP_POWER_060]
        tensor = settings[experiments.PRESET_EXP_TENSOR_075]

        self.assertEqual(mean["tap_normalization"], "mean_gain")
        self.assertEqual(static_rms["tap_normalization"], "rms_gain")
        self.assertEqual(power["tap_normalization"], "none")
        self.assertEqual(power["tap_formula"], "geometric_power")
        self.assertEqual(power["tap_strength"], 0.60)
        self.assertEqual(tensor["tap_normalization"], "tensor_rms")
        self.assertEqual(tensor["tap_strength"], 0.75)

    def test_backend_accepts_experimental_label_as_ui_helper(self):
        StableNode.calls.clear()
        name = experiments.PRESET_EXP_MEAN
        result = experiments.DonutKrea2FusionControl().apply(compatibility_preset=name)
        self.assertIn(f"preset_label={name}", result[-1])
        self.assertEqual(StableNode.calls[-1][1]["compatibility_preset"], "Custom")


if __name__ == "__main__":
    unittest.main()

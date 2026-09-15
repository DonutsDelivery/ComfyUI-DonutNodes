import unittest
from unittest.mock import patch

import donut_krea2_fusion_experiments as experiments


class Krea2FusionExperimentTests(unittest.TestCase):
    def test_schema_appends_all_experiments_without_removing_stable_presets(self):
        values = experiments.DonutKrea2FusionControl.INPUT_TYPES()["required"]["compatibility_preset"][0]
        self.assertIn("Balanced", values)
        self.assertIn("Rebalance", values)
        for name in experiments.EXPERIMENTAL_PRESETS:
            self.assertIn(name, values)
            self.assertEqual(values.count(name), 1)

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
        name = experiments.PRESET_EXP_MEAN
        fake = ("model", "c1", None, None, None, "preset_label=Custom; preset_is_ui_only=true")
        with patch.object(experiments.stable.DonutKrea2FusionControl, "apply", return_value=fake) as delegated:
            result = experiments.DonutKrea2FusionControl().apply(compatibility_preset=name)
        self.assertIn(f"preset_label={name}", result[-1])
        self.assertEqual(delegated.call_args.kwargs["compatibility_preset"], experiments.stable.PRESET_CUSTOM)


if __name__ == "__main__":
    unittest.main()

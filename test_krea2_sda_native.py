import unittest
from unittest.mock import patch

import donut_krea2_sda as sda


class NativeKrea2SDATests(unittest.TestCase):
    def test_sampler_exposes_appended_native_controls(self):
        optional = sda.DonutSampler.INPUT_TYPES()["optional"]
        names = list(optional)
        self.assertEqual(names[-2:], ["sda_enabled", "sda_strength"])
        self.assertFalse(optional["sda_enabled"][1]["default"])
        self.assertEqual(optional["sda_strength"][1]["default"], 1.0)

    def test_native_gate_reuses_existing_two_model_phase_engine(self):
        sampler = sda.DonutSampler()
        clean = object()
        sda_model = object()
        with patch.object(sda, "apply_krea2_sda", return_value=(sda_model, "Experimental bypass")), \
             patch.object(sda._BaseDonutSampler, "sample", return_value=("latent", "base info")) as base:
            latent, info = sampler.sample(
                clean,
                sda_enabled=True,
                sda_strength=0.9,
                steps=8,
                turbo_mode=True,
                denoise=1.0,
                mode="simple",
            )
        self.assertEqual(latent, "latent")
        self.assertIn("gate 2/8", info)
        self.assertIn("Experimental bypass", info)
        forwarded = base.call_args.kwargs
        self.assertIs(forwarded["model"], sda_model)
        self.assertIs(forwarded["model_2"], clean)
        self.assertIsNone(forwarded["model_3"])
        self.assertEqual(forwarded["mode"], "multi_model")
        self.assertEqual(forwarded["switch_at_step_1"], 2)
        self.assertEqual(forwarded["randomize_seed_per_model"], "disable")

    def test_invalid_sda_schedules_fail_before_loading_adapter(self):
        cases = [
            ({"steps": 8, "turbo_mode": False, "denoise": 1.0}, "Turbo mode"),
            ({"steps": 6, "turbo_mode": True, "denoise": 1.0}, "8-step"),
            ({"steps": 8, "turbo_mode": True, "denoise": 0.5}, "full-denoise"),
            ({"steps": 8, "turbo_mode": True, "denoise": 1.0, "edit_mode": True}, "editing"),
            ({"steps": 8, "turbo_mode": True, "denoise": 1.0, "mode": "multi_model"}, "two-model"),
            ({"steps": 8, "turbo_mode": True, "denoise": 1.0, "start_at_step": 1}, "step 0"),
            ({"steps": 8, "turbo_mode": True, "denoise": 1.0, "end_at_step": 6}, "complete 8-step"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message), \
                     patch.object(sda, "apply_krea2_sda") as apply:
                    sda.DonutSampler().sample(object(), sda_enabled=True, **kwargs)
                apply.assert_not_called()

    def test_experimental_bypass_uses_runtime_adapter_path(self):
        import DonutSafeApplyLoRAStack as safe

        model = object()
        result = object()
        lora = {"diffusion_model.blocks.0.example": object()}
        with patch.object(sda.folder_paths, "get_full_path", return_value="/models/sda.safetensors"), \
             patch.object(sda.comfy.utils, "load_torch_file", return_value=lora), \
             patch.object(sda, "resolve_execution_mode", return_value="Experimental bypass"), \
             patch.object(sda, "publish_execution_mode") as publish, \
             patch.object(safe, "_apply_bypass_applications", return_value=result) as bypass:
            actual, mode = sda.apply_krea2_sda(model, 1.0)
        self.assertIs(actual, result)
        self.assertEqual(mode, "Experimental bypass")
        applications = bypass.call_args.args[1]
        self.assertEqual(len(applications), 1)
        self.assertIs(applications[0][0], lora)
        self.assertEqual(applications[0][1], 1.0)
        self.assertEqual(len(applications[0][2].split(",")), 29)
        publish.assert_called_once_with(result, "Experimental bypass")

    def test_regular_mode_uses_existing_block_weight_loader_model_only(self):
        model = object()
        result = object()
        lora = {"diffusion_model.blocks.0.example": object()}
        with patch.object(sda.folder_paths, "get_full_path", return_value="/models/sda.safetensors"), \
             patch.object(sda.comfy.utils, "load_torch_file", return_value=lora), \
             patch.object(sda, "resolve_execution_mode", return_value="Comfy patches"), \
             patch.object(sda, "publish_execution_mode"), \
             patch.object(sda.LoraLoaderBlockWeight, "load_lora_for_models", return_value=(result, None, "vector")) as loader:
            actual, mode = sda.apply_krea2_sda(model, 0.75)
        self.assertIs(actual, result)
        self.assertEqual(mode, "Comfy patches")
        args = loader.call_args.args
        self.assertIs(args[0], model)
        self.assertIsNone(args[1])
        self.assertIs(args[2], lora)
        self.assertEqual(args[3], 0.75)
        self.assertEqual(args[4], 0.0)


if __name__ == "__main__":
    unittest.main()

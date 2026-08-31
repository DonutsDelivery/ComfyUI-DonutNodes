import unittest
from unittest.mock import Mock, patch

import torch

import comfy.samplers
import DonutKSamplerCFGLinear as sampler_module


class _FakePatcher:
    model_options = {}


class _FakeModelSampling:
    sigma_max = torch.tensor(10.0)

    def noise_scaling(self, sigma, noise, latent_image, max_denoise):
        return noise

    def inverse_noise_scaling(self, sigma, samples):
        return samples


class _InstalledSamplerProbe:
    inner_model = type("InnerModel", (), {"model_sampling": _FakeModelSampling()})()
    model_patcher = type(
        "ModelPatcher", (), {"model": type("Model", (), {})()}
    )()
    cfg = 1.0

    def __init__(self, guider):
        self.guider = guider
        self.cfg_indices = []

    def __call__(self, x, sigma, **kwargs):
        self.cfg_indices.append(self.guider._step_index)
        return torch.zeros_like(x)


class DynamicCFGGuiderTests(unittest.TestCase):
    def test_diagnostic_output_ignores_closed_manager_pipe(self):
        with patch.object(
            sampler_module.builtins,
            "print",
            side_effect=BrokenPipeError(32, "Broken pipe"),
        ):
            sampler_module._safe_print("diagnostic")

    def test_simple_sampler_does_not_return_source_latent_after_failure(self):
        engine = sampler_module._DonutSamplerEngine()
        latent = {"samples": torch.zeros(1, 16, 8, 8)}
        sampler_name = comfy.samplers.KSampler.SAMPLERS[0]

        with patch.object(
            sampler_module,
            "_common_ksampler_with_dynamic_cfg",
            side_effect=BrokenPipeError(32, "Broken pipe"),
        ):
            with self.assertRaises(BrokenPipeError):
                engine.run_simple(
                    object(), 1, 3, 1.0, 1.0, 1.0, 1, sampler_name,
                    "normal", object(), object(), latent, 1.0,
                )

    def test_cfg_schedule_advances_only_after_completed_sampler_steps(self):
        observed_cfg = []

        def parent_predict(guider, x, timestep, model_options=None, seed=None):
            observed_cfg.append(guider.cfg)
            return x

        guider = sampler_module._DynamicCFGGuider(
            _FakePatcher(), [8.0, 4.0, 1.0]
        )
        guider.set_cfg(99.0)

        with patch.object(
            comfy.samplers.CFGGuider, "predict_noise", new=parent_predict
        ):
            guider.predict_noise(torch.tensor([0.0]), torch.tensor([10.0]))
            guider.predict_noise(torch.tensor([0.0]), torch.tensor([5.0]))
            guider.set_completed_step(0)
            guider.predict_noise(torch.tensor([0.0]), torch.tensor([5.0]))
            guider.predict_noise(torch.tensor([0.0]), torch.tensor([2.0]))
            guider.set_completed_step(1)
            guider.predict_noise(torch.tensor([0.0]), torch.tensor([2.0]))

        self.assertEqual(observed_cfg, [8.0, 8.0, 4.0, 4.0, 1.0])
        self.assertEqual(guider.cfg, 99.0)

    def test_cfg_is_restored_when_parent_prediction_raises(self):
        guider = sampler_module._DynamicCFGGuider(_FakePatcher(), [8.0])
        guider.set_cfg(3.0)
        model_options = {
            "transformer_options": {"sample_sigmas": torch.tensor([1.0, 0.0])}
        }

        with patch.object(
            comfy.samplers.CFGGuider,
            "predict_noise",
            side_effect=RuntimeError("prediction failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                guider.predict_noise(
                    torch.tensor([0.0]), torch.tensor([1.0]), model_options
                )

        self.assertEqual(guider.cfg, 3.0)

    def test_repeated_or_out_of_order_callbacks_cannot_move_schedule_backward(self):
        guider = sampler_module._DynamicCFGGuider(
            _FakePatcher(), [8.0, 4.0, 1.0]
        )
        guider.set_completed_step(1)
        self.assertEqual(guider._step_index, 2)
        guider.set_completed_step(0)
        self.assertEqual(guider._step_index, 2)
        guider.set_completed_step(99)
        self.assertEqual(guider._step_index, 2)

    def test_cfg_alignment_preserves_partial_run_schedule_origin(self):
        self.assertEqual(
            sampler_module._aligned_cfg_values([8.0, 7.0, 6.0, 5.0], 2),
            [8.0, 7.0],
        )

    def test_installed_heun_callbacks_advance_once_per_sampler_step(self):
        guider = sampler_module._DynamicCFGGuider(
            _FakePatcher(), [8.0, 6.0, 4.0, 2.0]
        )
        probe = _InstalledSamplerProbe(guider)
        callbacks = []

        def callback(step, denoised, x, total_steps):
            callbacks.append(step)
            guider.set_completed_step(step)

        comfy.samplers.sampler_object("heun").sample(
            probe,
            torch.tensor([10.0, 5.0, 2.0, 1.0, 0.0]),
            {},
            callback,
            torch.zeros((1, 1, 2, 2)),
            disable_pbar=True,
        )

        self.assertEqual(callbacks, [0, 1, 2, 3])
        self.assertEqual(probe.cfg_indices, [0, 1, 1, 2, 2, 3, 3])


class DynamicCFGDispatchTests(unittest.TestCase):
    def test_simple_mode_uses_local_guider_without_replacing_global_class(self):
        engine = sampler_module._DonutSamplerEngine()
        original_guider = comfy.samplers.CFGGuider
        sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
        latent = {"samples": object()}

        with patch.object(
            sampler_module,
            "_common_ksampler_with_dynamic_cfg",
            return_value=(latent,),
        ) as dynamic_sample:
            result = engine.run_simple(
                object(), 1, 3, 8.0, 8.0, 2.0, 1, sampler_name,
                "normal", object(), object(), latent, 1.0,
            )

        self.assertIs(comfy.samplers.CFGGuider, original_guider)
        self.assertIs(result[0], latent)
        self.assertEqual(dynamic_sample.call_args.args[3], [8.0, 5.0, 2.0])

    def test_standard_lifecycle_receives_cfg_values_for_executed_sigma_slice(self):
        model = Mock()
        model.load_device = torch.device("cpu")
        model.model_options = {}
        latent_samples = torch.zeros((1, 4, 2, 2))
        latent = {"samples": latent_samples}
        fake_ksampler = Mock()
        fake_ksampler.sigmas = torch.tensor([10.0, 5.0, 0.0])
        fake_ksampler.sampler = "euler"
        fake_guider = Mock()
        preview_callback = Mock()

        def run_fake_sampler(*args, **kwargs):
            kwargs["callback"](0, "denoised", "x", 1)
            return latent_samples

        fake_guider.sample.side_effect = run_fake_sampler

        with (
            patch.object(
                sampler_module.comfy.sample,
                "fix_empty_latent_channels",
                return_value=latent_samples,
            ),
            patch.object(
                sampler_module.comfy.sample,
                "prepare_noise",
                return_value=latent_samples,
            ),
            patch.object(
                sampler_module.latent_preview,
                "prepare_callback",
                return_value=preview_callback,
            ),
            patch.object(
                sampler_module.comfy.samplers,
                "KSampler",
                return_value=fake_ksampler,
            ),
            patch.object(
                sampler_module.comfy.samplers,
                "sampler_object",
                return_value=object(),
            ),
            patch.object(
                sampler_module, "_DynamicCFGGuider", return_value=fake_guider
            ) as guider_class,
            patch.object(
                sampler_module.comfy.model_management,
                "intermediate_device",
                return_value=torch.device("cpu"),
            ),
            patch.object(
                sampler_module.comfy.model_management,
                "intermediate_dtype",
                return_value=torch.float32,
            ),
        ):
            result = sampler_module._common_ksampler_with_dynamic_cfg(
                model, 1, 2, [8.0, 4.0], "euler", "normal",
                object(), object(), latent, start_step=1,
            )

        self.assertTrue(torch.equal(result[0]["samples"], latent_samples))
        guider_class.assert_called_once_with(
            model, [8.0], log_prefix="DonutSampler"
        )
        fake_guider.set_conds.assert_called_once()
        fake_guider.set_cfg.assert_called_once_with(8.0)
        fake_guider.sample.assert_called_once()
        preview_callback.assert_called_once_with(0, "denoised", "x", 1)
        fake_guider.set_completed_step.assert_called_once_with(0)

    def test_multi_model_mode_preserves_phase_constant_cfg_behavior(self):
        engine = sampler_module._DonutSamplerEngine()
        sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
        latent = {"samples": object()}
        model_1 = object()
        model_2 = object()

        with patch(
            "nodes.common_ksampler",
            side_effect=lambda *args, **kwargs: (args[8],),
        ) as common_sample:
            result = engine.run_multi_model(
                model_1, "enable", 4, 8.0, 8.0, 2.0, 2,
                sampler_name, "normal", object(), object(), latent, 1,
                0, 4, "disable", "disable", 1.0,
                model_2=model_2, switch_at_step_1=2,
            )

        self.assertIs(result[0], latent)
        self.assertEqual(common_sample.call_count, 2)
        first, second = common_sample.call_args_list
        self.assertEqual(first.args[3], 8.0)
        self.assertEqual(second.args[3], 4.0)


if __name__ == "__main__":
    unittest.main()

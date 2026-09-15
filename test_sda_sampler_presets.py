"""Bleh/V4 regression contracts with real CPU tensors and interface doubles.

The dispatch fixture follows Bleh's wrapper contract at b889683c: a live slot
contains (KSAMPLER, override_sigmas), and its extra_options are passed through.
This is not a GPU/installed-ComfyUI or actual ER-SDE numerical-parity test.
"""
from functools import partial, update_wrapper
import sys
import json
import types
import unittest
from unittest.mock import Mock, patch

import torch
import test_krea2_sda_native as fixtures


class BlehSDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SDATests.setUpClass.__func__(cls)
        cls.bleh = types.ModuleType('_test_bleh_package.py.nodes.samplers')
        cls.bleh.BLEH_PRESET_COUNT = 4
        cls.bleh.BLEH_PRESET = [None] * 4
        setter = type('BlehSetSamplerPreset', (), {'__module__': cls.bleh.__name__})
        cls.bleh.BlehSetSamplerPreset = setter
        registry = cls.bleh

        def preset_wrapper(index, model, x, sigmas, *args, **kwargs):
            sampler, override = registry.BLEH_PRESET[index]
            if override is not None:
                sigmas = override.detach().clone().to(sigmas)
            return sampler.sampler_function(model, x, sigmas, *args,
                                            **sampler.extra_options, **kwargs)

        cls.bleh.bleh_sampler_preset_wrapper = preset_wrapper
        cls.nodes = types.ModuleType('nodes')
        cls.nodes.NODE_CLASS_MAPPINGS = {'BlehSetSamplerPreset': setter}
        cls.registry_patch = patch.dict(sys.modules, {
            cls.bleh.__name__: cls.bleh, 'nodes': cls.nodes,
        })
        cls.registry_patch.start()
        cls.addClassCleanup(cls.registry_patch.stop)

    def setUp(self):
        fixtures.SDATests.setUp(self)
        self.bleh.BLEH_PRESET[:] = [None] * 4
        self.bleh.BLEH_PRESET_COUNT = 4
        self.kernel_calls, self.predictions, self.history, self.callback_steps = [], [], [], []

        def kernel(model, x, sigmas, *, extra_args=None, callback=None, disable=False, **options):
            self.kernel_calls.append((model, x, sigmas, extra_args, callback, disable, options))
            state = x.clone()
            for i, sigma in enumerate(sigmas[:-1]):
                self.history.append(state.clone())
                prediction = model(state, sigma)
                self.predictions.append(prediction.item())
                state = state * .5 + prediction
                if callback is not None:
                    callback(i, prediction, state, len(sigmas) - 1)
            return state

        self.kernel = kernel
        self.kernels.sample_er_sde = kernel
        self.kernels.sample_euler = lambda *a, **k: None
        self.kernels.sample_dpmpp_2m = lambda *a, **k: None

        def ode_noise_scaler(x):
            return x
        self.scaler = ode_noise_scaler
        # SamplerER_SDE("ODE", 3, eta=0, s_noise=1) resolves s_noise to 0.
        self.ode = self.sampler(self.kernel, s_noise=0.0, max_stage=3, noise_scaler=self.scaler)
        self.bleh.BLEH_PRESET[0] = (self.ode, None)
        self.outer = self.sampler(partial(self.bleh.bleh_sampler_preset_wrapper, 0))

    @staticmethod
    def sampler(function, **options):
        return types.SimpleNamespace(sampler_function=function, extra_options=options,
                                     inpaint_options={})

    def inspect(self, sampler=None):
        return self.sampler_compat.inspect_sda_sampler(
            self.outer if sampler is None else sampler, self.schedule.SDA_SAMPLERS)

    def test_reference_api_uses_v4_preset_and_orders_registration_before_sampling(self):
        graph = json.loads((fixtures.ROOT / "workflows/examples/krea2_sda_v4_bleh_api.json").read_text())
        self.assertEqual(graph['18']['inputs'],
                         dict(solver_type='ODE', max_stage=3, eta=0.0, s_noise=1.0))
        self.assertEqual(graph['19']['inputs'], dict(sampler=['18', 0], any_input=['1', 0],
                                                   preset=0, discard_penultimate_sigma=False))
        for name in ['7', '8']:
            self.assertEqual(graph[name]['inputs']['model'], ['19', 0])
            self.assertEqual(graph[name]['inputs']['sampler_name'], 'bleh_preset_0')
            self.assertEqual(graph[name]['inputs']['scheduler'], 'beta')
        baseline, active = graph['7']['inputs'], graph['8']['inputs']
        self.assertFalse(baseline['sda_enabled'])
        self.assertTrue(active['sda_enabled'])
        self.assertEqual({k: v for k, v in baseline.items() if k != 'sda_enabled'},
                         {k: v for k, v in active.items() if k != 'sda_enabled'})
        for node in graph.values():
            for value in node['inputs'].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)

    def test_v4_alias_resolves_ode_without_changing_options_or_registry(self):
        entry = self.bleh.BLEH_PRESET[0]
        function = self.outer.sampler_function
        route, options = self.inspect()
        self.assertEqual(route, 'bleh_preset_0 -> er_sde')
        self.assertEqual(options['s_noise'], 0.)
        self.assertEqual(options['max_stage'], 3)
        self.assertIs(options['noise_scaler'], self.scaler)
        self.assertIs(self.bleh.BLEH_PRESET[0], entry)
        self.assertIs(self.outer.sampler_function, function)
        options['s_noise'] = 99
        self.assertEqual(self.ode.extra_options['s_noise'], 0.)

    def test_direct_supported_kernels_do_not_require_bleh(self):
        with patch.dict(self.nodes.NODE_CLASS_MAPPINGS, {}, clear=True):
            for name in self.schedule.SDA_SAMPLERS:
                route, _ = self.inspect(self.sampler(getattr(self.kernels, 'sample_' + name)))
                self.assertEqual(route, name)

    def test_registry_is_reread_and_different_solver_is_not_assumed_safe(self):
        self.inspect()
        def sample_heun(*args, **kwargs):
            pass
        self.bleh.BLEH_PRESET[0] = (self.sampler(sample_heun), None)
        with self.assertRaisesRegex(ValueError, 'bleh_preset_0 -> sample_heun'):
            self.inspect()

    def test_missing_slot_registration_is_actionable(self):
        self.bleh.BLEH_PRESET[0] = None
        with self.assertRaisesRegex(ValueError, 'BlehSetSamplerPreset'):
            self.inspect()

    def test_sigma_override_fails_instead_of_gating_the_wrong_schedule(self):
        self.bleh.BLEH_PRESET[0] = (self.ode, fixtures.SIGMAS.clone())
        with self.assertRaisesRegex(ValueError, 'override_sigmas_opt'):
            self.inspect()

    def test_arbitrary_wrapper_with_stock_name_is_not_trusted(self):
        def extra_evaluations(*args, **kwargs):
            pass
        extra_evaluations.__name__ = 'sample_er_sde'
        with self.assertRaises(ValueError):
            self.inspect(self.sampler(extra_evaluations))
        wrapped = update_wrapper(partial(extra_evaluations), self.kernel)
        with self.assertRaisesRegex(ValueError, 'custom sampler wrapper'):
            self.inspect(self.sampler(wrapped))

    def test_foreign_partial_does_not_gain_trust_from_its_name(self):
        def fake_wrapper(*args, **kwargs):
            pass
        fake_wrapper.__name__ = 'bleh_sampler_preset_wrapper'
        with self.assertRaisesRegex(ValueError, 'custom sampler wrapper'):
            self.inspect(self.sampler(partial(fake_wrapper, 0)))

    def test_missing_bleh_node_mapping_fails_cleanly(self):
        with patch.dict(self.nodes.NODE_CLASS_MAPPINGS, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'installed BlehSetSamplerPreset'):
                self.inspect()

    def test_bad_slot_indices_binding_and_entry_layout_are_rejected(self):
        for index in [-1, 4, True, '0']:
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'preset index'):
                self.inspect(self.sampler(partial(self.bleh.bleh_sampler_preset_wrapper, index)))
        with self.assertRaisesRegex(ValueError, 'binding'):
            self.inspect(self.sampler(partial(self.bleh.bleh_sampler_preset_wrapper, preset_idx=0)))
        self.bleh.BLEH_PRESET[0] = (self.ode,)
        with self.assertRaisesRegex(ValueError, 'registry layout'):
            self.inspect()

    def test_duplicate_options_match_bleh_error_not_invented_precedence(self):
        self.outer.extra_options['s_noise'] = 1.
        with self.assertRaisesRegex(ValueError, 'duplicate options'):
            self.inspect()

    def test_churn_checked_inside_preset(self):
        self.bleh.BLEH_PRESET[0] = (self.sampler(self.kernels.sample_euler, s_churn=1.), None)
        with self.assertRaisesRegex(ValueError, 'churn'):
            self.inspect()
        self.bleh.BLEH_PRESET[0][0].extra_options['s_churn'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'churn'):
            self.inspect()

    def test_nested_presets_resolve_and_cycles_fail(self):
        self.bleh.BLEH_PRESET[1] = (self.ode, None)
        second = self.sampler(partial(self.bleh.bleh_sampler_preset_wrapper, 1))
        self.bleh.BLEH_PRESET[0] = (second, None)
        self.assertEqual(self.inspect()[0], 'bleh_preset_0 -> bleh_preset_1 -> er_sde')
        self.bleh.BLEH_PRESET[1] = (self.outer, None)
        with self.assertRaisesRegex(ValueError, 'cycle'):
            self.inspect()

    def test_native_node_keeps_v4_name_beta_seed_and_execution_mode(self):
        with patch.object(self.sda, '_sda_path', return_value='checked-fixture'), \
             patch.object(self.sda, '_file_identity', return_value=('fixture',)), \
             patch.object(self.sda, '_load_verified_lora', return_value={'target.lora_up.weight': torch.ones(1, 1)}):
            for execution in ['Comfy patches', 'Experimental bypass']:
                model = fixtures.Model()
                model.model_options['donut_lora_execution_mode'] = execution
                for mode in ['simple', 'advanced']:
                    with self.subTest(execution=execution, mode=mode):
                        before = len(fixtures.BaseSampler.calls)
                        args = fixtures.parameters(mode=mode, sampler_name='bleh_preset_0', scheduler='beta')
                        self.sda.DonutSampler().sample(model, sda_enabled=True, **args)
                        call = fixtures.BaseSampler.calls[-1]
                        self.assertEqual(len(fixtures.BaseSampler.calls), before + 1)
                        self.assertEqual(call['sampler_name'], 'bleh_preset_0')
                        self.assertEqual(call['scheduler'], 'beta')
                        self.assertEqual(call['seed'], 42)
                        self.assertEqual(call['mode'], mode)
                        self.assertEqual(call['model'].model_options['donut_lora_execution_mode'], execution)

    def test_invalid_sampler_error_includes_actual_selection(self):
        with self.assertRaisesRegex(ValueError, "sampler 'heun'"):
            self.sda._validate_sda_sampling(fixtures.parameters(
                edit_mode=False, mode='simple', sampler_name='heun'), self.schedule.SDA_SAMPLERS)

    def test_guard_calls_original_preset_once_with_same_sigmas_options_and_callbacks(self):
        # Exercise both physical SDA adapter paths through the preset dispatch.
        sigmas = fixtures.SIGMAS  # non-linear schedule supplied by caller, not replaced
        for execution in ['Comfy patches', 'Experimental bypass']:
            with self.subTest(execution=execution):
                self.kernel_calls.clear(); self.history.clear(); self.predictions.clear()
                model = fixtures.Model()
                layer = model.model.diffusion_model.blocks[0]
                original_forward = layer.forward
                patched, pos, _ = self.schedule.prepare_sda(model, [[torch.ones(1, 1), {}]], [],
                    {fixtures.TARGET: fixtures.ToyAdapter()}, 1., execution)
                if execution == 'Experimental bypass':
                    wrap = patched.wrappers[('apply_model', self.schedule.SDA_WRAPPER_KEY)]
                    class Physical:
                        class_obj = model.model
                        def __call__(self, x, *args, **kwargs):
                            return layer(torch.ones_like(x))
                    def denoiser(x, sigma):
                        return wrap(Physical(), x, sigma, transformer_options={'sample_sigmas': sigmas})
                else:
                    hook = pos[0][1]['hooks'].hooks[0]
                    def denoiser(x, sigma):
                        hook.hook_keyframe.prepare_current_keyframe(sigma, {'sample_sigmas': sigmas})
                        return torch.full_like(x, 2. + .5 * hook.hook_keyframe.strength)
                executor = Mock()
                executor.class_obj = self.outer
                def execute(model_wrap, schedule, extra_args, callback, noise, *args):
                    return self.outer.sampler_function(model_wrap, noise, schedule,
                        extra_args=extra_args, callback=callback, disable=False, **self.outer.extra_options)
                executor.side_effect = execute
                extra_args, noise, latent, callback = {'seed': 42}, torch.zeros(1, 1), torch.ones(1, 1), Mock()
                with self.assertLogs(level='INFO') as logs:
                    self.schedule._sampling_guard(executor, denoiser, sigmas, extra_args, callback, noise, latent)
                executor.assert_called_once()
                self.assertEqual(len(self.kernel_calls), 1)
                call = self.kernel_calls[0]
                self.assertIs(call[1], noise)
                self.assertIs(call[2], sigmas)
                self.assertIs(call[3], extra_args)
                self.assertIs(call[4], callback)
                self.assertEqual(call[6]['s_noise'], 0.)
                self.assertEqual(call[6]['max_stage'], 3)
                self.assertIs(call[6]['noise_scaler'], self.scaler)
                self.assertEqual(self.predictions, [2.5, 2.5] + [2.] * 6)
                self.assertEqual(self.history[2].item(), 3.75, 'no history restart at gate')
                self.assertEqual(callback.call_count, 8)
                self.assertEqual(layer.forward, original_forward)
                self.assertIn('bleh_preset_0 -> er_sde', '\n'.join(logs.output))

    def test_runtime_rejection_does_not_execute_or_substitute_sampler(self):
        self.bleh.BLEH_PRESET[0] = (self.sampler(lambda: None), None)
        executor = Mock()
        executor.class_obj = self.outer
        with self.assertRaises(ValueError):
            self.schedule._sampling_guard(executor, None, fixtures.SIGMAS, {}, None, torch.zeros(1))
        executor.assert_not_called()


if __name__ == '__main__':
    unittest.main()

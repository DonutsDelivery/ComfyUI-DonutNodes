"""CPU contracts using production CFG functions and explicit ComfyUI doubles.

AST loading avoids importing an unavailable ComfyUI/GPU application. The guider,
interval slicing, curve calculations, and diagnostic formatting are unmodified
source from the checkout; sampling and model predictions are test doubles.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


class CFGIntervalTests(unittest.TestCase):
    def setUp(self):
        self.predictions = []
        self.samples = []
        self.previews = []
        self.sigmas = torch.arange(8, -1, -1, dtype=torch.float32) / 8
        owner = self

        class GuiderBase:
            def __init__(self, model):
                self.cfg = 1.
            def set_conds(self, positive, negative):
                self.conds = (positive, negative)
            def set_cfg(self, cfg):
                self.cfg = cfg
            def predict_noise(self, x, timestep, model_options, seed):
                owner.predictions.append(self.cfg)
                return x
            def sample(self, noise, latent, sampler, sigmas, callback, **kwargs):
                owner.samples.append((tuple(self.cfg_values), sigmas.clone(), kwargs))
                for step, sigma in enumerate(sigmas[:-1]):
                    # Two evaluations in an interval must share its CFG.
                    self.predict_noise(latent, sigma, seed=kwargs['seed'])
                    self.predict_noise(latent, sigma, seed=kwargs['seed'])
                    callback(step, latent, latent, len(sigmas) - 1)
                return latent

        class KSampler:
            SAMPLERS = ['euler']
            def __init__(self, *args, **kwargs):
                self.sigmas = owner.sigmas.clone()
                self.sampler = 'euler'

        self.ns = dict(torch=torch, print=lambda *a, **k: None,
            comfy=NS(samplers=NS(CFGGuider=GuiderBase, KSampler=KSampler, sampler_object=lambda name: object()),
                sample=NS(fix_empty_latent_channels=lambda model, value, *a: value,
                          prepare_noise=lambda value, *a: torch.zeros_like(value)),
                utils=NS(PROGRESS_BAR_ENABLED=False),
                model_management=NS(intermediate_device=lambda: 'cpu', intermediate_dtype=lambda: torch.float32)),
            latent_preview=NS(prepare_callback=lambda *a: lambda *args: self.previews.append(args[0])))
        path = Path(__file__).resolve().parents[1] / 'DonutKSamplerCFGLinear.py'
        names = {'_aligned_cfg_values', '_common_ksampler_with_dynamic_cfg', '_DynamicCFGGuider', '_DonutSamplerEngine'}
        tree = ast.parse(path.read_text())
        body = [n for n in tree.body if getattr(n, 'name', '') in names]
        self.assertEqual({n.name for n in body}, names)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), self.ns)
        self.curves = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == '_CFG_CURVES' for t in n.targets)))
        self.engine = self.ns['_DonutSamplerEngine']()

    def run_range(self, start=None, end=None, values=None, **kwargs):
        count = len(self.sigmas) - 1
        return self.ns['_common_ksampler_with_dynamic_cfg'](
            NS(load_device='cpu', model_options={}), 123, count,
            list(range(count, 0, -1)) if values is None else values, 'euler', 'simple', 'pos', 'neg',
            {'samples': torch.zeros(1, 1, 2, 2), 'noise_mask': 'mask', 'custom': 'retained'},
            start_step=start, last_step=end, **kwargs)

    def test_partial_cfg_and_sigmas_use_same_global_interval_indices(self):
        for start, end in ((None, None), (0, 4), (2, 6), (4, 8), (7, 8)):
            with self.subTest(start=start, end=end):
                self.predictions.clear()
                self.previews.clear()
                result, = self.run_range(start, end)
                first, last = start or 0, 8 if end is None else end
                expected = list(range(8, 0, -1))[first:last]
                values, sigmas, options = self.samples[-1]
                self.assertEqual(list(values), expected)
                torch.testing.assert_close(sigmas, self.sigmas[first:last + 1])
                self.assertEqual(self.predictions, [v for v in expected for _ in range(2)])
                self.assertEqual(self.previews, list(range(last - first)))
                self.assertEqual(options['denoise_mask'], 'mask')
                self.assertEqual(options['seed'], 123)
                self.assertEqual(result['custom'], 'retained')

    def test_force_full_denoise_changes_terminal_sigma_not_cfg_indices(self):
        self.run_range(2, 6, force_full_denoise=True)
        values, sigmas, _ = self.samples[-1]
        self.assertEqual(values, (6, 5, 4, 3))
        self.assertEqual(sigmas[-1], 0)
        torch.testing.assert_close(sigmas[:-1], self.sigmas[2:6])

    def test_effective_tail_schedule_uses_its_own_cfg_then_explicit_offset(self):
        # Denoise/Turbo schedule construction is outside this double: supply
        # an already shortened four-interval sigma schedule as KSampler does.
        self.sigmas = self.sigmas[4:]
        self.run_range(1, 4, values=[4., 3., 2., 1.], denoise=.5)
        values, sigmas, _ = self.samples[-1]
        self.assertEqual(values, (3., 2., 1.))
        torch.testing.assert_close(sigmas, self.sigmas[1:])

    def test_constant_cfg_and_disabled_noise_keep_interval_selection(self):
        self.run_range(4, 8, values=[1.] * 8, disable_noise=True)
        self.assertEqual(self.samples[-1][0], (1.,) * 4)

    def test_empty_ranges_still_raise(self):
        for start, end in ((4, 4), (8, 8), (-1, 8), (None, 0)):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, 'no denoising steps'):
                self.run_range(start, end)
        self.assertEqual(self.samples, [])

    def test_alignment_repeats_last_cfg_when_custom_schedule_is_longer(self):
        align = self.ns['_aligned_cfg_values']
        self.assertEqual(align([8., 4., 1.], 4, start_step=1), [4., 1., 1., 1.])
        with self.assertRaises(ValueError):
            align([], 4)

    def test_midpoint_clamps_inside_range_in_all_modes(self):
        for mode in ('simple', 'advanced', 'multi'):
            for midpoint in (-1, 0, 3, 7, 8, 10000):
                with self.subTest(mode=mode, midpoint=midpoint):
                    fn = getattr(self.engine, mode + '_calculate_cfg_for_step')
                    values = [fn(i, 8, 8., 4., 1., midpoint) for i in range(8)]
                    self.assertEqual(values[0], 8.)
                    self.assertEqual(values[-1], 1.)
                    self.assertEqual(values[max(1, min(midpoint, 6))], 4.)

    def test_one_two_and_three_step_schedules_have_explicit_endpoints(self):
        for mode in ('simple', 'advanced', 'multi'):
            for steps, expected in ((1, [8.]), (2, [8., 1.]), (3, [8., 4., 1.])):
                with self.subTest(mode=mode, steps=steps):
                    fn = getattr(self.engine, mode + '_calculate_cfg_for_step')
                    self.assertEqual([fn(i, steps, 8., 4., 1., 8) for i in range(steps)], expected)

    def test_advanced_curve_endpoints_survive_midpoint_clamping(self):
        for curve in self.curves:
            with self.subTest(curve=curve):
                values = [self.engine.advanced_calculate_cfg_for_step(i, 8, 8., 4., 1., 8, curve)
                          for i in range(8)]
                self.assertAlmostEqual(values[0], 8.)
                self.assertAlmostEqual(values[-2], 4.)
                self.assertAlmostEqual(values[-1], 1.)

    def test_advanced_diagnostics_report_executed_steps_without_double_offset(self):
        self.engine.cfg_history = list(enumerate(range(8, 0, -1)))
        info = self.engine.format_advanced_cfg_info(8., 8., 1., 8, 8, 'euler', 'simple', 4, 8, 'enable', 'disable')
        self.assertIn('Step 5: CFG=4.00', info)
        self.assertIn('Step 8: CFG=1.00', info)
        self.assertNotIn('Step 1:', info)
        self.assertNotIn('Step 9:', info)

    def test_interior_midpoint_and_constant_cfg_controls(self):
        fn = self.engine.simple_calculate_cfg_for_step
        self.assertEqual([fn(i, 5, 8., 4., 1., 2) for i in range(5)], [8., 6., 4., 2.5, 1.])
        self.assertEqual([fn(i, 8, 1., 1., 1., 8) for i in range(8)], [1.] * 8)


if __name__ == '__main__':
    unittest.main()

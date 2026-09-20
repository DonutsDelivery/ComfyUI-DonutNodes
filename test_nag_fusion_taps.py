"""NAG's separate negative stream must receive the upstream tap transform once."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

fake = types.ModuleType('comfy')
ext = types.ModuleType('comfy.patcher_extension')
ext.WrappersMP = types.SimpleNamespace(DIFFUSION_MODEL='diffusion_model')
fake.patcher_extension = ext
with patch.dict(sys.modules, {'comfy': fake, 'comfy.patcher_extension': ext}):
    spec = importlib.util.spec_from_file_location('fusion_under_test', Path(__file__).with_name('DonutKrea2FusionControl.py'))
    fusion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion)


class NAGTapTests(unittest.TestCase):
    def test_balanced_preserves_rms_and_is_applied_once(self):
        torch.manual_seed(7)
        value = torch.randn(1, 3, 30720)
        original = value.clone()
        gains = (1.,)*7 + (2.5, 5., 1.1, 4., 1.)
        config = dict(tap_method=fusion.TAP_METHOD_DONUT, tap_gains=gains,
                      tap_normalization='tensor_rms')
        model = types.SimpleNamespace(model_options={'transformer_options': {fusion.FUSION_BUDGET_KEY: config}})
        raw = [[value, {'keep': 'metadata'}]]
        result = fusion.prepare_nag_conditioning(model, raw)
        taps = value.reshape(1, 3, 12, 2560)
        expected = taps * torch.tensor(gains).reshape(1, 1, 12, 1)
        expected *= taps.square().mean().sqrt() / expected.square().mean().sqrt()
        torch.testing.assert_close(result[0][0], expected.reshape_as(value))
        torch.testing.assert_close(value, original)
        self.assertEqual(raw[0][1], {'keep': 'metadata'})
        second = fusion.prepare_nag_conditioning(model, result)
        self.assertIs(second[0][0], result[0][0])

    def test_rebalance_preserves_upstream_cast_then_multiplier_order(self):
        value = torch.randn(1, 2, 30720).bfloat16()
        profile = (1.,)*7 + (2.5, 5., 1.1, 4., 1.)
        config = dict(tap_method=fusion.TAP_METHOD_REBALANCE,
                      tap_gains=tuple(x * .7 for x in profile), tap_normalization='none',
                      tap_multiplier=.7, tap_profile_values=profile)
        model = types.SimpleNamespace(model_options={'transformer_options': {fusion.FUSION_BUDGET_KEY: config}})
        actual = fusion.prepare_nag_conditioning(model, [[value, {}]])[0][0]
        expected = (value.float().reshape(1, 2, 12, 2560) * torch.tensor(profile).reshape(1, 1, 12, 1)).reshape_as(value).bfloat16() * .7
        self.assertTrue(torch.equal(actual, expected))

    def test_rebalance_ignores_tensor_rms_normalization(self):
        value = torch.randn(1, 2, 30720).bfloat16()
        profile = (1.,)*7 + (2.5, 5., 1.1, 4., 1.)
        base = dict(tap_method=fusion.TAP_METHOD_REBALANCE,
                    tap_gains=profile, tap_multiplier=1., tap_profile_values=profile)
        def run(normalization):
            config = dict(base, tap_normalization=normalization)
            model = types.SimpleNamespace(model_options={'transformer_options': {fusion.FUSION_BUDGET_KEY: config}})
            return fusion.prepare_nag_conditioning(model, [[value.clone(), {}]])[0][0]
        torch.testing.assert_close(run('tensor_rms').float(), run('none').float())

    def test_no_tap_change_is_passthrough(self):
        raw = [[torch.ones(1, 2, 30720), {}]]
        model = types.SimpleNamespace(model_options={})
        self.assertIs(fusion.prepare_nag_conditioning(model, raw), raw)
        model.model_options = {'transformer_options': {fusion.FUSION_BUDGET_KEY: {'tap_gains': (1.,)*12}}}
        self.assertIs(fusion.prepare_nag_conditioning(model, raw), raw)

    def test_nag_match_taps_off_leaves_negative_raw(self):
        value = torch.randn(1, 2, 30720)
        gains = (1.,)*7 + (2.5, 5., 1.1, 4., 1.)
        config = dict(tap_method=fusion.TAP_METHOD_DONUT, tap_gains=gains,
                      tap_normalization='tensor_rms', nag_match_taps=False)
        model = types.SimpleNamespace(model_options={'transformer_options': {fusion.FUSION_BUDGET_KEY: config}})
        raw = [[value, {}]]
        self.assertIs(fusion.prepare_nag_conditioning(model, raw), raw)

    def test_standalone_nag_node_receives_fusion_taps(self):
        captured = []
        class NAG:
            def patch(self, model, nag_negative, phi=4., tau=2.5, alpha=.25,
                      sigma_start=1000., sigma_end=0.):
                captured.append(nag_negative)
                return (model,)
        nodes = types.ModuleType('nodes')
        nodes.NODE_CLASS_MAPPINGS = {'Krea2NormalizedAttentionGuidance': NAG}
        value = torch.ones(1, 2, 30720)
        profile = (1.,)*7 + (2.5, 5., 1.1, 4., 1.)
        config = dict(tap_method=fusion.TAP_METHOD_REBALANCE,
                      tap_gains=profile, tap_normalization='none',
                      tap_multiplier=1., tap_profile_values=profile)
        model = types.SimpleNamespace(model_options={'transformer_options': {fusion.FUSION_BUDGET_KEY: config}})
        with patch.dict(sys.modules, {'nodes': nodes}):
            fusion.ensure_standalone_nag_uses_fusion_taps()
            NAG().patch(model, [[value, {}]])
            fusion.ensure_standalone_nag_uses_fusion_taps()
        expected = fusion.prepare_nag_conditioning(model, [[value, {}]])[0][0]
        torch.testing.assert_close(captured[0][0][0], expected)
        self.assertEqual(len(captured), 1)

    def test_rebalance_is_applied_once_when_prepare_runs_twice(self):
        value = torch.randn(1, 2, 30720)
        profile = (1.,)*7 + (2.5, 5., 1.1, 4., 1.)
        config = dict(tap_method=fusion.TAP_METHOD_REBALANCE,
                      tap_gains=profile, tap_normalization='none',
                      tap_multiplier=1., tap_profile_values=profile)
        model = types.SimpleNamespace(model_options={'transformer_options': {fusion.FUSION_BUDGET_KEY: config}})
        first = fusion.prepare_nag_conditioning(model, [[value, {}]])
        second = fusion.prepare_nag_conditioning(model, first)
        self.assertIs(second[0][0], first[0][0])
        expected = (value.float().reshape(1, 2, 12, 2560) * torch.tensor(profile).reshape(1, 1, 12, 1)).reshape_as(value)
        torch.testing.assert_close(first[0][0], expected)

    def test_donut_sampler_patch_callable_skips_the_standalone_wrap(self):
        seen = []
        class NAG:
            def patch(self, model, nag_negative, **kwargs):
                seen.append('original')
                return (model,)
        nodes = types.ModuleType('nodes')
        nodes.NODE_CLASS_MAPPINGS = {'Krea2NormalizedAttentionGuidance': NAG}
        with patch.dict(sys.modules, {'nodes': nodes}):
            fusion.ensure_standalone_nag_uses_fusion_taps()
            fn = fusion.nag_patch_callable(NAG)
            self.assertIs(fn, getattr(NAG.patch, fusion._NAG_PATCH_ORIGINAL))
            fn(NAG(), model=object(), nag_negative=[[torch.ones(1, 2, 30720), {}]])
        self.assertEqual(seen, ['original'])


if __name__ == '__main__':
    unittest.main()

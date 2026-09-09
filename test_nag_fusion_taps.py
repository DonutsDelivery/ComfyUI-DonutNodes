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

    def test_no_tap_change_is_passthrough(self):
        raw = [[torch.ones(1, 2, 30720), {}]]
        model = types.SimpleNamespace(model_options={})
        self.assertIs(fusion.prepare_nag_conditioning(model, raw), raw)
        model.model_options = {'transformer_options': {fusion.FUSION_BUDGET_KEY: {'tap_gains': (1.,)*12}}}
        self.assertIs(fusion.prepare_nag_conditioning(model, raw), raw)

if __name__ == '__main__':
    unittest.main()

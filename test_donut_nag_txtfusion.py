"""UncensorFix NAG grain: one txtfusion call for both streams; budget survives clone."""
import types
import unittest
from unittest.mock import patch
import torch

from donut_nag_txtfusion import _fused_text
from DonutKrea2FusionControl import FUSION_BUDGET_KEY, copy_fusion_budget, prepare_nag_conditioning


class TxtfusionBatchTests(unittest.TestCase):
    def test_matching_shapes_call_txtfusion_once(self):
        calls = []

        class Fusion:
            def __call__(self, ctx, mask=None, transformer_options=None):
                calls.append(ctx.shape[0])
                return ctx

        class Model:
            txtfusion = Fusion()

            def txtmlp(self, value):
                return value

        pos = torch.ones(1, 12, 2560)
        neg = torch.zeros(1, 12, 2560)
        out_pos, out_neg = _fused_text(Model(), pos, neg, {})
        self.assertEqual(calls, [2])
        torch.testing.assert_close(out_pos, pos)
        torch.testing.assert_close(out_neg, neg)

    def test_nag_alpha_blends_both_streams_toward_mid_rms(self):
        class Model:
            def txtfusion(self, ctx, mask=None, transformer_options=None):
                return ctx

            def txtmlp(self, value):
                return value

        pos = torch.ones(1, 4, 8)
        neg = torch.ones(1, 4, 8) * 4
        opts = {"donut_krea2_fusion_budget": {"nag_match_taps": True}}
        out_pos, out_neg = _fused_text(Model(), pos, neg, opts, nag_alpha=1.0)
        mid = (1.0 + 4.0) * 0.5
        torch.testing.assert_close(out_pos, torch.ones(1, 4, 8) * mid)
        torch.testing.assert_close(out_neg, torch.ones(1, 4, 8) * mid)
        half_pos, half_neg = _fused_text(Model(), pos, neg, opts, nag_alpha=0.5)
        torch.testing.assert_close(half_pos, torch.ones(1, 4, 8) * (1.0 * 0.5 + mid * 0.5))
        torch.testing.assert_close(half_neg, torch.ones(1, 4, 8) * (4.0 * 0.5 + mid * 0.5))

    def test_fused_energy_match_off_leaves_negative(self):
        class Model:
            def txtfusion(self, ctx, mask=None, transformer_options=None):
                return ctx

            def txtmlp(self, value):
                return value

        pos = torch.ones(1, 4, 8)
        neg = torch.ones(1, 4, 8) * 4
        _, out_neg = _fused_text(
            Model(), pos, neg, {"donut_krea2_fusion_budget": {"nag_match_taps": False}},
            nag_alpha=1.0,
        )
        torch.testing.assert_close(out_neg, neg)

    def test_mismatched_shapes_bake_then_separate_calls(self):
        calls = []

        class Fusion:
            def __call__(self, ctx, mask=None, transformer_options=None):
                calls.append(tuple(ctx.shape))
                return ctx

        class Model:
            txtfusion = Fusion()

            def txtmlp(self, value):
                return value

        pos = torch.ones(1, 12, 2560)
        neg = torch.zeros(1, 8, 2560)
        _fused_text(Model(), pos, neg, {})
        self.assertEqual(calls, [(1, 12, 2560), (1, 12, 2560), (1, 8, 2560)])


class FusionBudgetCopyTests(unittest.TestCase):
    def test_uncensorfix_clone_keeps_nag_match_budget(self):
        budget = dict(
            tap_method="nova452 Rebalance operation",
            tap_gains=(1.,) * 7 + (2.5, 5., 1.1, 4., 1.),
            tap_normalization="none",
            tap_multiplier=1.,
            tap_profile_values=(1.,) * 7 + (2.5, 5., 1.1, 4., 1.),
            nag_match_taps=True,
        )
        src = types.SimpleNamespace(model_options={"transformer_options": {FUSION_BUDGET_KEY: budget}})
        dst = types.SimpleNamespace(model_options={"transformer_options": {}})
        copy_fusion_budget(dst, src)
        raw = [[torch.ones(1, 2, 30720), {}]]
        tapped = prepare_nag_conditioning(dst, raw)
        self.assertIsNot(tapped, raw)
        self.assertIsNot(tapped[0][0], raw[0][0])


if __name__ == "__main__":
    unittest.main()

"""Opt-in NAG txtfusion helpers: missing config must be a no-op."""
import sys
import types
import unittest
from unittest.mock import patch
import torch

from donut_nag_txtfusion import (
    NAG_BATCH_TXTFUSION,
    NAG_RMS_COMPENSATION,
    _fused_text,
    _nudge_tap_energy,
    ensure_nag_txtfusion_is_batched,
)
from DonutKrea2FusionControl import FUSION_BUDGET_KEY, copy_fusion_budget, prepare_nag_conditioning


class TxtfusionHelperTests(unittest.TestCase):
    def test_absent_budget_does_not_rescale(self):
        p, n = torch.ones(1, 4, 12, 8), torch.full((1, 4, 12, 8), 4.)
        p2, n2 = _nudge_tap_energy(p, n, 0.26, {})
        self.assertIs(p2, p)
        self.assertIs(n2, n)

    def test_nag_match_taps_does_not_enable_rms_compensation(self):
        p, n = torch.ones(1, 4, 12, 8), torch.full((1, 4, 12, 8), 4.)
        p2, n2 = _nudge_tap_energy(p, n, 1.0, {"donut_krea2_fusion_budget": {"nag_match_taps": True}})
        self.assertIs(p2, p)
        self.assertIs(n2, n)

    def test_one_pair_unchanged_when_another_batch_item_is_added(self):
        on = {"donut_krea2_fusion_budget": {NAG_RMS_COMPENSATION: True}}
        p, n = torch.ones(1, 4, 12, 8), torch.ones(1, 4, 12, 8)
        solo_p, solo_n = _nudge_tap_energy(p, n, 0.26, on)
        both_p = torch.cat([p, p * 100], dim=0)
        both_n = torch.cat([n, n], dim=0)
        batch_p, batch_n = _nudge_tap_energy(both_p, both_n, 0.26, on)
        torch.testing.assert_close(batch_p[:1], solo_p)
        torch.testing.assert_close(batch_n[:1], solo_n)

    def test_zero_alpha_preserves_original_objects(self):
        on = {"donut_krea2_fusion_budget": {NAG_RMS_COMPENSATION: True}}
        p, n = torch.randn(2, 3, 12, 8), torch.randn(2, 2, 12, 8)
        p2, n2 = _nudge_tap_energy(p, n, 0, on)
        self.assertIs(p2, p)
        self.assertIs(n2, n)

    def test_default_fused_text_is_two_calls_even_when_shapes_match(self):
        calls = []

        class Model:
            def txtfusion(self, ctx, mask=None, transformer_options=None):
                calls.append(tuple(ctx.shape))
                return ctx

            def txtmlp(self, value):
                return value

        pos = torch.ones(1, 12, 2560)
        neg = torch.zeros(1, 12, 2560)
        out_pos, out_neg = _fused_text(Model(), pos, neg, {})
        self.assertEqual(calls, [(1, 12, 2560), (1, 12, 2560)])
        torch.testing.assert_close(out_pos, pos)
        torch.testing.assert_close(out_neg, neg)

    def test_unequal_lengths_use_two_calls_not_a_warmup(self):
        calls = []

        class Model:
            def txtfusion(self, ctx, mask=None, transformer_options=None):
                calls.append(tuple(ctx.shape))
                return ctx

            def txtmlp(self, value):
                return value

        pos = torch.ones(1, 12, 2560)
        neg = torch.zeros(1, 8, 2560)
        _fused_text(Model(), pos, neg, {})
        self.assertEqual(calls, [(1, 12, 2560), (1, 8, 2560)])

    def test_opt_in_batching_concatenates_equal_lengths_only(self):
        calls = []

        class Model:
            def txtfusion(self, ctx, mask=None, transformer_options=None):
                calls.append(ctx.shape[0])
                return ctx

            def txtmlp(self, value):
                return value

        opts = {"donut_krea2_fusion_budget": {NAG_BATCH_TXTFUSION: True}}
        pos = torch.ones(1, 12, 2560)
        neg = torch.zeros(1, 12, 2560)
        _fused_text(Model(), pos, neg, opts)
        self.assertEqual(calls, [2])
        calls.clear()
        _fused_text(Model(), pos, torch.zeros(1, 8, 2560), opts)
        self.assertEqual(calls, [1, 1])

    def test_compensation_gain_is_capped(self):
        on = {"donut_krea2_fusion_budget": {NAG_RMS_COMPENSATION: True}}
        p, n = torch.ones(1, 4, 12, 8), torch.full((1, 4, 12, 8), 1e-5)
        _, n2 = _nudge_tap_energy(p, n, 0.26, on)
        gain = (n2.square().mean() / n.square().mean()).sqrt().item()
        self.assertLessEqual(gain, 4.0 + 1e-5)
        self.assertTrue(torch.isfinite(n2).all())

    def test_installing_does_not_redirect_upstream_wrapper(self):
        upstream = types.ModuleType("_review_upstream_nag")
        exec(
            "def krea2_nag_forward(*args, **kwargs):\n    return \"original\"\n"
            "def krea2_nag_wrapper(*args, **kwargs):\n    return krea2_nag_forward(*args, **kwargs)\n",
            upstream.__dict__,
        )
        node_module = types.ModuleType("_review_nag_node")
        exec("class Krea2NormalizedAttentionGuidance:\n    pass\n", node_module.__dict__)
        node_module.krea2_nag_wrapper = upstream.krea2_nag_wrapper
        nodes = types.ModuleType("nodes")
        nodes.NODE_CLASS_MAPPINGS = {
            "Krea2NormalizedAttentionGuidance": node_module.Krea2NormalizedAttentionGuidance,
        }
        already_registered = upstream.krea2_nag_wrapper
        fake_modules = {
            "nodes": nodes,
            upstream.__name__: upstream,
            node_module.__name__: node_module,
        }
        with patch.dict(sys.modules, fake_modules):
            self.assertEqual(already_registered(*([None] * 9)), "original")
            ensure_nag_txtfusion_is_batched()
            self.assertEqual(already_registered(*([None] * 9)), "original")


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

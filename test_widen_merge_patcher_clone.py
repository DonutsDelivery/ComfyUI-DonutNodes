import copy
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

import DonutWidenMerge as module


class FakePatcher:
    def __init__(self, model):
        self.model = model
        self.patches = {}
        self.clone_count = 0

    def __deepcopy__(self, memo):
        raise AssertionError("ComfyUI patchers must not be deep-copied")

    def clone(self):
        self.clone_count += 1
        clone = FakePatcher(self.model)
        clone.patches = copy.deepcopy(self.patches)
        return clone

    def add_patches(self, patches, strength_patch=1.0):
        self.patches.update(patches)
        self.patch_strength = strength_patch
        return list(patches)


class WidenMergePatcherCloneTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        self.patcher = FakePatcher(self.model)

    def test_uses_clone_and_lazy_diff_without_mutating_base_parameters(self):
        merged = {"weight": torch.tensor([[4.0, 7.0]])}

        output, applied_count, mismatch_count = module._clone_with_merged_parameter_diffs(
            self.patcher,
            self.model,
            merged,
        )

        self.assertIsNot(output, self.patcher)
        self.assertEqual(self.patcher.clone_count, 1)
        self.assertEqual(applied_count, 1)
        self.assertEqual(mismatch_count, 0)
        self.assertTrue(torch.equal(self.model.weight, torch.tensor([[1.0, 2.0]])))
        patch_type, (delta,) = output.patches["weight"]
        self.assertEqual(patch_type, "diff")
        self.assertTrue(torch.equal(delta, torch.tensor([[3.0, 5.0]])))
        self.assertEqual(output.patch_strength, 1.0)

    def test_reports_shape_mismatch_without_creating_a_patch(self):
        output, applied_count, mismatch_count = module._clone_with_merged_parameter_diffs(
            self.patcher,
            self.model,
            {"weight": torch.ones(2, 2)},
        )

        self.assertEqual(applied_count, 0)
        self.assertEqual(mismatch_count, 1)
        self.assertEqual(output.patches, {})

    def test_unet_execution_uses_lazy_diffs_instead_of_deepcopy(self):
        other_model = torch.nn.Linear(2, 1, bias=False)
        other_patcher = FakePatcher(other_model)
        diagnostics = {
            "compatibility_scores": [{"parameter": "weight", "compatibility": 1.0}],
            "varied_score_count": 1,
            "uniform_score_count": 0,
            "parameters_skipped_threshold": 0,
            "strength_distribution": {},
            "applied_strengths": {"weight": 1.0},
        }

        def merge(**kwargs):
            self.assertIs(kwargs["merged_model"], self.model)
            return {"weight": torch.tensor([[4.0, 7.0]])}, diagnostics

        with (
            patch.object(module, "compute_merge_hash", return_value="test-key"),
            patch.object(module, "check_cache_for_merge_with_bypass", return_value=None),
            patch.object(module, "memory_cleanup_context", side_effect=lambda _name: nullcontext()),
            patch.object(module, "enhanced_widen_merging_with_dynamic_strength", side_effect=merge),
            patch.object(module, "sanitize_strength_distribution", return_value={"display_text": "1.0", "count": 1}),
            patch.object(module, "_analyze_compatibility_patterns_and_recommend_threshold", return_value=(0.0, "ok")),
            patch.object(module, "force_cleanup"),
            patch.object(module, "store_merge_result"),
            patch.object(module, "gentle_cleanup"),
        ):
            output, _, _ = module.DonutWidenMergeUNet().execute(
                self.patcher,
                other_patcher,
                merge_strength=1.0,
                min_strength=0.0,
                max_strength=1.0,
                normalization_mode="none",
                importance_threshold=1.0,
                importance_boost=1.0,
                rank_sensitivity=1.0,
                skip_threshold=0.0,
            )

        self.assertIsNot(output, self.patcher)
        self.assertTrue(torch.equal(self.model.weight, torch.tensor([[1.0, 2.0]])))
        _, (delta,) = output.patches["weight"]
        self.assertTrue(torch.equal(delta, torch.tensor([[3.0, 5.0]])))


if __name__ == "__main__":
    unittest.main()

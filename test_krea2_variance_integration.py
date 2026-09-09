import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import nodes
import folder_paths
import donut_prompt
import DonutKSamplerCFGLinear as samplers
import krea2_variance_integration as variance


class SeedVarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [Path(root) / 'krea-seed-variance-enhancer' / 'krea_seed_variance_enhancer.py'
                      for root in folder_paths.get_folder_paths('custom_nodes')]
        pack = next((path for path in candidates if path.exists()), None)
        if pack is None:
            raise unittest.SkipTest('Optional krea-seed-variance-enhancer is not installed')
        spec = importlib.util.spec_from_file_location('variance_pack_tested', pack)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.node = module.KreaSeedVarianceEnhancer

    def setUp(self):
        self.registry = patch.dict(nodes.NODE_CLASS_MAPPINGS, {'KreaSeedVarianceEnhancer': self.node})
        self.registry.start()
        self.addCleanup(self.registry.stop)
        self.positive = [[torch.linspace(-1, 1, 48).reshape(1, 6, 8), {'pooled_output': torch.ones(1, 8)}]]

    def test_pair_matches_standalone_with_independent_seeds_and_preserves_input_rng(self):
        before = self.positive[0][0].clone()
        state = torch.get_rng_state().clone()
        full, face = variance.enhance_prompt_pair(self.positive, self.positive,
                                                variance_enabled=True, variance_seed=42)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(before, self.positive[0][0]))
        self.assertNotIn(variance.VARIANCE_KEY, self.positive[0][1])
        for actual, seed in [(full, 42), (face, 43)]:
            settings = actual[0][1][variance.VARIANCE_KEY]
            self.assertEqual(settings['seed'], seed)
            expected, _ = self.node().randomize_conditioning(self.positive, **settings)
            for (a, am), (b, bm) in zip(actual, expected):
                self.assertTrue(torch.equal(a, b))
                self.assertEqual({k:v for k,v in am.items() if k != variance.VARIANCE_KEY}, bm)
        self.assertFalse(torch.equal(full[0][0], face[0][0]))
        self.assertEqual(full[0][1]['end_percent'], .25)
        self.assertEqual(full[1][1]['start_percent'], .25)

    def test_edit_reapplies_recipe_to_fresh_grounded_embeddings_once(self):
        full, _ = variance.enhance_prompt_pair(self.positive, self.positive,
                                             variance_enabled=True, variance_seed=7)
        grounded = [[torch.randn(1, 4, 8), {'grounded': True}]]
        actual = variance.reapply_edit_variance(grounded, full)
        expected = variance.apply_seed_variance(grounded, full[0][1][variance.VARIANCE_KEY])
        self.assertEqual(len(actual), 2)
        for (a, am), (b, bm) in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(am['grounded'])
        self.assertIs(variance.reapply_edit_variance(grounded, self.positive), grounded)

    def test_sampler_reencodes_edit_then_applies_variance(self):
        original, _ = variance.enhance_prompt_pair(self.positive, self.positive,
                                                 variance_enabled=True, variance_seed=19)
        grounded = [[torch.randn(1, 4, 8), {"grounded": True}]]
        target = {"samples": torch.zeros(1, 16, 8, 8)}
        sampler = samplers.DonutSampler()
        with patch.object(samplers, "prepare_krea2_edit", return_value=(
            "model", grounded, self.positive, target, None,
        )), patch.object(sampler, "run_simple", return_value=(target, "info")) as run:
            sampler.sample("model", 19, 8, 1, 1, 1, 4, "euler", "simple",
                           original, self.positive, target, 1., edit_mode=True,
                           source_image=torch.zeros(1, 64, 64, 3), clip="clip", vae="vae")
        forwarded = run.call_args.args[9]
        self.assertEqual(len(forwarded), 2)
        self.assertEqual(forwarded[0][0].shape, grounded[0][0].shape)
        self.assertEqual(forwarded[0][1][variance.VARIANCE_KEY]["seed"], 19)
        self.assertTrue(forwarded[0][1]["grounded"])
        self.assertIs(run.call_args.args[10], self.positive)

    def test_disabled_and_missing_dependency(self):
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {}, clear=True):
            a, b = variance.enhance_prompt_pair(self.positive, self.positive)
            self.assertIs(a, self.positive)
            self.assertIs(b, self.positive)
            with self.assertRaisesRegex(RuntimeError, 'Install/enable'):
                variance.enhance_prompt_pair(a, b, variance_enabled=True)

    def test_prompt_node_only_enhances_positives_and_seed_wraps(self):
        with patch.object(nodes.CLIPTextEncode, 'encode', return_value=(self.positive,)):
            result = donut_prompt.DonutPromptConditioning().encode(
                object(), 'face', 'scene', 'negative', variance_enabled=True,
                variance_seed=(1 << 64) - 1,
            )['result']
        full, face, zeroed, raw = result[3:]
        self.assertEqual(face[0][1][variance.VARIANCE_KEY]['seed'], 0)
        self.assertIs(raw, self.positive)
        self.assertEqual(torch.count_nonzero(zeroed[0][0]), 0)
        self.assertNotIn(variance.VARIANCE_KEY, raw[0][1])
        self.assertNotIn(variance.VARIANCE_KEY, zeroed[0][1])


if __name__ == '__main__':
    unittest.main()

"""The master variance toggle must bypass the enhancer, regardless of parameters."""
import unittest
from unittest.mock import patch
import torch
import krea2_variance_integration as variance


class VarianceOffTests(unittest.TestCase):
    def test_off_returns_original_conditioning_and_never_invokes_enhancer(self):
        positive = [[torch.randn(1, 4, 8), {"pooled_output": torch.randn(1, 8)}]]
        face = [[torch.randn(1, 4, 8), {}]]
        grounded = [[torch.randn(1, 4, 8), {"grounded": True}]]
        original = positive[0][0].clone()
        rng = torch.get_rng_state().clone()
        with patch.object(variance, "apply_seed_variance", side_effect=AssertionError("Enhancer ran while off")) as enhance:
            for noise_insert in variance.variance_input_types()["variance_noise_insert"][0]:
                full, facial = variance.enhance_prompt_pair(
                    positive, face, variance_enabled=False,
                    variance_noise_insert=noise_insert, variance_strength=100,
                    variance_auto_strength_factor=100,
                )
                self.assertIs(full, positive)
                self.assertIs(facial, face)
                self.assertIs(variance.reapply_edit_variance(grounded, full), grounded)
            enhance.assert_not_called()
        self.assertTrue(torch.equal(original, positive[0][0]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertNotIn(variance.VARIANCE_KEY, positive[0][1])


if __name__ == "__main__":
    unittest.main()

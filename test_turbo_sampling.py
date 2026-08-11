import unittest

from turbo_sampling import resolve_turbo_sampling, turbo_denoise_points


class TurboSamplingTests(unittest.TestCase):
    def test_linear_eight_step_twenty_percent_snaps_to_two_at_twenty_five(self):
        self.assertEqual(resolve_turbo_sampling(8, 0.20, "simple"), (2, 0.25, 0.25))

    def test_linear_tie_rounds_up(self):
        self.assertEqual(resolve_turbo_sampling(8, 0.1875, "normal"), (2, 0.25, 0.25))

    def test_bong_tangent_four_step_row_executes_from_eight_step_schedule(self):
        self.assertEqual(resolve_turbo_sampling(8, 0.40, "bong_tangent"), (4, 0.5, 0.4))

    def test_bong_tangent_preserves_verified_eight_step_table(self):
        expected = [0.225, 0.308, 0.359, 0.400, 0.444, 0.502, 0.606, 1.0]
        actual = [denoise for _, denoise in turbo_denoise_points(8, "bong_tangent")]
        for value, target in zip(actual, expected):
            self.assertAlmostEqual(value, target)

    def test_other_supported_step_counts_scale_the_scheduler_curve(self):
        points = turbo_denoise_points(4, "bong_tangent")
        self.assertEqual([step for step, _ in points], [1, 2, 3, 4])
        self.assertEqual([round(value, 3) for _, value in points], [0.308, 0.400, 0.502, 1.0])

    def test_requested_bounds_still_run_at_least_one_step(self):
        self.assertEqual(resolve_turbo_sampling(8, 0.0, "simple"), (1, 0.125, 0.125))
        self.assertEqual(resolve_turbo_sampling(8, 2.0, "simple"), (8, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()

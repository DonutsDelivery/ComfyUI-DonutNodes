import math
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


class TurboScheduleReconstructionTests(unittest.TestCase):
    @staticmethod
    def reconstructed_steps(effective_steps, execution_denoise):
        # KSampler.set_steps' full-denoise branch and integer-truncating
        # schedule-length contract. This is not a ComfyUI/GPU integration test.
        if execution_denoise > 0.9999:
            return effective_steps
        return int(effective_steps / execution_denoise)

    def test_fourteen_step_schedule_does_not_reconstruct_thirteen(self):
        effective, execution, matched = resolve_turbo_sampling(14, 0.65, "beta")
        self.assertEqual(effective, 9)
        self.assertEqual(matched, 9 / 14)
        self.assertEqual(self.reconstructed_steps(effective, execution), 14)

    def test_fifty_step_schedule_does_not_reconstruct_forty_nine(self):
        for selected in (7, 14, 17, 28, 34):
            with self.subTest(selected=selected):
                effective, execution, matched = resolve_turbo_sampling(
                    50, selected / 50, "beta"
                )
                self.assertEqual(effective, selected)
                self.assertEqual(matched, selected / 50)
                self.assertEqual(self.reconstructed_steps(effective, execution), 50)

    def test_every_row_round_trips_for_step_counts_one_through_sixty_four(self):
        for supported in range(1, 65):
            for scheduler in ("simple", "beta", "bong_tangent"):
                for selected, requested in turbo_denoise_points(supported, scheduler):
                    effective, execution, matched = resolve_turbo_sampling(
                        supported, requested, scheduler
                    )
                    context = (supported, scheduler, selected)
                    self.assertEqual(effective, selected, context)
                    self.assertEqual(matched, requested, context)
                    self.assertEqual(
                        self.reconstructed_steps(effective, execution), supported, context
                    )

    def test_large_counts_and_near_full_denoise_boundary(self):
        for supported in (100, 250, 1000, 8192, 9999, 10000):
            selected_rows = (
                1, 2, 3, supported // 3, supported // 2,
                supported - 2, supported - 1, supported,
            )
            for selected in selected_rows:
                with self.subTest(supported=supported, selected=selected):
                    effective, execution, matched = resolve_turbo_sampling(
                        supported, selected / supported, "beta"
                    )
                    self.assertEqual(effective, selected)
                    self.assertEqual(matched, selected / supported)
                    self.assertEqual(
                        self.reconstructed_steps(effective, execution), supported
                    )

    def test_eight_step_rows_are_bit_for_bit_unchanged(self):
        for scheduler in ("simple", "normal", "beta", "bong_tangent"):
            for selected, requested in turbo_denoise_points(8, scheduler):
                with self.subTest(scheduler=scheduler, selected=selected):
                    self.assertEqual(
                        resolve_turbo_sampling(8, requested, scheduler),
                        (selected, selected / 8, requested),
                    )

    def test_full_denoise_stays_exactly_one(self):
        for supported in (1, 8, 14, 25, 50, 10000):
            for scheduler in ("beta", "bong_tangent"):
                with self.subTest(supported=supported, scheduler=scheduler):
                    self.assertEqual(
                        resolve_turbo_sampling(supported, 1.0, scheduler),
                        (supported, 1.0, 1.0),
                    )

    def test_only_execution_ratio_moves_by_one_float_when_needed(self):
        for supported, selected in ((14, 9), (25, 7), (50, 7)):
            ratio = selected / supported
            effective, execution, matched = resolve_turbo_sampling(
                supported, ratio, "beta"
            )
            self.assertEqual(effective, selected)
            self.assertEqual(matched, ratio)
            self.assertEqual(execution, math.nextafter(ratio, 0.0))


if __name__ == "__main__":
    unittest.main()

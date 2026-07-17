#!/usr/bin/env python3
"""
DonutSampler - Custom KSampler with Linear CFG Progression

A custom KSampler that allows defining start and end CFG values with linear
interpolation between them throughout the sampling steps.

This implementation creates a custom sampling loop with dynamic CFG values per step.

Consolidated node: DonutSampler now hosts three modes ("simple", "advanced",
"multi_model") behind a single mode selector. The simple mode reproduces the
original DonutSampler behaviour byte-for-byte; "advanced" reproduces the original
DonutSampler (Advanced) node; "multi_model" reproduces DonutMultiModelSampler.
The two legacy classes are kept as thin alias subclasses (with their ORIGINAL
INPUT_TYPES) so saved graphs continue to deserialize, and they pin a mode and
forward to the shared engine.
"""

import torch
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.k_diffusion.sampling as k_diffusion_sampling
import latent_preview

try:
    from .krea2_edit_integration import (
        make_krea2_edit_target,
        patch_krea2_edit_model,
        prepare_krea2_edit,
        validate_krea2_edit_target,
    )
except ImportError:
    from krea2_edit_integration import (
        make_krea2_edit_target,
        patch_krea2_edit_model,
        prepare_krea2_edit,
        validate_krea2_edit_target,
    )


class _DonutSamplerEngine:
    """
    Shared engine holding the verbatim math/pipeline for all three sampler modes.

    The per-mode CFG schedules are preserved exactly:
      - simple: piecewise-linear (no easing curve)
      - advanced / multi_model: curve-then-3point
    Per-call state (cfg_history, model_phases) is reset at the start of each run.
    """

    def __init__(self):
        self.cfg_history = []
        self.model_phases = []

    # ------------------------------------------------------------------
    # Shared curve helpers (verbatim from DonutKSamplerAdvanced)
    # ------------------------------------------------------------------
    def apply_curve(self, progress, curve_type):
        """Apply different mathematical curves to the progression."""
        import math

        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, progress))

        if curve_type == "linear":
            return progress

        # Basic easing curves
        elif curve_type == "ease_in":
            return progress * progress
        elif curve_type == "ease_out":
            return 1 - (1 - progress) * (1 - progress)
        elif curve_type == "ease_in_out":
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - 2 * (1 - progress) * (1 - progress)

        # Mathematical function curves
        elif curve_type == "exponential":
            # Exponential curve: starts slow, accelerates rapidly
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                # Use a gentler exponential curve
                return (math.exp(3 * progress) - 1) / (math.exp(3) - 1)

        elif curve_type == "logarithmic":
            # Logarithmic curve: starts fast, decelerates
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                # Use a gentler logarithmic curve
                return math.log(1 + 2 * progress) / math.log(3)

        elif curve_type == "sine_wave":
            # Sine wave: smooth S-curve
            return 0.5 * (1 - math.cos(progress * math.pi))

        elif curve_type == "cosine_wave":
            # Cosine wave: inverted sine
            return 0.5 * (1 + math.cos((1 - progress) * math.pi))

        elif curve_type == "smooth_step":
            # Smooth step function: 3t²-2t³
            return progress * progress * (3 - 2 * progress)

        elif curve_type == "smoother_step":
            # Even smoother step: 6t⁵-15t⁴+10t³
            return progress * progress * progress * (progress * (progress * 6 - 15) + 10)

        # Circular curves
        elif curve_type == "circular_in":
            return 1 - math.sqrt(1 - progress * progress)

        elif curve_type == "circular_out":
            return math.sqrt(1 - (progress - 1) * (progress - 1))

        # Back curves (overshoot)
        elif curve_type == "back_in":
            c1 = 1.70158
            c3 = c1 + 1
            return c3 * progress * progress * progress - c1 * progress * progress

        elif curve_type == "back_out":
            c1 = 1.70158
            c3 = c1 + 1
            return 1 + c3 * math.pow(progress - 1, 3) + c1 * math.pow(progress - 1, 2)

        # Elastic curves (spring-like)
        elif curve_type == "elastic_in":
            c4 = (2 * math.pi) / 3
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                return -math.pow(2, 10 * progress - 10) * math.sin((progress * 10 - 10.75) * c4)

        elif curve_type == "elastic_out":
            c4 = (2 * math.pi) / 3
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                return math.pow(2, -10 * progress) * math.sin((progress * 10 - 0.75) * c4) + 1

        # Bounce curves
        elif curve_type == "bounce_in":
            return 1 - self.bounce_out_helper(1 - progress)

        elif curve_type == "bounce_out":
            return self.bounce_out_helper(progress)

        # Dramatic curves for more visible differences
        elif curve_type == "dramatic_exponential":
            # Very steep exponential - stays high for most of the time
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                return math.pow(progress, 4)  # x^4 curve

        elif curve_type == "dramatic_logarithmic":
            # Very steep logarithmic - drops quickly then levels off
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                return math.pow(progress, 0.25)  # x^0.25 curve (4th root)

        # Default fallback
        return progress

    def bounce_out_helper(self, x):
        """Helper function for bounce curves."""
        n1 = 7.5625
        d1 = 2.75

        if x < 1 / d1:
            return n1 * x * x
        elif x < 2 / d1:
            return n1 * (x - 1.5 / d1) * (x - 1.5 / d1) + 0.75
        elif x < 2.5 / d1:
            return n1 * (x - 2.25 / d1) * (x - 2.25 / d1) + 0.9375
        else:
            return n1 * (x - 2.625 / d1) * (x - 2.625 / d1) + 0.984375

    def create_ascii_chart(self, cfg_values, cfg_start, cfg_end, width=50, height=12):
        """Create an ASCII art chart of the CFG progression."""
        if not cfg_values:
            return "No CFG data available"

        # Create the chart grid
        chart = []

        # Calculate scaling
        cfg_min = min(cfg_values)
        cfg_max = max(cfg_values)
        cfg_range = cfg_max - cfg_min if cfg_max != cfg_min else 1

        # Create chart lines from top to bottom
        for row in range(height):
            line = []
            # Y-axis labels (CFG values)
            y_value = cfg_max - (row / (height - 1)) * cfg_range
            line.append(f"{y_value:4.1f}│")

            # Plot the curve
            for col in range(width):
                step_index = int((col / (width - 1)) * (len(cfg_values) - 1))
                cfg_val = cfg_values[step_index]

                # Check if this point should be plotted
                expected_y = cfg_max - (row / (height - 1)) * cfg_range
                tolerance = cfg_range / (height * 2)

                if abs(cfg_val - expected_y) <= tolerance:
                    line.append("●")
                elif row == height - 1:  # Bottom line
                    line.append("─")
                elif col == 0:  # Left edge
                    line.append("│")
                else:
                    line.append(" ")

            chart.append("".join(line))

        # Add bottom axis
        bottom_line = "    └" + "─" * width
        chart.append(bottom_line)

        # Add step labels
        step_labels = "     "
        for i in range(0, width, width//5):
            step_num = int((i / (width - 1)) * (len(cfg_values) - 1))
            step_labels += f"{step_num:2d}" + " " * (width//5 - 2)
        chart.append(step_labels[:len(bottom_line)])
        chart.append("     Steps →")

        return "\n".join(chart)

    # ==================================================================
    # SIMPLE MODE (verbatim from original DonutSampler)
    # ==================================================================
    def simple_calculate_cfg_for_step(self, step, total_steps, cfg_start, cfg_halfway, cfg_end, halfway_step):
        """Calculate the CFG value for a specific step with optional halfway point."""
        if total_steps <= 1:
            return cfg_start

        # Check if halfway CFG is disabled (same as start or end)
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)

        if halfway_disabled:
            # Simple linear interpolation between start and end
            progress = step / (total_steps - 1)
            current_cfg = cfg_start + (cfg_end - cfg_start) * progress
        else:
            # Clamp halfway_step to valid range
            halfway_step = max(1, min(halfway_step, total_steps - 1))

            if step <= halfway_step:
                # First segment: start -> halfway
                if halfway_step == 0:
                    current_cfg = cfg_start
                else:
                    progress = step / halfway_step
                    current_cfg = cfg_start + (cfg_halfway - cfg_start) * progress
            else:
                # Second segment: halfway -> end
                remaining_steps = total_steps - 1 - halfway_step
                if remaining_steps == 0:
                    current_cfg = cfg_halfway
                else:
                    progress = (step - halfway_step) / remaining_steps
                    current_cfg = cfg_halfway + (cfg_end - cfg_halfway) * progress

        return current_cfg

    def run_simple(self, model, seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
                   scheduler, positive, negative, latent_image, denoise):
        """
        Basic sampling using ComfyUI's common_ksampler with dynamic CFG injection.
        """
        try:
            # Pre-calculate CFG values for each step
            self.cfg_history = []
            cfg_values = []
            for i in range(steps):
                cfg_val = self.simple_calculate_cfg_for_step(i, steps, cfg_start, cfg_halfway, cfg_end, halfway_step)
                cfg_values.append(cfg_val)
                self.cfg_history.append((i, cfg_val))

            # Check if halfway is enabled
            halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)
            if halfway_disabled:
                print(f"DonutSampler: CFG progression {cfg_start:.2f} -> {cfg_end:.2f} over {steps} steps (halfway disabled)")
            else:
                clamped_halfway_step = max(1, min(halfway_step, steps - 1))
                print(f"DonutSampler: CFG progression {cfg_start:.2f} -> {cfg_halfway:.2f} (step {clamped_halfway_step}) -> {cfg_end:.2f} over {steps} steps")

            # Show CFG values for debugging
            if len(cfg_values) >= 10:
                print(f"CFG linear progression preview:")
                indices = [0, len(cfg_values)//4, len(cfg_values)//2, 3*len(cfg_values)//4, len(cfg_values)-1]
                for i in indices:
                    if i < len(cfg_values):
                        progress = i / (len(cfg_values) - 1) if len(cfg_values) > 1 else 0
                        print(f"  Step {i}: progress={progress:.2f} -> CFG={cfg_values[i]:.2f}")
            else:
                print(f"CFG values: {[f'{v:.2f}' for v in cfg_values]}")

            # Create a custom CFGGuider that handles dynamic CFG
            import comfy.samplers

            class DynamicCFGGuider(comfy.samplers.CFGGuider):
                def __init__(self, model_patcher, cfg_values):
                    super().__init__(model_patcher)
                    self.cfg_values = cfg_values
                    self.step_count = 0

                def predict_noise(self, x, timestep, model_options={}, seed=None):
                    # Get dynamic CFG for current step
                    current_cfg = self.cfg_values[min(self.step_count, len(self.cfg_values) - 1)]
                    print(f"DonutSampler: Step {self.step_count}, CFG {current_cfg:.2f}")

                    # Temporarily set CFG and call parent
                    old_cfg = self.cfg
                    self.cfg = current_cfg
                    result = super().predict_noise(x, timestep, model_options, seed)
                    self.cfg = old_cfg

                    self.step_count += 1
                    return result

            # Monkey patch CFGGuider creation
            original_cfg_guider_class = comfy.samplers.CFGGuider
            comfy.samplers.CFGGuider = lambda model_patcher: DynamicCFGGuider(model_patcher, cfg_values)

            try:
                # Validate sampler exists to prevent fallback
                import comfy.samplers
                available_samplers = comfy.samplers.KSampler.SAMPLERS
                if sampler_name not in available_samplers:
                    raise ValueError(f"Sampler '{sampler_name}' not available. Available samplers: {available_samplers}")

                print(f"Using validated sampler: {sampler_name}")

                # Use ComfyUI's standard common_ksampler
                from nodes import common_ksampler
                result = common_ksampler(model, seed, steps, cfg_start, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

            finally:
                # Restore original CFGGuider
                comfy.samplers.CFGGuider = original_cfg_guider_class

            # Create CFG progression info
            cfg_info = self.format_simple_cfg_info(cfg_start, cfg_halfway, cfg_end, halfway_step, steps, sampler_name, scheduler)

            return (result[0], cfg_info)

        except Exception as e:
            # Handle interruption gracefully - let ComfyUI handle it properly
            if "InterruptProcessingException" in str(type(e)) or "InterruptProcessingException" in str(e):
                print("Sampling interrupted by user")
                raise e  # Re-raise to let ComfyUI handle interruption properly
            else:
                import traceback
                error_msg = f"Sampling failed: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                return (latent_image, f"ERROR: {error_msg}")

    def format_simple_cfg_info(self, cfg_start, cfg_halfway, cfg_end, halfway_step, steps, sampler_name, scheduler):
        """Format the CFG progression information."""
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)

        if halfway_disabled:
            cfg_info = f"DonutSampler - Linear CFG Progression\n"
        else:
            cfg_info = f"DonutSampler - Three-Point CFG Progression\n"

        cfg_info += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        cfg_info += f"Start CFG: {cfg_start:.2f}\n"

        if not halfway_disabled:
            clamped_halfway_step = max(1, min(halfway_step, steps - 1))
            cfg_info += f"Halfway CFG: {cfg_halfway:.2f} (at step {clamped_halfway_step})\n"

        cfg_info += f"End CFG: {cfg_end:.2f}\n"
        cfg_info += f"Steps: {steps}\n"
        cfg_info += f"Sampler: {sampler_name}\n"
        cfg_info += f"Scheduler: {scheduler}\n"

        # Add ASCII chart
        if self.cfg_history:
            cfg_values = [cfg_val for _, cfg_val in self.cfg_history]
            cfg_info += f"\nCFG Progression Chart:\n"
            cfg_info += self.create_ascii_chart(cfg_values, cfg_start, cfg_end)
            cfg_info += f"\n"

            # Show key numerical values
            cfg_info += f"\nKey Steps:\n"
            if len(self.cfg_history) > 8:
                shown = self.cfg_history[:4] + [(-1, "...")] + self.cfg_history[-4:]
            else:
                shown = self.cfg_history

            for step, cfg_val in shown:
                if step == -1:
                    cfg_info += f"  ...\n"
                else:
                    cfg_info += f"  Step {step+1}: CFG={cfg_val:.2f}\n"

        return cfg_info

    # ==================================================================
    # ADVANCED MODE (verbatim from original DonutKSamplerAdvanced)
    # ==================================================================
    def advanced_calculate_cfg_for_step(self, step, total_steps, cfg_start, cfg_halfway, cfg_end, halfway_step, curve_type="linear"):
        """Calculate the CFG value for a specific step with optional halfway point and curve."""
        if total_steps <= 1:
            return cfg_start

        # Check if halfway CFG is disabled (same as start or end)
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)

        if halfway_disabled:
            # Simple curved interpolation between start and end
            progress = step / (total_steps - 1)
            curved_progress = self.apply_curve(progress, curve_type)
            current_cfg = cfg_start + (cfg_end - cfg_start) * curved_progress
        else:
            # Clamp halfway_step to valid range
            halfway_step = max(1, min(halfway_step, total_steps - 1))

            if step <= halfway_step:
                # First segment: start -> halfway with curve
                if halfway_step == 0:
                    current_cfg = cfg_start
                else:
                    progress = step / halfway_step
                    curved_progress = self.apply_curve(progress, curve_type)
                    current_cfg = cfg_start + (cfg_halfway - cfg_start) * curved_progress
            else:
                # Second segment: halfway -> end with curve
                remaining_steps = total_steps - 1 - halfway_step
                if remaining_steps == 0:
                    current_cfg = cfg_halfway
                else:
                    progress = (step - halfway_step) / remaining_steps
                    curved_progress = self.apply_curve(progress, curve_type)
                    current_cfg = cfg_halfway + (cfg_end - cfg_halfway) * curved_progress

        # Debug: Print curve calculations for first few steps
        if step < 3 or step >= total_steps - 2:
            if halfway_disabled:
                progress = step / (total_steps - 1)
                curved_progress = self.apply_curve(progress, curve_type)
                print(f"Step {step}: progress={progress:.3f} -> curved={curved_progress:.3f} -> CFG={current_cfg:.3f} (curve: {curve_type})")
            else:
                segment = "first" if step <= halfway_step else "second"
                print(f"Step {step}: {segment} segment -> CFG={current_cfg:.3f} (curve: {curve_type})")

        return current_cfg

    def common_ksampler_with_dynamic_cfg(self, model, seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name, scheduler,
                                        positive, negative, latent, denoise=1.0, disable_noise=False,
                                        start_step=None, last_step=None, force_full_denoise=False, curve_type="linear"):
        """
        Advanced sampling with dynamic CFG using CFGGuider approach.
        """
        # Pre-calculate CFG values for each step
        self.cfg_history = []
        cfg_values = []
        for i in range(steps):
            cfg_val = self.advanced_calculate_cfg_for_step(i, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, curve_type)
            cfg_values.append(cfg_val)
            self.cfg_history.append((i, cfg_val))

        # Check if halfway is enabled
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)
        if halfway_disabled:
            print(f"DonutSampler Advanced: CFG {curve_type} progression {cfg_start:.2f} -> {cfg_end:.2f} over {steps} steps (halfway disabled)")
        else:
            clamped_halfway_step = max(1, min(halfway_step, steps - 1))
            print(f"DonutSampler Advanced: CFG {curve_type} progression {cfg_start:.2f} -> {cfg_halfway:.2f} (step {clamped_halfway_step}) -> {cfg_end:.2f} over {steps} steps")

        # Show curve comparison for debugging
        if len(cfg_values) >= 10:
            print(f"CFG progression preview ({curve_type}):")
            indices = [0, len(cfg_values)//4, len(cfg_values)//2, 3*len(cfg_values)//4, len(cfg_values)-1]
            for i in indices:
                if i < len(cfg_values):
                    progress = i / (len(cfg_values) - 1) if len(cfg_values) > 1 else 0
                    curved = self.apply_curve(progress, curve_type)
                    print(f"  Step {i}: progress={progress:.2f} -> curved={curved:.2f} -> CFG={cfg_values[i]:.2f}")
        else:
            print(f"CFG values: {[f'{v:.2f}' for v in cfg_values]}")

        # Create a custom CFGGuider that handles dynamic CFG
        import comfy.samplers

        class DynamicCFGGuiderAdvanced(comfy.samplers.CFGGuider):
            def __init__(self, model_patcher, cfg_values, curve_type):
                super().__init__(model_patcher)
                self.cfg_values = cfg_values
                self.curve_type = curve_type
                self.step_count = 0

            def predict_noise(self, x, timestep, model_options={}, seed=None):
                # Get dynamic CFG for current step
                current_cfg = self.cfg_values[min(self.step_count, len(self.cfg_values) - 1)]
                print(f"DonutSampler Advanced: Step {self.step_count}, CFG {current_cfg:.2f} ({self.curve_type})")

                # Temporarily set CFG and call parent
                old_cfg = self.cfg
                self.cfg = current_cfg
                result = super().predict_noise(x, timestep, model_options, seed)
                self.cfg = old_cfg

                self.step_count += 1
                return result

        # Monkey patch CFGGuider creation
        original_cfg_guider_class = comfy.samplers.CFGGuider
        comfy.samplers.CFGGuider = lambda model_patcher: DynamicCFGGuiderAdvanced(model_patcher, cfg_values, curve_type)

        try:
            # Validate sampler exists to prevent fallback
            import comfy.samplers
            available_samplers = comfy.samplers.KSampler.SAMPLERS
            if sampler_name not in available_samplers:
                raise ValueError(f"Sampler '{sampler_name}' not available. Available samplers: {available_samplers}")

            print(f"Using validated sampler: {sampler_name}")

            # Use ComfyUI's standard common_ksampler
            from nodes import common_ksampler
            result = common_ksampler(model, seed, steps, cfg_start, sampler_name, scheduler, positive, negative, latent,
                                   denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                   force_full_denoise=force_full_denoise)

        finally:
            # Restore original CFGGuider
            comfy.samplers.CFGGuider = original_cfg_guider_class

        return result

    def run_advanced(self, model, add_noise, noise_seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
                     scheduler, positive, negative, latent_image, start_at_step, end_at_step,
                     return_with_leftover_noise, cfg_curve="linear", denoise=1.0):
        """
        Main advanced sampling function with linear CFG progression.
        Based on KSamplerAdvanced.sample() but with dynamic CFG.
        """
        try:
            # Process advanced parameters (from KSamplerAdvanced.sample)
            force_full_denoise = True
            if return_with_leftover_noise == "enable":
                force_full_denoise = False

            disable_noise = False
            if add_noise == "disable":
                disable_noise = True

            # Use our modified common_ksampler with dynamic CFG
            result = self.common_ksampler_with_dynamic_cfg(
                model, noise_seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name, scheduler,
                positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise,
                start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise,
                curve_type=cfg_curve
            )

            # Create CFG progression info
            cfg_info = self.format_advanced_cfg_info(cfg_start, cfg_halfway, cfg_end, halfway_step, steps, sampler_name, scheduler,
                                                   start_at_step, end_at_step, add_noise, return_with_leftover_noise, cfg_curve)

            return (result[0], cfg_info)

        except Exception as e:
            # Handle interruption gracefully - let ComfyUI handle it properly
            if "InterruptProcessingException" in str(type(e)) or "InterruptProcessingException" in str(e):
                print("Advanced sampling interrupted by user")
                raise e  # Re-raise to let ComfyUI handle interruption properly
            else:
                import traceback
                error_msg = f"Advanced sampling failed: {str(e)}\n{traceback.format_exc()}"
                return (latent_image, f"ERROR: {error_msg}")

    def format_advanced_cfg_info(self, cfg_start, cfg_halfway, cfg_end, halfway_step, steps, sampler_name, scheduler,
                                start_at_step, end_at_step, add_noise, return_with_leftover_noise, cfg_curve="linear"):
        """Format the advanced CFG progression information."""
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)

        if halfway_disabled:
            cfg_info = f"DonutSampler Advanced - {cfg_curve.title()} CFG Progression\n"
        else:
            cfg_info = f"DonutSampler Advanced - Three-Point {cfg_curve.title()} CFG Progression\n"

        cfg_info += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        if halfway_disabled:
            cfg_info += f"CFG Range: {cfg_start:.2f} → {cfg_end:.2f} | Curve: {cfg_curve}\n"
        else:
            clamped_halfway_step = max(1, min(halfway_step, steps - 1))
            cfg_info += f"CFG Range: {cfg_start:.2f} → {cfg_halfway:.2f} (step {clamped_halfway_step}) → {cfg_end:.2f} | Curve: {cfg_curve}\n"

        cfg_info += f"Steps: {steps} | Sampler: {sampler_name} | Scheduler: {scheduler}\n"
        cfg_info += f"Step Range: {start_at_step} to {end_at_step}\n"
        cfg_info += f"Add Noise: {add_noise} | Return w/ Leftover Noise: {return_with_leftover_noise}\n"

        # Add ASCII chart
        if self.cfg_history:
            cfg_values = [cfg_val for _, cfg_val in self.cfg_history]
            cfg_info += f"\nCFG Progression Chart ({cfg_curve}):\n"
            cfg_info += self.create_ascii_chart(cfg_values, cfg_start, cfg_end)
            cfg_info += f"\n"

            # Show key numerical values
            cfg_info += f"\nKey Steps:\n"
            if len(self.cfg_history) > 8:
                shown = self.cfg_history[:4] + [(-1, "...")] + self.cfg_history[-4:]
            else:
                shown = self.cfg_history

            for step, cfg_val in shown:
                if step == -1:
                    cfg_info += f"  ...\n"
                else:
                    actual_step = start_at_step + step
                    cfg_info += f"  Step {actual_step+1}: CFG={cfg_val:.2f}\n"

        return cfg_info

    # ==================================================================
    # MULTI-MODEL MODE (verbatim from original DonutMultiModelSampler)
    # ==================================================================
    def multi_calculate_cfg_for_step(self, step, total_steps, cfg_start, cfg_halfway, cfg_end, halfway_step, curve_type="linear"):
        """Calculate CFG value for a specific step across the entire sampling process."""
        # Preserve original DonutMultiModelSampler behavior: these 8 curves were
        # never implemented in multi-model mode and fell through to linear. The
        # shared apply_curve now implements them, so downgrade here to keep saved
        # multi-model workflows producing identical output.
        if curve_type in ("circular_in", "circular_out", "back_in", "back_out",
                          "elastic_in", "elastic_out", "bounce_in", "bounce_out"):
            curve_type = "linear"

        if total_steps <= 1:
            return cfg_start

        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)

        if halfway_disabled:
            progress = step / (total_steps - 1)
            curved_progress = self.apply_curve(progress, curve_type)
            current_cfg = cfg_start + (cfg_end - cfg_start) * curved_progress
        else:
            halfway_step = max(1, min(halfway_step, total_steps - 1))

            if step <= halfway_step:
                if halfway_step == 0:
                    current_cfg = cfg_start
                else:
                    progress = step / halfway_step
                    curved_progress = self.apply_curve(progress, curve_type)
                    current_cfg = cfg_start + (cfg_halfway - cfg_start) * curved_progress
            else:
                remaining_steps = total_steps - 1 - halfway_step
                if remaining_steps == 0:
                    current_cfg = cfg_halfway
                else:
                    progress = (step - halfway_step) / remaining_steps
                    curved_progress = self.apply_curve(progress, curve_type)
                    current_cfg = cfg_halfway + (cfg_end - cfg_halfway) * curved_progress

        return current_cfg

    def plan_model_phases(self, total_steps, model_1, model_2, model_3, switch_at_step_1, switch_at_step_2):
        """Plan the phases for each model based on switch points."""
        phases = []
        models = [model_1]

        # Add available models
        if model_2 is not None:
            models.append(model_2)
        if model_3 is not None:
            models.append(model_3)

        # Validate and adjust switch points
        if len(models) == 1:
            # Single model - use all steps
            phases.append({
                'model': model_1,
                'start_step': 0,
                'end_step': total_steps,
                'steps': total_steps,
                'phase_name': 'Single Model'
            })
        elif len(models) == 2:
            # Two models - use switch_at_step_1
            switch_1 = max(1, min(switch_at_step_1, total_steps - 1))
            phases.append({
                'model': model_1,
                'start_step': 0,
                'end_step': switch_1,
                'steps': switch_1,
                'phase_name': 'Primary Model'
            })
            phases.append({
                'model': model_2,
                'start_step': switch_1,
                'end_step': total_steps,
                'steps': total_steps - switch_1,
                'phase_name': 'Secondary Model'
            })
        else:
            # Three models - use both switch points
            switch_1 = max(1, min(switch_at_step_1, total_steps - 2))
            switch_2 = max(switch_1 + 1, min(switch_at_step_2, total_steps - 1))

            phases.append({
                'model': model_1,
                'start_step': 0,
                'end_step': switch_1,
                'steps': switch_1,
                'phase_name': 'Primary Model'
            })
            phases.append({
                'model': model_2,
                'start_step': switch_1,
                'end_step': switch_2,
                'steps': switch_2 - switch_1,
                'phase_name': 'Secondary Model'
            })
            phases.append({
                'model': model_3,
                'start_step': switch_2,
                'end_step': total_steps,
                'steps': total_steps - switch_2,
                'phase_name': 'Tertiary Model'
            })

        return phases

    def run_multi_model(self, model_1, add_noise, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
                        scheduler, positive, negative, latent_image, noise_seed, start_at_step, end_at_step,
                        return_with_leftover_noise, randomize_seed_per_model, denoise, cfg_curve="linear",
                        model_2=None, model_3=None, switch_at_step_1=10, switch_at_step_2=15):
        """
        Main multi-model sampling function with model switching and noise refinement.
        """
        import time
        execution_id = int(time.time() * 1000) % 10000  # Last 4 digits of timestamp

        print(f"[DonutMultiModelSampler] Starting sampling with {sampler_name}/{scheduler}")

        # Basic validation only - don't modify the latent
        if latent_image is None or "samples" not in latent_image:
            raise ValueError("ERROR: Invalid latent input")
        try:
            # Process advanced step parameters
            force_full_denoise = True
            if return_with_leftover_noise == "enable":
                force_full_denoise = False

            # Clamp end_at_step to reasonable range
            actual_end_step = min(end_at_step, steps)
            effective_steps = actual_end_step - start_at_step

            if effective_steps <= 0:
                raise ValueError(f"Invalid step range: start_at_step={start_at_step}, end_at_step={end_at_step}, steps={steps}")

            # Plan model phases based on effective steps
            phases = self.plan_model_phases(effective_steps, model_1, model_2, model_3, switch_at_step_1, switch_at_step_2)
            self.model_phases = phases

            # Pre-calculate CFG values for the effective sampling process
            self.cfg_history = []
            all_cfg_values = []
            for i in range(effective_steps):
                cfg_val = self.multi_calculate_cfg_for_step(i, effective_steps, cfg_start, cfg_halfway, cfg_end, halfway_step, cfg_curve)
                all_cfg_values.append(cfg_val)
                self.cfg_history.append((start_at_step + i, cfg_val))

            print(f"DonutMultiModelSampler: Planned {len(phases)} phases across {effective_steps} effective steps")
            print(f"  Total steps: {steps}, Step range: {start_at_step} to {actual_end_step}, Effective: {effective_steps}")
            print(f"  Sampler: {sampler_name}, Scheduler: {scheduler}")
            print(f"  Switch points: {switch_at_step_1}, {switch_at_step_2} (relative to effective steps)")
            print(f"  Add noise: {add_noise}, Return leftover: {return_with_leftover_noise}, Randomize seeds: {randomize_seed_per_model}")

            # Debug: Show what models are actually being used in each phase
            for i, phase in enumerate(phases):
                model_status = "HAS MODEL" if phase['model'] is not None else "MODEL IS NONE"
                print(f"  Phase {i+1}: {phase['phase_name']} - Steps {phase['start_step']}-{phase['end_step']} - {model_status}")
                if phase['model'] is None:
                    print(f"    ⚠️ WARNING: Phase {i+1} has None model but will still be executed!")

            # Debug: Check available samplers
            import comfy.samplers
            try:
                if hasattr(comfy.samplers.KSampler.SAMPLERS, 'keys'):
                    print(f"  Available samplers: {list(comfy.samplers.KSampler.SAMPLERS.keys())[:5]}...")
                    sampler_check = sampler_name not in comfy.samplers.KSampler.SAMPLERS
                else:
                    print(f"  Available samplers: {comfy.samplers.KSampler.SAMPLERS[:5]}")
                    sampler_check = sampler_name not in comfy.samplers.KSampler.SAMPLERS
            except Exception as e:
                print(f"  Could not list samplers: {e}")
                sampler_check = False

            if sampler_check:
                print(f"  WARNING: Custom sampler '{sampler_name}' not in standard list!")
            for i, phase in enumerate(phases):
                actual_start = start_at_step + phase['start_step']
                actual_end = start_at_step + phase['end_step']
                print(f"  Phase {i+1}: {phase['phase_name']} - Steps {actual_start} to {actual_end} ({phase['steps']} steps)")

            # Current latent starts with input
            current_latent = latent_image
            current_seed = noise_seed

            # Process each phase sequentially
            for phase_idx, phase in enumerate(phases):
                is_first_phase = (phase_idx == 0)
                is_last_phase = (phase_idx == len(phases) - 1)

                # Get CFG values for this phase
                phase_cfg_values = all_cfg_values[phase['start_step']:phase['end_step']]

                print(f"\nStarting {phase['phase_name']} (Phase {phase_idx + 1}/{len(phases)})")
                print(f"  Steps: {phase['steps']}, CFG range: {phase_cfg_values[0]:.2f} -> {phase_cfg_values[-1]:.2f}")

                # Validate model before processing
                if phase['model'] is None:
                    print(f"  ERROR: {phase['phase_name']} model is None - skipping this phase")
                    continue

                try:
                    from nodes import common_ksampler
                    import comfy.samplers

                    # Validate sampler exists to prevent fallback
                    available_samplers = comfy.samplers.KSampler.SAMPLERS
                    if sampler_name not in available_samplers:
                        raise ValueError(f"Sampler '{sampler_name}' not available. Available samplers: {available_samplers}")

                    print(f"  Using validated sampler: {sampler_name}")

                    # Sampler validation passed

                    # Configure parameters for this phase
                    phase_start_step = start_at_step + phase['start_step']
                    phase_last_step = start_at_step + phase['end_step']

                    if is_first_phase:
                        phase_disable_noise = (add_noise == "disable")
                        phase_denoise = denoise
                    else:
                        # For subsequent phases, only disable noise if we're doing leftover noise mode
                        # Otherwise, each phase should add its own noise
                        phase_disable_noise = True  # Standard behavior: later phases work with previous output
                        phase_denoise = 1.0

                    print(f"  Phase {phase_idx + 1} noise config: disable_noise={phase_disable_noise}, denoise={phase_denoise}, add_noise_param={add_noise}")

                    phase_force_full_denoise = is_last_phase and (return_with_leftover_noise == "disable")

                    print(f"  🔍 Calling common_ksampler with sampler='{sampler_name}', scheduler='{scheduler}'")

                    result = common_ksampler(
                        phase['model'], current_seed, steps, phase_cfg_values[0],
                        sampler_name, scheduler, positive, negative, current_latent,
                        denoise=phase_denoise,
                        disable_noise=phase_disable_noise,
                        start_step=phase_start_step,
                        last_step=phase_last_step,
                        force_full_denoise=phase_force_full_denoise
                    )

                    # Update latent for next phase
                    current_latent = result[0] if isinstance(result, tuple) else result

                    # Update seed if randomization is enabled
                    if not is_last_phase and randomize_seed_per_model == "enable":
                        current_seed = (current_seed + 1) % 0xffffffffffffffff

                except Exception as phase_error:
                    # Handle interruption gracefully like standard samplers
                    if "InterruptProcessingException" in str(type(phase_error)) or "InterruptProcessingException" in str(phase_error):
                        print(f"  Phase {phase_idx + 1} interrupted by user")
                        # Re-raise interruption to let ComfyUI handle it properly
                        raise phase_error
                    else:
                        print(f"  Phase {phase_idx + 1} failed: {str(phase_error)}")
                        import traceback
                        traceback.print_exc()
                        break

            # Create comprehensive info
            info = self.format_multi_model_info(phases, effective_steps, cfg_start, cfg_halfway, cfg_end,
                                               halfway_step, sampler_name, scheduler, cfg_curve,
                                               start_at_step, end_at_step, add_noise, return_with_leftover_noise,
                                               randomize_seed_per_model)

            return (current_latent, info)

        except Exception as e:
            # Handle interruption gracefully - let ComfyUI handle it properly
            if "InterruptProcessingException" in str(type(e)) or "InterruptProcessingException" in str(e):
                print("Multi-model sampling interrupted by user")
                raise e  # Re-raise to let ComfyUI handle interruption properly
            else:
                import traceback
                error_msg = f"Multi-model sampling failed: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                return (latent_image, f"ERROR: {error_msg}")

    def format_multi_model_info(self, phases, steps, cfg_start, cfg_halfway, cfg_end, halfway_step,
                               sampler_name, scheduler, cfg_curve, start_at_step=0, end_at_step=10000,
                               add_noise="enable", return_with_leftover_noise="disable",
                               randomize_seed_per_model="enable"):
        """Format comprehensive multi-model sampling information."""
        info = f"DonutMultiModelSampler - {len(phases)} Model Phases\n"
        info += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        # CFG progression info
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)
        if halfway_disabled:
            info += f"CFG Progression: {cfg_start:.2f} → {cfg_end:.2f} | Curve: {cfg_curve}\n"
        else:
            info += f"CFG Progression: {cfg_start:.2f} → {cfg_halfway:.2f} (step {halfway_step}) → {cfg_end:.2f} | Curve: {cfg_curve}\n"

        info += f"Total Steps: {steps} | Step Range: {start_at_step} to {end_at_step}\n"
        info += f"Sampler: {sampler_name} | Scheduler: {scheduler}\n"
        info += f"Add Noise: {add_noise} | Return w/ Leftover Noise: {return_with_leftover_noise}\n"
        info += f"Randomize Seed Per Model: {randomize_seed_per_model}\n\n"

        # Model phases breakdown
        info += f"Model Phases:\n"
        for i, phase in enumerate(phases):
            info += f"  Phase {i+1}: {phase['phase_name']}\n"
            info += f"    Steps: {phase['start_step']} → {phase['end_step']} ({phase['steps']} steps)\n"
            if self.cfg_history:
                start_cfg = self.cfg_history[phase['start_step']][1] if phase['start_step'] < len(self.cfg_history) else 0
                end_cfg = self.cfg_history[min(phase['end_step']-1, len(self.cfg_history)-1)][1]
                info += f"    CFG: {start_cfg:.2f} → {end_cfg:.2f}\n"

        # Add ASCII chart if available
        if self.cfg_history:
            cfg_values = [cfg_val for _, cfg_val in self.cfg_history]
            info += f"\nCFG Progression Chart ({cfg_curve}):\n"
            info += self.create_ascii_chart(cfg_values, cfg_start, cfg_end)

            # Show key numerical values
            info += f"\n\nKey Steps:\n"
            if len(self.cfg_history) > 8:
                shown = self.cfg_history[:4] + [(-1, "...")] + self.cfg_history[-4:]
            else:
                shown = self.cfg_history

            for step, cfg_val in shown:
                if step == -1:
                    info += f"  ...\n"
                else:
                    info += f"  Step {step+1}: CFG={cfg_val:.2f}\n"

        return info


# Shared curve list used by advanced/multi widgets.
_CFG_CURVES = ["linear", "exponential", "logarithmic", "ease_in", "ease_out", "ease_in_out", "sine_wave", "cosine_wave", "smooth_step", "smoother_step", "circular_in", "circular_out", "back_in", "back_out", "elastic_in", "elastic_out", "bounce_in", "bounce_out", "dramatic_exponential", "dramatic_logarithmic"]


class DonutSampler(_DonutSamplerEngine):
    """
    Unified Donut sampler with a mode selector.

    Modes:
      - simple:      original DonutSampler (piecewise-linear CFG)
      - advanced:    original DonutSampler (Advanced) (curve-based CFG + step range)
      - multi_model: original DonutMultiModelSampler (model switching phases)

    CRITICAL: The first 13 required widgets reproduce the original DonutSampler
    widget order (seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step,
    sampler_name, scheduler, positive, negative, latent_image, denoise). All new
    widgets are APPENDED after them so already-saved plain DonutSampler nodes
    restore their values positionally.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg_start": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_halfway": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "halfway_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # APPENDED widgets - mode selector + mode-specific params.
                "mode": (["simple", "advanced", "multi_model"], {"default": "simple"}),
                "cfg_curve": (_CFG_CURVES, {"default": "linear"}),
                "add_noise": (["enable", "disable"],),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "randomize_seed_per_model": (["disable", "enable"], {"default": "enable"}),
                "switch_at_step_1": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "switch_at_step_2": ("INT", {"default": 15, "min": 1, "max": 10000}),
                "model_2": ("MODEL",),
                "model_3": ("MODEL",),
                "edit_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use the supplied latent_image shape as an empty Krea2 edit target; source_image provides clean reference conditioning.",
                }),
                "source_image": ("IMAGE", {"tooltip": "Required when edit_mode is enabled."}),
                "vae": ("VAE", {"tooltip": "Required when edit_mode is enabled."}),
                "clip": ("CLIP", {"tooltip": "Required when edit_mode is enabled."}),
                "edit_model": ("MODEL", {
                    "tooltip": "Optional Krea2 model with the Identity Edit LoRA already applied. Falls back to model.",
                }),
                "edit_prompt": ("STRING", {"forceInput": True}),
                "edit_negative_prompt": ("STRING", {"forceInput": True}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "cfg_progression_info")
    FUNCTION = "sample"
    CATEGORY = "donut/sampling"

    def sample(self, model, seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
               scheduler, positive, negative, latent_image, denoise,
               mode="simple", cfg_curve="linear", add_noise="enable", start_at_step=0,
               end_at_step=10000, return_with_leftover_noise="disable",
               randomize_seed_per_model="enable", switch_at_step_1=10, switch_at_step_2=15,
               model_2=None, model_3=None, edit_mode=False, source_image=None,
               vae=None, clip=None, edit_prompt="", edit_negative_prompt="", grounding_px=768,
               edit_model=None):
        # Reset per-call state before dispatching to the selected mode.
        self.cfg_history = []
        self.model_phases = []

        if edit_mode:
            target_samples, target_width, target_height = validate_krea2_edit_target(
                latent_image, source_image, mode, denoise, add_noise, start_at_step,
            )
            model, positive, negative, source_latent, conditioning_image = prepare_krea2_edit(
                edit_model if edit_model is not None else model,
                clip, vae, source_image, edit_prompt,
                edit_negative_prompt, grounding_px,
                target_width, target_height,
                target_batch=int(target_samples.shape[0]),
            )
            latent_image = make_krea2_edit_target(latent_image)
            if mode == "multi_model":
                if model_2 is not None:
                    model_2 = patch_krea2_edit_model(
                        model_2, source_latent, target_batch=int(target_samples.shape[0]),
                    )
                if model_3 is not None:
                    model_3 = patch_krea2_edit_model(
                        model_3, source_latent, target_batch=int(target_samples.shape[0]),
                    )

        if mode == "advanced":
            return self.run_advanced(
                model, add_noise, seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
                scheduler, positive, negative, latent_image, start_at_step, end_at_step,
                return_with_leftover_noise, cfg_curve=cfg_curve, denoise=denoise)
        elif mode == "multi_model":
            return self.run_multi_model(
                model, add_noise, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
                scheduler, positive, negative, latent_image, seed, start_at_step, end_at_step,
                return_with_leftover_noise, randomize_seed_per_model, denoise, cfg_curve=cfg_curve,
                model_2=model_2, model_3=model_3, switch_at_step_1=switch_at_step_1, switch_at_step_2=switch_at_step_2)
        else:
            return self.run_simple(
                model, seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
                scheduler, positive, negative, latent_image, denoise)


class DonutKSamplerAdvanced(_DonutSamplerEngine):
    """
    DEPRECATED alias kept so saved "DonutSampler (Advanced)" graphs reload.

    Keeps the ORIGINAL INPUT_TYPES (exact widget names AND order) and original
    RETURN_TYPES/RETURN_NAMES, pins the advanced mode, and forwards to the shared
    engine.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg_start": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_halfway": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "halfway_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "cfg_curve": (["linear", "exponential", "logarithmic", "ease_in", "ease_out", "ease_in_out", "sine_wave", "cosine_wave", "smooth_step", "smoother_step", "circular_in", "circular_out", "back_in", "back_out", "elastic_in", "elastic_out", "bounce_in", "bounce_out", "dramatic_exponential", "dramatic_logarithmic"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "cfg_progression_info")
    FUNCTION = "sample_advanced"
    CATEGORY = "donut/sampling"

    def sample_advanced(self, model, add_noise, noise_seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
                       scheduler, positive, negative, latent_image, start_at_step, end_at_step,
                       return_with_leftover_noise, cfg_curve="linear", denoise=1.0):
        self.cfg_history = []
        self.model_phases = []
        return self.run_advanced(
            model, add_noise, noise_seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
            scheduler, positive, negative, latent_image, start_at_step, end_at_step,
            return_with_leftover_noise, cfg_curve=cfg_curve, denoise=denoise)


class DonutMultiModelSampler(_DonutSamplerEngine):
    """
    DEPRECATED alias kept so saved "DonutMultiModelSampler" graphs reload.

    Keeps the ORIGINAL INPUT_TYPES (exact widget names AND order) and original
    RETURN_TYPES/RETURN_NAMES, pins the multi-model mode, and forwards to the
    shared engine.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_1": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg_start": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_halfway": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "halfway_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "randomize_seed_per_model": (["disable", "enable"], {"default": "enable"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg_curve": (["linear", "exponential", "logarithmic", "ease_in", "ease_out", "ease_in_out", "sine_wave", "cosine_wave", "smooth_step", "smoother_step", "circular_in", "circular_out", "back_in", "back_out", "elastic_in", "elastic_out", "bounce_in", "bounce_out", "dramatic_exponential", "dramatic_logarithmic"], {"default": "linear"}),
            },
            "optional": {
                "model_2": ("MODEL",),
                "model_3": ("MODEL",),
                "switch_at_step_1": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "switch_at_step_2": ("INT", {"default": 15, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sampling_info")
    FUNCTION = "sample_multi_model"
    CATEGORY = "donut/sampling"

    def sample_multi_model(self, model_1, add_noise, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
                          scheduler, positive, negative, latent_image, noise_seed, start_at_step, end_at_step,
                          return_with_leftover_noise, randomize_seed_per_model, denoise, cfg_curve="linear",
                          model_2=None, model_3=None, switch_at_step_1=10, switch_at_step_2=15):
        self.cfg_history = []
        self.model_phases = []
        return self.run_multi_model(
            model_1, add_noise, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name,
            scheduler, positive, negative, latent_image, noise_seed, start_at_step, end_at_step,
            return_with_leftover_noise, randomize_seed_per_model, denoise, cfg_curve=cfg_curve,
            model_2=model_2, model_3=model_3, switch_at_step_1=switch_at_step_1, switch_at_step_2=switch_at_step_2)


NODE_CLASS_MAPPINGS = {
    "DonutSampler": DonutSampler,
    "DonutSampler (Advanced)": DonutKSamplerAdvanced,
    "DonutMultiModelSampler": DonutMultiModelSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Only the unified node is surfaced in the menu. The two legacy ids are
    # intentionally omitted here (deprecated displays) but remain in
    # NODE_CLASS_MAPPINGS above so already-saved graphs still deserialize.
    "DonutSampler": "DonutSampler",
}

import copy
import importlib.util
import math
import re
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


fake_comfy = types.ModuleType("comfy")
fake_patcher_extension = types.ModuleType("comfy.patcher_extension")


class _WrappersMP:
    DIFFUSION_MODEL = "diffusion_model"


fake_patcher_extension.WrappersMP = _WrappersMP
fake_comfy.patcher_extension = fake_patcher_extension
old_comfy = sys.modules.get("comfy")
old_patcher_extension = sys.modules.get("comfy.patcher_extension")
sys.modules["comfy"] = fake_comfy
sys.modules["comfy.patcher_extension"] = fake_patcher_extension
try:
    spec = importlib.util.spec_from_file_location(
        "donut_krea2_fusion_control_tested",
        ROOT / "DonutKrea2FusionControl.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reference_spec = importlib.util.spec_from_file_location(
        "krea2t_enhancer_reference",
        "/home/user/Programs/ComfyUI-new/ComfyUI/custom_nodes/ComfyUI-Krea2T-Enhancer/__init__.py",
        submodule_search_locations=[
            "/home/user/Programs/ComfyUI-new/ComfyUI/custom_nodes/ComfyUI-Krea2T-Enhancer"
        ],
    )
    enhancer_reference = importlib.util.module_from_spec(reference_spec)
    sys.modules["krea2t_enhancer_reference"] = enhancer_reference
    reference_spec.loader.exec_module(enhancer_reference)

    rebalance_spec = importlib.util.spec_from_file_location(
        "krea2_rebalance_reference",
        "/home/user/Programs/ComfyUI-new/ComfyUI/custom_nodes/"
        "ComfyUI-ConditioningKrea2Rebalance/conditioning_rebalance.py",
    )
    rebalance_reference = importlib.util.module_from_spec(rebalance_spec)
    rebalance_spec.loader.exec_module(rebalance_reference)
finally:
    if old_comfy is None:
        del sys.modules["comfy"]
    else:
        sys.modules["comfy"] = old_comfy
    if old_patcher_extension is None:
        del sys.modules["comfy.patcher_extension"]
    else:
        sys.modules["comfy.patcher_extension"] = old_patcher_extension


class FakeModel:
    def __init__(self):
        self.model_options = {}
        self.wrapper = None
        self.clone_count = 0
        self.patches = {}
        self.patch_strength = None

    def clone(self):
        self.clone_count += 1
        cloned = FakeModel()
        cloned.model_options = copy.deepcopy(self.model_options)
        cloned.wrapper = self.wrapper
        cloned.patches = copy.deepcopy(self.patches)
        cloned.patch_strength = self.patch_strength
        return cloned

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrapper = None

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrapper = wrapper

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.patches.update(patches)
        self.patch_strength = strength_patch
        return list(patches)


class FakeDiffusionModel:
    def __init__(self):
        self.txtlayers = 12
        self.txtdim = 2560
        self.txtmlp = object()
        self.blocks = []
        self._unpack_context = lambda value: value
        self.txtfusion = types.SimpleNamespace(projector=torch.nn.Linear(12, 1, bias=False))
        with torch.no_grad():
            self.txtfusion.projector.weight.fill_(1.0)


class FakeExecutor:
    def __init__(self, diffusion_model):
        self.class_obj = diffusion_model

    def __call__(self, x, timesteps, context, attention_mask, transformer_options, **kwargs):
        return self.class_obj.txtfusion.projector(context)


class AffineBlock:
    def __init__(self, scale, bias):
        self.scale = scale
        self.bias = bias

    def __call__(self, value, mask=None, transformer_options=None):
        return value * self.scale + self.bias


class FakeTextFusion:
    def __init__(self):
        self.layerwise_blocks = [AffineBlock(1.01, 0.02), AffineBlock(0.99, -0.01)]
        self.projector = torch.nn.Linear(12, 1, bias=False)
        self.refiner_blocks = [AffineBlock(1.02, 0.01), AffineBlock(0.98, -0.02)]
        with torch.no_grad():
            self.projector.weight.copy_(torch.linspace(-0.8, 0.7, 12).reshape(1, 12))


class StrengthFormulaTests(unittest.TestCase):
    def test_scale_around_one_uses_neutral_baseline(self):
        profile = (1.0, 2.0, 5.0)
        self.assertEqual(module._strength_gains(profile, 0.0, "scale_around_1"), (1.0, 1.0, 1.0))
        self.assertEqual(module._strength_gains(profile, 0.5, "scale_around_1"), (1.0, 1.5, 3.0))
        self.assertEqual(module._strength_gains(profile, 1.0, "scale_around_1"), profile)

    def test_geometric_power_interpolates_ratios(self):
        gains = module._strength_gains((1.0, 4.0, 9.0), 0.5, "geometric_power")
        self.assertEqual(gains, (1.0, 2.0, 3.0))

    def test_geometric_power_rejects_non_positive_profile(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            module._strength_gains((1.0, 0.0), 0.5, "geometric_power")

    def test_raw_multiply_preserves_legacy_behavior(self):
        gains = module._strength_gains((1.0, 2.0, 5.0), 0.5, "raw_multiply")
        self.assertEqual(gains, (0.5, 1.0, 2.5))

    def test_gain_normalization_modes(self):
        mean_normalized = module._normalize_gains((1.0, 2.0, 3.0), "mean_gain")
        self.assertAlmostEqual(sum(mean_normalized) / 3.0, 1.0)

        rms_normalized = module._normalize_gains((1.0, 2.0, 3.0), "rms_gain")
        rms = math.sqrt(sum(value * value for value in rms_normalized) / 3.0)
        self.assertAlmostEqual(rms, 1.0)


class ConditioningControlTests(unittest.TestCase):
    def test_tensor_rms_preserves_actual_batch_rms(self):
        torch.manual_seed(5)
        conditioning = torch.randn(2, 3, module.KREA2_CONDITIONING_DIM)
        metadata = {"pooled_output": torch.ones(1)}
        gains = module._resolve_gains(
            "classic",
            module._format_profile_string(module._PROFILE_CLASSIC),
            1.0,
            "scale_around_1",
            "tensor_rms",
        )

        result = module._rebalance_conditioning([[conditioning, metadata]], gains, "tensor_rms")
        output, output_metadata = result[0]

        before_rms = conditioning.float().square().mean(dim=(1, 2)).sqrt()
        after_rms = output.float().square().mean(dim=(1, 2)).sqrt()
        self.assertTrue(torch.allclose(before_rms, after_rms, atol=1e-5, rtol=1e-5))
        self.assertFalse(torch.equal(conditioning, output))
        self.assertIsNot(metadata, output_metadata)
        self.assertIs(metadata["pooled_output"], output_metadata["pooled_output"])

    def test_neutral_gains_are_exact_no_op(self):
        conditioning = [[torch.randn(1, 2, module.KREA2_CONDITIONING_DIM), {}]]
        result = module._rebalance_conditioning(conditioning, (1.0,) * 12, "tensor_rms")
        self.assertIs(result, conditioning)

    def test_non_krea_conditioning_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "Expected Krea 2 conditioning"):
            module._rebalance_conditioning(
                [[torch.randn(1, 2, 2560), {}]],
                module._resolve_gains(
                    "classic",
                    module._format_profile_string(module._PROFILE_CLASSIC),
                    1.0,
                    "scale_around_1",
                    "none",
                ),
                "none",
            )


class ProjectorControlTests(unittest.TestCase):
    def _apply_node(self, projector_profile="deep_2", projector_normalization="none"):
        model = FakeModel()
        conditioning = [[torch.randn(1, 2, module.KREA2_CONDITIONING_DIM), {}]]
        node = module.DonutKrea2FusionControl()
        return model, node.apply(
            model=model,
            conditioning_in_1=conditioning,
            tap_profile="off",
            tap_strength=1.0,
            tap_formula="scale_around_1",
            tap_normalization="tensor_rms",
            projector_profile=projector_profile,
            projector_strength=1.0,
            projector_formula="scale_around_1",
            projector_normalization=projector_normalization,
            per_layer_weights=",".join(["1"] * 12),
            projector_layer_weights=module._format_profile_string(module._PROFILE_DEEP_2),
        )

    def test_deep_two_wrapper_exactly_doubles_two_live_projector_inputs(self):
        original_model, result = self._apply_node()
        patched_model, conditioning_1, conditioning_2, conditioning_3, conditioning_4, diagnostics = result
        self.assertIsNot(patched_model, original_model)
        self.assertIsNotNone(patched_model.wrapper)
        self.assertIn("external_files_loaded=none", diagnostics)
        self.assertIsNotNone(conditioning_1)
        self.assertIsNone(conditioning_2)
        self.assertIsNone(conditioning_3)
        self.assertIsNone(conditioning_4)

        diffusion_model = FakeDiffusionModel()
        executor = FakeExecutor(diffusion_model)
        projector_input = torch.ones(1, 1, 1, 12)
        original_forward = diffusion_model.txtfusion.projector.forward.__func__

        output = patched_model.wrapper(
            executor,
            None,
            None,
            projector_input,
            None,
            patched_model.model_options["transformer_options"],
        )

        self.assertEqual(float(output.item()), 14.0)
        self.assertIs(diffusion_model.txtfusion.projector.forward.__func__, original_forward)

    def test_projector_tensor_rms_preserves_live_input_rms(self):
        value = torch.ones(2, 3, 4, 12)
        gains = module._resolve_gains(
            "deep_3",
            module._format_profile_string(module._PROFILE_DEEP_3),
            1.0,
            "scale_around_1",
            "tensor_rms",
        )
        scaled = module._apply_tensor_gains(value, gains, axis=-1, normalization="tensor_rms")
        before = value.square().mean(dim=(1, 2, 3)).sqrt()
        after = scaled.square().mean(dim=(1, 2, 3)).sqrt()
        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-6))

    def test_fully_off_node_does_not_clone_or_validate_conditioning(self):
        model = FakeModel()
        conditioning = [[torch.randn(1, 2, 17), {}]]
        node = module.DonutKrea2FusionControl()
        result = node.apply(
            model=model,
            conditioning_in_1=conditioning,
            tap_profile="off",
            tap_strength=7.0,
            tap_formula="raw_multiply",
            tap_normalization="tensor_rms",
            projector_profile="off",
            projector_strength=7.0,
            projector_formula="raw_multiply",
            projector_normalization="none",
            per_layer_weights=",".join(["1"] * 12),
            projector_layer_weights=",".join(["1"] * 12),
        )
        output_model, output_conditioning, output_2, output_3, output_4, diagnostics = result
        self.assertIs(output_model, model)
        self.assertIs(output_conditioning, conditioning)
        self.assertIsNone(output_2)
        self.assertIsNone(output_3)
        self.assertIsNone(output_4)
        self.assertEqual(model.clone_count, 0)
        self.assertIn("; off;", diagnostics)

    def test_numbered_conditioning_inputs_route_to_matching_outputs(self):
        model = FakeModel()
        conditioning_1 = [[torch.ones(1, 1, module.KREA2_CONDITIONING_DIM), {"slot": 1}]]
        conditioning_3 = [[torch.full((1, 1, module.KREA2_CONDITIONING_DIM), 2.0), {"slot": 3}]]
        node = module.DonutKrea2FusionControl()

        result = node.apply(
            model=model,
            conditioning_in_1=conditioning_1,
            conditioning_in_2=None,
            conditioning_in_3=conditioning_3,
            conditioning_in_4=None,
            tap_profile="deep_2",
            tap_strength=1.0,
            tap_formula="scale_around_1",
            tap_normalization="none",
            projector_profile="off",
            projector_strength=1.0,
            projector_formula="scale_around_1",
            projector_normalization="none",
            per_layer_weights=module._format_profile_string(module._PROFILE_DEEP_2),
            projector_layer_weights=",".join(["1"] * 12),
        )

        output_model, output_1, output_2, output_3, output_4, diagnostics = result
        self.assertIs(output_model, model)
        self.assertEqual(output_1[0][1]["slot"], 1)
        self.assertEqual(output_3[0][1]["slot"], 3)
        self.assertIsNone(output_2)
        self.assertIsNone(output_4)
        self.assertAlmostEqual(float(output_1[0][0].mean()), 7.0 / 6.0, places=6)
        self.assertAlmostEqual(float(output_3[0][0].mean()), 7.0 / 3.0, places=6)
        self.assertIn("conditioning_routes=2/4", diagnostics)

    def test_numbered_port_schema_is_stable(self):
        schema = module.DonutKrea2FusionControl.INPUT_TYPES()
        self.assertIn("conditioning_in_1", schema["required"])
        self.assertEqual(
            list(schema["optional"]),
            ["conditioning_in_2", "conditioning_in_3", "conditioning_in_4"],
        )
        self.assertEqual(
            module.DonutKrea2FusionControl.RETURN_NAMES,
            (
                "model",
                "conditioning_out_1",
                "conditioning_out_2",
                "conditioning_out_3",
                "conditioning_out_4",
                "diagnostics",
            ),
        )


class CompatibilityPresetTests(unittest.TestCase):
    def _apply_settings(self, preset, conditioning, **overrides):
        settings = dict(
            model=FakeModel(),
            conditioning_in_1=conditioning,
            compatibility_preset=preset,
            tap_method=module.TAP_METHOD_DONUT,
            tap_profile="off",
            tap_strength=1.0,
            tap_formula="scale_around_1",
            tap_normalization="none",
            projector_method=module.PROJECTOR_METHOD_DONUT,
            projector_profile="off",
            projector_strength=1.0,
            projector_formula="scale_around_1",
            projector_normalization="none",
            fusion_method=module.FUSION_METHOD_STANDARD,
            fusion_strength=1.0,
            per_layer_weights=",".join(["1"] * 12),
            projector_layer_weights=",".join(["1"] * 12),
        )
        settings.update(overrides)
        return module.DonutKrea2FusionControl().apply(**settings)

    def test_bypass_copy_presets_embed_exact_file_tensors_at_strength_one(self):
        conditioning = [[torch.randn(1, 1, 17), {}]]
        cases = (
            (module.PRESET_BYPASS_2, module.PROJECTOR_METHOD_BYPASS_2, module._BYPASS_2_DIFF),
            (module.PRESET_BYPASS_3, module.PROJECTOR_METHOD_BYPASS_3, module._BYPASS_3_DIFF),
        )
        for preset, method, expected in cases:
            with self.subTest(preset=preset):
                patched_model, output, _, _, _, diagnostics = self._apply_settings(
                    preset,
                    conditioning,
                    projector_method=method,
                )
                patch_type, patch_args = patched_model.patches[module._PROJECTOR_WEIGHT_KEY]
                self.assertEqual(patch_type, "diff")
                self.assertTrue(torch.equal(patch_args[0], torch.tensor([expected], dtype=torch.float32)))
                self.assertEqual(patched_model.patch_strength, 1.0)
                self.assertIs(output, conditioning)
                self.assertIn(preset, diagnostics)

    def test_rebalance_copy_matches_nova452_profile_at_strength_one(self):
        torch.manual_seed(11)
        value = torch.randn(1, 2, module.KREA2_CONDITIONING_DIM, dtype=torch.bfloat16)
        conditioning = [[value, {"slot": 1}]]
        expected = rebalance_reference.ConditioningKrea2Rebalance().main(
            conditioning,
            1.0,
            rebalance_reference.ConditioningKrea2Rebalance.DEFAULT_WEIGHTS,
        )[0]

        _, actual, _, _, _, diagnostics = self._apply_settings(
            module.PRESET_REBALANCE,
            conditioning,
            tap_method=module.TAP_METHOD_REBALANCE,
            tap_profile="classic",
            tap_strength=1.0,
            per_layer_weights=module._format_profile_string(module._PROFILE_CLASSIC),
        )

        self.assertTrue(torch.equal(actual[0][0], expected[0][0]))
        self.assertEqual(actual[0][1], expected[0][1])
        self.assertIn("nova452", diagnostics)
        self.assertIn("multiplier=1", diagnostics)

    def test_enhancer_copy_matches_capitan01r_reference_algorithm(self):
        torch.manual_seed(13)
        value = torch.randn(1, 2, 12, 2560)
        reference_fusion = FakeTextFusion()
        donut_fusion = copy.deepcopy(reference_fusion)

        expected, _ = enhancer_reference._enhanced_txtfusion_forward(
            reference_fusion,
            value,
            transformer_options={},
            strength=1.0,
            collect_debug=False,
        )
        actual = module._capitan01r_enhancer_forward(
            donut_fusion,
            value,
            transformer_options={},
            strength=1.0,
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_enhancer_copy_preset_attaches_at_upstream_strength_one(self):
        conditioning = [[torch.randn(1, 1, 17), {}]]
        patched_model, output, _, _, _, diagnostics = self._apply_settings(
            module.PRESET_ENHANCER,
            conditioning,
            fusion_method=module.FUSION_METHOD_ENHANCER,
            fusion_strength=1.0,
        )
        config = patched_model.model_options["transformer_options"][module.CONFIG_KEY]
        self.assertEqual(config["enhancer_strength"], 1.0)
        self.assertIs(output, conditioning)
        self.assertIn("capitan01R", diagnostics)

    def test_preset_labels_are_explicitly_attributed(self):
        options = module.DonutKrea2FusionControl.INPUT_TYPES()["required"]["compatibility_preset"][0]
        self.assertEqual(tuple(options), module.COMPATIBILITY_PRESETS)
        allowed_prefixes = ("COPY settings:", "HYBRID settings:", "DONUT settings:")
        self.assertTrue(all(label == module.PRESET_MANUAL or label.startswith(allowed_prefixes) for label in options))
        self.assertIn(module.PRESET_REBALANCE_ENHANCER, options)
        self.assertIn(module.PRESET_REBALANCE_BYPASS_2, options)
        self.assertIn(module.PRESET_REBALANCE_BYPASS_3, options)
        self.assertNotIn("DONUT settings: Rebalance profile @ tap strength 1", options)
        self.assertNotIn("DONUT settings: Rebalance @ 1 + Krea2T-Enhancer", options)
        self.assertIn(module.PRESET_DONUT_BALANCED, options)
        self.assertIn(module.PRESET_DONUT_BALANCED_ENHANCER, options)

    def test_ui_presets_use_tap_strength_one_without_duplicate_labels(self):
        source = (ROOT / "web/donut_krea2_fusion_control.js").read_text()
        self.assertNotIn("tap_strength: 4.0", source)
        labels = re.findall(r'^  "([^"]+)": \{$', source, flags=re.MULTILINE)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(set(labels), set(module.COMPATIBILITY_PRESETS) - {module.PRESET_MANUAL})

    def test_preset_label_never_overrides_visible_settings_in_backend(self):
        conditioning = [[torch.randn(1, 1, 17), {}]]
        model, output, _, _, _, diagnostics = self._apply_settings(module.PRESET_BYPASS_2, conditioning)
        self.assertEqual(model.patches, {})
        self.assertIsNone(model.wrapper)
        self.assertIs(output, conditioning)
        self.assertIn("preset_is_ui_only=true", diagnostics)

    def test_visible_per_layer_string_is_the_executed_tap_profile(self):
        conditioning = [[torch.ones(1, 1, module.KREA2_CONDITIONING_DIM), {}]]
        _, output, _, _, _, _ = self._apply_settings(
            module.PRESET_MANUAL,
            conditioning,
            tap_profile="classic",
            tap_strength=1.0,
            tap_normalization="none",
            per_layer_weights=module._format_profile_string(module._PROFILE_DEEP_2),
        )
        self.assertAlmostEqual(float(output[0][0].mean()), 7.0 / 6.0, places=6)

    def test_visible_weight_string_widgets_are_in_schema(self):
        required = module.DonutKrea2FusionControl.INPUT_TYPES()["required"]
        names = list(required)
        self.assertEqual(
            required["per_layer_weights"][1]["default"],
            "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0",
        )
        self.assertIn("projector_layer_weights", required)
        self.assertEqual(names.index("per_layer_weights"), names.index("tap_profile") + 1)
        self.assertEqual(names.index("projector_layer_weights"), names.index("projector_profile") + 1)

    def test_visible_projector_and_fusion_methods_can_be_combined(self):
        conditioning = [[torch.randn(1, 1, 17), {}]]
        model, _, _, _, _, _ = self._apply_settings(
            module.PRESET_MANUAL,
            conditioning,
            projector_method=module.PROJECTOR_METHOD_BYPASS_2,
            fusion_method=module.FUSION_METHOD_ENHANCER,
            fusion_strength=1.0,
        )
        self.assertIn(module._PROJECTOR_WEIGHT_KEY, model.patches)
        self.assertEqual(
            model.model_options["transformer_options"][module.CONFIG_KEY]["enhancer_strength"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

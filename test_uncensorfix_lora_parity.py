"""CPU contract regressions for UncensorFix / Donut Apply parity.

Run: python test_uncensorfix_lora_parity.py -v
The preset and bypass bridge are real; ComfyUI, the base fusion node, and the
Donut Apply helper's environment are doubles. These tests establish routing,
not GPU/image-generation or quantized-kernel equivalence.
"""

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent
PACKAGE = "_donut_uncensorfix_parity"
BASE_NAMES = (
    "PRESET_MANUAL", "PRESET_BYPASS_2", "PRESET_BYPASS_3", "PRESET_REBALANCE",
    "PRESET_ENHANCER", "PRESET_REBALANCE_ENHANCER", "PRESET_REBALANCE_BYPASS_2",
    "PRESET_REBALANCE_BYPASS_3", "PRESET_DONUT_BALANCED", "PRESET_DONUT_BALANCED_ENHANCER",
)


class Adapter:
    def __init__(self, loaded_keys, weights):
        self.loaded_keys, self.weights = loaded_keys, weights


class Patcher:
    """Append-only Comfy-like patch lists; never overwrite existing patches."""
    def __init__(self, factors):
        self.state = {key: torch.zeros(up.shape[0], down.shape[1])
                      for key, up, down, _ in factors}
        self.model = self
        self.patches = {}
        self.injections = {}
        self.additional_models = {}
        self.attachments = {}
        self.model_options = {"transformer_options": {"upstream": object()}}
        self.bypass = {}
        self.merge_info = None
        self.patches_uuid = None

    def state_dict(self):
        return self.state

    def clone(self):
        result = copy.copy(self)
        result.model = result
        result.patches = {key: list(value) for key, value in self.patches.items()}
        result.injections = dict(self.injections)
        result.additional_models = dict(self.additional_models)
        result.attachments = dict(self.attachments)
        result.model_options = copy.deepcopy(self.model_options)
        result.bypass = dict(self.bypass)
        return result

    def add_patches(self, patches, strength_patch=1.0):
        accepted = []
        for key, patch in patches.items():
            if key not in self.state:
                continue
            self.patches.setdefault(key, []).append((strength_patch, patch, 1.0, None, None))
            accepted.append(key)
        return accepted

    def set_additional_models(self, key, models):
        self.additional_models[key] = models

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def effective_weight(self, key):
        weight = self.state[key].clone()
        for strength, adapter, _, _, _ in self.patches.get(key, ()):
            up, down, alpha, *_ = adapter.weights
            weight += (strength * (alpha / down.shape[0])) * (up @ down)
        return weight


class UncensorFixParityTests(unittest.TestCase):
    def setUp(self):
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        base = types.ModuleType(PACKAGE + ".DonutKrea2FusionControl")
        for name in BASE_NAMES:
            setattr(base, name, name)
        base.PRESET_MANUAL = "Custom settings"
        self.base_calls = []
        calls = self.base_calls

        class Base:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {
                    "model": ("MODEL",), "conditioning_in_1": ("CONDITIONING",),
                    "compatibility_preset": (["Custom settings"], {}),
                    "tap_strength": ("FLOAT", {"default": 1.0}),
                }, "optional": {"conditioning_in_2": ("CONDITIONING",)}}

            def apply(self, model, conditioning_in_1, tap_strength=1.0,
                      compatibility_preset="Custom settings", conditioning_in_2=None,
                      conditioning_in_3=None, conditioning_in_4=None, **kwargs):
                calls.append(kwargs)
                model = model.clone()
                model.model_options["fusion_controls_executed"] = True
                return (model, conditioning_in_1 * 2, conditioning_in_2, conditioning_in_3,
                        conditioning_in_4,
                        "preset_label=Custom settings; preset_is_ui_only=true\nexternal_files_loaded=none")

        base.DonutKrea2FusionControl = Base
        package.DonutKrea2FusionControl = base
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        weight_adapter = types.ModuleType("comfy.weight_adapter")
        weight_adapter.__path__ = []
        lora = types.ModuleType("comfy.weight_adapter.lora")
        lora.LoRAAdapter = Adapter
        serialization = types.ModuleType(PACKAGE + ".donut_krea2_merge_serialization")
        serialization.KREA2_MERGE_INJECTION_KEY = "merge_swap"
        serialization.KREA2_MERGE_SOURCE_KEY = "merge_source"
        serialization.get_krea2_merge_bypass_info = lambda model: model.merge_info
        nodes = types.ModuleType(PACKAGE + ".donut_lora_nodes")
        nodes._TEXT_MERGE_VECTOR = ",".join(["1"] * 13)
        block = types.ModuleType(PACKAGE + ".lora_block_weight")

        def load_lbw(model, clip, state, **kwargs):
            patches = {}
            for key in state:
                if key.endswith(".lora_up.weight"):
                    stem = key[:-len(".lora_up.weight")]
                    target = stem + ".weight"
                    if target in model.state:
                        patches[target] = (Adapter(set(), (
                            state[key], state[stem + ".lora_down.weight"],
                            float(state[stem + ".alpha"]), None, None, None)), 1.0)
            return patches, [], kwargs["block_vector"]

        block.LoraLoaderBlockWeight = types.SimpleNamespace(load_lbw=mock.Mock(side_effect=load_lbw))
        self.loader = block.LoraLoaderBlockWeight
        safe = types.ModuleType(PACKAGE + ".DonutSafeApplyLoRAStack")

        def shared_bypass(model, applications):
            result = model.clone()
            for state, strength, vector in applications:
                weights, _, _ = load_lbw(model, None, state, block_vector=vector)
                for key, (adapter, ratio) in weights.items():
                    if any(model.injections.values()):
                        result.add_patches({key: adapter}, strength * ratio)
                    else:
                        result.bypass[key] = (adapter, strength * ratio)
            if result.bypass:
                result.injections["donut_bypass_lora"] = [object()]
            return result

        safe._apply_bypass_applications = mock.Mock(side_effect=shared_bypass)
        self.shared_bypass = safe._apply_bypass_applications
        self.modules = mock.patch.dict(sys.modules, {
            PACKAGE: package, base.__name__: base,
            "comfy": comfy, "comfy.weight_adapter": weight_adapter,
            "comfy.weight_adapter.lora": lora,
            serialization.__name__: serialization, nodes.__name__: nodes,
            block.__name__: block, safe.__name__: safe,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.module = self.load("DonutKrea2FusionPreset")
        self.bridge = self.load("donut_uncensorfix_lora")
        self.factors = tuple(
            (f"diffusion_model.txtfusion.test{i}.weight",
             torch.arange(8, dtype=torch.float32).reshape(2, 4) / (i + 5),
             torch.arange(12, dtype=torch.float32).reshape(4, 3) / 20, 4.0)
            for i in range(33)
        )
        patch = mock.patch.object(self.module, "_uncensorfix_factors", return_value=self.factors)
        self.factors_mock = patch.start()
        self.addCleanup(patch.stop)
        self.model = Patcher(self.factors)
        self.cond = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)

    def load(self, name):
        spec = importlib.util.spec_from_file_location(PACKAGE + "." + name, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def apply(self, **kwargs):
        inputs = dict(model=self.model, conditioning_in_1=self.cond,
                      compatibility_preset="UncensorFix", tap_strength=1.0)
        inputs.update(kwargs)
        return self.module.DonutKrea2FusionControl().apply(**inputs)

    def test_default_is_lora_only_in_both_ui_modes(self):
        for mode in ("Simple", "Advanced"):
            result = self.apply(ui_mode=mode, tap_profile="classic", projector_strength=2.0,
                                fusion_method="enhancer", fusion_strength=2.0)
            self.assertIs(result[1], self.cond)
            self.assertEqual(len(result[0].patches), 33)
            self.assertNotIn("fusion_controls_executed", result[0].model_options)
            self.assertIn("uncensorfix_controls=LoRA only", result[-1])
        self.assertEqual(self.base_calls, [])

    def test_upstream_mode_controls_uncensorfix_execution(self):
        self.model.model_options["donut_lora_execution_mode"] = "Comfy patches"
        result = self.apply(execution_mode="Experimental bypass")
        self.assertEqual(len(result[0].patches), 33)
        self.shared_bypass.assert_not_called()
        self.assertIn("uncensorfix_execution_mode=Comfy patches", result[-1])

    def test_uncensorfix_publishes_selected_mode_for_downstream_nodes(self):
        result = self.apply(execution_mode="Experimental bypass")
        self.assertEqual(
            result[0].model_options["donut_lora_execution_mode"],
            "Experimental bypass",
        )

    def test_inactive_malformed_controls_do_not_affect_lora_only(self):
        result = self.apply(per_layer_weights="not a profile", projector_strength=float("nan"))
        self.assertIs(result[1], self.cond)
        self.assertFalse(self.base_calls)

    def test_all_conditioning_routes_preserved_by_identity(self):
        routes = (self.cond, [], None, [[object(), {"mask": object()}]])
        result = self.apply(**{f"conditioning_in_{i}": route for i, route in enumerate(routes, 1)})
        for actual, expected in zip(result[1:5], routes):
            self.assertIs(actual, expected)
        self.assertIn("conditioning_routes=3/4", result[-1])

    def test_fusion_controls_remain_available_by_explicit_opt_in(self):
        result = self.apply(uncensorfix_controls="LoRA + fusion controls", tap_profile="classic")
        self.assertEqual(len(self.base_calls), 1)
        self.assertTrue(torch.equal(result[1], self.cond * 2))
        self.assertEqual(len(result[0].patches), 33)

    def test_zero_lora_only_is_exact_noop_even_with_dirty_controls(self):
        self.factors_mock.side_effect = AssertionError("decoded")
        result = self.apply(tap_strength=0.0, tap_profile="classic", fusion_strength=2.0,
                            execution_mode="Experimental bypass")
        self.assertIs(result[0], self.model)
        self.assertIs(result[1], self.cond)
        self.assertFalse(self.base_calls)
        self.shared_bypass.assert_not_called()

    def test_off_preserves_all_upstream_state_and_ignores_execution_settings(self):
        self.model.patches["upstream"] = [object()]
        self.model.injections["upstream"] = [object()]
        before = dict(vars(self.model))
        self.factors_mock.side_effect = AssertionError("decoded")
        result = self.apply(compatibility_preset="Off", execution_mode="invalid",
                            uncensorfix_controls="invalid", tap_strength=float("nan"))
        self.assertIs(result[0], self.model)
        self.assertIs(result[1], self.cond)
        for key, value in before.items():
            self.assertIs(getattr(self.model, key), value)
        self.assertFalse(self.base_calls)

    def test_native_matches_ordinary_loaded_factor_patch_stack(self):
        for strength in (-0.5, 0.75, 1.0, 2.0):
            reference = self.model.clone()
            for key, up, down, alpha in self.factors:
                reference.add_patches({key: Adapter(set(), (up, down, alpha, None, None, None))}, strength)
            actual = self.apply(tap_strength=strength)[0]
            for key in reference.patches:
                self.assertTrue(torch.equal(actual.effective_weight(key), reference.effective_weight(key)))
                self.assertEqual(actual.patches[key][0][0], strength)
        self.shared_bypass.assert_not_called()

    def test_native_appends_same_target_patches_and_does_not_mutate_input(self):
        key, up, down, alpha = self.factors[0]
        previous = Adapter(set(), (up * 2, down, alpha, None, None, None))
        self.model.add_patches({key: previous}, 0.3)
        result = self.apply(tap_strength=0.75)[0]
        self.assertEqual(len(self.model.patches[key]), 1)
        self.assertEqual(len(result.patches[key]), 2)
        self.assertIs(result.patches[key][0][1], previous)
        self.assertTrue(torch.equal(result.effective_weight(key),
                                   (0.3 * (up * 2 @ down)) + (0.75 * (up @ down))))

    def test_cached_factors_are_not_shared_with_native_adapters(self):
        result = self.apply()[0]
        for key, up, down, _ in self.factors:
            copied = result.patches[key][0][1].weights
            self.assertNotEqual(copied[0].data_ptr(), up.data_ptr())
            self.assertNotEqual(copied[1].data_ptr(), down.data_ptr())

    def test_bypass_reuses_donut_apply_helper_with_exact_factors_and_text_strength(self):
        result = self.apply(execution_mode="Experimental bypass", tap_strength=0.75)[0]
        self.shared_bypass.assert_called_once()
        model, applications = self.shared_bypass.call_args.args
        self.assertIs(model, self.model)
        state, strength, vector = applications[0]
        self.assertEqual(strength, 0.75)
        self.assertEqual(vector, ",".join(["1"] * 13))
        self.assertEqual(len(state), 99)
        for key, up, down, alpha in self.factors:
            stem = key[:-7]
            self.assertTrue(torch.equal(state[stem + ".lora_up.weight"], up))
            self.assertTrue(torch.equal(state[stem + ".lora_down.weight"], down))
            self.assertEqual(float(state[stem + ".alpha"]), alpha)
        self.assertEqual(len(result.bypass), 33)
        self.assertFalse(result.patches)
        self.assertFalse(self.model.bypass)

    def test_bypass_retains_shared_helper_regular_fallback(self):
        self.model.injections["upstream"] = [object()]
        result = self.apply(execution_mode="Experimental bypass")[0]
        self.assertEqual(len(result.patches), 33)
        self.assertIs(result.injections["upstream"], self.model.injections["upstream"])
        self.assertFalse(result.bypass)

    def test_bypass_preflight_rejects_missing_remapped_or_muted_targets(self):
        valid_load = self.loader.load_lbw.side_effect
        for failure in ("missing", "remapped", "muted", "ratio"):
            def broken(*args, **kwargs):
                weights, muted, vector = valid_load(*args, **kwargs)
                key = next(iter(weights))
                if failure == "missing":
                    weights.pop(key)
                elif failure == "remapped":
                    weights["wrong.weight"] = weights.pop(key)
                elif failure == "muted":
                    muted.append(key)
                else:
                    weights[key] = (weights[key][0], 0.5)
                return weights, muted, vector
            with self.subTest(failure=failure):
                self.loader.load_lbw.side_effect = broken
                with self.assertRaisesRegex(RuntimeError, "unit block weight"):
                    self.apply(execution_mode="Experimental bypass")
        self.shared_bypass.assert_not_called()

    def test_bypass_rejects_silent_noop_helper(self):
        self.shared_bypass.side_effect = lambda model, applications: model.clone()
        with self.assertRaisesRegex(RuntimeError, "did not install"):
            self.apply(execution_mode="Experimental bypass")

    def test_modes_and_legacy_labels_keep_strength_and_diagnostics(self):
        for preset in ("UncensorFix", "TeacherFix", self.module.LEGACY_TEACHERFIX):
            for mode in ("Comfy patches", "Experimental bypass"):
                result = self.apply(compatibility_preset=preset, execution_mode=mode, tap_strength=-0.5)
                self.assertIn("uncensorfix_targets=33", result[-1])
                self.assertIn(f"uncensorfix_execution_mode={mode}", result[-1])
                self.assertIn("reference_text_weight=-0.5", result[-1])

    def test_full_legacy_positional_preset_and_strength_are_resolved(self):
        result = self.module.DonutKrea2FusionControl().apply(self.model, self.cond, 0.75, "UncensorFix")
        self.assertEqual(len(result[0].patches), 33)
        self.assertEqual(next(iter(result[0].patches.values()))[0][0], 0.75)
        self.assertIs(result[1], self.cond)
        self.assertFalse(self.base_calls)

    def test_positional_off_does_not_run_base(self):
        result = self.module.DonutKrea2FusionControl().apply(self.model, self.cond, 3.0, "Off")
        self.assertIs(result[0], self.model)
        self.assertFalse(self.base_calls)

    def test_invalid_active_modes_and_strength_fail(self):
        for kwargs in ({"execution_mode": "wrong"}, {"uncensorfix_controls": "wrong"},
                       {"tap_strength": float("inf")}, {"tap_strength": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.apply(**kwargs)

    def test_optional_controls_are_appended_and_existing_node_id_is_retained(self):
        schema = self.module.DonutKrea2FusionControl.INPUT_TYPES()
        self.assertEqual(list(schema["required"])[-1], "ui_mode")
        self.assertEqual(list(schema["optional"])[-3:], ["uncensorfix_controls", "execution_mode", "uncensorfix_strength"])
        self.assertEqual(schema["optional"]["uncensorfix_controls"][1]["default"], "Fusion only")
        self.assertEqual(list(self.module.NODE_CLASS_MAPPINGS), ["DonutKrea2FusionControl"])

    def test_composition_controls_lora_independently_of_preset(self):
        for preset in ("Balanced", "Custom", "UncensorFix", "Off"):
            with self.subTest(preset=preset):
                result = self.apply(compatibility_preset=preset, uncensorfix_controls="Fusion + LoRA")
                self.assertEqual(len(result[0].patches), 33)
                result = self.apply(compatibility_preset=preset, uncensorfix_controls="Fusion only")
                self.assertEqual(len(result[0].patches), 0)

    def test_weight_strength_is_independent_of_tap_strength(self):
        for tap in (0.0, 0.4, 2.0):
            result = self.apply(compatibility_preset="Balanced", uncensorfix_controls="Fusion + UncensorFix weights",
                                tap_strength=tap, uncensorfix_strength=0.75)
            self.assertIn("uncensorfix_strength=0.75", result[-1])
            self.assertEqual(len(result[0].patches), 33)
        result = self.apply(compatibility_preset="Balanced", uncensorfix_controls="Fusion + UncensorFix weights",
                            tap_strength=2, uncensorfix_strength=0)
        self.assertEqual(len(result[0].patches), 0)

    def test_partial_merge_routes_both_modes_without_mutating_model2(self):
        for mode in ("Comfy patches", "Experimental bypass"):
            with self.subTest(mode=mode):
                model = Patcher(self.factors)
                source = Patcher(self.factors)
                keys = [self.factors[i][0] for i in range(5)]
                # Match the false-y yet non-empty composable swap list.
                class Composable(list):
                    def __bool__(self):
                        return False
                model.injections["merge_swap"] = Composable([object()])
                model.merge_info = (source, [(None, key, None) for key in keys], None)
                result = self.apply(model=model, execution_mode=mode)[0]
                patched_source = result.additional_models["merge_source"][0]
                main_keys = set(result.patches) | set(result.bypass)
                source_keys = set(patched_source.patches) | set(patched_source.bypass)
                self.assertEqual(main_keys, {x[0] for x in self.factors} - set(keys))
                self.assertEqual(source_keys, set(keys))
                self.assertFalse(source.patches)
                self.assertFalse(source.bypass)
                self.assertFalse(model.patches)
                self.assertTrue(result.attachments)
                self.assertIsNotNone(result.patches_uuid)

    def test_native_off_and_zero_never_call_bypass_loader(self):
        self.loader.load_lbw.side_effect = AssertionError("bypass loader called")
        self.apply()
        self.apply(compatibility_preset="Off")
        self.apply(tap_strength=0.0, execution_mode="Experimental bypass")
        self.shared_bypass.assert_not_called()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

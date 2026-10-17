"""Tests of real embedded tensors and patch plumbing with minimal Comfy doubles.

No original safetensors file is needed. These are CPU unit tests, not a full
ComfyUI/GPU image-generation test.
"""
import hashlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent
EXPECTED_PAYLOAD_SHA = "f3c817bd957e6d47883346237b5e067697f0b9e1c9909bd06353da455949aacf"
REFUSAL_PAYLOAD_SHA = "8d2798e11b79f90c28744e6615a40ff1d8d2ee30fa86f47e2dfddb71467480ba"
BASE_NAMES = {
    "PRESET_MANUAL": "Custom settings",
    "PRESET_BYPASS_2": "COPY settings: Krea2FilterBypass 2vector",
    "PRESET_BYPASS_3": "COPY settings: Krea2FilterBypass 3vector",
    "PRESET_REBALANCE": "COPY settings: nova452 ConditioningKrea2Rebalance profile @ tap strength 1",
    "PRESET_ENHANCER": "COPY settings: capitan01R Krea2T-Enhancer defaults",
    "PRESET_REBALANCE_ENHANCER": "HYBRID settings: Rebalance + Krea2T-Enhancer",
    "PRESET_REBALANCE_BYPASS_2": "HYBRID settings: Rebalance + Krea2FilterBypass 2vector",
    "PRESET_REBALANCE_BYPASS_3": "HYBRID settings: Rebalance + Krea2FilterBypass 3vector",
    "PRESET_DONUT_BALANCED": "DONUT settings: RMS-balanced classic",
    "PRESET_DONUT_BALANCED_RAW_NAG": "DONUT settings: RMS-balanced classic, raw NAG negative",
    "PRESET_DONUT_BALANCED_ENHANCER": "DONUT settings: RMS-balanced classic + Krea2T-Enhancer",
}
SHORT_NAMES = ["Off", "Custom", "Bypass 2", "Bypass 3", "Rebalance", "Enhancer",
               "Rebalance + Enhancer", "Rebalance + Bypass 2", "Rebalance + Bypass 3",
               "Balanced", "Balanced + raw NAG", "Balanced + Enhancer", "Neutral", "UncensorFix",
               "Refusal Reduction v1", "Refusal Reduction v2"]
LEGACY_WIDGETS = ["model", "conditioning_in_1", "compatibility_preset", "tap_method",
                  "tap_profile", "per_layer_weights", "tap_strength", "tap_formula",
                  "tap_normalization", "projector_method", "projector_profile",
                  "projector_layer_weights", "projector_strength", "projector_formula",
                  "projector_normalization", "fusion_method", "fusion_strength"]


class FakeAdapter:
    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights


class FakeModel:
    def __init__(self, factors):
        self.state = {key: types.SimpleNamespace(shape=(up.shape[0], down.shape[1]))
                      for key, up, down, _ in factors}
        self.model = self
        self.patches = {}
        self.strength = None

    def state_dict(self):
        return self.state

    def clone(self):
        clone = FakeModel(())
        clone.state = dict(self.state)
        clone.patches = dict(self.patches)
        return clone

    def add_patches(self, patches, strength_patch=1.0):
        self.patches.update(patches)
        self.strength = strength_patch
        return list(patches)


class EmbeddedUncensorFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name = "_donut_embedded_testpkg"
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT)]
        base = types.ModuleType(name + ".DonutKrea2FusionControl")
        for key, value in BASE_NAMES.items():
            setattr(base, key, value)
        base.calls = []

        class BaseNode:
            @classmethod
            def INPUT_TYPES(cls):
                required = {key: ("STRING", {}) for key in LEGACY_WIDGETS}
                required["compatibility_preset"] = (list(BASE_NAMES.values()), {"default": base.PRESET_MANUAL})
                required["tap_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": .05})
                return {"required": required, "optional": {"conditioning_in_2": ("CONDITIONING",)}}

            def apply(self, **kwargs):
                base.calls.append(kwargs)
                return (kwargs["model"], kwargs["conditioning_in_1"], None, None, None,
                        f"preset_label={kwargs['compatibility_preset']}; preset_is_ui_only=true\n"
                        "external_files_loaded=none")

        base.DonutKrea2FusionControl = BaseNode
        package.DonutKrea2FusionControl = base
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        adapter_package = types.ModuleType("comfy.weight_adapter")
        adapter_package.__path__ = []
        adapter_module = types.ModuleType("comfy.weight_adapter.lora")
        adapter_module.LoRAAdapter = FakeAdapter
        cls.modules = mock.patch.dict(sys.modules, {
            name: package, base.__name__: base,
            "comfy": comfy, "comfy.weight_adapter": adapter_package,
            "comfy.weight_adapter.lora": adapter_module,
            "folder_paths": None, "comfy.lora": None, "comfy.utils": None,
            "safetensors": None, "safetensors.torch": None,
        })
        cls.modules.start()
        cls.addClassCleanup(cls.modules.stop)
        spec = importlib.util.spec_from_file_location(name + ".DonutKrea2FusionPreset", ROOT / "DonutKrea2FusionPreset.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.weights = __import__(name + ".uncensorfix_weights", fromlist=["*"])
        cls.factors = cls.weights.get_uncensorfix_factors()
        cls.refusal = __import__(name + ".refusal_reduction_weights", fromlist=["*"])
        cls.refusal_v1 = cls.refusal.get_refusal_reduction_v1_factors()
        cls.refusal_v2 = cls.refusal.get_refusal_reduction_v2_factors()

    def test_all_33_embedded_targets_and_exact_payload(self):
        self.assertEqual(len(self.factors), 33)
        payload = b"".join(tensor.numpy().astype("<f4", copy=False).tobytes()
                           for _, up, down, _ in self.factors for tensor in (up, down))
        self.assertEqual(len(payload), 3_457_232)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), EXPECTED_PAYLOAD_SHA)

    def test_rank_alpha_dtype_scope_and_finiteness(self):
        for key, up, down, alpha in self.factors:
            self.assertTrue(key.startswith("diffusion_model.txtfusion."))
            self.assertEqual(up.dtype, torch.float32)
            self.assertEqual(down.dtype, torch.float32)
            self.assertEqual(up.shape[1], 4)
            self.assertEqual(down.shape[0], 4)
            self.assertEqual(alpha, 4.0)
            self.assertTrue(torch.isfinite(up).all())
            self.assertTrue(torch.isfinite(down).all())

    def test_no_original_container_or_metadata(self):
        raw = self.weights._read_payload()
        self.assertNotIn(b"source_checkpoint", raw)
        self.assertNotIn(b"__metadata__", raw)
        source = (ROOT / "uncensorfix_weights.py").read_text()
        self.assertNotIn("krea2_c33_teacherfix_ema5000.safetensors", source)
        self.assertNotIn("source_checkpoint", source)

    def test_runtime_needs_no_safetensors_or_lora_loader(self):
        # Those module imports are blocked for the entire test class.
        result, count, name = self.module._apply_uncensorfix(FakeModel(self.factors), 1.0)
        self.assertEqual(count, 33)
        self.assertEqual(name, "embedded")
        self.assertEqual(len(result.patches), 33)

    def test_decode_is_cached(self):
        with mock.patch.object(self.weights, "_read_payload", side_effect=AssertionError("decoded twice")):
            self.assertIs(self.factors, self.weights.get_uncensorfix_factors())

    def test_refusal_payload_is_exact_bf16_and_hash_checked(self):
        raw = self.refusal._read_payload()
        self.assertEqual(len(raw), 30_344_832)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), REFUSAL_PAYLOAD_SHA)
        self.assertNotIn(b"__metadata__", raw)
        for key, up, down, alpha in self.refusal_v2:
            self.assertEqual(up.dtype, torch.float32)
            self.assertEqual((up.shape[1], down.shape[0], alpha), (64, 64, 64.0))
            self.assertTrue(torch.isfinite(up).all() and torch.isfinite(down).all())
            # Decoded from bfloat16, so every value must round-trip through it.
            self.assertTrue(torch.equal(up, up.to(torch.bfloat16).to(torch.float32)))
            self.assertTrue(torch.equal(down, down.to(torch.bfloat16).to(torch.float32)))

    def test_refusal_v1_is_the_block_subset_of_v2(self):
        self.assertEqual((len(self.refusal_v1), len(self.refusal_v2)), (32, 35))
        v2 = {key: (up, down) for key, up, down, _ in self.refusal_v2}
        for key, up, down, _ in self.refusal_v1:
            self.assertIn("_blocks.", key)
            self.assertIs(up, v2[key][0])
            self.assertIs(down, v2[key][1])
        extra = {key for key, *_ in self.refusal_v2} - {key for key, *_ in self.refusal_v1}
        self.assertEqual(extra, {"diffusion_model.txtfusion.projector.weight",
                                 "diffusion_model.txtmlp.1.weight", "diffusion_model.txtmlp.3.weight"})

    def test_refusal_weights_do_not_duplicate_uncensorfix(self):
        # Same targets, unrelated weights: no identical factor tensors, tiny cosine overlap.
        fix = {key: (up, down, alpha) for key, up, down, alpha in self.factors}
        for key, up, down, alpha in self.refusal_v2:
            if key not in fix:
                continue
            fu, fd, fa = fix[key]
            mine, theirs = up @ down, (fa / fd.shape[0]) * (fu @ fd)
            cosine = torch.nn.functional.cosine_similarity(mine.flatten(), theirs.flatten(), dim=0)
            self.assertLess(abs(float(cosine)), 0.3, key)
            self.assertFalse(torch.equal(mine, theirs))

    def test_refusal_presets_apply_their_own_targets_in_both_compositions(self):
        cases = (("Refusal Reduction v1", 32, "diffusion_model.txtmlp.1.weight"),
                 ("Refusal Reduction v2", 35, None))
        for preset, expected, absent in cases:
            for composition in ("LoRA only", "LoRA + fusion controls", "Fusion + UncensorFix weights"):
                with self.subTest(preset=preset, composition=composition):
                    cond = object()
                    model = FakeModel(self.refusal_v2)
                    result = self.module.DonutKrea2FusionControl().apply(
                        model=model, conditioning_in_1=cond, compatibility_preset=preset,
                        tap_strength=.75, uncensorfix_strength=.75, uncensorfix_controls=composition)
                    self.assertEqual(len(result[0].patches), expected)
                    self.assertEqual(result[0].strength, .75)
                    if absent:
                        self.assertNotIn(absent, result[0].patches)
                    self.assertIn(f"uncensorfix_targets={expected}", result[-1])
                    if composition == "Fusion + UncensorFix weights":
                        self.assertIn(f"embedded_lora={preset}", result[-1])
                    else:
                        self.assertIn(f"preset_label={preset};", result[-1])
                        self.assertIn("bundled_weight_asset=assets/refusal_reduction_v2.bf16", result[-1])
                    for key, up, down, alpha in self.refusal_v2:
                        if key in result[0].patches:
                            values = result[0].patches[key].weights
                            self.assertTrue(torch.equal(values[0], up) and torch.equal(values[1], down))
                            self.assertEqual(values[2], 64.0)

    def test_refusal_presets_are_not_applied_by_fusion_only_or_other_presets(self):
        with mock.patch.object(self.module, "_refusal_v2_factors", side_effect=AssertionError("data accessed")), \
                mock.patch.object(self.module, "_refusal_v1_factors", side_effect=AssertionError("data accessed")):
            for preset in ("Rebalance", "Balanced", "UncensorFix", "Refusal Reduction v1", "Refusal Reduction v2"):
                original = FakeModel(self.factors)
                self.module.DonutKrea2FusionControl().apply(
                    model=original, conditioning_in_1=object(), compatibility_preset=preset,
                    uncensorfix_controls="Fusion only")
        # Other presets with the weights composition still apply UncensorFix, unchanged.
        result = self.module.DonutKrea2FusionControl().apply(
            model=FakeModel(self.factors), conditioning_in_1=object(), compatibility_preset="Balanced",
            uncensorfix_controls="Fusion + UncensorFix weights")
        self.assertIn("uncensorfix_targets=33", result[-1])

    def test_embedded_weights_are_chosen_independently_of_the_preset(self):
        cases = (("UncensorFix", 33, self.factors), ("Refusal Reduction v1", 32, self.refusal_v2),
                 ("Refusal Reduction v2", 35, self.refusal_v2))
        for preset in ("Neutral", "Rebalance", "Balanced + Enhancer"):
            for weights, expected, factors in cases:
                with self.subTest(preset=preset, weights=weights):
                    result = self.module.DonutKrea2FusionControl().apply(
                        model=FakeModel(factors), conditioning_in_1=object(), compatibility_preset=preset,
                        uncensorfix_controls="Fusion + UncensorFix weights", embedded_weights=weights)
                    self.assertEqual(len(result[0].patches), expected)
                    self.assertIn(f"uncensorfix_targets={expected}", result[-1])

    def test_embedded_weights_none_and_fusion_only_apply_nothing(self):
        with mock.patch.object(self.module, "_uncensorfix_factors", side_effect=AssertionError("data accessed")), \
                mock.patch.object(self.module, "_refusal_v2_factors", side_effect=AssertionError("data accessed")), \
                mock.patch.object(self.module, "_refusal_v1_factors", side_effect=AssertionError("data accessed")):
            for controls, weights in (("Fusion + UncensorFix weights", "None"),
                                      ("Fusion only", "UncensorFix"), ("Fusion only", "Refusal Reduction v2")):
                with self.subTest(controls=controls, weights=weights):
                    result = self.module.DonutKrea2FusionControl().apply(
                        model=FakeModel(self.factors), conditioning_in_1=object(), compatibility_preset="Neutral",
                        uncensorfix_controls=controls, embedded_weights=weights)
                    self.assertEqual(result[0].patches, {})
                    self.assertNotIn("uncensorfix_targets", result[-1])

    def test_unknown_embedded_weights_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown embedded weights"):
            self.module.DonutKrea2FusionControl().apply(
                model=FakeModel(self.factors), conditioning_in_1=object(), compatibility_preset="Neutral",
                uncensorfix_controls="Fusion + UncensorFix weights", embedded_weights="Other")

    def test_embedded_weights_widget_is_appended_after_every_older_widget(self):
        optional = self.module.DonutKrea2FusionControl.INPUT_TYPES()["optional"]
        self.assertEqual(list(optional)[-1], "embedded_weights")
        self.assertEqual(optional["embedded_weights"][0], ["None", "UncensorFix", "Refusal Reduction v1", "Refusal Reduction v2"])
        self.assertEqual(optional["embedded_weights"][1]["default"], "UncensorFix")

    def test_refusal_v2_rejects_a_model_missing_its_targets(self):
        model = FakeModel(self.refusal_v1)  # no projector or txtmlp targets
        with self.assertRaisesRegex(RuntimeError, "Refusal Reduction v2 target missing or wrong shape"):
            self.module._apply_uncensorfix(model, 1.0, lora=self.module.REFUSAL_V2_LORA)
        result, count, _ = self.module._apply_uncensorfix(FakeModel(self.refusal_v1), 1.0, lora=self.module.REFUSAL_V1_LORA)
        self.assertEqual(count, 32)

    def test_corrupt_payload_fails(self):
        with mock.patch.object(Path, "open", mock.mock_open(read_data=b"corrupt")):
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self.weights._read_payload()

    def test_missing_asset_fails(self):
        with mock.patch.object(Path, "open", side_effect=FileNotFoundError("missing")):
            with self.assertRaisesRegex(RuntimeError, "missing or unreadable"):
                self.weights._read_payload()

    def test_wrong_checksum_fails(self):
        with mock.patch.object(self.weights, "PAYLOAD_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self.weights._read_payload()

    def test_wrong_size_fails(self):
        with mock.patch.object(self.weights, "PAYLOAD_SIZE_BYTES", 64):
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self.weights._read_payload()

    def test_schema_short_names_and_node_id_preserved(self):
        required = self.module.DonutKrea2FusionControl.INPUT_TYPES()["required"]
        self.assertEqual(list(required), LEGACY_WIDGETS + ["ui_mode"])
        self.assertEqual(required["compatibility_preset"][0], SHORT_NAMES)
        self.assertEqual(required["compatibility_preset"][1]["default"], "Custom")
        self.assertEqual(required["ui_mode"][1]["default"], "Advanced")
        self.assertEqual(required["tap_strength"][1]["max"], 10.0)
        self.assertEqual(list(self.module.NODE_CLASS_MAPPINGS), ["DonutKrea2FusionControl"])

    def test_zero_strength_never_decodes_or_inspects_model(self):
        original = object()
        with mock.patch.object(self.module, "_uncensorfix_factors", side_effect=AssertionError("data accessed")):
            result, count, _ = self.module._apply_uncensorfix(original, 0.0)
        self.assertIs(result, original)
        self.assertEqual(count, 0)

    def test_nonfinite_strength_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module._apply_uncensorfix(object(), value)

    def test_all_factors_alphas_and_strengths_reach_patch_path(self):
        for strength in (-0.5, 0.75, 1.0, 2.0):
            original = FakeModel(self.factors)
            original.patches["unrelated.weight"] = ("diff", ())
            result, count, _ = self.module._apply_uncensorfix(original, strength)
            self.assertIsNot(result, original)
            self.assertEqual(count, 33)
            self.assertEqual(result.strength, strength)
            self.assertEqual(len(original.patches), 1)
            self.assertIn("unrelated.weight", result.patches)
            for key, up, down, alpha in self.factors:
                values = result.patches[key].weights
                self.assertEqual(len(values), 6)
                self.assertTrue(torch.equal(values[0], up))
                self.assertTrue(torch.equal(values[1], down))
                self.assertEqual(values[2], alpha)
                self.assertEqual(values[3:], (None, None, None))
                self.assertNotEqual(values[0].data_ptr(), up.data_ptr())
                self.assertNotEqual(values[1].data_ptr(), down.data_ptr())

    def test_short_legacy_labels_both_modes_and_diagnostics(self):
        for preset in ("UncensorFix", "TeacherFix", self.module.LEGACY_TEACHERFIX):
            for mode in ("Simple", "Advanced"):
                cond = object()
                result = self.module.DonutKrea2FusionControl().apply(
                    model=FakeModel(self.factors), conditioning_in_1=cond,
                    compatibility_preset=preset, tap_strength=.75, ui_mode=mode)
                self.assertEqual(len(result[0].patches), 33)
                self.assertEqual(result[0].strength, .75)
                self.assertIs(result[1], cond)
                self.assertIn("preset_label=UncensorFix;", result[-1])
                self.assertIn("uncensorfix_source=embedded", result[-1])
                self.assertIn("uncensorfix_targets=33", result[-1])
                self.assertIn("external_files_loaded=none", result[-1])

    def test_other_presets_never_decode_weights(self):
        with mock.patch.object(self.module, "_uncensorfix_factors", side_effect=AssertionError("data accessed")):
            for short, legacy in self.module.SIMPLE_PRESET_TO_LEGACY.items():
                original = object()
                result = self.module.DonutKrea2FusionControl().apply(
                    model=original, conditioning_in_1=object(), compatibility_preset=short)
                self.assertIs(result[0], original)
                self.assertEqual(self.module.base.calls[-1]["compatibility_preset"], legacy)

    def test_current_composition_forwards_nag_experiment_flags(self):
        original = object()
        self.module.DonutKrea2FusionControl().apply(
            model=original,
            conditioning_in_1=object(),
            compatibility_preset="Rebalance",
            uncensorfix_controls=self.module.FUSION_ONLY,
            nag_text_energy_compensation=True,
            nag_batch_txtfusion=True,
        )
        forwarded = self.module.base.calls[-1]
        self.assertIs(forwarded["model"], original)
        self.assertIs(forwarded["nag_text_energy_compensation"], True)
        self.assertIs(forwarded["nag_batch_txtfusion"], True)

    def test_uncensorfix_weights_mode_forwards_nag_experiment_flags(self):
        # Regression: explicit wrapper parameters were previously dropped in
        # the Fusion-only / Fusion + UncensorFix weights delegation branch, so
        # toggling the experiment in the UI silently ran it off.
        cond = object()
        result = self.module.DonutKrea2FusionControl().apply(
            model=FakeModel(self.factors), conditioning_in_1=cond,
            compatibility_preset="Rebalance",
            uncensorfix_controls="Fusion + UncensorFix weights",
            nag_text_energy_compensation=True, nag_batch_txtfusion=True)
        sent = self.module.base.calls[-1]
        self.assertTrue(sent["nag_text_energy_compensation"])
        self.assertTrue(sent["nag_batch_txtfusion"])
        self.assertIn("uncensorfix_targets=33", result[-1])

    def test_default_apply_sends_experiment_flags_as_false(self):
        self.module.DonutKrea2FusionControl().apply(
            model=FakeModel(self.factors), conditioning_in_1=object(),
            compatibility_preset="Rebalance",
            uncensorfix_controls="Fusion + UncensorFix weights")
        sent = self.module.base.calls[-1]
        self.assertIs(sent["nag_text_energy_compensation"], False)
        self.assertIs(sent["nag_batch_txtfusion"], False)

    def test_lora_only_mode_reports_inactive_experiment_flags(self):
        result = self.module.DonutKrea2FusionControl().apply(
            model=FakeModel(self.factors), conditioning_in_1=object(),
            compatibility_preset="UncensorFix",
            uncensorfix_controls="LoRA only",
            nag_text_energy_compensation=True)
        self.assertIn("nag_experiment_flags_inactive=no fusion budget in LoRA-only mode", result[-1])

    def test_missing_model_target_rejected_before_clone(self):
        model = FakeModel(self.factors)
        model.state.pop(next(iter(model.state)))
        with mock.patch.object(model, "clone", side_effect=AssertionError("clone too soon")):
            with self.assertRaisesRegex(RuntimeError, "missing or wrong shape"):
                self.module._apply_uncensorfix(model, 1.0)

    def test_wrong_model_shape_rejected(self):
        model = FakeModel(self.factors)
        model.state[next(iter(model.state))] = types.SimpleNamespace(shape=(2, 3))
        with self.assertRaisesRegex(RuntimeError, "wrong shape"):
            self.module._apply_uncensorfix(model, 1.0)

    def test_incomplete_embedded_target_set_rejected(self):
        with mock.patch.object(self.module, "_uncensorfix_factors", return_value=self.factors[:-1]):
            with self.assertRaisesRegex(RuntimeError, "exactly 33"):
                self.module._apply_uncensorfix(FakeModel(self.factors), 1.0)

    def test_partial_patch_acceptance_rejected(self):
        with mock.patch.object(FakeModel, "add_patches", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "0/33"):
                self.module._apply_uncensorfix(FakeModel(self.factors), 1.0)

    def test_same_count_wrong_accepted_keys_rejected(self):
        with mock.patch.object(FakeModel, "add_patches", return_value=[f"wrong.{i}" for i in range(33)]):
            with self.assertRaisesRegex(RuntimeError, "could not patch every target"):
                self.module._apply_uncensorfix(FakeModel(self.factors), 1.0)

    def test_off_returns_all_inputs_by_identity_in_both_modes(self):
        model = object()  # No cloning, state_dict or patch APIs required.
        routes = ([[object(), {"nested": [1, 2]}]], [], None, [[object(), {}]])
        for mode in ("Simple", "Advanced"):
            with self.subTest(mode=mode):
                result = self.module.DonutKrea2FusionControl().apply(
                    model=model, compatibility_preset="Off", ui_mode=mode,
                    **{f"conditioning_in_{i}": value for i, value in enumerate(routes, 1)})
                self.assertEqual(len(result), 6)
                for actual, expected in zip(result[:-1], (model, *routes)):
                    self.assertIs(actual, expected)
                self.assertIn("preset_label=Off; preset_is_ui_only=false", result[-1])
                self.assertIn("fusion_control=off", result[-1])
                self.assertIn("conditioning_routes=3/4", result[-1])
                self.assertIn("uncensorfix_targets=0", result[-1])

    def test_off_does_not_run_hidden_controls_or_decode_embedded_weights(self):
        with (
            mock.patch.object(self.module.base.DonutKrea2FusionControl, "apply",
                              side_effect=AssertionError("base fusion executed")),
            mock.patch.object(self.module, "_uncensorfix_factors",
                              side_effect=AssertionError("embedded data decoded")),
            mock.patch.object(self.module, "_apply_uncensorfix",
                              side_effect=AssertionError("embedded patches applied")),
        ):
            for composition in (
                self.module.FUSION_ONLY,
                self.module.FUSION_WITH_LORA,
                self.module.FUSION_WITH_WEIGHTS,
                self.module.UNCENSORFIX_LORA_ONLY,
                self.module.UNCENSORFIX_WITH_CONTROLS,
            ):
                with self.subTest(composition=composition):
                    model, conditioning = object(), object()
                    result = self.module.DonutKrea2FusionControl().apply(
                        model=model, conditioning_in_1=conditioning,
                        compatibility_preset="Off", uncensorfix_controls=composition,
                        tap_method=BASE_NAMES["PRESET_REBALANCE"], tap_profile="custom",
                        per_layer_weights="invalid dormant profile", tap_strength=3.,
                        projector_method="Krea2FilterBypass 3vector diff", projector_strength=2.,
                        fusion_method="capitan01R Krea2T-Enhancer operation", fusion_strength=2.)
                    self.assertIs(result[0], model)
                    self.assertIs(result[1], conditioning)
                    self.assertEqual(result[2:5], (None, None, None))

    def test_off_is_safe_without_any_optional_conditioning(self):
        result = self.module.DonutKrea2FusionControl().apply(
            model=object(), conditioning_in_1=object(), compatibility_preset="Off")
        self.assertEqual(result[2:5], (None, None, None))
        self.assertIn("conditioning_routes=1/4", result[-1])

    def test_off_preserves_upstream_model_state_without_mutation(self):
        model = FakeModel(())
        model.patches = {"existing": [object()]}
        model.injections = {"runtime": [object()]}
        model.additional_models = {"source": [object()]}
        model.attachments = {"keep": object()}
        model.model_options = {"transformer_options": {"upstream": object()}}
        before = {key: value for key, value in vars(model).items()}
        with mock.patch.object(model, "clone", side_effect=AssertionError("cloned")):
            result = self.module.DonutKrea2FusionControl().apply(
                model=model, conditioning_in_1=object(), compatibility_preset="Off")
        self.assertIs(result[0], model)
        for key, value in before.items():
            self.assertIs(getattr(model, key), value)

    def test_off_supports_legacy_positional_routes_without_running_base(self):
        def base_apply(self, model, conditioning_in_1, ignored_control,
                       conditioning_in_2=None, conditioning_in_3=None,
                       conditioning_in_4=None, compatibility_preset="Custom settings"):
            raise AssertionError("base called")
        inputs = (object(), object(), "unused", object(), None, object())
        with mock.patch.object(self.module.base.DonutKrea2FusionControl, "apply", base_apply):
            result = self.module.DonutKrea2FusionControl().apply(
                *inputs, compatibility_preset="Off")
        for actual, expected in zip(result[:-1], (inputs[0], inputs[1], *inputs[3:])):
            self.assertIs(actual, expected)

    def test_switching_to_off_does_not_mutate_prior_uncensorfix_output(self):
        model, cond = FakeModel(self.factors), object()
        node = self.module.DonutKrea2FusionControl()
        enabled = node.apply(model=model, conditioning_in_1=cond,
                             compatibility_preset="UncensorFix", tap_strength=.75)
        disabled = node.apply(model=model, conditioning_in_1=cond,
                              compatibility_preset="Off", tap_strength=.75)
        self.assertIs(disabled[0], model)
        self.assertFalse(model.patches)
        self.assertEqual(len(enabled[0].patches), 33)

    def test_invalid_ui_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "UI mode"):
            self.module.DonutKrea2FusionControl().apply(ui_mode="wrong")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

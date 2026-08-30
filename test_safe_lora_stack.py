import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PACKAGE = "donut_safe_lora_test_package"


class _FakeTensor:
    def __init__(self, value, dtype=None):
        self.array = np.asarray(value, dtype=dtype or np.float32)
        self.device = "cpu"
        self.dtype = self.array.dtype

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    def is_floating_point(self):
        return np.issubdtype(self.array.dtype, np.floating)

    def reshape(self, *shape):
        return _FakeTensor(self.array.reshape(*shape), dtype=self.dtype)

    def __mul__(self, other):
        value = other.array if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.array * value, dtype=self.dtype)

    def __getitem__(self, item):
        return _FakeTensor(self.array[item], dtype=self.dtype)


fake_torch = types.ModuleType("torch")
fake_torch.Tensor = _FakeTensor
fake_torch.tensor = lambda value, device=None, dtype=None: _FakeTensor(value, dtype=dtype)
fake_torch.ones = lambda *shape: _FakeTensor(np.ones(shape, dtype=np.float32))
fake_torch.is_tensor = lambda value: isinstance(value, _FakeTensor)
fake_torch.equal = lambda left, right: np.array_equal(left.array, right.array)
fake_torch.allclose = lambda left, right: np.allclose(left.array, right.array)
setattr(fake_torch, "nn", types.SimpleNamespace(Linear=type("Linear", (), {})))


def _load_module():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]

    comfy = types.ModuleType("comfy")
    comfy_sd = types.ModuleType("comfy.sd")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_weight_adapter = types.ModuleType("comfy.weight_adapter")
    class WeightAdapterBase:
        def h(self, x, base_out):
            return 0

        def g(self, value):
            return value

        def bypass_forward(self, org_forward, x, *args, **kwargs):
            base_out = org_forward(x, *args, **kwargs)
            return self.g(base_out + self.h(x, base_out))

    class LoRAAdapter(WeightAdapterBase):
        def __init__(self, weights):
            self.loaded_keys = set()
            self.weights = weights

    class LoKrAdapter(WeightAdapterBase):
        def __init__(self, weights):
            self.loaded_keys = set()
            self.weights = weights

    setattr(comfy_weight_adapter, "WeightAdapterBase", WeightAdapterBase)
    setattr(comfy_weight_adapter, "LoRAAdapter", LoRAAdapter)
    setattr(comfy_weight_adapter, "LoKrAdapter", LoKrAdapter)
    setattr(comfy_weight_adapter, "BypassInjectionManager", object)
    comfy.sd = comfy_sd
    comfy.utils = comfy_utils
    comfy.weight_adapter = comfy_weight_adapter

    folder_paths = types.ModuleType("folder_paths")
    donut_lora_nodes = types.ModuleType(f"{PACKAGE}.donut_lora_nodes")
    donut_lora_nodes._TEXT_MERGE_VECTOR = ",".join(["1"] * 60)
    donut_lora_nodes._lora_has_real_text_encoder = lambda lora: False
    donut_lora_nodes._split_fused_text = lambda lora: (lora, {})

    lora_block_weight = types.ModuleType(f"{PACKAGE}.lora_block_weight")
    lora_block_weight.LoraLoaderBlockWeight = object

    injected = {
        PACKAGE: package,
        "comfy": comfy,
        "comfy.sd": comfy_sd,
        "comfy.utils": comfy_utils,
        "comfy.weight_adapter": comfy_weight_adapter,
        "folder_paths": folder_paths,
        "torch": fake_torch,
        f"{PACKAGE}.donut_lora_nodes": donut_lora_nodes,
        f"{PACKAGE}.lora_block_weight": lora_block_weight,
    }
    previous = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    try:
        spec = importlib.util.spec_from_file_location(
            f"{PACKAGE}.DonutSafeApplyLoRAStack",
            ROOT / "DonutSafeApplyLoRAStack.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


module = _load_module()


def _projector_entry(weight=1.0):
    return {
        "cw": weight,
        "fused_text": True,
        "is_krea_text": True,
        "projector_text": {"diffusion_model.txtfusion.projector.diff": fake_torch.ones(1, 12)},
        "has_projector_text": True,
        "has_other_text": False,
    }


class FusionMetadataTests(unittest.TestCase):
    def test_reads_valid_fusion_metadata_from_model_options(self):
        model = types.SimpleNamespace(model_options={
            "transformer_options": {
                module._FUSION_BUDGET_KEY: {
                    "version": 1,
                    "projector_gains": [1] * 12,
                    "projector_normalization": "none",
                }
            }
        })

        metadata = module._read_fusion_budget(model)
        self.assertEqual(metadata["projector_gains"], (1.0,) * 12)

    def test_rejects_wrong_sized_projector_profile(self):
        model = types.SimpleNamespace(model_options={
            "transformer_options": {
                module._FUSION_BUDGET_KEY: {"version": 1, "projector_gains": [1] * 11}
            }
        })
        self.assertIsNone(module._read_fusion_budget(model))


class ProjectorBudgetTests(unittest.TestCase):
    def test_attenuation_accounts_for_each_projector_gain(self):
        gains = (2.0, 0.5) + (1.0,) * 10
        scales, report = module._projector_column_scales(
            [_projector_entry(), _projector_entry()],
            gains,
            "Attenuate only",
            max_boost=2.0,
        )

        self.assertAlmostEqual(scales[0], 1.0 / (2.0 * math.sqrt(2.0)))
        self.assertEqual(scales[1], 1.0)
        self.assertAlmostEqual(scales[2], 1.0 / math.sqrt(2.0))
        self.assertTrue(report["limited"])
        self.assertFalse(report["boosted"])

    def test_headroom_boosts_quiet_columns_up_to_budget(self):
        gains = (2.0, 0.5) + (1.0,) * 10
        scales, report = module._projector_column_scales(
            [_projector_entry(), _projector_entry()],
            gains,
            "Use headroom",
            max_boost=2.0,
        )

        self.assertAlmostEqual(scales[1], math.sqrt(2.0))
        self.assertTrue(report["boosted"])

    def test_dynamic_tensor_rms_metadata_disables_headroom_boosts(self):
        scales, report = module._projector_column_scales(
            [_projector_entry(weight=0.25)],
            (0.5,) * 12,
            "Use headroom",
            max_boost=4.0,
            dynamic=True,
        )
        self.assertEqual(scales, (1.0,) * 12)
        self.assertTrue(report["dynamic"])
        self.assertFalse(report["boosted"])

    def test_tensor_rms_profile_is_nominally_rms_normalized(self):
        gains, dynamic = module._nominal_projector_gains({
            "projector_gains": (1.0,) * 10 + (2.0, 2.0),
            "projector_normalization": "tensor_rms",
        })
        rms = math.sqrt(sum(value * value for value in gains) / len(gains))
        self.assertAlmostEqual(rms, 1.0)
        self.assertTrue(dynamic)


class ProjectorTensorScalingTests(unittest.TestCase):
    def test_standard_lora_down_matrix_is_scaled_by_input_column(self):
        down_key = "diffusion_model.txtfusion.projector.lora_down.weight"
        up_key = "diffusion_model.txtfusion.projector.lora_up.weight"
        original = {
            down_key: fake_torch.ones(2, 12),
            up_key: fake_torch.ones(1, 2),
        }
        scales = tuple(float(index + 1) for index in range(12))

        adjusted, transformed = module._scale_projector_lora_columns(original, scales)

        self.assertTrue(transformed)
        self.assertTrue(fake_torch.equal(adjusted[down_key][0], fake_torch.tensor(scales)))
        self.assertTrue(fake_torch.equal(adjusted[up_key], original[up_key]))
        self.assertTrue(fake_torch.equal(original[down_key], fake_torch.ones(2, 12)))

    def test_direct_projector_diff_is_scaled_by_column(self):
        key = "diffusion_model.txtfusion.projector.diff"
        scales = tuple(0.1 * (index + 1) for index in range(12))
        adjusted, transformed = module._scale_projector_lora_columns(
            {key: fake_torch.ones(1, 12)},
            scales,
        )
        self.assertTrue(transformed)
        self.assertTrue(fake_torch.allclose(adjusted[key][0], fake_torch.tensor(scales)))


class SafetyCompatibilityTests(unittest.TestCase):
    def test_overlong_numeric_vector_keeps_ignored_tail(self):
        vector = ",".join(str(index + 1) for index in range(30))
        entries = [{
            "name": "overlong.safetensors",
            "mw": 0.0,
            "vector": vector,
            "is_krea": True,
            "krea_blocks": set(range(28)),
        }]

        adjusted, limited = module._normalise_krea_vectors(entries)

        self.assertEqual(adjusted[0], vector)
        self.assertEqual(limited, [])

    def test_projector_only_krea_lora_participates_in_text_budget(self):
        weights, limited = module._normalise_fused_text_weights(
            [_projector_entry(), _projector_entry()]
        )
        expected = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(weights[0], expected)
        self.assertAlmostEqual(weights[1], expected)
        self.assertIsNotNone(limited)

    def test_new_controls_default_to_legacy_off(self):
        inputs = module.DonutApplyLoRAStackSafe.INPUT_TYPES()
        required = inputs["required"]
        self.assertEqual(required["fusion_aware"][1]["default"], "Off")
        self.assertEqual(required["max_fusion_boost"][1]["default"], 2.0)
        self.assertEqual(inputs["optional"]["execution_mode"][1]["default"], "Comfy patches")

    def test_composite_bypass_sums_overlapping_additive_adapters(self):
        class Adapter(module.comfy_weight_adapter.WeightAdapterBase):
            def __init__(self, value):
                self.value = value
                self.loaded_keys = set()
                self.weights = ()

            def h(self, x, base_out):
                return self.value * self.multiplier

        composite = module._CompositeBypassAdapter([
            (Adapter(2.0), 0.5),
            (Adapter(3.0), 2.0),
        ])
        composite.multiplier = 1.0

        self.assertEqual(composite.h(None, None), 7.0)

    def test_composite_bypass_preserves_each_adapter_weight_layout(self):
        class Adapter(module.comfy_weight_adapter.WeightAdapterBase):
            def __init__(self, weights):
                self.loaded_keys = set()
                self.weights = weights

        first = Adapter((1, 2))
        second = Adapter([3, 4, 5])
        composite = module._CompositeBypassAdapter([(first, 1.0), (second, 1.0)])

        self.assertEqual(composite.weights, (1, 2, 3, 4, 5))
        composite.weights = (10, 20, 30, 40, 50)
        self.assertEqual(first.weights, (10, 20))
        self.assertEqual(second.weights, [30, 40, 50])

    def test_partition_keeps_mixed_target_wholly_on_regular_path(self):
        linear = fake_torch.nn.Linear()
        root = types.SimpleNamespace(layer=linear, other=fake_torch.nn.Linear())
        lora = module.comfy_weight_adapter.LoRAAdapter((1, 2, 1.0, None, None, None))
        lokr = module.comfy_weight_adapter.LoKrAdapter(
            (1, 2, 1.0, None, None, None, None, None, None)
        )
        direct = ("diff", (3,))

        bypass, regular, reasons = module._partition_bypass_targets(
            root,
            {"layer.weight", "other.weight"},
            {
                "layer.weight": [(lora, 0.5), (direct, 1.0)],
                "other.weight": [(lokr, 2.0)],
            },
        )

        self.assertEqual(bypass, {"other.weight": [(lokr, 2.0)]})
        self.assertEqual(regular, {"layer.weight": [(lora, 0.5), (direct, 1.0)]})
        self.assertIn("regular patch type tuple", reasons["layer.weight"])

    def test_bypass_allowlist_rejects_dora_reshape_and_unknown_adapters(self):
        linear = fake_torch.nn.Linear()
        dora = module.comfy_weight_adapter.LoRAAdapter((1, 2, 1.0, None, 3, None))
        reshape = module.comfy_weight_adapter.LoRAAdapter((1, 2, 1.0, None, None, (2, 2)))

        class Unknown(module.comfy_weight_adapter.WeightAdapterBase):
            weights = ()

        self.assertIn("DoRA", module._bypass_compatibility_error(dora, linear))
        self.assertIn("reshape_weight", module._bypass_compatibility_error(reshape, linear))
        self.assertIn("unsupported adapter type", module._bypass_compatibility_error(Unknown(), linear))

    def test_bypass_accepts_comfy_mixed_precision_linear_without_input_scale(self):
        comfy_linear_type = type(
            "Linear",
            (),
            {
                "__module__": "comfy.ops",
                "_forward": lambda self, input_value, weight, bias: input_value,
            },
        )
        linear = comfy_linear_type()
        lora = module.comfy_weight_adapter.LoRAAdapter((1, 2, 1.0, None, None, None))

        self.assertIsNone(module._bypass_compatibility_error(lora, linear))
        setattr(linear, "pre_quant_scale", object())
        self.assertIn(
            "pre-quantization scale",
            module._bypass_compatibility_error(lora, linear),
        )

    def test_bypass_uses_regular_path_with_existing_runtime_injections(self):
        cloned = types.SimpleNamespace(
            model=types.SimpleNamespace(state_dict=lambda: {}),
            added=[],
        )
        cloned.add_patches = lambda patches, strength: cloned.added.append((patches, strength))
        model = types.SimpleNamespace(
            injections={"existing": [object()]},
            is_injected=False,
            clone=lambda: cloned,
        )
        had_load_lbw = hasattr(module.LoraLoaderBlockWeight, "load_lbw")
        original = getattr(module.LoraLoaderBlockWeight, "load_lbw", None)
        setattr(module.LoraLoaderBlockWeight, "load_lbw", lambda *args, **kwargs: (
            {"layer.weight": (("diff", (1,)), 0.5)},
            None,
            None,
        ))
        try:
            result = module._apply_bypass_applications(model, [(object(), 2.0, "1")])
        finally:
            if had_load_lbw:
                setattr(module.LoraLoaderBlockWeight, "load_lbw", original)
            else:
                delattr(module.LoraLoaderBlockWeight, "load_lbw")

        self.assertIs(result, cloned)
        self.assertEqual(cloned.added, [({"layer.weight": ("diff", (1,))}, 1.0)])

    def test_apply_splits_and_column_scales_projector_lora(self):
        down_key = "diffusion_model.txtfusion.projector.lora_down.weight"
        up_key = "diffusion_model.txtfusion.projector.lora_up.weight"
        loaded_lora = {
            down_key: fake_torch.ones(2, 12),
            up_key: fake_torch.ones(1, 2),
        }
        calls = []

        class Loader:
            def load_lora_for_models(self, model, clip, lora, **kwargs):
                calls.append((lora, kwargs["strength_model"]))
                return model, clip, kwargs["block_vector"]

        module.LoraLoaderBlockWeight = Loader
        module.folder_paths.get_full_path = lambda category, name: f"/{name}"
        module.comfy.utils.load_torch_file = lambda path, safe_load=True: loaded_lora
        module._lora_has_real_text_encoder = lambda lora: False
        module._split_fused_text = lambda lora: ({}, dict(lora))

        model = types.SimpleNamespace(model_options={
            "transformer_options": {
                module._FUSION_BUDGET_KEY: {
                    "version": 1,
                    "projector_gains": (2.0, 0.5) + (1.0,) * 10,
                    "projector_normalization": "none",
                }
            }
        })

        module.DonutApplyLoRAStackSafe().apply_stack(
            model,
            object(),
            [("projector.safetensors", 0.0, 1.0, "")],
            safe_stack="On",
            fusion_aware="Use headroom",
            max_fusion_boost=2.0,
        )

        self.assertEqual(len(calls), 1)
        applied_lora, applied_strength = calls[0]
        self.assertEqual(applied_strength, 1.0)
        expected = (0.5, 2.0) + (1.0,) * 10
        self.assertTrue(fake_torch.allclose(applied_lora[down_key][0], fake_torch.tensor(expected)))


if __name__ == "__main__":
    unittest.main()

"""CPU regression contracts for FP8 model2 merges, in both input orders.

Uses real torch layers, FP8 storage tensors, and the shipped Donut merge code.
Comfy patcher, scaled-weight layout, and injection-manager interfaces are fixtures,
not a full ComfyUI/GPU run. The failing accessor/patch-list contracts follow
ComfyUI 3216c62e (comfy/model_patcher.py get_key_weight/get_key_patches).
No model downloads or network access are needed.
"""
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "diffusion_model."
WEIGHT = PREFIX + "blocks.0.weight"
BIAS = PREFIX + "blocks.0.bias"


def get_attr(root, path):
    for part in path.split("."):
        root = getattr(root, part)
    return root


def get_key_weight(root, key):
    """Strict live-attribute accessor, including the core conversion callback."""
    owner_path, _, name = key.rpartition(".")
    owner = get_attr(root, owner_path) if owner_path else root
    setter = getattr(owner, "set_" + name, None)
    converter = getattr(owner, "convert_" + name, None)
    weight = getattr(owner, name)  # Reproduces the reported AttributeError.
    if converter is not None:
        weight = get_attr(root, key)
    return weight, setter, converter


class QuantizedTensor:
    """Small layout fixture: scales are internal, state_dict exports sidecars.

    Packed values deliberately differ from dequantized values. Tests therefore
    fail if merge uses serialized storage rather than the live weight converter.
    """
    def __init__(self, packed, scale=2.0, scale2=None, dtype=torch.float8_e4m3fn):
        self.packed = torch.as_tensor(packed, dtype=torch.float32).to(dtype)
        self.scale = torch.as_tensor(scale, dtype=torch.float32)
        self.scale2 = None if scale2 is None else torch.tensor(scale2)
        self.dequantizations = 0
        self.exports = 0
        self.shape = self.packed.shape

    def state_dict(self, prefix):
        self.exports += 1
        values = {prefix: self.packed, prefix + "_scale": self.scale}
        if self.scale2 is not None:
            values[prefix + "_scale_2"] = self.scale2
        return values

    def dequantize(self):
        self.dequantizations += 1
        value = self.packed.float() * self.scale
        return value if self.scale2 is None else value * self.scale2


class Linear(torch.nn.Module):
    """MixedPrecisionOps-like module, not an nn.Linear subclass."""
    __module__ = "comfy.ops"

    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = torch.nn.Parameter(torch.as_tensor(bias, dtype=torch.float32), requires_grad=False)
        self.out_features, self.in_features = weight.shape

    def state_dict(self, *args, destination=None, prefix="", **kwargs):
        sd = destination if destination is not None else {}
        sd[prefix + "bias"] = self.bias
        sd.update(self.weight.state_dict(prefix + "weight"))
        sd[prefix + "comfy_quant"] = torch.tensor([1], dtype=torch.uint8)
        for key in ("input_scale", "pre_quant_scale"):
            if hasattr(self, key):
                sd[prefix + key] = getattr(self, key)
        return sd

    def convert_weight(self, weight, **kwargs):
        return weight.dequantize() if isinstance(weight, QuantizedTensor) else weight

    def _forward(self, x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias)

    def forward(self, x):
        return self._forward(x, self.convert_weight(self.weight), self.bias)


def model(quantized=False, value=2.0, scale2=None):
    root = torch.nn.Module()
    root.diffusion_model = torch.nn.Module()
    weights = [[value, value + 1], [value + 2, value + 3]]
    bias = [value / 2, -value / 2]
    if quantized:
        layer = Linear(QuantizedTensor(weights, [[2.0], [3.0]], scale2), bias)
    else:
        layer = torch.nn.Linear(2, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(weights))
            layer.bias.copy_(torch.tensor(bias))
    root.diffusion_model.blocks = torch.nn.ModuleList([layer])
    return Patcher(root)


class Patcher:
    def __init__(self, root):
        self.model = root
        self.patches, self.backup, self.hook_backup = {}, {}, {}
        self.injections, self.additional_models, self.attachments = {}, {}, {}
        self.load_device = torch.device("cpu")
        self.is_injected = False
        self.patches_uuid = None

    def clone(self):
        other = copy(self)
        for name in ("patches", "backup", "hook_backup", "injections", "additional_models", "attachments"):
            setattr(other, name, {k: v.copy() if isinstance(v, list) else v for k, v in getattr(self, name).items()})
        return other

    def model_state_dict(self):
        return self.model.state_dict()

    def get_key_patches(self, filter_prefix=None):
        result = {}
        for key in self.model_state_dict():
            if filter_prefix is not None and not key.startswith(filter_prefix):
                continue
            weight, _, convert = get_key_weight(self.model, key)
            if key in self.backup:
                weight = self.backup[key].weight
            if key in self.hook_backup:
                weight = self.hook_backup[key][0]
            result[key] = [(weight, convert or (lambda w, **kwargs: w))] + self.patches.get(key, [])
        return result

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        keys = self.model_state_dict()
        accepted = []
        for key, value in patches.items():
            if key in keys:
                self.patches.setdefault(key, []).append((strength_patch, value, strength_model, None, None))
                accepted.append(key)
        return accepted

    def set_injections(self, key, value):
        self.injections[key] = value

    def set_additional_models(self, key, value):
        self.additional_models[key] = value

    def get_additional_models_with_key(self, key):
        return self.additional_models.get(key, [])

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def is_clone(self, other):
        return self.model is other.model

    def use_ejected(self):
        return nullcontext()


@dataclass
class Injection:
    inject: object
    eject: object


class BypassManager:
    def __init__(self):
        self.adapters = []

    def add_adapter(self, key, adapter, strength=1.0):
        self.adapters.append((key, adapter))

    def get_hook_count(self):
        return len(self.adapters)

    def create_injections(self, root):
        result = []
        for key, adapter in self.adapters:
            layer = get_attr(root, key[:-7])
            original = layer.forward
            result.append(Injection(
                lambda _p, m=layer, a=adapter, f=original: setattr(m, "forward", lambda *args, **kw: a.bypass_forward(f, *args, **kw)),
                lambda _p, m=layer, f=original: setattr(m, "forward", f),
            ))
        return result


def evaluate_patch_list(items):
    weight, convert = items[0]
    # Core materializes ordinary FP8 in the merge compute dtype as well.
    value = convert(weight).float().clone()
    for sp, payload, sm, offset, fn in items[1:]:
        assert offset is None and fn is None
        source = evaluate_patch_list(payload) if isinstance(payload, list) else payload[1][0]
        value = value * sm + source * sp
    return value


def effective_weight(patcher, key):
    value, _, convert = get_key_weight(patcher.model, key)
    return evaluate_patch_list([(value, convert or (lambda w: w))] + patcher.patches.get(key, []))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FP8MergeTests(unittest.TestCase):
    def setUp(self):
        comfy = types.ModuleType("comfy"); comfy.__path__ = []
        adapters = types.ModuleType("comfy.weight_adapter")
        adapters.WeightAdapterBase = type("WeightAdapterBase", (), {})
        adapters.BypassInjectionManager = BypassManager
        injections = types.ModuleType("comfy.patcher_extension"); injections.PatcherInjection = Injection
        quant = types.ModuleType("comfy.quant_ops"); quant.QuantizedTensor = QuantizedTensor
        patcher = types.ModuleType("comfy.model_patcher"); patcher.get_key_weight = get_key_weight
        package = types.ModuleType("_donut_quant_merge_test"); package.__path__ = [str(ROOT)]
        modules = {"comfy": comfy, "comfy.weight_adapter": adapters,
                   "comfy.patcher_extension": injections, "comfy.quant_ops": quant,
                   "comfy.model_patcher": patcher, package.__name__: package}
        self.scope = patch.dict(sys.modules, modules); self.scope.start(); self.addCleanup(self.scope.stop)
        self.merge = load_module(package.__name__ + ".DonutModelMergeKrea2", ROOT / "DonutModelMergeKrea2.py")
        self.grouped = load_module(package.__name__ + ".donut_grouped_merge", ROOT / "donut_grouped_merge.py")

    def assertTensor(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_original_accessor_reproduces_exact_error_after_input_swap(self):
        a, b = model(True), model(False)
        a.clone()  # Quantized model as model1 does not require source enumeration.
        self.assertIn(WEIGHT, b.get_key_patches(PREFIX))
        with self.assertRaisesRegex(AttributeError, "'Linear' object has no attribute 'weight_scale'"):
            a.get_key_patches(PREFIX)

    def test_export_metadata_is_excluded_but_live_wrapper_and_converter_survive(self):
        b = model(True, scale2=0.5); layer = b.model.diffusion_model.blocks[0]
        layer.register_buffer("input_scale", torch.tensor(0.25))
        layer.register_buffer("pre_quant_scale", torch.ones(2))
        result = self.merge._get_merge_key_patches(b)
        self.assertEqual(set(result), {WEIGHT, BIAS, PREFIX + "blocks.0.input_scale", PREFIX + "blocks.0.pre_quant_scale"})
        self.assertIs(result[WEIGHT][0][0], layer.weight)
        self.assertEqual(layer.weight.dequantizations, 0)
        self.assertTensor(evaluate_patch_list(result[WEIGHT]), layer.weight.packed.float() * layer.weight.scale * 0.5)
        self.assertFalse(hasattr(layer, "weight_scale"))

    def test_regular_numeric_ratios_in_both_input_orders(self):
        for reversed_inputs in (False, True):
            for ratio in (0.0, 0.25, 0.5, 1.0):
                with self.subTest(reversed=reversed_inputs, ratio=ratio):
                    a, b = model(False, 2), model(True, 5)
                    if reversed_inputs: a, b = b, a
                    expected = effective_weight(a, WEIGHT) * ratio + effective_weight(b, WEIGHT) * (1-ratio)
                    out, = self.merge.DonutModelMergeKrea2().merge(a, b, **{"first.": ratio})
                    self.assertTensor(effective_weight(out, WEIGHT), expected)
                    self.assertEqual(a.patches, {}); self.assertEqual(b.patches, {})

    def test_hard_swaps_forward_to_source_in_both_orders_and_eject_cleanly(self):
        x = torch.tensor([[0.5, -2.0]])
        for reversed_inputs in (False, True):
            with self.subTest(reversed=reversed_inputs):
                a, b = model(False, 2), model(True, 5)
                if reversed_inputs: a, b = b, a
                layer = a.model.diffusion_model.blocks[0]
                before, expected = layer(x).detach(), b.model.diffusion_model.blocks[0](x).detach()
                out, = self.merge.DonutModelMergeKrea2().merge(a, b, "Experimental bypass", **{"first.": 0.0})
                self.assertEqual(out.patches, {})
                injection, = out.injections[self.merge._INJECTION_KEY]
                injection.inject(out)
                try: self.assertTensor(layer(x), expected)
                finally: injection.eject(out)
                self.assertTensor(layer(x), before)
                self.assertFalse(hasattr(layer, "weight_scale"))

    def test_hybrid_partial_blend_uses_regular_patches_not_two_forwards(self):
        a, b = model(False, 2), model(True, 5)
        expected = effective_weight(a, WEIGHT)*0.4 + effective_weight(b, WEIGHT)*0.6
        out, = self.merge.DonutModelMergeKrea2().merge(a, b, "Experimental bypass", **{"first.": 0.4})
        self.assertEqual(out.injections, {})
        self.assertTensor(effective_weight(out, WEIGHT), expected)

    def test_hybrid_keep_model1_is_unmodified_with_quantized_model2(self):
        a, b = model(), model(True)
        out, = self.merge.DonutModelMergeKrea2().merge(a, b, "Experimental bypass", **{"first.": 1.0})
        self.assertEqual(out.patches, {}); self.assertEqual(out.injections, {})
        self.assertIs(out.model, a.model)
        self.assertEqual(b.model.diffusion_model.blocks[0].weight.dequantizations, 0)

    def test_hybrid_fallback_for_older_patchers_still_handles_quantized_source(self):
        a, b = model(), model(True)
        with patch.object(self.merge, "_BYPASS_MANAGER", None):
            out, = self.merge.DonutModelMergeKrea2().merge(a, b, "Experimental bypass", **{"first.": 0.5})
        self.assertTensor(effective_weight(out, WEIGHT), (effective_weight(a, WEIGHT)+effective_weight(b, WEIGHT))*0.5)

    def test_source_patches_and_backups_are_preserved_in_correct_order(self):
        b = model(True); layer = b.model.diffusion_model.blocks[0]
        backup = QuantizedTensor(torch.ones(2, 2), 4.0)
        hook = QuantizedTensor(torch.ones(2, 2), 8.0)
        b.backup[WEIGHT] = types.SimpleNamespace(weight=backup)
        b.hook_backup[WEIGHT] = (hook, None)
        diff = (0.5, ("diff", (torch.ones(2, 2),)), 1.0, None, None)
        b.patches[WEIGHT] = [diff]
        result = self.merge._get_merge_key_patches(b)
        self.assertIs(result[WEIGHT][0][0], hook)
        self.assertIs(result[WEIGHT][1], diff)
        self.assertTensor(evaluate_patch_list(result[WEIGHT]), torch.full((2, 2), 8.5))
        del b.hook_backup[WEIGHT]
        self.assertIs(self.merge._get_merge_key_patches(b)[WEIGHT][0][0], backup)
        self.assertIs(b.model.diffusion_model.blocks[0], layer)
        self.assertEqual(b.patches[WEIGHT], [diff])

    def test_grouped_body_and_fusion_ratios_keep_their_input_orientation(self):
        for mode in ("Comfy patches", "Experimental bypass"):
            for reversed_inputs in (False, True):
                with self.subTest(mode=mode, reversed=reversed_inputs):
                    a, b = model(), model(True, 5)
                    for p in (a, b):
                        p.model.diffusion_model.txtfusion = torch.nn.Module()
                        p.model.diffusion_model.txtfusion.projector = torch.nn.Linear(2, 2)
                    if reversed_inputs: a, b = b, a
                    fusion_key = PREFIX + "txtfusion.projector.weight"
                    out, = self.grouped.DonutModelMergeKrea2Grouped().merge_grouped(
                        model1=a, model2=b, ratio_mode="Grouped", body_ratio=0.25,
                        fusion_ratio=0.75, execution_mode=mode)
                    self.assertTensor(effective_weight(out, WEIGHT), effective_weight(a, WEIGHT)*0.25+effective_weight(b, WEIGHT)*0.75)
                    self.assertTensor(effective_weight(out, fusion_key), effective_weight(a, fusion_key)*0.75+effective_weight(b, fusion_key)*0.25)

    def test_existing_core_result_is_returned_without_inspection(self):
        marker = object()
        opaque = types.SimpleNamespace(get_key_patches=lambda prefix: marker)
        self.assertIs(self.merge._get_merge_key_patches(opaque), marker)

    def test_unknown_missing_attribute_is_not_silently_dropped(self):
        b = model(True)
        original = b.model_state_dict
        b.model_state_dict = lambda: {**original(), PREFIX + "blocks.0.missing_tensor": torch.tensor(1.0)}
        with self.assertRaisesRegex(AttributeError, "missing_tensor"):
            self.merge._get_merge_key_patches(b)

    def test_same_suffix_on_nonquantized_layer_is_not_ignored(self):
        b = model(True)
        b.model.diffusion_model.other = torch.nn.Linear(2, 2)
        original = b.model_state_dict
        b.model_state_dict = lambda: {**original(), PREFIX + "other.weight_scale": torch.tensor(1.0)}
        with self.assertRaisesRegex(AttributeError, "weight_scale"):
            self.merge._get_merge_key_patches(b)

    def test_real_scale_buffer_is_preserved_despite_its_name(self):
        b = model(True); layer = b.model.diffusion_model.blocks[0]
        layer.register_buffer("weight_scale", torch.tensor(0.7))
        result = self.merge._get_merge_key_patches(b)
        self.assertIs(result[PREFIX + "blocks.0.weight_scale"][0][0], layer.weight_scale)

    def test_real_failing_property_is_not_mistaken_for_export_metadata(self):
        class FailingScale(Linear):
            @property
            def weight_scale(self):
                raise AttributeError("scale property failure")
        b = model(True); old = b.model.diffusion_model.blocks[0]
        b.model.diffusion_model.blocks[0] = FailingScale(old.weight, old.bias.detach())
        with self.assertRaisesRegex(AttributeError, "weight_scale"):
            self.merge._get_merge_key_patches(b)

    def test_explicit_patch_or_backup_on_export_only_key_fails(self):
        for name in ("patches", "backup", "hook_backup"):
            with self.subTest(storage=name):
                b = model(True)
                getattr(b, name)[PREFIX + "blocks.0.weight_scale"] = object()
                with self.assertRaisesRegex(RuntimeError, "export-only quantization metadata"):
                    self.merge._get_merge_key_patches(b)

    def test_no_global_core_monkeypatch_or_added_scale_attributes(self):
        b = model(True); original_method = Patcher.get_key_patches
        original_accessor = sys.modules["comfy.model_patcher"].get_key_weight
        self.merge._get_merge_key_patches(b)
        self.assertIs(Patcher.get_key_patches, original_method)
        self.assertIs(sys.modules["comfy.model_patcher"].get_key_weight, original_accessor)
        with self.assertRaisesRegex(AttributeError, "weight_scale"):
            b.get_key_patches(PREFIX)

    def test_unrelated_failures_propagate_unchanged(self):
        for scaled in (False, True):
            for error in (AttributeError("unrelated"), ValueError("bad format"), RuntimeError("allocation failed")):
                with self.subTest(error=type(error).__name__, scaled=scaled):
                    b = model(scaled)
                    def fail(_prefix): raise error
                    b.get_key_patches = fail
                    with self.assertRaises(type(error)) as caught:
                        self.merge._get_merge_key_patches(b)
                    self.assertIs(caught.exception, error)

    def test_prefix_filter_does_not_read_unrelated_missing_attributes(self):
        b = model(True); original = b.model_state_dict
        b.model_state_dict = lambda: {**original(), "unrelated.missing": torch.tensor(1.0)}
        self.assertEqual(set(self.merge._get_merge_key_patches(b)), {WEIGHT, BIAS})
        with self.assertRaises(AttributeError):
            self.merge._get_merge_key_patches(b, None)

    def test_source_and_destination_patch_lists_are_not_mutated(self):
        a, b = model(), model(True)
        prior = (1.0, ("diff", (torch.ones(2, 2),)), 1.0, None, None)
        a.patches[WEIGHT] = [prior]; b.patches[WEIGHT] = [prior]
        a_saved, b_saved = a.patches[WEIGHT], b.patches[WEIGHT]
        out, = self.merge.DonutModelMergeKrea2().merge(a, b, **{"first.": 0.5})
        self.assertIs(a.patches[WEIGHT], a_saved); self.assertIs(b.patches[WEIGHT], b_saved)
        self.assertEqual(len(a_saved), 1); self.assertEqual(len(b_saved), 1)
        self.assertEqual(len(out.patches[WEIGHT]), 2)

    def test_bias_is_merged_with_the_same_ratio_as_weight(self):
        a, b = model(False, 2), model(True, 8)
        out, = self.merge.DonutModelMergeKrea2().merge(a, b, **{"first.": 0.3})
        self.assertTensor(effective_weight(out, BIAS), effective_weight(a, BIAS)*0.3+effective_weight(b, BIAS)*0.7)

    def test_unfixed_merge_path_fails_only_when_quantized_model_is_source(self):
        for mode in ("Comfy patches", "Experimental bypass"):
            with self.subTest(mode=mode):
                a, b = model(True), model()
                with patch.object(self.merge, "_get_merge_key_patches", lambda p: p.get_key_patches(PREFIX)):
                    self.merge.DonutModelMergeKrea2().merge(a, b, mode, **{"first.": 0.0})
                    with self.assertRaisesRegex(AttributeError, "'Linear' object has no attribute 'weight_scale'"):
                        self.merge.DonutModelMergeKrea2().merge(b, a, mode, **{"first.": 0.0})

    def test_mixed_hybrid_keeps_hard_swap_partial_and_unchanged_layers_separate(self):
        a, b = model(), model(True)
        for p in (a, b):
            p.model.diffusion_model.blocks.extend([
                model(p is b, 6).model.diffusion_model.blocks[0],
                model(p is b, 9).model.diffusion_model.blocks[0],
            ])
        out, = self.merge.DonutModelMergeKrea2().merge(a, b, "Experimental bypass",
            **{"first.": 1.0, "blocks.0.": 0.0, "blocks.1.": 0.4})
        self.assertEqual(set(out.patches), {PREFIX+"blocks.1.weight", PREFIX+"blocks.1.bias"})
        plans = next(iter(out.attachments.values()))
        self.assertEqual(plans, ((PREFIX+"blocks.0", WEIGHT, 0.0),))
        key = PREFIX+"blocks.1.weight"
        self.assertTensor(effective_weight(out, key), effective_weight(a, key)*0.4+effective_weight(b, key)*0.6)

    def test_two_quantized_models_preserve_nonunit_scales(self):
        a, b = model(True, 2, scale2=0.25), model(True, 6, scale2=0.75)
        out, = self.merge.DonutModelMergeKrea2().merge(a, b, **{"first.": 0.25})
        self.assertTensor(effective_weight(out, WEIGHT), effective_weight(a, WEIGHT)*0.25+effective_weight(b, WEIGHT)*0.75)

    def test_dynamic_live_scale_attribute_is_not_dropped(self):
        class DynamicScale(Linear):
            def __getattr__(self, name):
                if name == "weight_scale": return torch.tensor(0.375)
                return super().__getattr__(name)
        b = model(True); old = b.model.diffusion_model.blocks[0]
        b.model.diffusion_model.blocks[0] = DynamicScale(old.weight, old.bias.detach())
        result = self.merge._get_merge_key_patches(b)
        self.assertTensor(result[PREFIX+"blocks.0.weight_scale"][0][0], torch.tensor(0.375))


    def test_plain_fp8_without_scale_metadata_keeps_native_accessor(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            with self.subTest(dtype=dtype):
                b = model()
                layer = b.model.diffusion_model.blocks[0]
                layer.weight = torch.nn.Parameter(layer.weight.detach().to(dtype), requires_grad=False)
                # Plain FP8 is a live tensor with no exported scale keys.
                self.assertEqual(set(b.model_state_dict()), {WEIGHT, BIAS})
                with patch.object(b, "get_key_patches", wraps=b.get_key_patches) as native:
                    with patch.object(b, "model_state_dict", wraps=b.model_state_dict) as state_dict:
                        values = self.merge._get_merge_key_patches(b)
                self.assertEqual(native.call_count, 1)
                self.assertEqual(state_dict.call_count, 1)  # No fallback scan.
                self.assertIs(values[WEIGHT][0][0], layer.weight)
                self.assertEqual(values[WEIGHT][0][0].dtype, dtype)

    def test_plain_fp8_numeric_ratios_in_either_socket(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for reverse in (False, True):
                for ratio in (0.0, 0.25, 1.0):
                    with self.subTest(dtype=dtype, reverse=reverse, ratio=ratio):
                        a, b = model(False, 8), model(False, 2)
                        layer = b.model.diffusion_model.blocks[0]
                        layer.weight = torch.nn.Parameter(layer.weight.detach().to(dtype), requires_grad=False)
                        if reverse: a, b = b, a
                        out, = self.merge.DonutModelMergeKrea2().merge(a, b, **{"first.": ratio})
                        self.assertTensor(effective_weight(out, WEIGHT),
                            effective_weight(a, WEIGHT)*ratio+effective_weight(b, WEIGHT)*(1-ratio))

    def test_scaled_fp8_e4m3_and_e5m2_preserve_nonunit_scale(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for mode in ("Comfy patches", "Experimental bypass"):
                with self.subTest(dtype=dtype, mode=mode):
                    a, b = model(False, 8), model(True, 2)
                    layer = b.model.diffusion_model.blocks[0]
                    layer.weight = QuantizedTensor([[1.25, 2.5], [4.0, 8.0]], scale=0.375, dtype=dtype)
                    self.assertEqual(layer.weight.packed.dtype, dtype)
                    with self.assertRaisesRegex(AttributeError, "weight_scale"):
                        b.get_key_patches(PREFIX)
                    out, = self.merge.DonutModelMergeKrea2().merge(a, b, mode, **{"first.": 0.25})
                    self.assertTensor(effective_weight(out, WEIGHT),
                        effective_weight(a, WEIGHT)*0.25+layer.weight.dequantize()*0.75)
                    self.assertIs(out.patches[WEIGHT][-1][1][0][0], layer.weight)
                    self.assertEqual(layer.weight.packed.dtype, dtype)

    def test_non_weight_parameters_are_not_lost_to_suffix_filtering(self):
        b = model(True)
        b.model.diffusion_model.register_parameter("modulation", torch.nn.Parameter(torch.tensor([2.0, 3.0])))
        values = self.merge._get_merge_key_patches(b)
        self.assertIn(PREFIX + "modulation", values)
        self.assertIs(values[PREFIX + "modulation"][0][0], b.model.diffusion_model.modulation)


if __name__ == "__main__":
    unittest.main()

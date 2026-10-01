"""Single-LoRA runtime chunking with actual ComfyUI adapters and hooks.

Set COMFYUI_ROOT to a ComfyUI checkout with comfy/weight_adapter support.
ModelPatcher ownership and device selection are explicit doubles; linear,
convolution, adapter arithmetic, injection hooks, and autograd use real PyTorch.
"""
import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import test_lokr_bypass_parity as parity


class Patcher:
    def __init__(self, root, chunk_lora=None):
        self.model = root
        self.model_options = {} if chunk_lora is None else {"donut_chunk_lora": chunk_lora}

    def clone(self, chunk_lora):
        result = copy.copy(self)
        result.model_options = {"donut_chunk_lora": chunk_lora}
        return result


class SingleLoRAMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.LoKrBypassParityTests.setUpClass.__func__(cls)

    def make_plan(self, layer=None, strength=-0.65, alpha=0.8, rank=2):
        if layer is None:
            layer = torch.nn.Linear(8, 15)
        root = torch.nn.Module()
        root.layer = layer
        input_features = layer.in_channels if isinstance(layer, torch.nn.Conv1d) else layer.in_features
        output_features = layer.out_channels if isinstance(layer, torch.nn.Conv1d) else layer.out_features
        adapter = self.native.LoRAAdapter({"test.up", "test.down"}, (
            torch.randn(output_features, rank) * 0.1,
            torch.randn(rank, input_features) * 0.1,
            alpha, None, None, None,
        ))
        manager = self.native.BypassInjectionManager()
        self.safe._register_bypass_adapters(manager, {"layer.weight": [(adapter, strength)]})
        injection, = self.safe._make_rebinding_bypass_injections(manager, root)
        return root, adapter, manager, injection

    def inject(self, injection, owner):
        management = self.native.BypassInjectionManager.create_injections.__globals__["comfy"].model_management
        with patch.object(management, "get_torch_device", return_value=owner.model.layer.weight.device):
            injection.inject(owner)
        self.addCleanup(injection.eject, owner)
        return owner.model.layer.forward.__self__.adapter

    def record_calls(self, calls):
        original = self.native.LoRAAdapter.h
        def recorded(adapter, x, base_out):
            calls.append((tuple(x.shape), None if base_out is None else tuple(base_out.shape)))
            return original(adapter, x, base_out)
        return patch.object(self.native.LoRAAdapter, "h", recorded)

    def test_enabled_single_lora_chunks_and_preserves_scaled_hook_output(self):
        torch.manual_seed(7)
        for strength, alpha in ((-0.65, 0.8), (1.7, None), (0.0, 0.8), (1.0, 0.0)):
            with self.subTest(strength=strength, alpha=alpha):
                root, adapter, _, injection = self.make_plan(strength=strength, alpha=alpha)
                owner = Patcher(root, True)
                x = torch.randn(2, 2051, 16)[..., ::2]
                weights = root.layer.weight.detach().clone()
                expected_weight = adapter.calculate_weight(weights, "layer.weight", strength, 1., None, lambda x: x)
                with torch.inference_mode():
                    expected = F.linear(x, expected_weight, root.layer.bias)
                    runtime = self.inject(injection, owner)
                    calls = []
                    with self.record_calls(calls):
                        actual = root.layer(x)
                self.assertIsInstance(runtime, self.safe._CompositeBypassAdapter)
                self.assertEqual([shape[1] for shape, _ in calls], [1024, 1024, 3])
                self.assertEqual([shape[1] for _, shape in calls], [1024, 1024, 3])
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                injection.eject(owner)

    def test_disabled_and_absent_flag_keep_native_single_adapter(self):
        for flag in (False, None):
            with self.subTest(flag=flag):
                root, _, _, injection = self.make_plan()
                owner = Patcher(root, flag)
                runtime = self.inject(injection, owner)
                self.assertIs(type(runtime), self.native.LoRAAdapter)
                calls = []
                with torch.inference_mode(), self.record_calls(calls):
                    root.layer(torch.randn(1, 2051, 8))
                self.assertEqual([shape[1] for shape, _ in calls], [2051])
                injection.eject(owner)

    def test_shared_root_on_off_on_and_repeat_injection_do_not_stack(self):
        root, _, _, injection = self.make_plan()
        owner = Patcher(root, True)
        x = torch.randn(1, 2051, 8)
        with torch.inference_mode():
            baseline = root.layer(x)
            outputs = []
            previous = None
            for flag in (True, False, True):
                current = owner.clone(flag)
                runtime = self.inject(injection, current)
                forward = root.layer.forward
                injection.inject(current)
                self.assertEqual(root.layer.forward, forward)
                self.assertEqual(isinstance(runtime, self.safe._CompositeBypassAdapter), flag)
                if previous is not None:
                    injection.eject(previous)
                    self.assertEqual(root.layer.forward, forward)
                calls = []
                with self.record_calls(calls):
                    outputs.append(root.layer(x))
                self.assertEqual([shape[1] for shape, _ in calls], [1024, 1024, 3] if flag else [2051])
                previous = current
            injection.eject(previous)
            torch.testing.assert_close(root.layer(x), baseline)
        for result in outputs[1:]:
            torch.testing.assert_close(result, outputs[0])

    def test_runtime_wrap_does_not_change_canonical_adapter_or_save_strength(self):
        root, original, manager, injection = self.make_plan(strength=1.7)
        saved, strength = manager.adapters["layer"]
        original_weights = saved.weights
        expected = original.calculate_weight(torch.zeros_like(root.layer.weight), "layer.weight", strength, 1., None, lambda x: x)
        owner = Patcher(root, True)
        runtime = self.inject(injection, owner)
        self.assertIs(manager.adapters["layer"][0], saved)
        self.assertIs(type(saved), self.native.LoRAAdapter)
        self.assertIs(saved.weights, original_weights)
        self.assertFalse(hasattr(saved, "low_vram_chunking"))
        child, child_strength = runtime.components[0]
        self.assertIsNot(child, saved)
        self.assertEqual(child_strength, 1.)
        # Simulate a runtime device/dtype conversion without altering canonical weights.
        runtime.weights = tuple(value.double() if torch.is_tensor(value) else value for value in runtime.weights)
        self.assertEqual(saved.weights[0].dtype, torch.float32)
        actual = saved.calculate_weight(torch.zeros_like(root.layer.weight), "layer.weight", strength, 1., None, lambda x: x)
        torch.testing.assert_close(actual, expected)

    def test_enabled_single_lora_keeps_full_forward_and_gradients_with_autograd(self):
        root, adapter, _, injection = self.make_plan()
        owner = Patcher(root, True)
        x = torch.randn(1, 2051, 8, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        weight = adapter.calculate_weight(root.layer.weight.detach().clone(), "layer.weight", -.65, 1., None, lambda x: x)
        expected = F.linear(reference_x, weight, root.layer.bias.detach())
        expected.sum().backward()
        self.inject(injection, owner)
        calls = []
        with self.record_calls(calls):
            actual = root.layer(x)
            actual.sum().backward()
        self.assertEqual([shape[1] for shape, _ in calls], [2051])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, reference_x.grad)

    def test_small_empty_and_two_dimensional_inputs_keep_direct_forward(self):
        root, _, _, injection = self.make_plan()
        owner = Patcher(root, True)
        self.inject(injection, owner)
        for shape in ((2, 1024, 8), (2, 0, 8), (2051, 8)):
            calls = []
            with torch.inference_mode(), self.record_calls(calls):
                result = root.layer(torch.randn(*shape))
            self.assertEqual([input_shape for input_shape, _ in calls], [shape])
            self.assertEqual(result.shape, (*shape[:-1], 15))

    def test_convolution_remains_unchunked_and_preserves_native_geometry(self):
        root, adapter, _, injection = self.make_plan(torch.nn.Conv1d(8, 15, 1))
        owner = Patcher(root, True)
        x = torch.randn(2, 8, 2051)
        expected_weight = adapter.calculate_weight(root.layer.weight.detach().clone(), "layer.weight", -.65, 1., None, lambda x: x)
        expected = F.conv1d(x, expected_weight, root.layer.bias)
        self.inject(injection, owner)
        calls = []
        with torch.inference_mode(), self.record_calls(calls):
            actual = root.layer(x)
        self.assertEqual([shape for shape, _ in calls], [(2, 8, 2051)])
        torch.testing.assert_close(actual, expected)

    def test_other_adapter_types_are_not_wrapped(self):
        root, _, _, _ = self.make_plan()
        adapter = self.native.LoHaAdapter(set(), (torch.randn(15, 2), torch.randn(2, 8),
            1., torch.randn(15, 2), torch.randn(2, 8), None, None, None))
        manager = self.native.BypassInjectionManager()
        self.safe._register_bypass_adapters(manager, {"layer.weight": [(adapter, 1.)]})
        injection, = self.safe._make_rebinding_bypass_injections(manager, root)
        runtime = self.inject(injection, Patcher(root, True))
        self.assertIs(type(runtime), self.native.LoHaAdapter)

    def test_runtime_wrap_rebinds_to_a_different_sampling_root(self):
        root, _, _, injection = self.make_plan()
        delegate = torch.nn.Module()
        delegate.layer = copy.deepcopy(root.layer)
        owner = Patcher(delegate, True)
        x = torch.randn(1, 2051, 8)
        with torch.inference_mode():
            baseline = root.layer(x)
            runtime = self.inject(injection, owner)
            calls = []
            with self.record_calls(calls):
                actual = delegate.layer(x)
            self.assertIsInstance(runtime, self.safe._CompositeBypassAdapter)
            self.assertEqual([shape[1] for shape, _ in calls], [1024, 1024, 3])
            torch.testing.assert_close(root.layer(x), baseline)
            self.assertFalse(torch.equal(actual, baseline))
            injection.eject(owner)
            torch.testing.assert_close(delegate.layer(x), baseline)

    def test_sampling_owner_collection_ejects_single_chunked_adapter(self):
        import gc
        import weakref
        root, _, _, injection = self.make_plan()
        owner = Patcher(root, True)
        x = torch.randn(1, 2051, 8)
        with torch.inference_mode():
            baseline = root.layer(x)
            injection.inject(owner)
            self.assertFalse(torch.equal(root.layer(x), baseline))
            reference = weakref.ref(owner)
            del owner
            gc.collect()
            self.assertIsNone(reference())
            torch.testing.assert_close(root.layer(x), baseline)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for peak-allocation check")
    def test_cuda_single_lora_chunking_lowers_incremental_peak(self):
        torch.manual_seed(15)
        root, _, _, injection = self.make_plan(torch.nn.Linear(256, 2048, device="cuda"), rank=128)
        owner = Patcher(root)
        x = torch.randn(1, 8193, 256, device="cuda")
        results = []
        for flag in (False, True):
            current = owner.clone(flag)
            self.inject(injection, current)
            with torch.inference_mode():
                warmup = root.layer(x)
                del warmup
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                actual = root.layer(x)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - baseline
                results.append((actual.cpu(), peak))
                del actual
            injection.eject(current)
        torch.testing.assert_close(results[1][0], results[0][0], atol=1e-4, rtol=1e-4)
        self.assertLess(results[1][1], results[0][1])
        print(f"Single-LoRA CUDA incremental peak: native={results[0][1]} chunked={results[1][1]} bytes")


if __name__ == "__main__":
    unittest.main()

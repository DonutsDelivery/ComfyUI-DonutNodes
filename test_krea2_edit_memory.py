"""CPU regressions for edit-only pointwise memory patches; no ComfyUI required."""
import copy
from contextlib import ExitStack
import types
import unittest
from unittest.mock import patch
import weakref

import torch

from krea2_memory import TokenChunkedMLP, patch_krea2_upscale_memory
from test_krea2_memory import MLP


class Norm(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],)).to(x.dtype)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qknorm = torch.nn.Module()
        self.qknorm.qnorm = Norm()
        self.qknorm.knorm = Norm()
        self.seen = []

    def forward(self, x):
        self.seen.append(tuple(x.shape))
        q = self.qknorm.qnorm(x.unsqueeze(1))
        k = self.qknorm.knorm(x.unsqueeze(1))
        return torch.nn.functional.scaled_dot_product_attention(q, k, x.unsqueeze(1)).squeeze(1)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.prenorm = Norm()
        self.postnorm = Norm()
        self.attn = Attention()

    def forward(self, x):
        x = x + self.attn(self.prenorm(x))
        return x + self.mlp(self.postnorm(x))


class TextFusion(torch.nn.Module):
    """Mirror upstream's (B*T),layers,C layout and full-sequence refiners."""
    def __init__(self):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList([Block(), Block()])
        self.projector = torch.nn.Linear(12, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList([Block(), Block()])

    def forward(self, x):
        batch, tokens, layers, channels = x.shape
        x = x.reshape(batch * tokens, layers, channels)
        for block in self.layerwise_blocks:
            x = block(x.contiguous())
        x = x.reshape(batch, tokens, layers, channels).transpose(-2, -1)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x)
        return x


class Patcher:
    def __init__(self, *, active=True, options=None, storage="patcher", key="donut_krea2_edit"):
        self.model = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(
            blocks=[Block()], txtfusion=TextFusion(), tproj=None,
        ))
        self.model_options = dict(options or {})
        self.object_patches = {}
        self.wrappers = {}
        if active:
            wrappers = {"diffusion_model": {key: [object()]}}
            if storage == "patcher":
                self.wrappers = wrappers
            else:
                self.model_options["transformer_options"] = {"wrappers": wrappers}

    def clone(self):
        cloned = copy.copy(self)
        cloned.model_options = copy.deepcopy(self.model_options)
        cloned.object_patches = dict(self.object_patches)
        cloned.wrappers = copy.deepcopy(self.wrappers)
        return cloned

    def resolve(self, path):
        value = self.model
        for name in path.split("."):
            value = value[int(name)] if name.isdecimal() else getattr(value, name)
        return value

    def get_model_object(self, path):
        return self.object_patches[path] if path in self.object_patches else self.resolve(path)

    def add_object_patch(self, path, value):
        self.object_patches[path] = value

    def applied(self):
        stack = ExitStack()
        for path, forward in self.object_patches.items():
            owner_path, name = path.rsplit(".", 1)
            stack.enter_context(patch.object(self.resolve(owner_path), name, forward))
        return stack


class EditMemoryTests(unittest.TestCase):
    def test_active_edit_enables_pointwise_patches_in_both_stores(self):
        for storage in ("patcher", "transformer"):
            for key in ("donut_krea2_edit", "krea2_edit"):
                with self.subTest(storage=storage, key=key):
                    original = Patcher(storage=storage, key=key)
                    patched = patch_krea2_upscale_memory(original)
                    self.assertEqual(len(patched.object_patches), 25)
                    self.assertEqual(original.object_patches, {})
                    self.assertIs(original.model, patched.model)
                    self.assertFalse(any(path.endswith("attn.forward") for path in patched.object_patches))

    def test_non_edit_and_unused_edit_studio_branch_are_unchanged(self):
        for options in ({}, {"donut_krea2_edit_branch": True}):
            model = Patcher(active=False, options=options)
            self.assertIs(patch_krea2_upscale_memory(model), model)

    def test_empty_or_unrelated_wrappers_do_not_enable_chunking(self):
        for keyed in ({"donut_krea2_edit": []}, {"fusion": [object()]}):
            model = Patcher(active=False)
            model.wrappers = {"diffusion_model": keyed}
            self.assertIs(patch_krea2_upscale_memory(model), model)

    def test_explicit_opt_outs_are_independent(self):
        for mlp, norm, expected in ((False, False, 0), (True, False, 5), (False, True, 20)):
            with self.subTest(mlp=mlp, norm=norm):
                model = Patcher(options={"donut_chunk_edit_mlp": mlp, "donut_chunk_edit_norm": norm})
                patched = patch_krea2_upscale_memory(model)
                self.assertEqual(len(patched.object_patches), expected)
                if expected == 0:
                    self.assertIs(patched, model)

    def test_explicit_non_edit_opt_in_still_works(self):
        model = Patcher(active=False, options={"donut_chunk_edit_mlp": True})
        self.assertEqual(len(patch_krea2_upscale_memory(model).object_patches), 5)

    def test_layerwise_and_refiner_axes_match_upstream_layout(self):
        patched = patch_krea2_upscale_memory(Patcher())
        for path, wrapper in patched.object_patches.items():
            if ".layerwise_blocks." in path:
                self.assertEqual((wrapper.axis, wrapper.chunk_size), (0, 64))
            else:
                expected_axis = -2 if ".qknorm." in path else 1
                self.assertEqual((wrapper.axis, wrapper.chunk_size), (expected_axis, 1024))

    def test_repeated_patching_does_not_nest_or_clone(self):
        patched = patch_krea2_upscale_memory(Patcher())
        repeated = patch_krea2_upscale_memory(patched)
        self.assertIs(repeated, patched)
        self.assertTrue(all(not isinstance(value.forward, TokenChunkedMLP)
                            for value in repeated.object_patches.values()))

    def test_disabling_one_family_restores_only_that_family(self):
        patched = patch_krea2_upscale_memory(Patcher())
        disabled = patched.clone()
        disabled.model_options["donut_chunk_edit_mlp"] = False
        restored = patch_krea2_upscale_memory(disabled)
        for path, forward in restored.object_patches.items():
            self.assertEqual(isinstance(forward, TokenChunkedMLP), ".mlp." not in path)
        self.assertTrue(all(isinstance(value, TokenChunkedMLP) for value in patched.object_patches.values()))
        restored.model_options["donut_chunk_edit_mlp"] = True
        reenabled = patch_krea2_upscale_memory(restored)
        self.assertTrue(all(isinstance(value, TokenChunkedMLP) for value in reenabled.object_patches.values()))

    def test_leaving_edit_mode_restores_original_forwards(self):
        patched = patch_krea2_upscale_memory(Patcher())
        ordinary = patched.clone()
        ordinary.wrappers = {}
        restored = patch_krea2_upscale_memory(ordinary)
        self.assertTrue(all(not isinstance(value, TokenChunkedMLP) for value in restored.object_patches.values()))

    def test_existing_upstream_forward_is_preserved_and_restored(self):
        model = Patcher()
        path = "diffusion_model.blocks.0.mlp.forward"
        original = model.get_model_object(path)
        def custom(x):
            return original(x) + 0.125
        model.object_patches[path] = custom
        patched = patch_krea2_upscale_memory(model)
        self.assertIs(patched.object_patches[path].forward, custom)
        disabled = patched.clone()
        disabled.model_options["donut_chunk_edit_mlp"] = False
        restored = patch_krea2_upscale_memory(disabled)
        self.assertIs(restored.object_patches[path], custom)

    def test_textfusion_parity_hooks_and_full_attention_sequences(self):
        torch.manual_seed(19)
        model = Patcher()
        fusion = model.model.diffusion_modelfusion if False else model.model.diffusion_model.txtfusion
        seen = []
        def adapter(layer, args, output):
            seen.append(args[0].shape[0])
            return output + args[0].sum(-1, keepdim=True) * 0.03
        hook = fusion.layerwise_blocks[0].mlp.up.register_forward_hook(adapter)
        self.addCleanup(hook.remove)
        # B*T=130 crosses two 64-row boundaries; layer count remains 12.
        x = torch.randn(2, 65, 12, 8)
        original_x = x.clone()
        patched = patch_krea2_upscale_memory(model)
        with torch.inference_mode():
            expected = fusion(x)
            seen.clear()
            for block in [*fusion.layerwise_blocks, *fusion.refiner_blocks]:
                block.attn.seen.clear()
            with patched.applied():
                actual = fusion(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(x, original_x, rtol=0, atol=0)
        self.assertEqual(seen, [64, 64, 2])
        for block in fusion.layerwise_blocks:
            self.assertEqual(block.attn.seen, [(130, 12, 8)])
        for block in fusion.refiner_blocks:
            self.assertEqual(block.attn.seen, [(2, 65, 8)])

    def test_layerwise_bfloat16_norm_bounds_flattened_batch(self):
        seen = []
        def norm(x):
            seen.append(x.shape[0])
            return torch.nn.functional.rms_norm(x.float(), (8,)).to(x.dtype)
        x = torch.randn(130, 2, 12, 16, dtype=torch.bfloat16)[..., ::2]
        with torch.inference_mode():
            expected = norm(x)
            seen.clear()
            actual = TokenChunkedMLP(norm, 64, axis=0)(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(seen, [64, 64, 2])
        self.assertEqual(actual.dtype, x.dtype)

    def test_grad_enabled_forward_is_not_chunked(self):
        seen = []
        def forward(x):
            seen.append(x.shape[0])
            return x.square()
        x = torch.randn(130, 12, 8, requires_grad=True)
        actual = TokenChunkedMLP(forward, 64, axis=0)(x)
        actual.sum().backward()
        self.assertEqual(seen, [130])
        torch.testing.assert_close(x.grad, 2 * x.detach())

    def test_chunk_outputs_are_released_before_next_forward(self):
        previous = []
        def forward(x):
            self.assertTrue(all(ref() is None for ref in previous))
            result = x + 1
            previous.append(weakref.ref(result))
            return result
        with torch.inference_mode():
            actual = TokenChunkedMLP(forward, 4, axis=0)(torch.zeros(13, 12, 8))
        self.assertEqual(len(previous), 4)
        self.assertTrue(all(ref() is None for ref in previous))
        torch.testing.assert_close(actual, torch.ones_like(actual))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for peak-allocation check")
    def test_cuda_chunking_lowers_layerwise_mlp_peak_allocation(self):
        class WideMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Linear(256, 768)
                self.up = torch.nn.Linear(256, 768)
                self.down = torch.nn.Linear(768, 256)

            def forward(self, x):
                return self.down(torch.nn.functional.silu(self.gate(x)).mul_(self.up(x)))

        device = torch.device("cuda", torch.cuda.current_device())
        mlp = WideMLP().to(device)
        x = torch.randn(1025, 12, 256, device=device)
        chunked = TokenChunkedMLP(mlp.forward, 64, axis=0)

        def measured(forward):
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            result = forward(x)
            torch.cuda.synchronize(device)
            return result, torch.cuda.max_memory_allocated(device) - baseline

        with torch.inference_mode():
            # Warm both shapes before measuring allocated (not reserved) bytes.
            del_warmup = mlp(x)
            del del_warmup
            del_warmup = chunked(x)
            del del_warmup
            expected, full_peak = measured(mlp)
            actual, chunked_peak = measured(chunked)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        self.assertLess(chunked_peak, full_peak)

    def test_small_empty_and_lower_rank_inputs_use_direct_forward(self):
        for shape in ((64, 12, 8), (0, 12, 8), (130, 8)):
            calls = []
            def forward(x):
                calls.append(x)
                return x
            x = torch.zeros(shape)
            with torch.inference_mode():
                self.assertIs(TokenChunkedMLP(forward, 64, axis=0)(x), x)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()

import copy
import types
import unittest
import torch
from krea2_memory import TokenChunkedMLP, patch_krea2_upscale_memory


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(8, 24)
        self.up = torch.nn.Linear(8, 24)
        self.down = torch.nn.Linear(24, 8)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)).mul_(self.up(x)))


class MemoryTests(unittest.TestCase):
    def test_chunks_preserve_outputs_and_execute_linear_hooks(self):
        torch.manual_seed(5)
        mlp = MLP()
        seen = []
        def adapter(layer, args, output):
            seen.append(args[0].shape[1])
            return output + args[0].sum(-1, keepdim=True) * .03
        hook = mlp.up.register_forward_hook(adapter)
        self.addCleanup(hook.remove)
        x = torch.randn(2, 19, 8)
        with torch.inference_mode():
            expected = mlp(x)
            seen.clear()
            actual = TokenChunkedMLP(mlp.forward, 4)(x)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(seen, [4, 4, 4, 4, 3])

    def test_reuse_stages_and_releases_once_per_chunked_call(self):
        torch.manual_seed(9)
        mlp = MLP()
        x = torch.randn(2, 19, 8)
        wrapped = TokenChunkedMLP(mlp.forward, 4, reuse_weights=True)
        staged = []
        released = []
        wrapped._stage_weights = lambda value: staged.append(tuple(value.shape)) or object()
        wrapped._release_weights = lambda value: released.append(value)
        with torch.inference_mode():
            expected = mlp(x)
            actual = wrapped(x)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(staged, [(2, 19, 8)])
        self.assertEqual(len(released), 1)

    def test_object_patches_are_clone_local_and_idempotent(self):
        mlp = MLP()
        root = types.SimpleNamespace(blocks=[types.SimpleNamespace(mlp=mlp)], txtfusion=None, tproj=None)
        class Patcher:
            def __init__(self):
                self.model = types.SimpleNamespace(diffusion_model=root)
                self.object_patches = {}
            def clone(self):
                clone = copy.copy(self)
                clone.object_patches = dict(self.object_patches)
                return clone
            def get_model_object(self, path):
                return self.object_patches.get(path, mlp.forward)
            def add_object_patch(self, path, value):
                self.object_patches[path] = value
        original = Patcher()
        patched = patch_krea2_upscale_memory(original)
        self.assertFalse(original.object_patches)
        self.assertEqual(len(patched.object_patches), 1)
        repeated = patch_krea2_upscale_memory(patched)
        self.assertIs(next(iter(patched.object_patches.values())), next(iter(repeated.object_patches.values())))


if __name__ == '__main__':
    unittest.main()

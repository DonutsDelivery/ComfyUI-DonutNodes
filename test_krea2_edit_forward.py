import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


class FakeWrapperExecutor:
    def __init__(self, original, class_obj, wrappers, idx=0):
        self.original = original
        self.class_obj = class_obj
        self.wrappers = list(wrappers)
        self.idx = idx
        self.is_last = idx == len(self.wrappers)

    def __call__(self, *args, **kwargs):
        return type(self)(
            self.original, self.class_obj, self.wrappers, self.idx + 1,
        ).execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if self.is_last:
            return self.original(*args, **kwargs)
        return self.wrappers[self.idx](self, *args, **kwargs)

    @classmethod
    def new_class_executor(cls, original, class_obj, wrappers, idx=0):
        return cls(original, class_obj, wrappers, idx)


fake_nodes = types.ModuleType("nodes")
fake_nodes.NODE_CLASS_MAPPINGS = {}
fake_comfy = types.ModuleType("comfy")
fake_ldm = types.ModuleType("comfy.ldm")
fake_common_dit = types.ModuleType("comfy.ldm.common_dit")
fake_common_dit.pad_to_patch_size = lambda value, patch: value
fake_flux = types.ModuleType("comfy.ldm.flux")
fake_flux_layers = types.ModuleType("comfy.ldm.flux.layers")
fake_flux_layers.timestep_embedding = lambda timesteps, dim: timesteps.reshape(-1, 1).repeat(1, dim)
fake_patcher_extension = types.ModuleType("comfy.patcher_extension")
fake_patcher_extension.WrappersMP = types.SimpleNamespace(DIFFUSION_MODEL="diffusion_model")
fake_patcher_extension.WrapperExecutor = FakeWrapperExecutor
fake_comfy.ldm = fake_ldm
fake_comfy.patcher_extension = fake_patcher_extension
fake_ldm.common_dit = fake_common_dit
fake_ldm.flux = fake_flux
fake_flux.layers = fake_flux_layers
_fake_modules = {
    "nodes": fake_nodes,
    "comfy": fake_comfy,
    "comfy.ldm": fake_ldm,
    "comfy.ldm.common_dit": fake_common_dit,
    "comfy.ldm.flux": fake_flux,
    "comfy.ldm.flux.layers": fake_flux_layers,
    "comfy.patcher_extension": fake_patcher_extension,
}
_missing = object()
_previous_modules = {name: sys.modules.get(name, _missing) for name in _fake_modules}
sys.modules.update(_fake_modules)
try:
    spec = importlib.util.spec_from_file_location(
        "krea2_edit_forward_tested",
        Path(__file__).with_name("krea2_edit_integration.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    for name, previous in _previous_modules.items():
        if previous is _missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class FakeBlock:
    def __init__(self):
        self.mod_inputs = []
        self.attention_input = None
        self.mlp_input = None

    def __call__(self, value, vec, freqs, mask=None,
                 transformer_options=None):
        pre_scale, pre_shift, pre_gate, post_scale, post_shift, post_gate = self.mod(vec)
        attention_input = (1 + pre_scale) * self.prenorm(value) + pre_shift
        attention = self.attn(
            attention_input, freqs, mask,
            transformer_options=transformer_options,
        )
        value = value + pre_gate * attention
        mlp_input = (1 + post_scale) * self.postnorm(value) + post_shift
        return value + post_gate * self.mlp(mlp_input)

    def mod(self, value):
        self.mod_inputs.append(value.detach().clone())
        return value.chunk(6, dim=-1)

    def prenorm(self, value):
        return value

    def postnorm(self, value):
        return value

    def attn(self, value, freqs, mask, transformer_options=None):
        self.attention_input = value.detach().clone()
        return torch.zeros_like(value)

    def mlp(self, value):
        self.mlp_input = value.detach().clone()
        return torch.zeros_like(value)


class FakeDiffusionModel:
    patch = 1
    channels = 1
    tdim = 1

    def __init__(self):
        self.block = FakeBlock()
        self.blocks = [self.block]
        self.last_t = None
        self.first_batches = []
        self.first_values = []

    def _unpack_context(self, context):
        return context

    def first(self, value):
        self.first_batches.append(value.shape[0])
        self.first_values.append(value.detach().clone())
        return value

    def tmlp(self, value):
        return value

    def tproj(self, value):
        return value.repeat(1, 1, 6)

    def txtfusion(self, context, mask=None, transformer_options=None):
        return context

    def txtmlp(self, context):
        return context

    def pe_embedder(self, positions):
        return positions

    def last(self, value, timestep):
        self.last_t = timestep.detach().clone()
        return value


class FakeProcessModel:
    def __init__(self):
        self.diffusion_model = FakeDiffusionModel()

    def process_latent_in(self, samples):
        return samples


class FakeModelPatcher:
    def __init__(self, model_options=None, wrappers=None):
        self.model = FakeProcessModel()
        self.model_options = model_options or {"transformer_options": {}}
        self.wrappers = wrappers or {}

    def clone(self):
        return type(self)(
            copy.deepcopy(self.model_options),
            copy.deepcopy(self.wrappers),
        )

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)


def all_diffusion_wrappers(model_patcher):
    keyed = model_patcher.wrappers.get("diffusion_model", {})
    return [wrapper for wrappers in keyed.values() for wrapper in wrappers]


class Krea2EditForwardTests(unittest.TestCase):
    def test_reference_text_and_target_use_upstream_active_timestep(self):
        model = FakeDiffusionModel()
        target = torch.tensor([[[[30.0]]]])
        source = torch.tensor([[[[20.0]]]])
        context = torch.tensor([[[10.0]]])
        timesteps = torch.tensor([2.0])

        output = module.krea2_edit_forward(
            model, target, timesteps, context, source,
            target_batch=1, transformer_options={},
        )

        expected = torch.tensor([[[32.0], [62.0], [92.0]]])
        self.assertTrue(torch.equal(model.block.attention_input, expected))
        self.assertTrue(torch.equal(model.block.mlp_input, expected))
        self.assertEqual(
            [tuple(value.shape) for value in model.block.mod_inputs],
            [(1, 1, 6)],
        )
        self.assertEqual(model.block.mod_inputs[0].flatten().tolist(), [2.0] * 6)
        self.assertEqual(tuple(output.shape), (1, 1, 1, 1))
        self.assertEqual(float(output.item()), 30.0)
        self.assertEqual(float(model.last_t.item()), 2.0)

    def test_singleton_source_is_projected_once_then_broadcast(self):
        model = FakeDiffusionModel()

        module.krea2_edit_forward(
            model,
            torch.tensor([[[[30.0]]], [[[40.0]]]]),
            torch.tensor([2.0, 3.0]),
            torch.tensor([[[10.0]], [[11.0]]]),
            torch.tensor([[[[20.0]]]]),
            target_batch=2,
            transformer_options={},
        )

        self.assertEqual(model.first_batches, [2, 1])
        self.assertEqual(
            model.block.attention_input[:, 1, 0].tolist(),
            [62.0, 83.0],
        )

    def test_odd_target_resizes_source_before_matching_patch_padding(self):
        model = FakeDiffusionModel()
        model.patch = 2
        source = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)

        def pad_to_patch_size(value, patch):
            patch_height, patch_width = patch
            pad_height = (-value.shape[-2]) % patch_height
            pad_width = (-value.shape[-1]) % patch_width
            return torch.nn.functional.pad(value, (0, pad_width, 0, pad_height))

        old_pad = fake_common_dit.pad_to_patch_size
        fake_common_dit.pad_to_patch_size = pad_to_patch_size
        try:
            module.krea2_edit_forward(
                model,
                torch.zeros(1, 1, 3, 5),
                torch.tensor([1.0]),
                torch.zeros(1, 1, 4),
                source,
                target_batch=1,
                transformer_options={},
            )
        finally:
            fake_common_dit.pad_to_patch_size = old_pad

        expected = torch.nn.functional.interpolate(
            source, size=(3, 5), mode="bilinear",
        )
        expected = pad_to_patch_size(expected, (2, 2))
        expected = module._patchify(expected, 2)
        self.assertTrue(torch.allclose(model.first_values[1], expected))

    def test_wrapper_moves_only_its_source_tensor(self):
        wrapper = module._Krea2EditWrapper(
            torch.ones(1, 1, 1, 1), target_batch=3,
        )

        moved = wrapper.to(torch.float64)

        self.assertIsNot(moved, wrapper)
        self.assertEqual(moved.source_samples.dtype, torch.float64)
        self.assertEqual(wrapper.source_samples.dtype, torch.float32)
        self.assertEqual(moved.target_batch, 3)

    def test_patch_rejects_non_krea_model_before_cloning(self):
        patcher = FakeModelPatcher()
        del patcher.model.diffusion_model

        with self.assertRaisesRegex(RuntimeError, "Krea 2 diffusion model"):
            module.patch_krea2_edit_model(
                patcher, {"samples": torch.zeros(1, 1, 1, 1)},
            )

    def test_local_edit_wrapper_is_idempotent_and_composes_with_other_wrappers(self):
        events = []

        def sentinel(name):
            def wrapped(executor, x, timesteps, context,
                        attention_mask=None, transformer_options=None, **kwargs):
                events.append(f"{name}:in")
                result = executor(
                    x=x,
                    timesteps=timesteps,
                    context=context,
                    attention_mask=attention_mask,
                    transformer_options=transformer_options,
                    **kwargs,
                )
                events.append(f"{name}:out")
                return result
            return wrapped

        base = FakeModelPatcher()
        base.add_wrapper_with_key("diffusion_model", "before", sentinel("before"))
        legacy_wrappers = (
            base.model_options.setdefault("transformer_options", {})
            .setdefault("wrappers", {})
            .setdefault("diffusion_model", {})
        )
        legacy_wrappers["krea2_edit"] = [sentinel("legacy")]
        source_latent = {"samples": torch.tensor([[[[20.0]]]])}

        patched = module.patch_krea2_edit_model(base, source_latent, target_batch=1)
        patched.add_wrapper_with_key("diffusion_model", "after", sentinel("after"))
        repatched = module.patch_krea2_edit_model(patched, source_latent, target_batch=1)
        repatched.add_wrapper_with_key(
            "diffusion_model", "trailing", sentinel("trailing"),
        )

        keyed = repatched.wrappers["diffusion_model"]
        nested_keyed = repatched.model_options["transformer_options"]["wrappers"][
            "diffusion_model"
        ]
        self.assertNotIn("krea2_edit", nested_keyed)
        self.assertEqual(len(keyed[module._EDIT_WRAPPER_KEY]), 1)

        diffusion_model = FakeDiffusionModel()
        executor = FakeWrapperExecutor.new_class_executor(
            lambda *args, **kwargs: self.fail("native forward should be replaced"),
            diffusion_model,
            all_diffusion_wrappers(repatched),
        )
        output = executor.execute(
            torch.tensor([[[[30.0]]]]),
            torch.tensor([2.0]),
            torch.tensor([[[10.0]]]),
            None,
            repatched.model_options["transformer_options"],
        )

        self.assertEqual(
            events,
            [
                "before:in", "after:in", "trailing:in",
                "trailing:out", "after:out", "before:out",
            ],
        )
        self.assertEqual(float(output.item()), 30.0)


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


# Minimal Comfy/folder stubs so pure extraction helpers can be imported in a
# normal Python test process without starting ComfyUI.
fake_comfy = types.ModuleType("comfy")
fake_comfy.__path__ = []
fake_lora = types.ModuleType("comfy.lora")
fake_utils = types.ModuleType("comfy.utils")
fake_comfy.lora = fake_lora
fake_comfy.utils = fake_utils


def fake_calculate_weight(patches, weight, key, **kwargs):
    out = weight.clone()
    for patch in patches:
        strength, adapter = patch[0], patch[1]
        if hasattr(adapter, "delta"):
            out = out + float(strength) * adapter.delta.to(out)
    return out


fake_lora.calculate_weight = fake_calculate_weight
fake_utils.save_torch_file = lambda *args, **kwargs: None

fake_folder_paths = types.ModuleType("folder_paths")
fake_folder_paths.get_output_directory = lambda: "/tmp"
fake_folder_paths.get_save_image_path = lambda prefix, output: (
    output, "test", 1, "", prefix,
)

_previous = {
    name: sys.modules.get(name)
    for name in ("comfy", "comfy.lora", "comfy.utils", "folder_paths")
}
sys.modules["comfy"] = fake_comfy
sys.modules["comfy.lora"] = fake_lora
sys.modules["comfy.utils"] = fake_utils
sys.modules["folder_paths"] = fake_folder_paths

try:
    helper_spec = importlib.util.spec_from_file_location(
        "donut_bypass_materialization",
        Path(__file__).with_name("donut_bypass_materialization.py"),
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    sys.modules["donut_bypass_materialization"] = helper

    extract_spec = importlib.util.spec_from_file_location(
        "donut_extract_lora_tested",
        Path(__file__).with_name("DonutExtractLoRA.py"),
    )
    extract = importlib.util.module_from_spec(extract_spec)
    extract_spec.loader.exec_module(extract)
finally:
    for name, previous in _previous.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class Adapter:
    def __init__(self, delta):
        self.delta = delta


class CompositeAdapter:
    def __init__(self, components):
        self.components = components


class FakeInjectionModel:
    def __init__(self, manager):
        def inject_all(model_patcher):
            return manager

        def eject_all(model_patcher):
            return manager

        injection = types.SimpleNamespace(inject=inject_all, eject=eject_all)
        self.injections = {helper.BYPASS_INJECTION_KEY: [injection]}
        self.attachments = {}
        self.added = []

    def get_attachment(self, key):
        return self.attachments.get(key)

    def get_injections(self, key):
        return self.injections.get(key, [])

    def clone(self):
        clone = FakeInjectionModel.__new__(FakeInjectionModel)
        clone.injections = {k: list(v) for k, v in self.injections.items()}
        clone.attachments = dict(self.attachments)
        clone.added = []
        return clone

    def remove_injections(self, key):
        self.injections.pop(key, None)

    def remove_attachments(self, key):
        self.attachments.pop(key, None)

    def add_patches(self, patches, strength):
        self.added.append((patches, strength))


class DonutExtractLoRATests(unittest.TestCase):
    def test_rank_factorization_reconstructs_low_rank_matrix(self):
        left = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
        right = torch.tensor([[2.0, 0.0, 1.0, -1.0], [1.0, 3.0, -2.0, 0.5]])
        delta = left @ right

        up, down, rank = extract._factorize_delta(delta, 2)

        self.assertEqual(rank, 2)
        self.assertTrue(torch.allclose(up @ down, delta, atol=1e-4, rtol=1e-4))

    def test_conv_factorization_uses_standard_lora_shapes(self):
        torch.manual_seed(4)
        up_true = torch.randn(5, 2)
        down_true = torch.randn(2, 3 * 3 * 3)
        delta = (up_true @ down_true).reshape(5, 3, 3, 3)

        up, down, rank = extract._factorize_delta(delta, 2)

        self.assertEqual(rank, 2)
        self.assertEqual(tuple(up.shape), (5, 2, 1, 1))
        self.assertEqual(tuple(down.shape), (2, 3, 3, 3))
        reconstructed = (up.flatten(1) @ down.flatten(1)).reshape(delta.shape)
        self.assertTrue(torch.allclose(reconstructed, delta, atol=1e-4, rtol=1e-4))

    def test_effective_weight_includes_bypass_adapter(self):
        base = torch.zeros(3, 4)
        regular = Adapter(torch.ones(3, 4))
        bypass = Adapter(torch.full((3, 4), 2.0))
        key_patches = [
            (base, lambda value, **kwargs: value),
            (0.5, regular, 1.0, None, None),
        ]

        result = extract._effective_weight(
            "diffusion_model.layer.weight",
            key_patches,
            [(bypass, 0.25)],
        )

        expected = torch.ones(3, 4)
        self.assertTrue(torch.allclose(result, expected))

    def test_discovers_and_flattens_existing_bypass_manager(self):
        a = Adapter(torch.ones(2, 2))
        b = Adapter(torch.ones(2, 2))
        manager = types.SimpleNamespace(adapters={
            "diffusion_model.layer": (CompositeAdapter([(a, 0.5), (b, 0.25)]), 2.0),
        })
        model = FakeInjectionModel(manager)

        components = helper.get_bypass_components(model)

        self.assertIn("diffusion_model.layer.weight", components)
        self.assertEqual(components["diffusion_model.layer.weight"], [(a, 1.0), (b, 0.5)])

    def test_converts_bypass_to_regular_patches_without_dense_delta(self):
        adapter = Adapter(torch.ones(2, 2))
        manager = types.SimpleNamespace(adapters={
            "diffusion_model.layer": (adapter, 0.75),
        })
        model = FakeInjectionModel(manager)

        converted = helper.clone_with_bypass_as_regular_patches(model)

        self.assertNotIn(helper.BYPASS_INJECTION_KEY, converted.injections)
        self.assertEqual(len(converted.added), 1)
        patches, strength = converted.added[0]
        self.assertIs(patches["diffusion_model.layer.weight"], adapter)
        self.assertEqual(strength, 0.75)

    def test_node_schema_has_raw_patched_and_rank(self):
        required = extract.DonutExtractLoRA.INPUT_TYPES()["required"]
        self.assertIn("raw_model", required)
        self.assertIn("patched_model", required)
        self.assertIn("rank", required)
        self.assertEqual(required["rank"][1]["default"], 32)


if __name__ == "__main__":
    unittest.main()

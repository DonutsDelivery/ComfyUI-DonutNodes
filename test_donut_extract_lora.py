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
fake_model_patcher = types.ModuleType("comfy.model_patcher")
fake_comfy.lora = fake_lora
fake_comfy.utils = fake_utils
fake_comfy.model_patcher = fake_model_patcher


SAVED_FILES = []


def fake_calculate_weight(patches, weight, key, **kwargs):
    out = weight.clone()
    for patch in patches:
        strength, adapter = patch[0], patch[1]
        if hasattr(adapter, "delta"):
            out = out + float(strength) * adapter.delta.to(out)
    return out


def fake_get_key_weight(model, key):
    obj = model
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    # Deliberately use getattr here: requesting weight_scale from the fake
    # quantized Linear will raise exactly like the real ComfyUI failure.
    return getattr(obj, parts[-1]), None, None


def fake_save_torch_file(sd, path, metadata=None):
    SAVED_FILES.append((dict(sd), path, dict(metadata or {})))


fake_lora.calculate_weight = fake_calculate_weight
fake_model_patcher.get_key_weight = fake_get_key_weight
fake_utils.save_torch_file = fake_save_torch_file

fake_folder_paths = types.ModuleType("folder_paths")
fake_folder_paths.get_output_directory = lambda: "/tmp"
fake_folder_paths.get_save_image_path = lambda prefix, output: (
    output, "test", 1, "", prefix,
)

_previous = {
    name: sys.modules.get(name)
    for name in (
        "comfy",
        "comfy.lora",
        "comfy.model_patcher",
        "comfy.utils",
        "folder_paths",
        "donut_bypass_materialization",
        "donut_krea2_merge_serialization",
    )
}
sys.modules["comfy"] = fake_comfy
sys.modules["comfy.lora"] = fake_lora
sys.modules["comfy.model_patcher"] = fake_model_patcher
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

    krea_spec = importlib.util.spec_from_file_location(
        "donut_krea2_merge_serialization",
        Path(__file__).with_name("donut_krea2_merge_serialization.py"),
    )
    krea = importlib.util.module_from_spec(krea_spec)
    krea_spec.loader.exec_module(krea)
    sys.modules["donut_krea2_merge_serialization"] = krea

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


class FalseyInjectionList(list):
    def __bool__(self):
        return False


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


class FakeQuantizedPatcher:
    """State dict exposes scale metadata that the live Linear does not own."""

    def __init__(self):
        layer = types.SimpleNamespace(weight=torch.randn(4, 3))
        diffusion_model = types.SimpleNamespace(layer=layer)
        self.model = types.SimpleNamespace(diffusion_model=diffusion_model)
        self.patches = {}
        self.backup = {}
        self.hook_backup = {}

        def state_dict():
            return {
                "diffusion_model.layer.weight": layer.weight,
                "diffusion_model.layer.weight_scale": torch.tensor(0.125),
                "diffusion_model.layer.input_scale": torch.tensor(0.5),
            }

        self.model.state_dict = state_dict


class FakeQuantizedValue:
    def __init__(self, tensor):
        self.tensor = tensor

    def dequantize(self):
        return self.tensor


class FakeModelTree:
    def __init__(self, layer_weight, layer_bias, other_weight=None):
        self.diffusion_model = types.SimpleNamespace(
            layer=types.SimpleNamespace(
                weight=layer_weight.clone(),
                bias=None if layer_bias is None else layer_bias.clone(),
            ),
            other=types.SimpleNamespace(
                weight=(
                    torch.eye(layer_weight.shape[0])
                    if other_weight is None
                    else other_weight.clone()
                ),
            ),
        )

    def state_dict(self):
        state = {
            "diffusion_model.layer.weight": self.diffusion_model.layer.weight,
            "diffusion_model.other.weight": self.diffusion_model.other.weight,
        }
        if self.diffusion_model.layer.bias is not None:
            state["diffusion_model.layer.bias"] = self.diffusion_model.layer.bias
        return state


class FakeMergePatcher:
    def __init__(self, layer_weight, layer_bias, other_weight=None, clone_id="base"):
        self.model = FakeModelTree(layer_weight, layer_bias, other_weight)
        self.patches = {}
        self.backup = {}
        self.hook_backup = {}
        self.injections = {}
        self.attachments = {}
        self.additional_models = {}
        self.clone_base_uuid = clone_id

    def get_attachment(self, key):
        return self.attachments.get(key)

    def get_injections(self, key):
        return self.injections.get(key, [])

    def get_additional_models_with_key(self, key):
        return self.additional_models.get(key, [])


def attach_krea2_swap(patched, source):
    plan = (
        "diffusion_model.layer",
        "diffusion_model.layer.weight",
        0.0,
    )
    patched.injections[krea.KREA2_MERGE_INJECTION_KEY] = FalseyInjectionList([object()])
    patched.additional_models[krea.KREA2_MERGE_SOURCE_KEY] = [source]
    patched.attachments[krea.KREA2_MERGE_PLAN_PREFIX + "test"] = (plan,)
    return plan


class DonutExtractLoRATests(unittest.TestCase):
    def setUp(self):
        SAVED_FILES.clear()

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

    def test_quantized_metadata_keys_are_not_resolved_as_module_weights(self):
        model = FakeQuantizedPatcher()

        patches = extract._get_extractable_key_patches(model)

        self.assertEqual(list(patches), ["diffusion_model.layer.weight"])
        self.assertTrue(torch.equal(
            patches["diffusion_model.layer.weight"][0][0],
            model.model.diffusion_model.layer.weight,
        ))

    def test_quantized_value_is_dequantized_before_float32_extraction(self):
        expected = torch.tensor([[1.25, -2.5], [3.75, 4.0]], dtype=torch.float16)
        value = FakeQuantizedValue(expected)

        result = extract._convert_base_weight(value, lambda x, **kwargs: x)

        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result, expected.float()))

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

    def test_falsey_krea2_runtime_is_not_treated_as_invisible(self):
        model = FakeMergePatcher(torch.eye(2), torch.zeros(2))
        model.injections[krea.KREA2_MERGE_INJECTION_KEY] = FalseyInjectionList([object()])

        self.assertIn(
            krea.KREA2_MERGE_INJECTION_KEY,
            extract._runtime_injection_keys(model),
        )

    def test_effective_sources_use_model2_for_hard_swap_and_model1_elsewhere(self):
        raw = FakeMergePatcher(
            torch.eye(2),
            torch.zeros(2),
            other_weight=torch.eye(2),
        )
        source = FakeMergePatcher(
            4.0 * torch.eye(2),
            torch.tensor([2.0, -1.0]),
            other_weight=99.0 * torch.eye(2),
        )
        patched = FakeMergePatcher(
            torch.eye(2),
            torch.zeros(2),
            other_weight=torch.eye(2),
        )
        attach_krea2_swap(patched, source)
        patched.patches["diffusion_model.other.weight"] = [
            (1.0, Adapter(3.0 * torch.eye(2)), 1.0, None, None),
        ]
        bypass = {
            "diffusion_model.layer.weight": [
                (Adapter(torch.ones(2, 2)), 0.5),
            ],
        }

        effective, swapped, swap_count = extract._get_effective_key_patches(
            patched,
            bypass,
        )

        swapped_weight = extract._effective_weight(
            "diffusion_model.layer.weight",
            effective["diffusion_model.layer.weight"],
            bypass["diffusion_model.layer.weight"],
        )
        swapped_bias = extract._effective_weight(
            "diffusion_model.layer.bias",
            effective["diffusion_model.layer.bias"],
            None,
        )
        other_weight = extract._effective_weight(
            "diffusion_model.other.weight",
            effective["diffusion_model.other.weight"],
            None,
        )

        self.assertEqual(swap_count, 1)
        self.assertIn("diffusion_model.layer.weight", swapped)
        self.assertIn("diffusion_model.layer.bias", swapped)
        self.assertTrue(torch.allclose(
            swapped_weight,
            4.0 * torch.eye(2) + 0.5 * torch.ones(2, 2),
        ))
        self.assertTrue(torch.equal(swapped_bias, torch.tensor([2.0, -1.0])))
        self.assertTrue(torch.equal(other_weight, 4.0 * torch.eye(2)))
        self.assertNotEqual(id(raw), id(patched))

    def test_full_extraction_includes_krea2_swap_lora_and_bias_diff(self):
        raw = FakeMergePatcher(
            torch.eye(2),
            torch.zeros(2),
            clone_id="same-base",
        )
        source = FakeMergePatcher(
            4.0 * torch.eye(2),
            torch.tensor([2.0, -1.0]),
            clone_id="model2",
        )
        patched = FakeMergePatcher(
            torch.eye(2),
            torch.zeros(2),
            clone_id="same-base",
        )
        attach_krea2_swap(patched, source)
        bypass_delta = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        patched.attachments[helper.BYPASS_ATTACHMENT_KEY] = {
            "diffusion_model.layer.weight": [
                (Adapter(bypass_delta), 0.5),
            ],
        }

        _path, report = extract.DonutExtractLoRA().extract(
            raw,
            patched,
            rank=2,
            filename_prefix="loras/test",
            dtype="fp32",
        )

        self.assertEqual(len(SAVED_FILES), 1)
        sd, _saved_path, metadata = SAVED_FILES[0]
        up = sd["diffusion_model.layer.lora_up.weight"]
        down = sd["diffusion_model.layer.lora_down.weight"]
        reconstructed_delta = up @ down
        expected_target = 4.0 * torch.eye(2) + 0.5 * bypass_delta
        expected_delta = expected_target - torch.eye(2)

        self.assertTrue(torch.allclose(
            reconstructed_delta,
            expected_delta,
            atol=1e-4,
            rtol=1e-4,
        ))
        self.assertTrue(torch.equal(
            sd["diffusion_model.layer.diff_b"],
            torch.tensor([2.0, -1.0]),
        ))
        self.assertEqual(metadata["donut.krea2_merge_aware"], "true")
        self.assertIn("included 1 Krea2 model2 hard swap", report)
        self.assertNotIn("unsupported runtime injections", report)

    def test_node_schema_has_raw_patched_and_rank(self):
        required = extract.DonutExtractLoRA.INPUT_TYPES()["required"]
        self.assertIn("raw_model", required)
        self.assertIn("patched_model", required)
        self.assertIn("rank", required)
        self.assertEqual(required["rank"][1]["default"], 32)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from shared.lora_processing import LoRADelta, _default_block_vector, _named_parameter_module


class ModelPatcherDynamic:
    """Minimal ComfyUI-style wrapper with no named_parameters() method."""
    def __init__(self, model):
        self.model = model


class LoRAProcessingTests(unittest.TestCase):
    def test_delta_resolves_model_patcher_dynamic_wrappers(self):
        base_module = torch.nn.Linear(2, 1, bias=False)
        enhanced_module = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            base_module.weight.copy_(torch.tensor([[1.0, 2.0]]))
            enhanced_module.weight.copy_(torch.tensor([[4.0, 7.0]]))

        delta = LoRADelta(
            ModelPatcherDynamic(base_module),
            ModelPatcherDynamic(enhanced_module),
            "krea2-test",
        )

        self.assertIs(delta.base_model, base_module)
        self.assertEqual(delta.get_delta_info()["delta_count"], 1)
        self.assertTrue(torch.equal(delta.deltas["weight"], torch.tensor([[3.0, 5.0]])))

    def test_default_vector_covers_all_krea2_blocks_and_non_block_weights(self):
        lora = {
            "diffusion_model.blocks.0.attn.lora_A.weight": torch.empty(1),
            "diffusion_model.blocks.27.attn.lora_A.weight": torch.empty(1),
            "diffusion_model.txtfusion.lora_A.weight": torch.empty(1),
        }

        vector = _default_block_vector(lora)

        self.assertEqual(vector.split(","), ["1"] * 29)

    def test_resolver_rejects_objects_without_a_parameter_module(self):
        with self.assertRaisesRegex(TypeError, "does not expose a parameter module"):
            _named_parameter_module(object())


if __name__ == "__main__":
    unittest.main()

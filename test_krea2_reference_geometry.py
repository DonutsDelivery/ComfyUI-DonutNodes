import importlib.util
import sys
import types
import unittest

import torch


fake_nodes = types.ModuleType("nodes")
fake_nodes.NODE_CLASS_MAPPINGS = {}
sys.modules.setdefault("nodes", fake_nodes)

spec = importlib.util.spec_from_file_location(
    "krea2_edit_integration_tested",
    "/home/user/Programs/ComfyUI/custom_nodes/donutnodes/krea2_edit_integration.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AlignedReferenceCropTests(unittest.TestCase):
    def test_natural_one_megapixel_conditioning_is_not_grid_snapped(self):
        source = torch.zeros(1, 2433, 1664, 3)

        scaled = module.scale_image_to_megapixels(source)

        self.assertEqual(tuple(scaled.shape), (1, 1238, 847, 3))

    def test_target_solver_prioritizes_aspect_ratio_on_32_pixel_grid(self):
        width, height = module.solve_multiple_target(847, 1238)

        self.assertEqual((width, height), (832, 1216))
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)

    def test_patch_source_is_center_cropped_to_exact_target_geometry(self):
        x = torch.arange(200, dtype=torch.float32).view(1, 1, 200, 1)
        source = x.expand(1, 100, 200, 3)

        cropped = module.resize_center_crop(source, 100, 100)

        self.assertEqual(tuple(cropped.shape), (1, 100, 100, 3))
        self.assertEqual(float(cropped[0, 0, 0, 0]), 50.0)
        self.assertEqual(float(cropped[0, 0, -1, 0]), 149.0)

    def test_prepare_splits_patch_and_grounding_image_geometry(self):
        captured = {"grounded": []}

        class FakeVAEEncode:
            def encode(self, vae, image):
                captured["patch_image"] = tuple(image.shape[1:3])
                return ({"samples": torch.zeros(1, 16, 8, 8)},)

        class FakePatch:
            def patch(self, model, source_latent):
                return ("patched",)

        class FakeGrounded:
            def encode(self, clip, prompt, image=None, grounding_px=768):
                captured["grounded"].append(tuple(image.shape[1:3]))
                return (prompt,)

        old_vae_encode = getattr(fake_nodes, "VAEEncode", None)
        old_mappings = fake_nodes.NODE_CLASS_MAPPINGS
        fake_nodes.VAEEncode = FakeVAEEncode
        fake_nodes.NODE_CLASS_MAPPINGS = {
            "Krea2EditModelPatch": FakePatch,
            "Krea2EditGroundedEncode": FakeGrounded,
        }
        try:
            source = torch.zeros(1, 100, 200, 3)
            natural = module.scale_image_to_megapixels(source)
            result = module.prepare_krea2_edit(
                "model", "clip", "vae", source, "edit", "", 768,
                64, 64,
            )
        finally:
            fake_nodes.NODE_CLASS_MAPPINGS = old_mappings
            if old_vae_encode is None:
                del fake_nodes.VAEEncode
            else:
                fake_nodes.VAEEncode = old_vae_encode

        natural_shape = tuple(natural.shape[1:3])
        self.assertEqual(captured["patch_image"], (64, 64))
        self.assertEqual(captured["grounded"], [natural_shape, natural_shape])
        self.assertEqual(result[0], "patched")

    def test_face_padding_round_trips_without_resizing_content(self):
        source = torch.arange(1 * 123 * 157 * 3, dtype=torch.float32).reshape(1, 123, 157, 3)

        padded, padding = module.pad_image_to_multiple(source)
        restored = module.crop_image_padding(padded, padding)

        self.assertEqual(tuple(padded.shape), (1, 128, 160, 3))
        self.assertTrue(torch.equal(restored, source))

    def test_reference_is_resized_to_target_canvas_then_cropped_by_exact_region(self):
        source = torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3)

        canvas = module.resize_reference_to_canvas(source, 8, 4)
        crop = module.crop_aligned_reference(source, 8, 4, (2, 1, 6, 3))

        self.assertEqual(tuple(canvas.shape), (1, 4, 8, 3))
        self.assertEqual(tuple(crop.shape), (1, 2, 4, 3))
        self.assertTrue(torch.equal(crop, canvas[:, 1:3, 2:6, :]))

    def test_bbox_is_clamped_to_canvas_and_never_returns_an_empty_crop(self):
        source = torch.ones(1, 3, 5, 3)

        crop = module.crop_aligned_reference(source, 10, 6, (-4, -2, 4, 3))

        self.assertEqual(tuple(crop.shape), (1, 3, 4, 3))

    def test_batch_index_uses_matching_reference_or_single_reference_fallback(self):
        source = torch.stack((torch.zeros(2, 2, 3), torch.ones(2, 2, 3)))

        second = module.crop_aligned_reference(source, 2, 2, (0, 0, 2, 2), batch_index=1)
        fallback = module.crop_aligned_reference(source[:1], 2, 2, (0, 0, 2, 2), batch_index=9)

        self.assertEqual(float(second.mean()), 1.0)
        self.assertEqual(float(fallback.mean()), 0.0)


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


fake_nodes = types.ModuleType("nodes")
fake_nodes.NODE_CLASS_MAPPINGS = {}

fake_comfy = types.ModuleType("comfy")
fake_ldm = types.ModuleType("comfy.ldm")
fake_common_dit = types.ModuleType("comfy.ldm.common_dit")
fake_common_dit.pad_to_patch_size = lambda value, patch: value
fake_flux = types.ModuleType("comfy.ldm.flux")
fake_flux_layers = types.ModuleType("comfy.ldm.flux.layers")
fake_flux_layers.timestep_embedding = lambda timesteps, dim: torch.zeros(
    timesteps.shape[0], dim, device=timesteps.device,
)
fake_patcher_extension = types.ModuleType("comfy.patcher_extension")
fake_patcher_extension.WrappersMP = types.SimpleNamespace(DIFFUSION_MODEL="diffusion_model")
fake_patcher_extension.WrapperExecutor = object
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
        "krea2_edit_integration_tested",
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

        def fake_patch(model, source_latent, target_batch=None):
            captured["target_batch"] = target_batch
            return "patched"

        class FakeGrounded:
            def encode(self, clip, prompt, image=None, grounding_px=768):
                captured["grounded"].append(tuple(image.shape[1:3]))
                return (prompt,)

        old_vae_encode = getattr(fake_nodes, "VAEEncode", None)
        old_mappings = fake_nodes.NODE_CLASS_MAPPINGS
        old_patch = module.patch_krea2_edit_model
        fake_nodes.VAEEncode = FakeVAEEncode
        fake_nodes.NODE_CLASS_MAPPINGS = {
            "Krea2EditGroundedEncode": FakeGrounded,
        }
        module.patch_krea2_edit_model = fake_patch
        try:
            source = torch.zeros(1, 100, 200, 3)
            natural = module.scale_image_to_megapixels(source)
            result = module.prepare_krea2_edit(
                "model", "clip", "vae", source, "edit", "", 768,
                64, 64, target_batch=2,
            )
        finally:
            module.patch_krea2_edit_model = old_patch
            fake_nodes.NODE_CLASS_MAPPINGS = old_mappings
            if old_vae_encode is None:
                del fake_nodes.VAEEncode
            else:
                fake_nodes.VAEEncode = old_vae_encode

        natural_shape = tuple(natural.shape[1:3])
        self.assertEqual(captured["patch_image"], (64, 64))
        self.assertEqual(captured["grounded"], [natural_shape, natural_shape])
        self.assertEqual(captured["target_batch"], 2)
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

    def test_edit_target_preserves_shape_dtype_and_metadata_without_mutating_input(self):
        samples = torch.randn(2, 16, 8, 12, dtype=torch.float16)
        mask = torch.ones(2, 64, 96)
        source = {
            "samples": samples,
            "noise_mask": mask,
            "batch_index": [7, 9],
            "downscale_ratio_spacial": 8,
        }

        target = module.make_krea2_edit_target(source)

        self.assertIsNot(target, source)
        self.assertEqual(tuple(target["samples"].shape), tuple(samples.shape))
        self.assertEqual(target["samples"].dtype, samples.dtype)
        self.assertEqual(int(torch.count_nonzero(target["samples"])), 0)
        self.assertTrue(torch.equal(source["samples"], samples))
        self.assertIs(target["noise_mask"], mask)
        self.assertEqual(target["batch_index"], [7, 9])

    def test_source_batch_broadcasts_single_reference(self):
        singleton = torch.tensor([[[[3.0]]]])

        broadcast = module._match_source_batch(
            singleton, runtime_batch=4, target_batch=2,
        )

        self.assertEqual(
            broadcast[:, 0, 0, 0].tolist(),
            [3.0, 3.0, 3.0, 3.0],
        )

    def test_source_batch_rejects_multiple_sources_and_invalid_runtime_batch(self):
        with self.assertRaisesRegex(ValueError, "one source image"):
            module._match_source_batch(torch.zeros(2, 1, 1, 1), 4, 2)
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            module._match_source_batch(torch.zeros(1, 1, 1, 1), 3, 2)

    def test_edit_target_validation_uses_latent_geometry_and_ratio(self):
        latent = {"samples": torch.zeros(2, 16, 32, 48)}
        source = torch.zeros(1, 80, 60, 3)

        samples, width, height = module.validate_krea2_edit_target(
            latent, source, "simple", 1.0, "disable", 17,
        )
        explicit = dict(latent, downscale_ratio_spacial=16)
        _, explicit_width, explicit_height = module.validate_krea2_edit_target(
            explicit, source, "simple", 1.0, "enable", 0,
        )

        self.assertIs(samples, latent["samples"])
        self.assertEqual((width, height), (384, 256))
        self.assertEqual((explicit_width, explicit_height), (768, 512))

    def test_edit_target_validation_treats_none_spatial_ratio_as_default(self):
        latent = {
            "samples": torch.zeros(1, 16, 8, 12),
            "downscale_ratio_spacial": None,
        }
        source = torch.zeros(1, 64, 64, 3)

        _, width, height = module.validate_krea2_edit_target(
            latent, source, "simple", 1.0, "enable", 0,
        )

        self.assertEqual((width, height), (96, 64))

    def test_edit_target_validation_rejects_temporal_and_oversize_targets(self):
        source = torch.zeros(1, 64, 64, 3)
        temporal = {
            "samples": torch.zeros(1, 16, 8, 8),
            "downscale_ratio_temporal": 4,
        }
        oversize = {
            "samples": torch.zeros(1, 16, 1, 1),
            "downscale_ratio_spacial": 2048,
        }

        with self.assertRaisesRegex(ValueError, "temporal target latents"):
            module.validate_krea2_edit_target(
                temporal, source, "simple", 1.0, "enable", 0,
            )
        with self.assertRaisesRegex(ValueError, "up to 2 MiP"):
            module.validate_krea2_edit_target(
                oversize, source, "simple", 1.0, "enable", 0,
            )

    def test_edit_target_validation_rejects_invalid_empty_target_controls(self):
        latent = {"samples": torch.zeros(2, 16, 8, 8)}
        source = torch.zeros(1, 64, 64, 3)

        with self.assertRaisesRegex(ValueError, "denoise=1.0"):
            module.validate_krea2_edit_target(
                latent, source, "simple", 0.5, "enable", 0,
            )
        with self.assertRaisesRegex(ValueError, "add_noise=enable"):
            module.validate_krea2_edit_target(
                latent, source, "advanced", 1.0, "disable", 0,
            )
        with self.assertRaisesRegex(ValueError, "start_at_step=0"):
            module.validate_krea2_edit_target(
                latent, source, "multi_model", 1.0, "enable", 2,
            )

    def test_edit_target_validation_preserves_full_mask_and_rejects_partial_mask(self):
        source = torch.zeros(1, 64, 64, 3)
        full = {
            "samples": torch.zeros(1, 16, 8, 8),
            "noise_mask": torch.ones(1, 64, 64),
        }
        partial = dict(full, noise_mask=torch.zeros(1, 64, 64))

        module.validate_krea2_edit_target(full, source, "simple", 1.0, "enable", 0)
        with self.assertRaisesRegex(ValueError, "partial noise_mask"):
            module.validate_krea2_edit_target(
                partial, source, "simple", 1.0, "enable", 0,
            )

    def test_edit_target_validation_rejects_mismatched_source_batch(self):
        latent = {"samples": torch.zeros(2, 16, 8, 8)}
        source = torch.zeros(3, 64, 64, 3)

        with self.assertRaisesRegex(ValueError, "one source image"):
            module.validate_krea2_edit_target(
                latent, source, "simple", 1.0, "enable", 0,
            )


if __name__ == "__main__":
    unittest.main()

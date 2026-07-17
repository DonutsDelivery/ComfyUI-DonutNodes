import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch


class FakeModel:
    def __init__(self, name, model_options=None):
        self.name = name
        self.model_options = model_options or {}


class FakeSegment:
    def __init__(self, name, bbox, mask_value=1.0):
        self.name = name
        self.bbox = bbox
        self.crop_region = bbox
        height = max(1, bbox[3] - bbox[1])
        width = max(1, bbox[2] - bbox[0])
        self.cropped_mask = np.full((height, width), mask_value, dtype=np.float32)
        self.cropped_image = torch.zeros(1, height, width, 3)


class FakeDetector:
    def __init__(self, target_segments, reference_segments=None):
        self.target_segments = target_segments
        self.reference_segments = reference_segments or []
        self.aux = None
        self.calls = []

    def setAux(self, value):
        self.aux = value

    def detect(self, image, threshold, dilation, crop_factor, drop_size,
               detailer_hook=None):
        is_reference = bool(float(image.mean()) > 0.5)
        self.calls.append("reference" if is_reference else "target")
        segments = self.reference_segments if is_reference else self.target_segments
        return ((image.shape[1], image.shape[2]), list(segments))


fake_comfy = types.ModuleType("comfy")
fake_comfy.__path__ = []
fake_samplers = types.ModuleType("comfy.samplers")
fake_samplers.KSampler = types.SimpleNamespace(
    SAMPLERS=["euler"], SCHEDULERS=["simple"],
)
fake_sample = types.ModuleType("comfy.sample")
fake_management = types.ModuleType("comfy.model_management")
fake_management.unload_all_models = lambda: None
fake_utils = types.ModuleType("comfy.utils")
fake_comfy.samplers = fake_samplers
fake_comfy.sample = fake_sample
fake_comfy.model_management = fake_management
fake_comfy.utils = fake_utils

fake_nodes = types.ModuleType("nodes")
fake_nodes.MAX_RESOLUTION = 16384

fake_detailer_core = types.ModuleType("donut_detailer_core")
fake_detailer_core.scale_to_megapixels = lambda w, h, resolution, maximum: (w, h)
fake_detailer_core.sample_and_decode = lambda *args, **kwargs: args[0]

fake_krea = types.ModuleType("krea2_edit_integration")
fake_krea.crop_image_padding = lambda image, padding: image
fake_krea.pad_image_to_multiple = lambda image: (image, (0, 0, 0, 0))
fake_krea.prepare_krea2_edit = lambda *args, **kwargs: (
    args[0], args[3], args[3], {"samples": torch.zeros(1, 1, 1, 1)}, args[3],
)

fake_impact = types.ModuleType("impact")
fake_impact.__path__ = []
fake_impact_core = types.ModuleType("impact.core")
fake_impact_utils = types.ModuleType("impact.utils")
fake_impact_sampling = types.ModuleType("impact.impact_sampling")
fake_impact.core = fake_impact_core
fake_impact.utils = fake_impact_utils
fake_impact.impact_sampling = fake_impact_sampling

fake_impact_core.segs_to_combined_mask = lambda segs: torch.zeros(
    1, segs[0][0], segs[0][1],
)
fake_impact_core.make_sam_mask = lambda *args, **kwargs: object()
fake_impact_core.segs_bitwise_and_mask = lambda segs, mask: segs
fake_impact_utils.tensor_gaussian_blur_mask = lambda mask, feather: torch.ones(
    1, mask.shape[-2], mask.shape[-1], 1,
)
fake_impact_utils.tensor_resize = lambda image, width, height: torch.zeros(
    image.shape[0], height, width, image.shape[-1],
)
fake_impact_utils.to_latent_image = lambda image, vae: {
    "samples": torch.zeros(image.shape[0], 1, image.shape[1], image.shape[2]),
}
fake_impact_utils.crop_image = lambda image, region: image[
    :, region[1]:region[3], region[0]:region[2], :
]
fake_impact_utils.to_tensor = lambda mask: torch.as_tensor(mask).float()
fake_impact_utils.tensor_paste = lambda *args, **kwargs: None
fake_impact_utils.empty_pil_tensor = lambda: torch.zeros(1, 1, 1, 3)

fake_comfy_extras = types.ModuleType("comfy_extras")
fake_comfy_extras.__path__ = []
fake_dd = types.ModuleType("comfy_extras.nodes_differential_diffusion")
fake_dd.DifferentialDiffusion = type(
    "DifferentialDiffusion",
    (),
    {"execute": lambda self, model: (model,)},
)
fake_comfy_extras.nodes_differential_diffusion = fake_dd

_fake_modules = {
    "comfy": fake_comfy,
    "comfy.samplers": fake_samplers,
    "comfy.sample": fake_sample,
    "comfy.model_management": fake_management,
    "comfy.utils": fake_utils,
    "nodes": fake_nodes,
    "donut_detailer_core": fake_detailer_core,
    "krea2_edit_integration": fake_krea,
    "impact": fake_impact,
    "impact.core": fake_impact_core,
    "impact.utils": fake_impact_utils,
    "impact.impact_sampling": fake_impact_sampling,
    "comfy_extras": fake_comfy_extras,
    "comfy_extras.nodes_differential_diffusion": fake_dd,
}
_missing = object()
_previous_modules = {name: sys.modules.get(name, _missing) for name in _fake_modules}
sys.modules.update(_fake_modules)
try:
    spec = importlib.util.spec_from_file_location(
        "donut_face_detailer_tested",
        Path(__file__).with_name("DonutFaceDetailer.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    for name, previous in _previous_modules.items():
        if previous is _missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class DonutFaceDetailerTests(unittest.TestCase):
    def setUp(self):
        self.original_prepare = module.prepare_krea2_edit
        self.original_sample = module.sample_and_decode
        self.original_dd = module.nodes_differential_diffusion.DifferentialDiffusion
        self.original_sam_and = module.core.segs_bitwise_and_mask
        self.original_enhance = module.DonutFaceDetailer.enhance_detail_megapixel

    def tearDown(self):
        module.prepare_krea2_edit = self.original_prepare
        module.sample_and_decode = self.original_sample
        module.nodes_differential_diffusion.DifferentialDiffusion = self.original_dd
        module.core.segs_bitwise_and_mask = self.original_sam_and
        module.DonutFaceDetailer.enhance_detail_megapixel = self.original_enhance

    def detail_kwargs(self, **overrides):
        values = {
            "image": torch.zeros(1, 32, 32, 3),
            "model": FakeModel("base"),
            "clip": object(),
            "vae": object(),
            "resolution": 1024 * 1024,
            "max_resolution": 0,
            "guide_size_for_bbox": False,
            "bbox": (0, 0, 32, 32),
            "seed": 7,
            "steps": 2,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": object(),
            "negative": object(),
            "denoise": 0.4,
            "noise_mask": torch.ones(32, 32),
            "force_inpaint": True,
            "noise_mask_feather": 8,
            "edit_mode": True,
            "edit_model": FakeModel("edit"),
            "face_reference_crop": torch.ones(1, 32, 32, 3),
        }
        values.update(overrides)
        return values

    def face_kwargs(self, detector, **overrides):
        values = {
            "image": torch.zeros(1, 64, 64, 3),
            "model": FakeModel("base"),
            "clip": object(),
            "vae": object(),
            "resolution": 1024 * 1024,
            "max_resolution": 0,
            "guide_size_for_bbox": False,
            "seed": 20,
            "steps": 2,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": object(),
            "negative": object(),
            "denoise": 0.4,
            "feather": 0,
            "noise_mask_enabled": False,
            "force_inpaint": True,
            "bbox_threshold": 0.2,
            "bbox_dilation": 0,
            "bbox_crop_factor": 1.0,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.9,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 1,
            "bbox_detector": detector,
            "max_faces": 2,
            "cycle": 1,
        }
        values.update(overrides)
        return values

    def test_differential_diffusion_patches_prepared_edit_model(self):
        prepared = FakeModel("prepared")
        patched = FakeModel("patched")
        calls = {"dd": [], "sample": []}

        module.prepare_krea2_edit = lambda *args, **kwargs: (
            prepared, object(), object(), {"samples": torch.zeros(1, 1, 1, 1)}, object(),
        )

        class FakeDifferentialDiffusion:
            def execute(self, model):
                calls["dd"].append(model)
                return (patched,)

        module.nodes_differential_diffusion.DifferentialDiffusion = FakeDifferentialDiffusion

        def sample(model, *args, **kwargs):
            calls["sample"].append(model)
            return torch.zeros(1, 32, 32, 3)

        module.sample_and_decode = sample
        module.DonutFaceDetailer.enhance_detail_megapixel(**self.detail_kwargs())

        self.assertEqual(calls["dd"], [prepared])
        self.assertEqual(calls["sample"], [patched])

    def test_existing_denoise_mask_function_skips_differential_diffusion(self):
        prepared = FakeModel("prepared", {"denoise_mask_function": object()})
        calls = []
        module.prepare_krea2_edit = lambda *args, **kwargs: (
            prepared, object(), object(), {"samples": torch.zeros(1, 1, 1, 1)}, object(),
        )

        class FakeDifferentialDiffusion:
            def execute(self, model):
                calls.append(model)
                return (model,)

        module.nodes_differential_diffusion.DifferentialDiffusion = FakeDifferentialDiffusion
        module.sample_and_decode = lambda *args, **kwargs: torch.zeros(1, 32, 32, 3)

        module.DonutFaceDetailer.enhance_detail_megapixel(**self.detail_kwargs())

        self.assertEqual(calls, [])

    def test_final_refinement_discards_empty_masks_then_applies_max_faces(self):
        large = FakeSegment("large", (0, 0, 50, 50))
        medium = FakeSegment("medium", (0, 0, 40, 40))
        small = FakeSegment("small", (0, 0, 30, 30))
        detector = FakeDetector([large, medium, small])
        sampled = []

        def refined(segs, mask):
            return (
                segs[0],
                [
                    FakeSegment("refined-large", large.bbox, mask_value=0),
                    FakeSegment("refined-medium", medium.bbox),
                    FakeSegment("refined-small", small.bbox),
                ],
            )

        module.core.segs_bitwise_and_mask = refined
        module.DonutFaceDetailer.enhance_detail_megapixel = staticmethod(
            lambda image, *args, **kwargs: (sampled.append(image.shape[2]) or image, None)
        )

        module.DonutFaceDetailer.enhance_face(**self.face_kwargs(
            detector, max_faces=1, sam_model_opt=object(),
        ))

        self.assertEqual(sampled, [40])
        self.assertTrue(np.any(large.cropped_mask))

    def test_segmentation_override_is_sorted_limited_and_detects_reference(self):
        reference_large = FakeSegment("reference-large", (0, 0, 45, 45))
        reference_small = FakeSegment("reference-small", (0, 0, 20, 20))
        detector = FakeDetector([], [reference_small, reference_large])
        target_small = FakeSegment("target-small", (0, 0, 15, 15))
        target_large = FakeSegment("target-large", (0, 0, 35, 35))
        target_medium = FakeSegment("target-medium", (0, 0, 25, 25))
        pairings = []

        class FakeSegmDetector:
            override_bbox_by_segm = True

            def detect(self, *args, **kwargs):
                return ((64, 64), [target_small, target_large, target_medium])

        def enhance(image, *args, **kwargs):
            face_reference_crop = args[-1]
            pairings.append((image.shape[2], face_reference_crop.shape[2]))
            return image, None

        module.DonutFaceDetailer.enhance_detail_megapixel = staticmethod(enhance)
        module.DonutFaceDetailer.enhance_face(**self.face_kwargs(
            detector,
            max_faces=2,
            segm_detector=FakeSegmDetector(),
            edit_mode=True,
            edit_model=FakeModel("edit"),
            face_reference=torch.ones(1, 64, 64, 3),
        ))

        self.assertEqual(detector.calls, ["target", "reference"])
        self.assertEqual(pairings, [(35, 45), (25, 20)])

    def test_face_seed_toggle_preserves_shared_default_and_can_increment(self):
        first = FakeSegment("first", (0, 0, 30, 30))
        second = FakeSegment("second", (0, 0, 20, 20))
        detector = FakeDetector([first, second])

        def run(vary, seed=20, cycle=2):
            seeds = []

            def enhance(image, model, clip, vae, resolution, max_resolution,
                        guide_size_for_bbox, bbox, seed, *args, **kwargs):
                seeds.append(seed)
                return image, None

            module.DonutFaceDetailer.enhance_detail_megapixel = staticmethod(enhance)
            module.DonutFaceDetailer.enhance_face(**self.face_kwargs(
                detector,
                seed=seed,
                cycle=cycle,
                vary_seed_per_face=vary,
            ))
            return seeds

        self.assertEqual(run(False), [20, 1020, 20, 1020])
        self.assertEqual(run(True), [20, 1020, 21, 1021])
        self.assertEqual(
            run(True, seed=0xffffffffffffffff, cycle=1),
            [0xffffffffffffffff, 0],
        )

    def test_seed_toggle_is_optional_and_defaults_to_shared(self):
        optional = module.DonutFaceDetailer.INPUT_TYPES()["optional"]
        names = list(optional)

        self.assertIn("vary_seed_per_face", optional)
        self.assertFalse(optional["vary_seed_per_face"][1]["default"])
        self.assertGreater(
            names.index("vary_seed_per_face"),
            names.index("grounding_px"),
        )


if __name__ == "__main__":
    unittest.main()

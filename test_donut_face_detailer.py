import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch
import torch.nn.functional as F


lifecycle_calls = []
sample_calls = []
resize_calls = []
encode_calls = []
wildcard_calls = []


class FakeModel:
    def __init__(self, name, model_options=None):
        self.name = name
        self.model_options = model_options or {}


class FakeVAE:
    def __init__(self):
        self.decode_calls = 0

    def decode(self, samples):
        self.decode_calls += 1
        return samples[:, :3].permute(0, 2, 3, 1).contiguous()


class FakeSegment:
    def __init__(self, name, bbox, mask_value=1.0):
        self.name = name
        self.bbox = bbox
        self.crop_region = bbox
        h = max(1, bbox[3] - bbox[1])
        w = max(1, bbox[2] - bbox[0])
        self.cropped_mask = np.full((h, w), mask_value, dtype=np.float32)
        self.cropped_image = torch.zeros(1, h, w, 3)
        self.confidence = 1.0
        self.label = "face"
        self.control_net_wrapper = None


class FakeDetector:
    def __init__(self, target_segments, reference_segments=None):
        self.target_segments = target_segments
        self.reference_segments = reference_segments or []
        self.aux = None
        self.calls = []

    def setAux(self, value):
        self.aux = value

    def detect(self, image, threshold, dilation, crop_factor, drop_size, detailer_hook=None):
        is_reference = bool(float(image.mean()) > 0.5)
        self.calls.append("reference" if is_reference else "target")
        segments = self.reference_segments if is_reference else self.target_segments
        return ((image.shape[1], image.shape[2]), list(segments))


fake_comfy = types.ModuleType("comfy")
fake_comfy.__path__ = []
fake_samplers = types.ModuleType("comfy.samplers")
fake_samplers.KSampler = types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
fake_sample = types.ModuleType("comfy.sample")
fake_management = types.ModuleType("comfy.model_management")
fake_management.unload_all_models = lambda: None
fake_utils_mod = types.ModuleType("comfy.utils")
fake_comfy.samplers = fake_samplers
fake_comfy.sample = fake_sample
fake_comfy.model_management = fake_management
fake_comfy.utils = fake_utils_mod

fake_nodes = types.ModuleType("nodes")
fake_nodes.MAX_RESOLUTION = 16384


class FakeInpaintModelConditioning:
    calls = []

    def encode(self, positive, negative, image, vae, mask=None, noise_mask=True):
        self.calls.append((image.shape, mask.shape, noise_mask))
        latent = fake_to_latent_image(image, vae)
        latent["inpaint"] = True
        return positive, negative, latent


class FakeConditioningConcat:
    def concat(self, a, b):
        return (list(a) + list(b),)


class FakeConditioningCombine:
    def combine(self, a, b):
        return (list(a) + list(b),)


fake_nodes.InpaintModelConditioning = FakeInpaintModelConditioning
fake_nodes.ConditioningConcat = FakeConditioningConcat
fake_nodes.ConditioningCombine = FakeConditioningCombine

fake_detailer_core = types.ModuleType("donut_detailer_core")
fake_detailer_core.offload_model_for_auxiliary_stage = (
    lambda model, management: lifecycle_calls.append(model)
)

fake_krea = types.ModuleType("krea2_edit_integration")
fake_krea.crop_image_padding = lambda image, padding: image
fake_krea.pad_image_to_multiple = lambda image: (image, (0, 0, 0, 0))
fake_krea.prepare_krea2_edit = lambda model, clip, vae, ref, prompt, negative, grounding, width, height: (
    FakeModel("prepared"), ["edit-positive"], ["edit-negative"],
    {"samples": torch.zeros(1, 4, 8, 8)}, ref,
)

fake_turbo = types.ModuleType("turbo_sampling")
fake_turbo.resolve_turbo_sampling = lambda steps, denoise, scheduler: (2, 0.25, 0.308)

fake_impact = types.ModuleType("impact")
fake_impact.__path__ = []
fake_impact_core = types.ModuleType("impact.core")
fake_impact_utils = types.ModuleType("impact.utils")
fake_impact_sampling = types.ModuleType("impact.impact_sampling")
fake_wildcards = types.ModuleType("impact.wildcards")
fake_impact.core = fake_impact_core
fake_impact.utils = fake_impact_utils
fake_impact.impact_sampling = fake_impact_sampling
fake_impact.wildcards = fake_wildcards


def fake_tensor_resize(image, width, height):
    resize_calls.append((image.shape[2], image.shape[1], width, height, image.shape[-1]))
    nchw = image.permute(0, 3, 1, 2).float()
    out = F.interpolate(nchw, size=(height, width), mode="bilinear", align_corners=False)
    return out.permute(0, 2, 3, 1).contiguous()


def fake_to_latent_image(image, vae):
    encode_calls.append(tuple(image.shape))
    nchw = image.permute(0, 3, 1, 2).float()
    if nchw.shape[1] < 4:
        extra = torch.zeros(nchw.shape[0], 4 - nchw.shape[1], nchw.shape[2], nchw.shape[3])
        nchw = torch.cat([nchw, extra], dim=1)
    return {"samples": nchw}


def fake_tensor_paste(dest, src, left_top, mask):
    x, y = left_top
    h, w = src.shape[1:3]
    m = mask
    if isinstance(m, np.ndarray):
        m = torch.from_numpy(m)
    if m.ndim == 2:
        m = m[None, ..., None]
    elif m.ndim == 3:
        m = m[..., None]
    if m.shape[1:3] != (h, w):
        m = fake_tensor_resize(m, w, h)
    dest[:, y:y+h, x:x+w] = dest[:, y:y+h, x:x+w] * (1 - m) + src * m


def fake_tensor_convert_rgba(image):
    if image.shape[-1] == 4:
        return image.clone()
    alpha = torch.ones(*image.shape[:-1], 1, dtype=image.dtype)
    return torch.cat([image[..., :3], alpha], dim=-1)


def fake_tensor_putalpha(image, mask):
    if mask.ndim == 3:
        mask = mask[..., None]
    image[..., 3:4] = mask[..., :1]


fake_impact_utils.tensor_resize = fake_tensor_resize
fake_impact_utils.to_latent_image = fake_to_latent_image
fake_impact_utils.crop_image = lambda image, region: image[:, region[1]:region[3], region[0]:region[2], :]
fake_impact_utils.to_tensor = lambda value: torch.as_tensor(value).float()
fake_impact_utils.tensor_gaussian_blur_mask = lambda mask, feather: (
    torch.as_tensor(mask).float()[None, ..., None]
    if torch.as_tensor(mask).ndim == 2 else torch.as_tensor(mask).float()
)
fake_impact_utils.tensor_paste = fake_tensor_paste
fake_impact_utils.tensor_convert_rgba = fake_tensor_convert_rgba
fake_impact_utils.tensor_putalpha = fake_tensor_putalpha
fake_impact_utils.empty_pil_tensor = lambda: torch.zeros(1, 1, 1, 3)

fake_impact_core.segs_to_combined_mask = lambda segs: torch.zeros(1, segs[0][0], segs[0][1])
fake_impact_core.make_sam_mask = lambda *args, **kwargs: object()
fake_impact_core.segs_bitwise_and_mask = lambda segs, mask: segs


def fake_ksampler_wrapper(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, **kwargs):
    sample_calls.append({
        "model": model, "seed": seed, "steps": steps,
        "positive": positive, "negative": negative, "latent": latent,
        "sampler_opt": kwargs.get("sampler_opt"), "noise": kwargs.get("noise"),
    })
    out = dict(latent)
    out["samples"] = latent["samples"] + 0.01
    return out


fake_impact_sampling.ksampler_wrapper = fake_ksampler_wrapper


class FakeChooser:
    def __init__(self, text):
        self.text = text

    def get(self, seg):
        return None, self.text


fake_wildcards.process_wildcard_for_segs = lambda text: (None, FakeChooser(text))


def fake_process_with_loras(text, model, clip):
    wildcard_calls.append((text, model))
    return FakeModel("wildcard-model", model.model_options), clip, ["wild-positive"]


fake_wildcards.process_with_loras = fake_process_with_loras

fake_comfy_extras = types.ModuleType("comfy_extras")
fake_comfy_extras.__path__ = []
fake_dd = types.ModuleType("comfy_extras.nodes_differential_diffusion")
fake_dd.DifferentialDiffusion = type(
    "DifferentialDiffusion", (), {"execute": lambda self, model: (model,)}
)
fake_comfy_extras.nodes_differential_diffusion = fake_dd

_modules = {
    "comfy": fake_comfy,
    "comfy.samplers": fake_samplers,
    "comfy.sample": fake_sample,
    "comfy.model_management": fake_management,
    "comfy.utils": fake_utils_mod,
    "nodes": fake_nodes,
    "donut_detailer_core": fake_detailer_core,
    "krea2_edit_integration": fake_krea,
    "turbo_sampling": fake_turbo,
    "impact": fake_impact,
    "impact.core": fake_impact_core,
    "impact.utils": fake_impact_utils,
    "impact.impact_sampling": fake_impact_sampling,
    "impact.wildcards": fake_wildcards,
    "comfy_extras": fake_comfy_extras,
    "comfy_extras.nodes_differential_diffusion": fake_dd,
}
_missing = object()
_previous = {name: sys.modules.get(name, _missing) for name in _modules}
sys.modules.update(_modules)
try:
    spec = importlib.util.spec_from_file_location(
        "donut_face_detailer_tested", Path(__file__).with_name("DonutFaceDetailer.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    for name, old in _previous.items():
        if old is _missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


class DonutFaceDetailerTests(unittest.TestCase):
    def setUp(self):
        lifecycle_calls.clear()
        sample_calls.clear()
        resize_calls.clear()
        encode_calls.clear()
        wildcard_calls.clear()
        FakeInpaintModelConditioning.calls.clear()
        self.original_enhance = module.DonutFaceDetailer.enhance_detail_megapixel
        self.original_dd = module.nodes_differential_diffusion.DifferentialDiffusion
        self.original_sam_and = module.core.segs_bitwise_and_mask
        self.original_prepare = module.prepare_krea2_edit

    def tearDown(self):
        module.DonutFaceDetailer.enhance_detail_megapixel = self.original_enhance
        module.nodes_differential_diffusion.DifferentialDiffusion = self.original_dd
        module.core.segs_bitwise_and_mask = self.original_sam_and
        module.prepare_krea2_edit = self.original_prepare

    def detail_kwargs(self, **overrides):
        values = {
            "image": torch.zeros(1, 300, 500, 3), "model": FakeModel("base"),
            "clip": object(), "vae": FakeVAE(), "resolution": 1024 * 1024,
            "max_resolution": 0, "guide_size_for_bbox": False,
            "bbox": (0, 0, 200, 200), "seed": 7, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["pos"],
            "negative": ["neg"], "denoise": 0.4, "noise_mask": None,
            "force_inpaint": True,
        }
        values.update(overrides)
        return values

    def face_kwargs(self, detector, **overrides):
        values = {
            "image": torch.zeros(1, 64, 64, 3), "model": FakeModel("base"),
            "clip": object(), "vae": FakeVAE(), "resolution": 1024 * 1024,
            "max_resolution": 0, "guide_size_for_bbox": False, "seed": 20,
            "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["pos"], "negative": ["neg"], "denoise": 0.4, "feather": 0,
            "noise_mask_enabled": False, "force_inpaint": True, "bbox_threshold": 0.2,
            "bbox_dilation": 0, "bbox_crop_factor": 1.0, "sam_detection_hint": "center-1",
            "sam_dilation": 0, "sam_threshold": 0.9, "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False",
            "drop_size": 1, "bbox_detector": detector, "max_faces": 2, "cycle": 1,
        }
        values.update(overrides)
        return values

    def test_sampling_canvas_is_64_aligned_and_cycles_stay_latent(self):
        vae = FakeVAE()
        result, _ = module.DonutFaceDetailer.enhance_detail_megapixel(
            **self.detail_kwargs(vae=vae, cycle=3))
        self.assertEqual(len(encode_calls), 1)
        self.assertEqual(len(sample_calls), 3)
        self.assertEqual(vae.decode_calls, 1)
        target_h, target_w = result.shape[1:3]
        self.assertEqual(target_w % 64, 0)
        self.assertEqual(target_h % 64, 0)
        self.assertEqual([call["seed"] for call in sample_calls], [7, 1007, 2007])

    def test_max_resolution_remains_aligned(self):
        result, _ = module.DonutFaceDetailer.enhance_detail_megapixel(
            **self.detail_kwargs(max_resolution=1024))
        h, w = result.shape[1:3]
        self.assertLessEqual(max(h, w), 1024)
        self.assertEqual(h % 64, 0)
        self.assertEqual(w % 64, 0)

    def test_inpaint_model_conditioning_is_used(self):
        module.DonutFaceDetailer.enhance_detail_megapixel(
            **self.detail_kwargs(noise_mask=torch.ones(300, 500),
                                 noise_mask_feather=0, inpaint_model=True))
        self.assertEqual(len(FakeInpaintModelConditioning.calls), 1)
        self.assertTrue(sample_calls[0]["latent"].get("inpaint"))

    def test_wildcard_conditioning_reaches_sampler(self):
        module.DonutFaceDetailer.enhance_detail_megapixel(
            **self.detail_kwargs(wildcard_opt="sharp portrait"))
        self.assertEqual(wildcard_calls[0][0], "sharp portrait")
        self.assertEqual(sample_calls[0]["model"].name, "wildcard-model")
        self.assertEqual(sample_calls[0]["positive"], ["wild-positive"])

    def test_differential_diffusion_patches_actual_edit_model(self):
        prepared, patched, dd_calls = FakeModel("prepared"), FakeModel("patched"), []
        module.prepare_krea2_edit = lambda *args, **kwargs: (
            prepared, ["edit-pos"], ["edit-neg"], {}, None)
        class DD:
            def execute(self, model):
                dd_calls.append(model)
                return (patched,)
        module.nodes_differential_diffusion.DifferentialDiffusion = DD
        module.DonutFaceDetailer.enhance_detail_megapixel(**self.detail_kwargs(
            image=torch.zeros(1, 64, 64, 3), bbox=(0, 0, 32, 32),
            noise_mask=torch.ones(64, 64), noise_mask_feather=8, edit_mode=True,
            edit_model=FakeModel("edit"), face_reference_crop=torch.ones(1, 64, 64, 3)))
        self.assertEqual(dd_calls, [prepared])
        self.assertIs(sample_calls[0]["model"], patched)

    def test_existing_denoise_mask_function_skips_differential_diffusion(self):
        calls = []
        class DD:
            def execute(self, model):
                calls.append(model)
                return (model,)
        module.nodes_differential_diffusion.DifferentialDiffusion = DD
        module.DonutFaceDetailer.enhance_detail_megapixel(**self.detail_kwargs(
            model=FakeModel("base", {"denoise_mask_function": object()}),
            noise_mask=torch.ones(300, 500), noise_mask_feather=8))
        self.assertEqual(calls, [])

    def test_progressive_overlapping_face_crops_use_current_image(self):
        detector = FakeDetector([
            FakeSegment("first", (0, 0, 32, 32)), FakeSegment("second", (0, 0, 32, 32))])
        means = []
        def enhance(image, *args, **kwargs):
            means.append(float(image.mean()))
            return image + 1.0, None
        module.DonutFaceDetailer.enhance_detail_megapixel = staticmethod(enhance)
        _, _, alpha, _, _ = module.DonutFaceDetailer.enhance_face(**self.face_kwargs(detector))
        self.assertEqual(means, [0.0, 1.0])
        self.assertEqual(alpha[0].shape[-1], 4)

    def test_final_refinement_removes_empty_masks_before_max_faces(self):
        large, medium, small = (FakeSegment("large", (0, 0, 50, 50)),
                                FakeSegment("medium", (0, 0, 40, 40)),
                                FakeSegment("small", (0, 0, 30, 30)))
        detector, sampled = FakeDetector([large, medium, small]), []
        def refined(segs, mask):
            return segs[0], [FakeSegment("large-empty", large.bbox, 0),
                             FakeSegment("medium", medium.bbox), FakeSegment("small", small.bbox)]
        module.core.segs_bitwise_and_mask = refined
        module.DonutFaceDetailer.enhance_detail_megapixel = staticmethod(
            lambda image, *args, **kwargs: (sampled.append(image.shape[2]) or image, None))
        module.DonutFaceDetailer.enhance_face(**self.face_kwargs(detector, max_faces=1, sam_model_opt=object()))
        self.assertEqual(sampled, [40])

    def test_cycles_are_passed_inside_once_and_seed_can_vary_per_face(self):
        detector = FakeDetector([FakeSegment("first", (0, 0, 30, 30)),
                                 FakeSegment("second", (32, 0, 52, 20))])
        def run(vary):
            seen = []
            def enhance(image, model, clip, vae, resolution, max_resolution,
                        guide_size_for_bbox, bbox, seed, *args, cycle=1, **kwargs):
                seen.append((seed, cycle))
                return image, None
            module.DonutFaceDetailer.enhance_detail_megapixel = staticmethod(enhance)
            module.DonutFaceDetailer.enhance_face(**self.face_kwargs(
                detector, cycle=3, vary_seed_per_face=vary))
            return seen
        self.assertEqual(run(False), [(20, 3), (20, 3)])
        self.assertEqual(run(True), [(20, 3), (21, 3)])

    def test_current_hook_lifecycle_is_used(self):
        events = []
        class Hook:
            def touch_scaled_size(self, w, h): events.append("touch_scaled_size"); return w, h
            def post_upscale(self, image, mask): events.append("post_upscale"); return image
            def post_encode(self, latent): events.append("post_encode"); return latent
            def get_skip_sampling(self): return False
            def get_custom_sampler(self): events.append("get_custom_sampler"); return "custom"
            def set_steps(self, step): events.append(("set_steps", step))
            def cycle_latent(self, latent): events.append("cycle_latent"); return latent
            def pre_ksample(self, model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise):
                events.append("pre_ksample")
                return model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise
            def get_custom_noise(self, seed, noise, is_touched=False): events.append("get_custom_noise"); return noise, True
            def pre_decode(self, latent): events.append("pre_decode"); return latent
            def post_decode(self, image): events.append("post_decode"); return image
        module.DonutFaceDetailer.enhance_detail_megapixel(
            **self.detail_kwargs(detailer_hook=Hook(), cycle=2))
        for name in ["touch_scaled_size", "post_upscale", "post_encode", "get_custom_sampler",
                     "cycle_latent", "pre_ksample", "get_custom_noise", "pre_decode", "post_decode"]:
            self.assertIn(name, events)
        self.assertEqual(sample_calls[0]["sampler_opt"], "custom")

    def test_detection_offloads_only_active_model(self):
        detector, model = FakeDetector([]), FakeModel("base")
        module.DonutFaceDetailer.enhance_face(**self.face_kwargs(detector, model=model))
        self.assertEqual(lifecycle_calls, [model])

    def test_turbo_mode_resolves_before_face_sampling(self):
        detector, seen = FakeDetector([FakeSegment("face", (0, 0, 32, 32))]), []
        def enhance(image, model, clip, vae, resolution, max_resolution,
                    guide_size_for_bbox, bbox, seed, steps, cfg, sampler_name,
                    scheduler, positive, negative, denoise, *args, **kwargs):
            seen.append((steps, denoise))
            return image, None
        module.DonutFaceDetailer.enhance_detail_megapixel = staticmethod(enhance)
        module.DonutFaceDetailer.enhance_face(**self.face_kwargs(
            detector, steps=8, denoise=0.2, turbo_mode=True))
        self.assertEqual(seen, [(2, 0.25)])

    def test_edit_reference_pairing_happens_after_final_face_selection(self):
        target_large, target_small = (FakeSegment("target-large", (0, 0, 35, 35)),
                                      FakeSegment("target-small", (40, 0, 60, 20)))
        ref_large, ref_small = (FakeSegment("ref-large", (0, 0, 45, 45)),
                                FakeSegment("ref-small", (48, 0, 64, 16)))
        detector, pairs = FakeDetector([target_small, target_large], [ref_small, ref_large]), []
        def enhance(image, *args, **kwargs):
            ref = kwargs.get("face_reference_crop")
            if ref is None and len(args) >= 27:
                ref = args[26]
            pairs.append((image.shape[2], ref.shape[2]))
            return image, None
        module.DonutFaceDetailer.enhance_detail_megapixel = staticmethod(enhance)
        module.DonutFaceDetailer.enhance_face(**self.face_kwargs(
            detector, edit_mode=True, edit_model=FakeModel("edit"),
            face_reference=torch.ones(1, 64, 64, 3)))
        self.assertEqual(detector.calls, ["target", "reference"])
        self.assertEqual(pairs, [(35, 45), (20, 16)])


if __name__ == "__main__":
    unittest.main()

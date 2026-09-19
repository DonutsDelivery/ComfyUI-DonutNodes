"""CPU contract tests; no model download, Comfy install, or GPU required.

Run directly: python tests/test_seedvr2_stage.py
Native nodes/loaders are stubbed; these are not image-quality/GPU acceptance tests.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StageTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(sys.modules)
        self.env.start()
        package = types.ModuleType("_seedvr2_test_pack")
        package.__path__ = [str(ROOT)]
        sys.modules[package.__name__] = package
        calls = self.legacy_calls = []
        class Base:
            FUNCTION = "upscale"
            @classmethod
            def INPUT_TYPES(cls):
                return {"required":{"image":("IMAGE",), "model":("MODEL",), "seed":("INT", {"default":0}),
                                    "rescale_factor":("FLOAT", {"default":2}), "resampling_method":(["lanczos"],)},
                        "optional":{"color_reference":("IMAGE",), "edit_source_image_b":("IMAGE",)}}
            def upscale(self, **kwargs):
                calls.append(kwargs)
                return ("legacy-image", "legacy-debug")
        base = types.ModuleType(package.__name__ + ".DonutTiledUpscale")
        base.NODE_CLASS_MAPPINGS = {"DonutTiledUpscale":Base}
        sys.modules[base.__name__] = base
        folders = types.ModuleType("folder_paths")
        folders.get_filename_list = lambda folder: []
        sys.modules["folder_paths"] = folders
        self.engine = load(package.__name__ + ".donut_seedvr2", ROOT / "donut_seedvr2.py")
        self.module = load(package.__name__ + ".donut_upscale_stage", ROOT / "donut_upscale_stage.py")
        self.stage = self.module.DonutTiledUpscaleStage()

    def tearDown(self):
        self.env.stop()

    def test_schema_preserves_old_widget_order_and_appends_engine(self):
        schema = self.stage.INPUT_TYPES()
        self.assertEqual(list(schema['optional'])[:3], ['color_reference', 'edit_source_image_b', 'enabled'])
        self.assertEqual(list(schema['optional'])[3], 'upscale_engine')
        self.assertEqual(schema['optional']['upscale_engine'][1]['default'], 'Donut')
        self.assertTrue(schema['required']['model'][1]['lazy'])

    def test_disabled_stage_is_identity_even_with_missing_models(self):
        image = object()
        self.assertEqual(self.stage.check_lazy_status(image, enabled=False, model=None), [])
        self.assertEqual(self.stage.run_stage(image, enabled=False, upscale_engine='SeedVR2'), (image, image))

    def test_legacy_path_filters_only_new_options(self):
        self.assertEqual(self.stage.run_stage('input', seed=9, model='krea', seedvr2_steps=3), ('legacy-image', 'legacy-debug'))
        self.assertEqual(self.legacy_calls, [dict(image='input', seed=9, model='krea')])

    def test_seedvr_lazy_path_does_not_request_krea_or_esrgan_inputs(self):
        self.assertEqual(self.stage.check_lazy_status('image', upscale_engine='SeedVR2',
            model=None, upscale_model=None, positive=None, vae=None, clip=None,
            edit_source_image_b=None, seed=None, seedvr2_model_name=None), ['seed', 'seedvr2_model_name'])

    def test_legacy_lazy_path_does_not_request_seedvr_inputs(self):
        self.assertEqual(self.stage.check_lazy_status('image', model=None, seedvr2_model_name=None), ['model'])

    def test_seedvr_route_preserves_two_image_output_and_drops_legacy_settings(self):
        with patch.object(self.engine, 'upscale', return_value='restored') as run:
            self.assertEqual(self.stage.run_stage('image', upscale_engine='SeedVR2', seed=42,
                cfg=7, model='krea', nag_enabled=True, seedvr2_steps=1), ('restored', 'restored'))
        self.assertEqual(run.call_args.kwargs, dict(seed=42, seedvr2_steps=1))
        self.assertEqual(self.legacy_calls, [])

    def test_unknown_engine_errors_instead_of_using_wrong_model(self):
        with self.assertRaises(ValueError):
            self.stage.run_stage('input', upscale_engine='typo')


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(sys.modules)
        self.env.start()
        self.engine = load('_seedvr2_recipe_test', ROOT / 'donut_seedvr2.py')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for name in ('dit', 'vae'):
            (root / name).write_bytes(b'fixture')
        folders = types.ModuleType('folder_paths')
        folders.get_full_path = lambda category, name: str(root / name) if (root / name).exists() else None
        sys.modules['folder_paths'] = folders
        self.events = events = []
        class Output:
            def __init__(self, *items): self.result = items
        class Pre:
            @classmethod
            def execute(cls, resized_images):
                events.append(('pre', tuple(resized_images.shape)))
                return Output(resized_images.unsqueeze(0))
        class Condition:
            @classmethod
            def execute(cls, model, vae_conditioning):
                events.append(('conditioning', model, vae_conditioning['samples'].ndim))
                return Output('positive', 'negative')
        class Post:
            @classmethod
            def execute(cls, images, original_resized_images, color_correction_method):
                events.append(('post', color_correction_method))
                return Output(images)
        class ModelLoader:
            def load_unet(self, name, dtype): events.append(('model', name, dtype)); return ('seed-model',)
        class VAELoader:
            def load_vae(self, name): events.append(('vae', name)); return ('seed-vae',)
        class Scale:
            def upscale(self, image, method, width, height, crop):
                events.append(('resize', method, width, height, crop))
                return (F.interpolate(image.movedim(-1, 1), (height, width), mode='nearest').movedim(1, -1),)
        class Encode:
            def encode(self, vae, pixels, tile, overlap, **kw):
                events.append(('encode', vae, tile, overlap, kw))
                return ({'samples':pixels},)
        class Decode:
            def decode(self, vae, samples, tile, overlap, **kw):
                events.append(('decode', vae, tile, overlap, kw))
                return (samples['samples'].reshape(-1, *samples['samples'].shape[-3:]),)
        class Sampler:
            def sample(self, model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise):
                events.append(('sample', model, seed, steps, cfg, sampler, scheduler, positive, negative, denoise))
                return (latent,)
        nodes = types.ModuleType('nodes')
        nodes.NODE_CLASS_MAPPINGS = dict(SeedVR2Preprocess=Pre, SeedVR2Conditioning=Condition, SeedVR2PostProcessing=Post)
        nodes.UNETLoader, nodes.VAELoader, nodes.ImageScale = ModelLoader, VAELoader, Scale
        nodes.VAEEncodeTiled, nodes.VAEDecodeTiled, nodes.KSampler = Encode, Decode, Sampler
        sys.modules['nodes'] = self.nodes = nodes
        self.owner = types.SimpleNamespace()
        self.options = dict(seedvr2_model_name='dit', seedvr2_vae_name='vae')

    def tearDown(self): self.env.stop()

    def run_engine(self, image, **kw):
        return self.engine.upscale(self.owner, image, **{**self.options, **kw})

    def test_independent_stills_keep_batch_order_and_do_not_become_video(self):
        image = torch.stack([torch.zeros(4, 6, 3), torch.ones(4, 6, 3)])
        result = self.run_engine(image, seed=42)
        self.assertEqual(tuple(result.shape), (2, 8, 12, 3))
        self.assertTrue(torch.equal(result[0], torch.zeros_like(result[0])))
        self.assertTrue(torch.equal(result[1], torch.ones_like(result[1])))
        self.assertEqual([e for e in self.events if e[0] == 'pre'], [('pre', (1, 8, 12, 3))] * 2)
        samples = [e for e in self.events if e[0] == 'sample']
        self.assertEqual([e[2] for e in samples], [42, 43])
        self.assertEqual(samples[0][3:], (1, 1.0, 'euler', 'simple', 'positive', 'negative', 1.0))

    def test_resource_pair_is_reused(self):
        image = torch.zeros(1, 4, 6, 3)
        self.run_engine(image); self.run_engine(image)
        self.assertEqual(sum(e[0] == 'model' for e in self.events), 1)
        self.assertEqual(sum(e[0] == 'vae' for e in self.events), 1)

    def test_even_dimensions_chosen_before_padding_and_alpha_preserved(self):
        image = torch.rand(1, 5, 7, 4)
        result = self.run_engine(image, rescale_factor=1.5)
        self.assertEqual(tuple(result.shape), (1, 8, 10, 4))
        self.assertEqual(self.events[2][2:4], (10, 8))

    def test_seed_wraps_uint64(self):
        self.run_engine(torch.zeros(2, 4, 6, 3), seed=2**64 - 1)
        self.assertEqual([e[2] for e in self.events if e[0] == 'sample'], [2**64 - 1, 0])

    def test_missing_native_nodes_fail_before_loading_any_weights(self):
        del self.nodes.NODE_CLASS_MAPPINGS['SeedVR2Conditioning']
        with self.assertRaisesRegex(RuntimeError, 'Update ComfyUI'):
            self.run_engine(torch.zeros(1, 4, 6, 3))
        self.assertEqual(self.events, [])

    def test_missing_model_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, 'models/diffusion_models'):
            self.run_engine(torch.zeros(1, 4, 6, 3), seedvr2_model_name='missing')
        self.assertEqual(self.events, [])

    def test_native_output_contracts(self):
        self.assertEqual(self.engine.outputs({'result':(1, 2)}), (1, 2))
        self.assertEqual(self.engine.outputs(types.SimpleNamespace(result=(3,))), (3,))
        with self.assertRaises(TypeError): self.engine.outputs('not a result')

    def test_rejects_bad_shapes_and_nonfinite_pixels(self):
        for image in (torch.zeros(0, 4, 6, 3), torch.zeros(1, 1, 6, 3), torch.zeros(4, 6, 3), torch.full((1, 4, 6, 3), float('nan'))):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError): self.run_engine(image)

    def test_rejects_invalid_options_before_model_load(self):
        for kw in (dict(seedvr2_denoise=float('nan')), dict(seedvr2_denoise=0), dict(seedvr2_steps=0),
                   dict(seedvr2_vae_tile_size=513), dict(rescale_factor=0.5), dict(seedvr2_color_correction='bad')):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.run_engine(torch.zeros(1, 4, 6, 3), **kw)
        self.assertEqual(self.events, [])


if __name__ == '__main__': unittest.main()

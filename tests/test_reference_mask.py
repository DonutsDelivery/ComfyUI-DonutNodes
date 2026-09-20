"""Reference-B mask tests with real CPU tensors/PIL and stubbed Comfy nodes.
Run directly: python tests/test_reference_mask.py. No GPU/model downloads.
"""
import asyncio
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from aiohttp import web
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(sys.modules); self.env.start()
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / 'birefnet.safetensors').write_bytes(b'fixture')
        package = types.ModuleType('_donut_mask_test'); package.__path__ = [str(ROOT)]
        sys.modules[package.__name__] = package
        folders = types.ModuleType('folder_paths')
        folders.get_user_directory = lambda: self.tmp.name
        folders.get_filename_list = lambda folder: ['birefnet.safetensors']
        folders.get_full_path = lambda folder, name: str(self.dir / name) if (self.dir / name).is_file() else None
        sys.modules['folder_paths'] = folders
        server = types.ModuleType('server')
        class Routes:
            def get(self, path): return lambda fn: fn
            def post(self, path): return lambda fn: fn
        server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=Routes()))
        sys.modules['server'] = server
        self.name = 'donutref:' + 'a' * 64
        self.source = Image.new('RGB', (8, 6), (230, 50, 10))
        self.sources = {self.name:self.source}
        crops = self.crops = []
        def crop(source, size, x, y):
            crops.append((source.copy(), size, x, y))
            return torch.from_numpy(np.asarray(source.resize(size)).astype(np.float32) / 255).unsqueeze(0)
        self.original = (object(), torch.ones(1, 4, 4, 3), True, 4, 4, 1088,
                         object(), 'prompt', object(), 'constant', 512, 1088)
        original = self.original
        class Base:
            @classmethod
            def INPUT_TYPES(cls): return {'required':{'enabled':('BOOLEAN', {'default':False})}, 'optional':{'grounding_end_px':('INT', {'default':1088})}}
            @classmethod
            def IS_CHANGED(cls, **kwargs): return 'original-cache-key'
            def check_lazy_status(self, enabled=False, model=None, **kwargs): return ['model'] if enabled and model is None else []
            def prepare(self, enabled=True, image_a='A', image_b='', use_reference_b=True, crop_b_x=.5, crop_b_y=.5, **unused):
                if not enabled or not use_reference_b: return (original[0], None, *original[2:])
                return original
        base = types.ModuleType(package.__name__ + '.DonutEditStudio')
        base.DonutEditStudio, base._crop_reference = Base, crop
        base._open_reference = lambda name: self.sources[name].copy()
        sys.modules[base.__name__] = base
        events = self.events = []
        class Output:
            def __init__(self, *items): self.result = items
        class Loader:
            @classmethod
            def execute(cls, bg_removal_name): events.append(('load', bg_removal_name)); return Output('biref-model')
        class Remove:
            @classmethod
            def execute(cls, bg_removal_model, image):
                events.append(('mask', bg_removal_model, tuple(image.shape)))
                mask = torch.zeros(image.shape[:3]); mask[:,1:-1,2:-2] = .5; mask[:,2:-2,3:-3] = 1
                return Output(mask)
        nodes = types.ModuleType('nodes'); nodes.NODE_CLASS_MAPPINGS = {'LoadBackgroundRemovalModel':Loader, 'RemoveBackground':Remove}
        sys.modules['nodes'] = self.nodes = nodes
        spec = importlib.util.spec_from_file_location(package.__name__ + '.donut_reference_mask', ROOT / 'donut_reference_mask.py')
        self.module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = self.module; spec.loader.exec_module(self.module)
        class StudioFixture(self.module.DonutSubjectMaskStudio):
            def prepare(self, **kwargs):
                values = dict(enabled=True, image_a='A', image_b='', use_reference_b=True,
                    prompt='', resolution_mode='Custom', aspect_ratio='1:1 Square',
                    megapixels=1., width=4, height=4, multiple=1, grounding_px=1088,
                    lora_name='None', lora_strength=1.)
                return super().prepare(**{**values, **kwargs})
        self.studio = StudioFixture()

    def tearDown(self): self.env.stop()

    def mask(self):
        mask = torch.zeros(1, 6, 8); mask[:,1:-1,2:-2] = .5; mask[:,2:-2,3:-3] = 1
        return mask

    def record(self): return self.module.store_mask(self.name, self.source, self.mask())

    def test_registration_places_mask_override_after_edit_studio(self):
        tree = ast.parse((ROOT / '__init__.py').read_text())
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in ('_NODE_MODULES', '_REQUIRED_OVERRIDES')}
        modules = values['_NODE_MODULES']
        self.assertLess(modules.index('DonutEditStudio'), modules.index('donut_reference_mask'))
        self.assertEqual(values['_REQUIRED_OVERRIDES']['donut_reference_mask'], ('DonutEditStudio',))

    def test_off_is_exact_existing_result_and_never_loads_model(self):
        self.assertIs(self.studio.prepare(image_b=self.name), self.original)
        self.assertEqual(self.events, [])

    def test_model_availability_uses_registered_background_removal_folder(self):
        self.assertTrue(self.module._model_available('birefnet.safetensors'))
        self.assertFalse(self.module._model_available('missing.safetensors'))

    def test_prompt_selection_rejects_empty_prompt_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'Enter a mask prompt'):
            self.module.prompt_mask(self.name, self.source, '  ')
        self.assertEqual(self.events, [])

    def test_prompt_selection_cache_depends_on_prompt_and_threshold(self):
        (self.dir / self.module.PROMPT_MODEL).write_bytes(b'model')
        calls = []
        class Loader:
            def load_checkpoint(self, name): return 'model', 'clip', None
        self.module.nodes.NODE_CLASS_MAPPINGS.update({'SAM3_Detect':object, 'CheckpointLoaderSimple':Loader})
        def native(node_id, **kwargs):
            if node_id == 'CLIPTextEncode': return kwargs['text']
            calls.append((kwargs['conditioning'], kwargs['threshold']))
            mask = torch.zeros(1, 6, 8)
            mask[:, :3 if kwargs['conditioning']=='hat' else 6, :4] = 1
            return mask
        with patch.object(self.module, '_native', side_effect=native):
            for prompt, threshold in [('hat',.5),('hat',.5),('shirt',.5),('shirt',.4)]:
                self.module.prompt_mask(self.name, self.source, prompt, threshold)
        self.assertEqual(calls, [('hat',.5),('shirt',.5),('shirt',.4)])

    def test_disabled_editing_or_unused_b_requires_nothing(self):
        for kwargs in (dict(enabled=False), dict(use_reference_b=False)):
            result = self.studio.prepare(image_b='missing', mask_b_mode='Auto subject', **kwargs)
            self.assertIsNone(result[1])
        self.assertEqual(self.events, [])

    def test_schema_only_appends_and_keeps_mask_independent(self):
        schema = self.studio.INPUT_TYPES()
        self.assertEqual(next(iter(schema['optional'])), 'grounding_end_px')
        self.assertEqual(schema['optional']['mask_b_mode'][1]['default'], 'Off')
        self.assertEqual(schema['optional']['mask_b'][0], 'MASK')

    def test_saved_mask_roundtrip_preserves_soft_alpha(self):
        record = self.record()
        restored = self.module.load_mask(json.loads(json.dumps(record)), self.name, self.source)
        self.assertTrue(torch.allclose(restored, self.mask(), atol=1/255))
        self.assertGreater(restored[0,1,2], 0); self.assertLess(restored[0,1,2], 1)

    def test_saved_mask_rejects_changed_image_pixels_even_with_same_reference_id(self):
        record = self.record()
        with self.assertRaisesRegex(ValueError, 'another version'):
            self.module.load_mask(record, self.name, Image.new('RGB', self.source.size, 'blue'))

    def test_saved_mask_rejects_other_reference_id(self):
        with self.assertRaisesRegex(ValueError, 'another version'):
            self.module.load_mask(self.record(), 'other', self.source)

    def test_mask_paths_reject_traversal(self):
        for token in ('../file', 'donutmask:../../file', 'donutmask:' + 'a'*64 + '/x', None):
            with self.subTest(token=token), self.assertRaises(ValueError): self.module._mask_path(token)

    def test_missing_saved_mask_is_actionable_and_never_falls_back_to_full_b(self):
        record = self.record(); self.module._mask_path(record['mask']).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Copy user/donut/edit_subject_masks'):
            self.studio.prepare(image_b=self.name, mask_b_mode='Saved mask', mask_b_data=json.dumps(record))

    def test_modified_mask_is_rejected(self):
        record = self.record(); self.module._mask_path(record['mask']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.module.load_mask(record, self.name, self.source)

    def test_composite_preserves_foreground_and_replaces_background(self):
        result = np.asarray(self.module.composite(self.source, self.mask()))
        self.assertTrue(np.array_equal(result[2,3], np.asarray(self.source)[2,3]))
        self.assertTrue(np.array_equal(result[0,0], [128,128,128]))
        self.assertFalse(np.array_equal(result[1,2], [128,128,128]))

    def test_composite_grow_shrink_and_feather(self):
        original = np.asarray(self.module.composite(self.source, self.mask(), background='Black'))
        grown = np.asarray(self.module.composite(self.source, self.mask(), grow=1, background='Black'))
        shrunken = np.asarray(self.module.composite(self.source, self.mask(), grow=-1, background='Black'))
        feathered = np.asarray(self.module.composite(self.source, self.mask(), feather=1, background='Black'))
        self.assertGreater(grown.sum(), original.sum()); self.assertLess(shrunken.sum(), original.sum())
        self.assertFalse(np.array_equal(feathered, original))

    def test_prepare_changes_only_b_and_neutralizes_before_existing_crop(self):
        record = self.record()
        result = self.studio.prepare(image_b=self.name, crop_b_x=.7, crop_b_y=.2,
            mask_b_mode='Saved mask', mask_b_data=json.dumps(record))['result']
        self.assertIs(result[0], self.original[0]); self.assertIs(result[8], self.original[8])
        self.assertEqual(result[2:], self.original[2:])
        source, size, x, y = self.crops[-1]
        self.assertEqual((size,x,y), ((4,4),.7,.2))
        self.assertEqual(source.size, self.source.size)
        self.assertEqual(source.getpixel((0,0)), (128,128,128))
        self.assertEqual(self.source.getpixel((0,0)), (230,50,10))

    def test_external_mask_uses_original_b_coordinates_not_a_or_output(self):
        result = self.studio.prepare(image_b=self.name, mask_b_mode='External mask', mask_b=self.mask())
        self.assertEqual(tuple(result['result'][1].shape), (1,4,4,3)); self.assertEqual(self.events, [])
        with self.assertRaisesRegex(ValueError, 'ORIGINAL'):
            self.studio.prepare(image_b=self.name, mask_b_mode='External mask', mask_b=torch.ones(1,4,4))

    def test_lazy_external_input_is_requested_only_when_selected(self):
        self.assertEqual(self.studio.check_lazy_status(enabled=True, model='model', use_reference_b=True, mask_b_mode='External mask', mask_b=None), ['mask_b'])
        self.assertEqual(self.studio.check_lazy_status(enabled=True, model='model', use_reference_b=True, mask_b_mode='Off', mask_b=None), [])
        self.assertEqual(self.studio.check_lazy_status(enabled=False, mask_b=None), [])

    def test_bad_or_empty_masks_are_rejected(self):
        for mask in (torch.zeros(1,6,8), torch.full((1,6,8), float('nan')), torch.full((1,6,8), 2.), torch.ones(2,6,8)):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError): self.module.validate_mask(mask, self.source.size)

    def test_native_auto_uses_full_source_and_disk_cache_not_gpu_tensor_cache(self):
        first, _ = self.module.auto_mask(self.name, self.source)
        second, _ = self.module.auto_mask(self.name, self.source)
        self.assertEqual(first, second)
        self.assertEqual(self.events, [('load','birefnet.safetensors'), ('mask','biref-model',(1,6,8,3))])
        self.module.auto_mask(self.name, self.source, force=True)
        self.assertEqual(len(self.events), 4)

    def test_missing_native_nodes_only_block_auto(self):
        self.nodes.NODE_CLASS_MAPPINGS.clear()
        with self.assertRaisesRegex(RuntimeError, 'Update ComfyUI'):
            self.module.auto_mask(self.name, self.source)
        result = self.studio.prepare(image_b=self.name, mask_b_mode='External mask', mask_b=self.mask())
        self.assertIn('result', result)

    def test_preview_is_queue_output_job_not_generation(self):
        preview = self.module.DonutSubjectMaskPreview()
        self.assertTrue(preview.OUTPUT_NODE)
        result = preview.preview(self.name, 'birefnet.safetensors', 'unique-request')
        self.assertEqual(result['result'], ())
        self.assertEqual(result['ui']['donut_subject_mask'][0]['image'], self.name)
        self.assertEqual(self.crops, [])

    def test_upload_route_is_cpu_only_and_binds_mask_to_source(self):
        encoded=io.BytesIO(); Image.fromarray((self.mask()[0].numpy()*255).astype(np.uint8)).save(encoded,format='PNG'); encoded.seek(0)
        file=web.FileField(name='mask',filename='mask.png',file=encoded,content_type='image/png',headers={})
        class Request:
            async def post(inner): return {'reference':self.name,'mask':file}
        response=asyncio.run(self.module.upload_subject_mask(Request()))
        record=json.loads(response.text)
        self.assertEqual(record['source'], self.module.fingerprint(self.source)); self.assertEqual(self.events, [])

    def test_upload_route_refuses_arbitrary_file_references(self):
        class Request:
            async def post(inner): return {'reference':'../../private','mask':None}
        with self.assertRaises(web.HTTPBadRequest): asyncio.run(self.module.upload_subject_mask(Request()))


if __name__ == '__main__': unittest.main()

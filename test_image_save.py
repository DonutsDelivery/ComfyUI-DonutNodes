"""Run with ComfyUI on PYTHONPATH and its Python environment."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from PIL import Image
import DonutImageSave as save_module
from DonutImageSave import DonutImageSave, args


class SaveTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch('folder_paths.get_output_directory', return_value=str(self.root / 'output')),
                        patch('folder_paths.get_temp_directory', return_value=str(self.root / 'temp')),
                        patch('folder_paths.get_user_directory', return_value=str(self.root / 'user')),
                        patch.object(args, 'disable_metadata', False)]
        for p in self.patches:
            p.start()
        self.node = DonutImageSave()
        self.images = torch.rand(2, 12, 16, 3)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()


class ImageSaveTests(SaveTestCase):
    def test_webp_numbering_batch_and_repeat(self):
        options = dict(filename_prefix='Final/453968006882751', filename_delimiter='',
                       filename_number_padding=1, extension='webp', quality=100)
        a = self.node.save_images(self.images, **options)
        b = self.node.save_images(self.images[:1], **options)
        self.assertEqual([Path(p).name for p in a['result'][1]], ['4539680068827511.webp', '4539680068827512.webp'])
        self.assertEqual(Path(b['result'][1][0]).name, '4539680068827513.webp')
        for p in a['result'][1] + b['result'][1]:
            with Image.open(p) as im:
                im.load()
                self.assertEqual(im.size, (16, 12))
        self.assertEqual(a['ui']['images'][0]['subfolder'], 'Final')

    def test_prefix_number_and_overwrite(self):
        out = self.node.save_images(self.images, filename_prefix='x', filename_number_start=True,
                                    filename_number_padding=3, extension='png')
        self.assertEqual([Path(p).name for p in out['result'][1]], ['001_x.png', '002_x.png'])
        out = self.node.save_images(self.images[:1], filename_prefix='x', overwrite_mode=True)
        again = self.node.save_images(self.images[:1] * 0, filename_prefix='x', overwrite_mode=True)
        self.assertEqual(out['result'][1], again['result'][1])
        with Image.open(again['result'][1][0]) as im:
            self.assertEqual(im.getpixel((0, 0)), (0, 0, 0))

    def test_metadata_roundtrip_png_and_webp(self):
        for ext in ('png', 'webp'):
            out = self.node.save_images(self.images[:1], extension=ext, prompt={'test': 1},
                                       extra_pnginfo={'workflow': {'nodes': []}})
            with Image.open(out['result'][1][0]) as im:
                if ext == 'png':
                    self.assertEqual(json.loads(im.info['workflow']), {'nodes': []})
                    self.assertEqual(json.loads(im.info['prompt']), {'test': 1})
                else:
                    self.assertEqual(im.getexif()[0x0110], 'prompt:{"test": 1}')
                    self.assertEqual(im.getexif()[0x010F], 'workflow:{"nodes": []}')
            with patch.object(args, 'disable_metadata', True):
                out = self.node.save_images(self.images[:1], extension=ext, prompt={'secret': 1})
            with Image.open(out['result'][1][0]) as im:
                self.assertNotIn('prompt', im.info)
                self.assertNotIn(0x0110, im.getexif())

    def test_temp_previews_and_formats(self):
        for ext in ('png', 'jpeg', 'jpg', 'gif', 'bmp', 'tiff', 'webp'):
            out = self.node.save_images(self.images[:1], root='temp', extension=ext)
            self.assertEqual(out['ui']['images'][0]['type'], 'temp')
            with Image.open(out['result'][1][0]) as im:
                im.load()
        out = self.node.save_images(self.images[:1], show_previews=False)
        self.assertEqual(out['ui']['images'], [])
        self.assertTrue(Path(out['result'][1][0]).is_file())

    def test_paths_cannot_escape_output(self):
        for options in ({'filename_prefix': '../escape'}, {'filename_prefix': str(self.root / 'escape')},
                        {'filename_delimiter': '/../'}, {'extension': '../png'}, {'root': '../output'}):
            with self.assertRaises(ValueError):
                self.node.save_images(self.images[:1], **options)
        output = self.root / 'output'
        output.mkdir(exist_ok=True)
        (output / 'outside').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.node.save_images(self.images[:1], filename_prefix='outside/escape')

    def test_write_errors_are_not_reported_as_success(self):
        with patch.object(Image.Image, 'save', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.node.save_images(self.images[:1])
        self.assertEqual(list((self.root / 'output').iterdir()), [])


    def test_percent_variables_expand_without_listing_the_folder(self):
        with patch('folder_paths.get_save_image_path', side_effect=AssertionError('lists the folder')):
            out = self.node.save_images(self.images[:1], filename_prefix='%width%x%height%/img')
        self.assertEqual(Path(out['result'][1][0]).relative_to(self.root / 'output').as_posix(), '16x12/img_0001.png')


class BackgroundSaveTests(SaveTestCase):
    """Saves nothing consumes are written after the node returns."""
    def drain(self):
        save_module._writer.submit(lambda: None).result(timeout=30)

    def save(self, prompt, **options):
        return self.node.save_images(self.images, filename_prefix='Final/img', extension='webp',
                                     prompt=prompt, unique_id='64', **options)

    def test_node_returns_before_a_blocked_disk_write(self):
        release = threading.Event()
        original = save_module._write_numbered

        def slow(data, job):
            release.wait(30)
            return original(data, job)

        with patch.object(save_module, '_write_numbered', slow):
            out = self.save({'64': {'inputs': {}}})
            self.assertEqual(out['result'][1], [])
            self.assertFalse((self.root / 'output' / 'Final').exists())
            previews = out['ui']['images']
            self.assertEqual([p['type'] for p in previews], ['temp', 'temp'])
            for preview in previews:
                self.assertTrue((self.root / 'temp' / preview['filename']).is_file())
            release.set()
            self.drain()
        names = sorted(p.name for p in (self.root / 'output' / 'Final').iterdir())
        self.assertEqual(names, ['img_0001.webp', 'img_0002.webp'])
        self.assertEqual(list((self.root / 'user' / save_module.STAGING_NAME).iterdir()), [])

    def test_consumed_files_output_saves_before_returning(self):
        out = self.save({'64': {'inputs': {}}, '9': {'inputs': {'paths': ['64', 1]}}})
        self.assertEqual([Path(p).name for p in out['result'][1]], ['img_0001.webp', 'img_0002.webp'])
        self.assertEqual(out['ui']['images'][0]['type'], 'output')

    def test_staged_save_left_by_a_restart_is_delivered(self):
        with patch.object(save_module._writer, 'submit'):
            self.save({'64': {'inputs': {}}}, show_previews=False)
        staged = sorted((self.root / 'user' / save_module.STAGING_NAME).glob('*.json'))
        self.assertEqual(len(staged), 2)
        save_module._resume_staged_saves()
        self.drain()
        self.assertEqual(sorted(p.name for p in (self.root / 'output' / 'Final').iterdir()),
                         ['img_0001.webp', 'img_0002.webp'])

    def test_failed_background_write_keeps_the_staged_image(self):
        with patch.object(save_module, '_write_numbered', side_effect=OSError('disk full')), \
                self.assertLogs(level='ERROR'):
            self.save({'64': {'inputs': {}}})
            self.drain()
        self.assertEqual(len(list((self.root / 'user' / save_module.STAGING_NAME).glob('*.data'))), 2)


if __name__ == '__main__':
    unittest.main()

"""Run with ComfyUI on PYTHONPATH and its Python environment."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from DonutImageSave import DonutImageSave, args


class ImageSaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch('folder_paths.get_output_directory', return_value=str(self.root / 'output')),
                        patch('folder_paths.get_temp_directory', return_value=str(self.root / 'temp')),
                        patch.object(args, 'disable_metadata', False)]
        for p in self.patches:
            p.start()
        self.node = DonutImageSave()
        self.images = torch.rand(2, 12, 16, 3)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

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


if __name__ == '__main__':
    unittest.main()

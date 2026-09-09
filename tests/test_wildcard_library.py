import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PKG = '_donut_wildcard_test'
package = types.ModuleType(PKG); package.__path__ = [str(ROOT)]
sys.modules[PKG] = package
fake_server = types.ModuleType('server')
fake_server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=types.SimpleNamespace(
    get=lambda route: lambda fn: fn, post=lambda route: lambda fn: fn)))
with patch.dict(sys.modules, {'server':fake_server}):
    spec = importlib.util.spec_from_file_location(PKG + '.donut_wildcards', ROOT / 'donut_wildcards.py')
    library = importlib.util.module_from_spec(spec); spec.loader.exec_module(library)

class WildcardFilesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'wildcards'
        p = patch.object(library.folder_paths, 'get_user_directory', return_value=self.directory.name);p.start();self.addCleanup(p.stop)
        p = patch.object(library, 'wildcard_roots', return_value=(self.root,));p.start();self.addCleanup(p.stop)

    def test_save_reload_and_edit_change_expansion(self):
        path = library.save_wildcard('haircolor', 'red hair\n\nblonde hair\n')
        self.assertEqual(path.read_text(), 'red hair\nblonde hair\n')
        self.assertEqual(library.list_wildcards()['haircolor'], path)
        self.assertEqual(library.expand_text('haircolor*', 1, roots=[self.root]), 'blonde hair')
        library.save_wildcard('haircolor', 'blue hair\ngreen hair')
        self.assertEqual(library.expand_text('haircolor*', 1, roots=[self.root]), 'green hair')
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_nested_names_and_path_containment(self):
        self.assertTrue(library.save_wildcard('clothes/shirt', 'linen').is_file())
        for name in ('../outside','a/../../outside','/absolute','bad.name','a//b'):
            with self.assertRaises(ValueError): library.save_wildcard(name, 'x')
        outside = Path(self.directory.name) / 'outside'; outside.mkdir()
        (self.root / 'escape').symlink_to(outside)
        with self.assertRaises(ValueError): library.save_wildcard('escape/file', 'x')

    def test_reject_empty_without_overwriting_saved_file(self):
        path = library.save_wildcard('haircolor', 'red')
        with self.assertRaises(ValueError): library.save_wildcard('haircolor', ' \n')
        self.assertEqual(path.read_text(), 'red\n')

if __name__ == '__main__': unittest.main()

"""Explicit adapters must retain their upstream positional and default contracts."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def signature(file, owner, method):
    tree = ast.parse((ROOT / file).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    args = fn.args
    return ([a.arg for a in args.posonlyargs], [a.arg for a in args.args],
            [ast.dump(d) for d in args.defaults])


class WrapperSignatureTests(unittest.TestCase):
    def test_edit_wrappers_preserve_positional_arguments_and_defaults(self):
        expected = signature('DonutEditStudio.py', 'DonutEditStudio', 'prepare')
        for file, owner in [('donut_reference_mask.py', 'DonutSubjectMaskStudio'),
                            ('donut_crop_studio.py', 'DonutCropEditStudio')]:
            with self.subTest(owner=owner):
                self.assertEqual(signature(file, owner, 'prepare'), expected)

    def test_grounding_preserves_sampler_and_engine_contracts(self):
        for method, file, owner in [('sample', 'donut_krea2_sda.py', 'DonutSampler'),
                                    ('sample', 'donut_txtfusion_guard_sampler.py', 'DonutSampler'),
                                    ('run_simple', 'DonutKSamplerCFGLinear.py', '_DonutSamplerEngine'),
                                    ('run_advanced', 'DonutKSamplerCFGLinear.py', '_DonutSamplerEngine')]:
            with self.subTest(method=method):
                self.assertEqual(signature('donut_grounding_schedule.py', 'DonutSampler', method),
                                 signature(file, owner, method))


if __name__ == '__main__':
    unittest.main()

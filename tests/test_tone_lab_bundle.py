"""Bundled real-checkpoint and shipped HTML conformance; CPU, synthetic pixels.

Run: python tests/test_tone_lab_bundle.py -v
These checks do not evaluate aesthetics or live ComfyUI panel serialization.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import donut_tone_engine as engine

NAME = "donut-tone-v4-r12.json"
CHECKPOINT = ROOT / "models/donut_tone" / NAME
TRAINER = ROOT / "tools/donut_tone_lab_feature_learner_v4.html"
CHECKPOINT_SHA = "5542880ab9b627d828e2849a86818f2ccc100afd72da19cf98636abef38f4f28"
TRAINER_GIT_SHA = "81d3af15976e0c4139a39d2f301c1fd0f98e899e"


def load_node(root):
    paths = types.ModuleType("folder_paths")
    paths.models_dir = str(root)
    paths.folder_names_and_paths = {}
    def add(key, path):
        values, extensions = paths.folder_names_and_paths.setdefault(key, ([], set()))
        if path not in values:
            values.append(path)
    paths.add_model_folder_path = add
    paths.get_folder_paths = lambda key: paths.folder_names_and_paths[key][0]
    paths.get_filename_list = lambda key: sorted({
        f.relative_to(p).as_posix()
        for p in map(Path, paths.get_folder_paths(key)) for f in p.rglob("*.json")
    })
    package = types.ModuleType("_tone_bundle_test")
    package.__path__ = [str(ROOT)]
    spec = importlib.util.spec_from_file_location("_tone_bundle_test.DonutToneLab", ROOT / "DonutToneLab.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"folder_paths": paths, "_tone_bundle_test": package}):
        spec.loader.exec_module(module)
    return module, paths


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.node, self.paths = load_node(self.root)
        self.document = json.loads(CHECKPOINT.read_bytes())

    def tearDown(self):
        self.temp.cleanup()

    def test_checkpoint_integrity_and_contract(self):
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), CHECKPOINT_SHA)
        model = engine.validate_export(self.document)
        self.assertEqual(model["revision"], 12)
        self.assertEqual(sum(w.size + b.size for w, b in model["layers"]), 2883)
        self.assertEqual(self.document["model"]["training"]["imageCount"], 395)
        self.assertEqual(self.document["model"]["training"]["validationCount"], 84)
        self.assertEqual(set(self.document), {"version", "type", "schema", "algorithm", "analysis", "model"})
        self.assertNotIn("records", self.document["model"])
        self.assertNotIn("history", self.document["model"])

    def test_shipped_trainer_integrity(self):
        raw = TRAINER.read_bytes()
        self.assertEqual(hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(), TRAINER_GIT_SHA)

    def test_bundle_discovered_without_copying_or_changing_defaults(self):
        inputs = self.node.DonutToneLab.INPUT_TYPES()["required"]
        self.assertIn(NAME, inputs["model_name"][0])
        self.assertEqual(inputs["model_name"][1]["default"], "None")
        self.assertFalse(inputs["enabled"][1]["default"])
        self.assertFalse(inputs["apply_to_edits"][1]["default"])
        self.assertEqual(self.node._model_path(NAME), CHECKPOINT.resolve())
        self.assertEqual(list(self.root.rglob("*.json")), [])

    def test_extra_paths_preserved_and_registration_idempotent(self):
        extra = self.root / "extra"
        self.paths.folder_names_and_paths["donut_tone"] = ([str(extra)], {".custom"})
        for _ in range(5):
            self.node._register_folder()
        roots, extensions = self.paths.folder_names_and_paths["donut_tone"]
        self.assertEqual(roots, [str(extra), str(CHECKPOINT.parent)])
        self.assertEqual(extensions, {".custom", ".json"})

    def test_explicit_user_checkpoint_has_priority(self):
        user = self.root / "donut_tone" / NAME
        user.parent.mkdir()
        user.write_bytes(CHECKPOINT.read_bytes())
        self.assertEqual(self.node._model_path(NAME), user.resolve())
        before = self.node.DonutToneLab.IS_CHANGED(enabled=True, model_name=NAME)
        self.document["model"]["layers"][-1]["b"][0] += .01
        user.write_text(json.dumps(self.document), encoding="utf-8")
        after = self.node.DonutToneLab.IS_CHANGED(enabled=True, model_name=NAME)
        self.assertNotEqual(before, after)
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), CHECKPOINT_SHA)

    def test_bypass_never_loads_weights(self):
        image = torch.ones(1, 12, 16, 3)
        with patch.object(self.node, "_read_model", side_effect=AssertionError("Unexpected read")):
            for opts in [{}, {"enabled": True, "strength": 0}, {"enabled": True, "edit_mode": True}]:
                result = self.node.DonutToneLab().apply(image, model_name=NAME, **opts)
                self.assertIs(result["result"][0], image)

    def test_real_checkpoint_batch_curve_alpha_and_dtype(self):
        image = torch.linspace(.01, .99, 2 * 24 * 32 * 3).reshape(2, 24, 32, 3)
        image = torch.cat((image, torch.ones_like(image[..., :1])), -1)
        for channels in [3, 4]:
            for dtype in [torch.float32, torch.float16]:
                source = image[..., :channels].to(dtype)
                before = source.clone()
                result = self.node.DonutToneLab().apply(source, True, NAME)
                output, text = result["result"]
                report = json.loads(text)
                self.assertEqual(report["sha256"], CHECKPOINT_SHA)
                self.assertEqual(report["revision"], 12)
                self.assertEqual(len(report["frames"]), 2)
                self.assertEqual(output.dtype, source.dtype)
                self.assertEqual(output.device, source.device)
                self.assertTrue(torch.equal(before, source))
                if channels == 4:
                    self.assertTrue(torch.equal(output[..., 3], source[..., 3]))
                for i, row in enumerate(report["frames"]):
                    expected = (source[i, ..., :3].float().pow(row["effectiveGamma"]) * row["effectiveGain"]).clamp(0, 1).to(dtype)
                    torch.testing.assert_close(output[i, ..., :3], expected)

    def test_path_escape_still_rejected(self):
        for name in ["../" + NAME, "/tmp/" + NAME, "C:/" + NAME, "x\\" + NAME, "model.py"]:
            with self.assertRaises(ValueError):
                self.node._model_path(name)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for HTML conformance")
    def test_real_checkpoint_matches_shipped_html_on_18_proxies(self):
        html = TRAINER.read_text(encoding="utf-8")
        source = html[html.index("function buildFeatureEngine(){"):html.index("const E=buildFeatureEngine();")]
        source += """
const E=buildFeatureEngine(), fs=require('fs');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(input.fixtures.map(f=>{
  const s=E.analyze(f.pixels,f.w,f.h);
  return {features:s.features,p:E.predict(s.features,input.model)};
})));
"""
        rng = np.random.default_rng(240922)
        fixtures = []
        for h, w in [(1, 1), (1, 32), (32, 1), (3, 5), (37, 61), (128, 256)]:
            a = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
            a[..., 3] = 255
            fixtures.append(a)
        for level in [0, 1, 64, 128, 192, 254, 255]:
            a = np.full((19, 23, 4), level, dtype=np.uint8)
            a[..., 3] = 255
            fixtures.append(a)
        for alpha in [0, 249, 250, 254, 255]:
            a = rng.integers(0, 256, (17, 13, 4), dtype=np.uint8)
            a[..., 3] = alpha
            fixtures.append(a)
        payload = {"model": self.document["model"], "fixtures": [
            {"w": a.shape[1], "h": a.shape[0], "pixels": a.ravel().tolist()} for a in fixtures]}
        process = subprocess.run(["node", "-e", source], input=json.dumps(payload),
                                 text=True, capture_output=True, check=True, timeout=30)
        references = json.loads(process.stdout)
        model = engine.validate_export(self.document)
        for a, ref in zip(fixtures, references):
            features = engine.analyze_rgba(a)["features"]
            np.testing.assert_allclose(features, ref["features"], atol=1e-11, rtol=1e-11)
            actual = engine.predict(features, model)
            self.assertEqual(actual["noop"], ref["p"]["noop"])
            for k in ["gamma", "gain", "rawGamma", "rawGain", "noOpScore", "coverage"]:
                self.assertAlmostEqual(actual[k], ref["p"][k], places=10)


if __name__ == "__main__":
    unittest.main()

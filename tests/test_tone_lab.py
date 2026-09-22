"""CPU tests; synthetic fixtures only. Run: python tests/test_tone_lab.py -v."""
import copy
import importlib.util
import json
import math
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


def fixture_model(gamma=1.35, gain=1.04, *, varying=False):
    m = {"format": engine.ALGORITHM, "schema": engine.SCHEMA,
         "featureCount": 169, "names": list(engine.FEATURE_NAMES), "trained": True,
         "revision": 1, "mu": [.3] * 169, "sd": [.2] * 169, "layers": []}
    for i, (ni, no) in enumerate(engine.SHAPES):
        weights = [math.sin(k * .31 + i) * .04 if varying else 0 for k in range(ni * no)]
        m["layers"].append({"ni": ni, "no": no, "w": weights, "b": [0] * no})
    m["layers"][-1]["b"] = [math.atanh(math.log(gamma) / math.log(3)),
                               math.atanh(math.log(gain) / math.log(1.25)), -2]
    return {"version": 4, "type": "donut-tone-model", "schema": engine.SCHEMA,
            "algorithm": engine.ALGORITHM,
            "analysis": {"longEdge": 256, "colorSpace": "srgb", "opaqueAlphaMinimum": 250}, "model": m}


def load_node(model_dir):
    paths = types.ModuleType("folder_paths")
    paths.models_dir = str(model_dir)
    paths.folder_names_and_paths = {}
    paths.add_model_folder_path = lambda key, path: paths.folder_names_and_paths.setdefault(key, ([path], set()))
    paths.get_folder_paths = lambda key: paths.folder_names_and_paths[key][0]
    paths.get_filename_list = lambda key: [p.name for p in (model_dir / key).glob("*.json")]
    package = types.ModuleType("_tone_test_pack")
    package.__path__ = [str(ROOT)]
    spec = importlib.util.spec_from_file_location("_tone_test_pack.DonutToneLab", ROOT / "DonutToneLab.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"folder_paths": paths, "_tone_test_pack": package}):
        spec.loader.exec_module(module)
    return module


class EngineTests(unittest.TestCase):
    def test_schema_and_empty_alpha(self):
        stats = engine.analyze_rgba(np.zeros((3, 5, 4), dtype=np.uint8))
        self.assertEqual(stats["sampleCount"], 0)
        self.assertEqual(len(stats["features"]), 169)
        self.assertTrue(np.isfinite(stats["features"]).all())
        self.assertEqual(tuple(stats["names"]), engine.FEATURE_NAMES)

    def test_invalid_pixels(self):
        for a in [np.zeros((0, 3, 4), dtype=np.uint8), np.zeros((3, 3, 3), dtype=np.uint8), np.zeros((3, 3, 4))]:
            with self.assertRaises(ValueError):
                engine.analyze_rgba(a)

    @unittest.skipUnless(shutil.which("node"), "Node.js required for original JS conformance")
    def test_original_javascript_conformance(self):
        rng = np.random.default_rng(471)
        fixtures = []
        for h, w in [(1, 1), (1, 32), (32, 1), (3, 5), (31, 37), (128, 256)]:
            a = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
            a[..., 3] = 255
            fixtures.append(a)
        for level in [0, 1, 128, 254, 255]:
            a = np.full((17, 19, 4), level, dtype=np.uint8); a[..., 3] = 255
            fixtures.append(a)
        for alpha in [0, 249, 250, 254, 255]:
            a = rng.integers(0, 256, (7, 13, 4), dtype=np.uint8); a[..., 3] = alpha
            fixtures.append(a)
        model = fixture_model(varying=True)
        code = """const fs=require('fs'), E=require(process.argv[1]);
const x=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(x.fixtures.map(f=>{const s=E.analyze(f.a,f.w,f.h);return {s,p:E.predict(s.features,x.model)}})));"""
        payload = {"model": model["model"], "fixtures": [{"w": a.shape[1], "h": a.shape[0], "a": a.ravel().tolist()} for a in fixtures]}
        result = subprocess.run(["node", "-e", code, str(ROOT / "tests/fixtures/tone_lab_v4_reference.cjs")],
                                input=json.dumps(payload), capture_output=True, text=True, check=True)
        expected = json.loads(result.stdout)
        validated = engine.validate_export(model)
        for a, ref in zip(fixtures, expected):
            s = engine.analyze_rgba(a)
            self.assertEqual(s["names"], ref["s"]["names"])
            self.assertEqual(s["sampleCount"], ref["s"]["sampleCount"])
            np.testing.assert_allclose(s["features"], ref["s"]["features"], atol=1e-11, rtol=1e-11)
            p = engine.predict(s["features"], validated)
            for key in p:
                self.assertAlmostEqual(p[key], ref["p"][key], places=10)

    def test_noop_gate_and_two_directions(self):
        for gamma, gain in [(1, 1), (.8, .93), (1.4, 1.08)]:
            p = engine.predict([.3] * 169, engine.validate_export(fixture_model(gamma, gain)))
            self.assertAlmostEqual(p["gamma"], gamma)
            self.assertAlmostEqual(p["gain"], gain)
            self.assertEqual(p["noop"], gamma == gain == 1)

    def test_high_noop_score_only_snaps_small_curve(self):
        for gamma, should_snap in [(1.02, True), (1.4, False)]:
            m = fixture_model(gamma, 1);m["model"]["layers"][-1]["b"][2] = 10
            p = engine.predict([.3] * 169, engine.validate_export(m))
            self.assertEqual(p["noop"], should_snap)

    def test_reject_bad_models(self):
        transforms = [lambda d: d.update(type="session"), lambda d: d.update(schema="future"),
                      lambda d: d["model"].update(trained=False), lambda d: d["model"]["names"].reverse(),
                      lambda d: d["model"]["sd"].__setitem__(0, 0), lambda d: d["model"]["mu"].__setitem__(0, float("nan")),
                      lambda d: d["model"]["layers"][0]["w"].pop(), lambda d: d["analysis"].update(longEdge=512),
                      lambda d: d["model"]["layers"][0]["b"].__setitem__(0, float("inf"))]
        for transform in transforms:
            d = fixture_model();transform(d)
            with self.assertRaises(ValueError):engine.validate_export(d)

    def test_features_and_weights_only(self):
        d = fixture_model(varying=True);m = engine.validate_export(d)
        p = engine.predict([.4] * 169, m)
        d.update(ratings={"any": "too_dark"}, history=["ignored"], filename="unrelated")
        self.assertEqual(p, engine.predict([.4] * 169, engine.validate_export(d)))


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "donut_tone").mkdir()
        self.path = self.root / "donut_tone/model.json"
        self.path.write_text(json.dumps(fixture_model()))
        self.module = load_node(self.root)
        self.node = self.module.DonutToneLab()
        self.image = torch.linspace(.01, .99, 12 * 16 * 3).reshape(1, 12, 16, 3)

    def tearDown(self):self.tmp.cleanup()

    def test_node_schema_defaults(self):
        inputs = self.node.INPUT_TYPES()
        self.assertFalse(inputs["required"]["enabled"][1]["default"])
        self.assertIn("model.json", inputs["required"]["model_name"][0])
        self.assertTrue(inputs["optional"]["edit_mode"][1]["forceInput"])

    def test_passthrough_and_edit_protection(self):
        for kwargs in [{}, {"enabled": True, "strength": 0}, {"enabled": True, "edit_mode": True}]:
            r = self.node.apply(self.image, **kwargs)
            self.assertIs(r["result"][0], self.image)
            self.assertFalse(json.loads(r["result"][1])["applied"])
            self.assertEqual(self.node.IS_CHANGED(**kwargs), "passthrough")

    def test_enabled_missing_model_fails(self):
        with self.assertRaises(ValueError):self.node.apply(self.image, enabled=True)
        with self.assertRaises(FileNotFoundError):self.node.apply(self.image, enabled=True, model_name="missing.json")

    def test_batch_alpha_strength_and_no_mutation(self):
        image = self.image.repeat(2, 1, 1, 1)
        image = torch.cat([image, torch.ones_like(image[..., :1])], dim=-1)
        before = image.clone()
        r = self.node.apply(image, True, "model.json", .5, True, True)
        out, text = r["result"]
        torch.testing.assert_close(out[..., :3], (image[..., :3].pow(math.sqrt(1.35)) * math.sqrt(1.04)).clamp(0, 1))
        self.assertTrue(torch.equal(before, image));self.assertTrue(torch.equal(out[..., 3], image[..., 3]))
        self.assertEqual(len(json.loads(text)["frames"]), 2)
        self.assertEqual(out.dtype, image.dtype);self.assertEqual(out.device, image.device)

    def test_float16(self):
        out = self.node.apply(self.image.half(), True, "model.json")["result"][0]
        self.assertEqual(out.dtype, torch.float16);self.assertTrue(torch.isfinite(out).all())

    def test_identity_model_and_transparency(self):
        self.path.write_text(json.dumps(fixture_model(1, 1)))
        self.assertIs(self.node.apply(self.image, True, "model.json")["result"][0], self.image)
        a = torch.cat([self.image, torch.zeros_like(self.image[..., :1])], -1)
        self.assertIs(self.node.apply(a, True, "model.json")["result"][0], a)

    def test_content_replacement_invalidates_cache(self):
        a = self.node.IS_CHANGED(enabled=True, model_name="model.json")
        x = self.node.apply(self.image, True, "model.json")["result"][0]
        self.path.write_text(json.dumps(fixture_model(1.5, 1.01)))
        b = self.node.IS_CHANGED(enabled=True, model_name="model.json")
        y = self.node.apply(self.image, True, "model.json")["result"][0]
        self.assertNotEqual(a, b);self.assertFalse(torch.equal(x, y))

    def test_paths_and_size(self):
        for name in ["../a.json", "/tmp/a.json", "C:/a.json", "x\\a.json", "a.py"]:
            with self.assertRaises(ValueError):self.module._read_model(name)
        self.path.write_bytes(b"x" * (self.module.MAX_MODEL_BYTES + 1))
        with self.assertRaises(ValueError):self.module._read_model("model.json")

    def test_escaping_symlink(self):
        outside = self.root / "outside.json";outside.write_text("{}")
        link = self.root / "donut_tone/link.json"
        try:link.symlink_to(outside)
        except OSError:self.skipTest("Symlinks unavailable")
        with self.assertRaises(ValueError):self.module._read_model("link.json")

    def test_invalid_image_and_strength(self):
        for x in [torch.ones(2, 2), torch.ones(1, 2, 2, 3, dtype=torch.int64), torch.empty(0, 2, 2, 3), self.image * float('nan')]:
            with self.assertRaises(ValueError):self.node.apply(x, True, "model.json")
        for s in [-1, 2, float('nan')]:
            with self.assertRaises(ValueError):self.node.apply(self.image, True, "model.json", s)

    def test_proxy_dimensions_and_identity_resize(self):
        for h, w in [(10, 12), (300, 600), (4032, 2304), (1, 600)]:
            frame = torch.full((h, w, 3), 128 / 255)
            proxy = self.module.analysis_proxy(frame)
            scale = min(1, 256 / max(h, w))
            self.assertEqual(proxy.shape, (max(1, math.floor(h * scale + .5)), max(1, math.floor(w * scale + .5)), 4))
            self.assertTrue((proxy[..., :3] == 128).all());self.assertTrue((proxy[..., 3] == 255).all())


if __name__ == '__main__':unittest.main()

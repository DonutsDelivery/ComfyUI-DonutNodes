"""Synthetic face-routing behavior; no ComfyUI, models, GPU or user data."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import random
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = load("dmc_face_detailer_fixture", "dmc_face_detailer.py")
ROWS = [{"character": "Ada", "position": "left", "face_prompt": "Ada smiles."},
        {"character": "Bo", "position": "right", "face_prompt": "Bo frowns."}]


class Image:
    def __init__(self, index=0, state=0):
        self.index, self.state = index, state

    def unsqueeze(self, axis):
        return self


class Mask:
    ndim = 2

    def unsqueeze(self, axis):
        result = Mask()
        result.ndim = 3
        return result


class Detector:
    def __init__(self, detections):
        self.detections, self.aux = detections, None

    def setAux(self, value):
        self.aux = value

    def detect(self, image, *args, **kwargs):
        return (100, 100), self.detections[image.index]


def segment(x, y=0, width=10, mask=1):
    return types.SimpleNamespace(bbox=(x, y, x + width, y + 10), crop_region=(x, y, x + width, y + 10),
                                 cropped_mask=mask, cropped_image="stale original crop")


def relative_rows(names, left=None, above=None):
    return [{"character": name, "position": {"left_of": (left or {}).get(name, []),
            "above": (above or {}).get(name, [])}, "face_prompt": name + " has a distinct expression."} for name in names]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.passes, self.encoded, self.legacy = [], [], []
        fixture = self
        self.module = types.ModuleType("synthetic_donut_detailer")
        class Upstream:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",)}, "optional": {"edit_mode": ("BOOLEAN", {"default": False})}}

            def doit(self, image, model="model", clip="clip", vae="vae", resolution=1024, max_resolution=1280,
                     guide_size_for=False, seed=42, steps=8, cfg=1, sampler_name="er_sde", scheduler="simple",
                     positive=None, negative="negative", denoise=.2, feather=5, noise_mask=True,
                     force_inpaint=False, bbox_threshold=.21, bbox_dilation=8, bbox_crop_factor=1.5,
                     sam_detection_hint="rect-4", sam_dilation=3, sam_threshold=.93, sam_bbox_expansion=6,
                     sam_mask_hint_threshold=.7, sam_mask_hint_use_negative="False", drop_size=12,
                     bbox_detector=None, wildcard="", cycle=2, max_faces=13, sam_model_opt=None,
                     segm_detector_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=20,
                     scheduler_func_opt=None, edit_mode=False, edit_prompt="whole image",
                     edit_negative_prompt="negative text", grounding_px=1088, edit_model=None,
                     face_reference=None, vary_seed_per_face=False, turbo_mode=True,
                     face_reference_b=None, vae_damage_correction=True, vae_damage_strength=.8, **nag_options):
                fixture.legacy.append((image, nag_options))
                fixture.delegated = locals()
                return ("legacy",) * 6

            @staticmethod
            def enhance_face(image, **kwargs):
                _, segs = kwargs["bbox_detector"].detect(image)
                assert len(segs) == 1
                fixture.passes.append((image.index, image.state, segs[0].bbox, kwargs))
                # Simulates a crop from the evolving full image, not seg.cropped_image.
                return Image(image.index, image.state + 1), ["crop"], ["alpha"], "mask", ["control"]
        Upstream.__module__ = self.module.__name__
        self.module.Upstream = Upstream
        self.module._ensure_impact = lambda: None
        self.module.offload_model_for_auxiliary_stage = lambda *args: None
        self.module.comfy = types.SimpleNamespace(model_management=object())
        self.module.torch = types.SimpleNamespace(is_tensor=lambda _: False, count_nonzero=lambda mask: mask,
                                                 cat=lambda values, dim=0: values)
        self.module.np = types.SimpleNamespace(count_nonzero=lambda mask: mask)
        self.module.core = types.SimpleNamespace(segs_to_combined_mask=lambda segs: Mask(),
            make_sam_mask=lambda *args: "sam", segs_bitwise_and_mask=lambda segs, mask: segs)
        self.module.reapply_edit_variance = lambda fresh, original: [[fresh[0][0], {"variance": original[0][1]["variance"]}]]
        self.module.prepare_positive_conditioning_taps = lambda model, fresh: [[fresh[0][0], {**fresh[0][1], "fusion": model}]]
        class Encoder:
            def encode(self, clip, text):
                fixture.encoded.append((clip, text))
                return ([[text, {}]],)
        self.nodes = types.ModuleType("nodes")
        self.nodes.NODE_CLASS_MAPPINGS = {"DonutFaceDetailer": Upstream}
        self.nodes.CLIPTextEncode = Encoder
        self.modules = patch.dict(sys.modules, {"nodes": self.nodes, self.module.__name__: self.module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def run_faces(self, detections=None, rows=ROWS, **kwargs):
        detections = detections or [[segment(70, width=25), segment(10)]]
        return runtime.DMCFaceDetailer().doit([Image(index) for index in range(len(detections))],
            bbox_detector=Detector(detections), positive=[["NEVER REUSE WHOLE PROMPT", {"variance": "recipe"}]],
            face_prompts_json=json.dumps(rows), **kwargs)

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_ordered_separate_conditioning_and_evolving_crops(self):
        result = self.run_faces(vary_seed_per_face=True, nag_alpha=3.4)
        self.assertEqual(len(result), 6)
        self.assertEqual(len(result[4]), 14)
        self.assertEqual([mask.ndim for mask in result[3]], [3])
        self.assertEqual([entry[2][0] for entry in self.passes], [10, 70])
        self.assertEqual([entry[1] for entry in self.passes], [0, 1])
        self.assertEqual(self.encoded, [("clip", "Ada smiles."), ("clip", "Bo frowns.")])
        for index, (_, _, _, inputs) in enumerate(self.passes):
            self.assertEqual(inputs["positive"], [[ROWS[index]["face_prompt"], {"variance": "recipe", "fusion": "model"}]])
            self.assertEqual(inputs["edit_prompt"], ROWS[index]["face_prompt"])
            self.assertEqual(inputs["seed"], 42 + index)
            self.assertEqual(inputs["resolution"], 1024 ** 2)
            for key, wanted in {"cycle": 2, "nag_alpha": 3.4, "vae_damage_strength": .8,
                                "scheduler": "simple", "noise_mask_enabled": True, "guide_size_for_bbox": False,
                                "denoise": .2, "turbo_mode": True}.items():
                self.assertEqual(inputs[key], wanted)

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_horizontal_plans_reject_equal_and_nearly_equal_x_instead_of_using_y_or_index(self):
        for faces in ([segment(50, 0), segment(50, 40)], [segment(50, 40), segment(50, 0)],
                      [segment(50, 0), segment(52.5, 40)], [segment(50, 0), segment(52.4, 40)],
                      [segment(10, 0, 10), segment(5, 40, 20)]):
            with self.subTest(faces=faces), self.assertRaisesRegex(ValueError, "ambiguous.*above/below"):
                self.run_faces([faces])
            self.assertEqual(self.passes, [])
            self.assertEqual(self.encoded, [])

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_later_batch_horizontal_tie_prevents_any_first_image_refinement(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.run_faces([[segment(0), segment(40)], [segment(50, 0), segment(50, 40)]])
        self.assertEqual(self.passes, [])
        self.assertEqual(self.encoded, [])

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_sixteen_ordinal_positions_keep_separate_character_conditioning(self):
        rows=[{"character": f"Person {index}", "position": position, "face_prompt": f"Face {index} smiles."}
              for index, position in enumerate(runtime.ORDINALS)]
        expected=[segment(index*40, (index%3)*20) for index in range(16)]
        self.run_faces([list(reversed(expected))], rows=rows, max_faces=16)
        self.assertEqual([entry[2] for entry in self.passes], [seg.bbox for seg in expected])
        self.assertEqual([entry[3]["positive"][0][0] for entry in self.passes], [row["face_prompt"] for row in rows])

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_every_batch_count_is_checked_before_any_sampling_or_encoding(self):
        for detections in [[[segment(0)]], [[segment(0), segment(20), segment(50)]],
                           [[segment(0), segment(20)], [segment(0)]], [[segment(0, mask=0), segment(20)]]]:
            with self.subTest(detections=detections), self.assertRaisesRegex(ValueError, "routing mismatch"):
                self.run_faces(detections)
            self.assertEqual(self.passes, [])
            self.assertEqual(self.encoded, [])
        with self.assertRaisesRegex(ValueError, "Maximum faces"):
            self.run_faces(max_faces=1)
        self.assertEqual(self.passes, [])

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_identity_edit_references_and_prompt_overrides_are_rejected(self):
        for options in ({"edit_mode": True}, {"face_reference": Image()}, {"face_reference_b": Image()},
                        {"wildcard": "replace prompt"}, {"detailer_hook": object()}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "not support"):
                self.run_faces(**options)
            self.assertEqual(self.passes, [])

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_batch_seeds_and_geometry_are_independent(self):
        self.run_faces([[segment(50), segment(0)], [segment(80), segment(20)]], vary_seed_per_face=True)
        self.assertEqual([(entry[0], entry[1], entry[2][0], entry[3]["seed"]) for entry in self.passes],
                         [(0, 0, 0, 42), (0, 1, 50, 43), (1, 0, 20, 43), (1, 1, 80, 44)])

    # AC: @create-multiple-face-prompts ac-single-face-compatible
    def test_empty_plan_delegates_without_rejecting_legacy_options(self):
        self.assertEqual(self.run_faces(rows=[], edit_mode=True, face_reference=Image(), wildcard="legacy"), ("legacy",) * 6)
        self.assertEqual(len(self.legacy), 1)
        self.assertEqual(self.passes, [])
        self.assertEqual(runtime.DMCFaceDetailer.INPUT_TYPES()["optional"]["face_prompts_json"][1]["default"], "[]")

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_strict_single_face_rejects_multi_faces_and_batch_mismatch_before_delegation(self):
        for detections in ([[segment(0), segment(20)]], [[segment(0)], [segment(0), segment(20)]],
                           [[segment(0, mask=0)]]):
            with self.subTest(detections=detections), self.assertRaisesRegex(ValueError, "single face prompt"):
                self.run_faces(detections=detections, rows=[], require_single_face=True)
            self.assertEqual(self.legacy, [])
            self.assertEqual(self.passes, [])
            self.assertEqual(self.encoded, [])

    # AC: @create-multiple-face-prompts ac-single-face-compatible
    def test_strict_single_face_preserves_original_edit_reference_and_sampling_inputs(self):
        reference, hook = Image(), object()
        result = self.run_faces(detections=[[segment(0)], [segment(20)]], rows=[], require_single_face=True,
            edit_mode=True, face_reference=reference, face_reference_b=reference, detailer_hook=hook,
            edit_prompt="Only Ada's face", wildcard="legacy", seed=123, cycle=3, nag_alpha=4.5)
        self.assertEqual(result, ("legacy",) * 6)
        self.assertEqual(len(self.legacy), 1)
        self.assertEqual(self.delegated["face_reference"], reference)
        self.assertEqual(self.delegated["face_reference_b"], reference)
        self.assertEqual(self.delegated["detailer_hook"], hook)
        self.assertTrue(self.delegated["edit_mode"])
        self.assertEqual(self.delegated["edit_prompt"], "Only Ada's face")
        self.assertEqual(self.delegated["wildcard"], "legacy")
        self.assertEqual(self.delegated["seed"], 123)
        self.assertEqual(self.delegated["cycle"], 3)
        self.assertEqual(self.legacy[0][1], {"nag_alpha": 4.5})
        self.assertEqual(runtime.DMCFaceDetailer.INPUT_TYPES()["optional"]["require_single_face"], ("BOOLEAN", {"default": False}))

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_strict_character_and_position_schema(self):
        for invalid in (ROWS[:1], [{**ROWS[0], "position": "order"}, ROWS[1]], [ROWS[1], ROWS[0]],
                        [ROWS[0], {**ROWS[1], "position": "left"}], [ROWS[0], {**ROWS[1], "character": "ada"}],
                        [{**ROWS[0], "position": "1st"}, ROWS[1]], [{**ROWS[0], "face_prompt": ""}, ROWS[1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runtime.face_prompts(json.dumps(invalid))
        self.assertEqual(len(runtime.face_prompts(json.dumps([{**row, "position": str(index + 1) + suffix}
            for index, (row, suffix) in enumerate(zip(ROWS, ["st", "nd"]))]))), 2)
        maximum = [{"character": "Character " + str(index), "position": position, "face_prompt": "\u263a" * 8000}
                   for index, position in enumerate(runtime.ORDINALS)]
        self.assertEqual(len(runtime.face_prompts(json.dumps(maximum))), 16)

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_relative_above_and_left_chains_follow_relations_in_arbitrary_record_order(self):
        for axis, detections in (("above", [segment(50, 80), segment(20, 0), segment(0, 40)]),
                                 ("left_of", [segment(80, 0), segment(0, 50), segment(40, 20)])):
            rows=relative_rows(["BottomOrRight", "TopOrLeft", "Middle"])
            by_name={row["character"]: row for row in rows}
            by_name["TopOrLeft"]["position"][axis]=["Middle"]
            by_name["Middle"]["position"][axis]=["BottomOrRight"]
            self.run_faces([detections], rows=rows)
            self.assertEqual([entry[2] for entry in self.passes], [detections[0].bbox, detections[1].bbox, detections[2].bbox])
            self.assertEqual([entry[3]["positive"][0][0] for entry in self.passes], [row["face_prompt"] for row in rows])
            self.passes.clear()

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_relative_family_and_six_people_work_without_fixed_grid_cells(self):
        names=["Father", "Son", "Mother", "Daughter"]
        rows=relative_rows(names, {"Father": ["Mother", "Daughter"], "Son": ["Mother", "Daughter"]},
                           {"Father": ["Son"], "Mother": ["Daughter"]})
        # Uneven rows and columns still satisfy the explicitly stated relations.
        expected=[segment(0, 0), segment(25, 45), segment(80, 8), segment(110, 80)]
        self.run_faces([[expected[3], expected[0], expected[2], expected[1]]], rows=rows)
        self.assertEqual([entry[2] for entry in self.passes], [seg.bbox for seg in expected])
        self.passes.clear()
        names=["A", "B", "C", "D", "E", "F"]
        rows=relative_rows(names, {"A": ["C", "D", "E", "F"], "B": ["C", "D", "E", "F"],
                                  "C": ["E", "F"], "D": ["E", "F"]},
                           {"A": ["B"], "C": ["D"], "E": ["F"]})
        expected=[segment(0, 0), segment(10, 40), segment(50, 5), segment(60, 50), segment(100, 0), segment(110, 70)]
        self.run_faces([[expected[index] for index in (5, 3, 0, 4, 1, 2)]], rows=rows)
        self.assertEqual([entry[2] for entry in self.passes], [seg.bbox for seg in expected])
        self.assertEqual([entry[3]["edit_prompt"] for entry in self.passes], [row["face_prompt"] for row in rows])

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_relative_ambiguity_and_near_ties_fail_entire_batch_before_conditioning(self):
        unconstrained=relative_rows(["A", "B"])
        above=relative_rows(["A", "B"], above={"A": ["B"]})
        left=relative_rows(["A", "B"], left={"A": ["B"]})
        cases=[(unconstrained, [[segment(0), segment(40)]]),
               (above, [[segment(0, 0), segment(40, 2.5)]]),
               (left, [[segment(0), segment(2.4)]]),
               (above, [[segment(0, 0), segment(0, 40)], [segment(0, 0), segment(20, 0)]]),
               (above, [[segment(0, 0)]])]
        for rows, detections in cases:
            with self.subTest(rows=rows, detections=detections), self.assertRaisesRegex(ValueError, "ambiguous|cannot match|routing mismatch"):
                self.run_faces(detections, rows=rows)
            self.assertEqual(self.passes, [])
            self.assertEqual(self.encoded, [])

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_relative_schema_rejects_unknown_self_cycles_and_mixed_positions(self):
        bad=[relative_rows(["A", "B"], left={"A": ["Unknown"]}),
             relative_rows(["A", "B"], above={"A": ["A"]}),
             relative_rows(["A", "B"], above={"A": ["B"], "B": ["A"]}),
             relative_rows(["A", "B"], left={"A": ["B", "B"]}),
             [relative_rows(["A"])[0], ROWS[1]],
             [{**ROWS[0], "position": {"column": 1, "row": 1}}, ROWS[1]]]
        for rows in bad:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                runtime.face_prompts(json.dumps(rows))
        # Cross-axis references can describe a valid diagonal, not an axis cycle.
        self.assertEqual(len(runtime.face_prompts(json.dumps(relative_rows(["A", "B"],
            left={"A": ["B"]}, above={"B": ["A"]})))), 2)

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_relative_search_budget_exhaustion_never_claims_unique_mapping(self):
        rows=relative_rows(["A", "B"])
        for budget in range(1, 12):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "search limit|ambiguous"):
                runtime.relative_faces([segment(0), segment(40)], rows, budget=budget)

    # AC: @create-multiple-face-prompts ac-spatial-matching
    # AC: @create-multiple-face-prompts ac-mismatch
    def test_relative_solver_agrees_with_independent_exhaustive_bijections(self):
        rng=random.Random(1946)
        for case in range(40):
            names=["A", "B", "C", "D"]
            rows=relative_rows(names)
            for index, row in enumerate(rows):
                for axis in ("left_of", "above"):
                    row["position"][axis]=[name for name in names[index+1:] if rng.random()<.4]
            segments=[segment(rng.randrange(0, 100), rng.randrange(0, 100), rng.randrange(5, 25)) for _ in names]
            expected=[]
            for assignment in itertools.permutations(segments):
                by_name=dict(zip(names, assignment))
                matched=True
                for row in rows:
                    own=by_name[row["character"]].bbox
                    for axis, coordinate in (("left_of", 0), ("above", 1)):
                        for name in row["position"][axis]:
                            other=by_name[name].bbox
                            gap=(other[coordinate]+other[coordinate+2]-own[coordinate]-own[coordinate+2])/2
                            margin=.25*min(own[coordinate+2]-own[coordinate], other[coordinate+2]-other[coordinate])
                            matched=matched and gap>margin
                if matched:
                    expected.append(assignment)
            with self.subTest(case=case):
                if len(expected)==1:
                    self.assertEqual(runtime.relative_faces(segments, rows), list(expected[0]))
                else:
                    with self.assertRaisesRegex(ValueError, "cannot match|ambiguous"):
                        runtime.relative_faces(segments, rows)

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_relative_margin_is_invariant_under_scaling_and_translation(self):
        rows=relative_rows(["A", "B", "C"], above={"A": ["B"], "B": ["C"]})
        for scale in (.1, 1, 100):
            segments=[segment(80, 80), segment(10, 0), segment(20, 40)]
            for seg in segments:
                seg.bbox=tuple(value*scale+123 for value in seg.bbox)
            self.assertEqual(runtime.relative_faces(segments, rows), [segments[1], segments[2], segments[0]])

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_relative_sixteen_character_chain_proves_unique_mapping_within_budget(self):
        names=[f"Person {index}" for index in range(16)]
        rows=relative_rows(names, left={name: [names[index+1]] for index, name in enumerate(names[:-1])})
        expected=[segment(index*40, index*3) for index in range(16)]
        self.run_faces([list(reversed(expected))], rows=rows, max_faces=16)
        self.assertEqual([entry[2] for entry in self.passes], [seg.bbox for seg in expected])


if __name__ == "__main__":
    unittest.main()

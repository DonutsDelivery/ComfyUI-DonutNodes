"""CPU-only unit/contract tests. They do not claim GPU or browser validation."""
from copy import deepcopy
from datetime import datetime as real_datetime
import importlib.util
import json
import gzip
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PKG = "_donut_streamline_test"
package = types.ModuleType(PKG); package.__path__ = [str(ROOT)]; sys.modules[PKG] = package


def load(name):
    full = f"{PKG}.{name}"
    spec = importlib.util.spec_from_file_location(full, ROOT / (name.replace(".", "/") + ".py"))
    module = importlib.util.module_from_spec(spec); sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


prompt = load("donut_prompt")
seed_plan = load("donut_seed_plan")
migration = load("tools.streamline_workflow")


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.roots = patch.object(prompt, "wildcard_roots", return_value=(self.root,)); self.roots.start()
    def tearDown(self): self.roots.stop(); self.tmp.cleanup()
    def file(self, name, text):
        path = self.root / (name + ".txt"); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8"); return path
    def test_literal_whitespace_is_preserved(self): self.assertEqual(prompt.expand_text(" face  \n scene "), " face  \n scene ")
    def test_nested_files_and_choices(self):
        for i in range(40): self.file(f"n{i}", f"__n{i+1}__")
        self.file("n40", "{red|blue} __subjects/bird__"); self.file("subjects/bird", "finch")
        self.assertIn(prompt.expand_text("__n0__", 4), ("red finch", "blue finch"))
    def test_exact_depth_choice(self): self.assertEqual(prompt.expand_text("{a}", max_depth=1), "a")
    def test_repeated_modifiers_and_multi_line(self):
        self.file("colors", "red\ngreen\nblue\n")
        self.assertEqual(prompt.expand_text("__colors__ __!colors__ __+colors__ __-colors__", 2), "blue blue red green")
        self.assertEqual(prompt.expand_text("2$$__colors__", 2), "blue,red")
    def test_filters_are_whole_word_or(self):
        self.file("birds", "redness\nred bird\nblue bird")
        self.assertEqual(prompt.expand_text("__birds|red__", 0), "red bird")
        self.assertEqual(prompt.expand_text("__birds|red|blue__", 2), "blue bird")
    def test_repeat_random_is_deterministic(self):
        self.file("n", "a\nb\nc\nd")
        self.assertEqual(prompt.expand_text("__n__ __*n__ __n__", 400), prompt.expand_text("__n__ __*n__ __n__", 400))
    def test_missing_policies(self):
        with self.assertRaisesRegex(ValueError, "not found"): prompt.expand_text("__missing__")
        self.assertEqual(prompt.expand_text("x __missing__", missing="keep"), "x __missing__")
        self.assertEqual(prompt.expand_text("x __missing__", missing="empty"), "x ")
    def test_empty_files(self):
        self.file("empty", "")
        with self.assertRaisesRegex(ValueError, "Empty wildcard"): prompt.expand_text("__empty__")
    def test_cycles(self):
        self.file("a", "__b__"); self.file("b", "__a__")
        with self.assertRaisesRegex(ValueError, "Cyclic"): prompt.expand_text("__a__")
        self.file("self", "__self__")
        with self.assertRaisesRegex(ValueError, "Cyclic"): prompt.expand_text("__self__")
    def test_depth_and_expansion_budget(self):
        self.file("a", "__b__"); self.file("b", "__c__"); self.file("c", "done")
        with self.assertRaisesRegex(ValueError, "max_depth"): prompt.expand_text("__a__", max_depth=1)
        with self.assertRaisesRegex(ValueError, "budget"): prompt.expand_text("20000$$__a__")
        with self.assertRaisesRegex(ValueError, "size limit"): prompt.expand_text("x" * (prompt.MAX_TEXT + 1))
    def test_path_traversal(self):
        for text in ("__../secret__", "__/etc/passwd__", "__C:/secret__"):
            with self.assertRaisesRegex(ValueError, "Unsafe"): prompt.expand_text(text)
    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "secret.txt"; target.write_text("secret")
            (self.root / "escape.txt").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "escapes"): prompt.expand_text("__escape__")
    def test_file_cache_invalidates(self):
        path = self.file("a", "one")
        before = prompt.DonutText.IS_CHANGED(text="__a__", seed=0)
        stat = path.stat(); path.write_text("two"); os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.assertNotEqual(before, prompt.DonutText.IS_CHANGED(text="__a__", seed=0))
        self.assertEqual(prompt.expand_text("__a__"), "two")
    def test_rng_is_local(self):
        self.file("n", "a\nb\nc")
        random.seed(892); state = random.getstate()
        prompt.expand_text("{x|y} <random:-1:1> __n__ __*n__", 32)
        self.assertEqual(state, random.getstate())
    def test_numeric_random_syntax(self):
        expected = str(round(random.Random(45).uniform(-1, 4), 4))
        self.assertEqual(prompt.expand_text("<random:-1:4>", 45), expected)
    def test_macros_inside_wildcard_files(self):
        self.file("a", "%Seed.value%")
        extra = {"workflow": {"nodes": [{"id": 7, "title": "Seed", "widgets_values_named": {"value": 42}}]}}
        self.assertEqual(prompt.expand_text("__a__", extra_pnginfo=extra), "42")
    def test_nested_date_invalidates_cache(self):
        self.file("date", "%date:yyyy-MM-dd hh:mm:ss%")
        class Clock:
            @classmethod
            def now(cls): return real_datetime(2026, 9, 7, 1, 2, cls.second)
        Clock.second = 3
        with patch.object(prompt, "datetime", Clock):
            first = prompt.DonutText.IS_CHANGED(text="__date__")
            Clock.second = 4
            self.assertNotEqual(first, prompt.DonutText.IS_CHANGED(text="__date__"))
    def test_prefix_suffix_expand_together(self):
        self.file("color", "red")
        result = prompt.DonutText().process("bird", prefix="__color__", suffix="photo", separator=" ")
        self.assertEqual(result["result"], ("red bird photo",))
    def test_blank_lines_preserve_selection_indices(self):
        self.file("a", "first\n\n# literal\nlast\n")
        self.assertEqual(prompt.expand_text("__a__", 1), "")
        self.assertEqual(prompt.expand_text("__a__", 2), "# literal")


class SeedTests(unittest.TestCase):
    def test_stage_seeds_are_unique_and_repeatable(self):
        self.assertEqual(seed_plan.stage_seeds(100), (100,102,103,104))
    def test_browser_boundary_wraps(self):
        values = seed_plan.stage_seeds(seed_plan.MAX_SEED)
        self.assertEqual(len(set(values)), 4)
        self.assertTrue(all(0 <= x <= seed_plan.MAX_SEED for x in values))
    def test_invalid_seeds(self):
        for seed in (-1, True, 1.2, "1", seed_plan.MAX_SEED+1):
            with self.assertRaises(ValueError): seed_plan.stage_seeds(seed)
    def test_independent_text_and_filename_domains(self):
        self.assertEqual(seed_plan.DonutSeedPlan().generate(7,100,55), (7,100,102,103,104,"55"))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        class Builder:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"model_type": (["Auto", "KREA2"],), "civitai_lookup": (["On","Off"],),
                    "lora_name_1": (["None","test.safetensors"],), "block_preset_1": (["None","KREA2-ALL:1,1"],)}}
            def build_stack(self, **kwargs):
                calls.append(("build",deepcopy(kwargs)))
                stack = list(kwargs.get("lora_stack", []))
                for i in range(1,4):
                    if kwargs[f"switch_{i}"] == "On" and kwargs[f"lora_name_{i}"] != "None":
                        stack.append(tuple(kwargs[f"{key}_{i}"] for key in ("lora_name", "model_weight", "clip_weight", "block_vector")))
                return {"result": (stack,), "ui": {"text": ["summary","triggers","urls","one","two","three"]}}
        class Applier:
            @classmethod
            def INPUT_TYPES(cls): return {"required": {"model":("MODEL",),"clip":("CLIP",),"lora_stack":("LORA_STACK",),"safe_stack":(["Off","On"],)},"optional":{"execution_mode":(["Comfy patches","Experimental bypass"],)}}
            def apply_stack(self, model, clip, stack, **kwargs):
                calls.append(("apply",deepcopy(stack),kwargs)); return model,clip,"help"
        class Upscaler:
            FUNCTION="upscale"
            @classmethod
            def INPUT_TYPES(cls): return {"required":{"image":("IMAGE",),"model":("MODEL",),"seed":("INT",{"default":0})},"optional":{"color_reference":("IMAGE",)}}
            def upscale(self, **kwargs): calls.append(("upscale",kwargs)); return ("upscaled","debug")
        class Merge:
            FUNCTION="merge"
            @classmethod
            def INPUT_TYPES(cls): return {"required":{"model1":("MODEL",),"model2":("MODEL",),"blocks.0.":("FLOAT",{"default":1})}}
            def merge(self, **kwargs): calls.append(("merge",kwargs)); return (kwargs,)
        class Injection:
            FUNCTION="inject"
            @classmethod
            def INPUT_TYPES(cls): return {"required":{"prompt":("STRING",{"multiline":True}),"seed":("INT",{"default":0})}}
            def inject(self, prompt, seed): return (prompt+" {a|b}","style preview")
        modules = {}
        for name, attrs in {
            "donut_lora_nodes":{"DonutLoRAStack":Builder}, "DonutSafeApplyLoRAStack":{"DonutApplyLoRAStackSafe":Applier},
            "DonutTiledUpscale":{"NODE_CLASS_MAPPINGS":{"DonutTiledUpscale":Upscaler}},
            "DonutModelMergeKrea2":{"NODE_CLASS_MAPPINGS":{"DonutModelMergeKrea2":Merge}},
            "DonutPromptInjection":{"NODE_CLASS_MAPPINGS":{"DonutPromptInjection":Injection}},
        }.items():
            module = types.ModuleType(f"{PKG}.{name}"); module.__dict__.update(attrs); modules[module.__name__] = module
        folders = types.ModuleType("folder_paths"); folders.get_full_path = lambda kind,name: "/nonexistent-test-path" if name != "missing" else None
        modules["folder_paths"] = folders
        self.mock_modules = patch.dict(sys.modules, modules); self.mock_modules.start()
        self.lora=load("donut_dynamic_lora"); self.stage=load("donut_upscale_stage"); self.merge=load("donut_grouped_merge"); self.injection=load("donut_prompt_injection_recursive")
    def tearDown(self): self.mock_modules.stop()
    def rows(self, n=7): return [dict(id=f"r{i}", lora_name=f"lora{i}", model_weight=i/10, clip_weight=-i/10, lora_hash=f"hash{i}") for i in range(n)]
    def test_more_than_three_slots_and_single_apply(self):
        rows=self.rows(); result=self.lora.DonutLoRALoader().load("MODEL","CLIP",json.dumps(rows), safe_stack="On", execution_mode="Experimental bypass")
        self.assertEqual(len(result["result"][2]),7)
        self.assertEqual(len([c for c in self.calls if c[0]=="build"]),3)
        applies=[c for c in self.calls if c[0]=="apply"]
        self.assertEqual(len(applies),1); self.assertEqual(len(applies[0][1]),7)
        self.assertEqual(applies[0][2]["execution_mode"],"Experimental bypass")
    def test_disabled_slots_and_inherited_vectors(self):
        rows=self.rows(4); rows[1]["enabled"]=False; rows[2].update(inherit_block_vector=True,block_vector="0,0")
        result=self.lora.DonutLoRALoader().load("m","c",rows,global_block_vector="1,1")
        self.assertEqual([r[0] for r in result["result"][2]],["lora0","lora2","lora3"])
        self.assertEqual(result["result"][2][1][3],"1,1")
        self.assertEqual(len(result["ui"]["donut_loras"]),4)
    def test_hashes_follow_rows_not_chunk_position(self):
        self.lora.DonutDynamicLoRAStack().build(slots_json=self.rows(),model_type="KREA2")
        calls=[c[1] for c in self.calls if c[0]=="build"]
        self.assertEqual(calls[1]["extra_pnginfo"]["workflow"]["nodes"][0]["properties"]["lora_hashes"],["hash3","hash4","hash5"])
    def test_upstream_stack_is_preserved(self):
        result=self.lora.DonutDynamicLoRAStack().build(slots_json=self.rows(1),lora_stack=[("upstream",1,0,"")])
        self.assertEqual(result["result"][0][0][0],"upstream")
    def test_empty_stack(self): self.assertEqual(self.lora.DonutLoRALoader().load("m","c")["result"][:3],("m","c",[]))
    def test_invalid_slot_state(self):
        bad_values=[{},"{}",[1],[{"model_weight":float("nan")}],[{"enabled":"false"}],[{"id":"a"},{"id":"a"}],[{"clip_weight":True}]]
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)): self.lora.parse_slots(value)
    def test_missing_active_lora_fails_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError,"missing"):
            self.lora.DonutLoRALoader().load("m","c",[{"lora_name":"missing"}])
    def test_disabled_stage_does_not_request_or_run_upscaler(self):
        node=self.stage.DonutTiledUpscaleStage(); image=object()
        self.assertEqual(node.check_lazy_status(image,enabled=False,model=None),[])
        result=node.run_stage(image,enabled=False,model=None)
        self.assertIs(result[0],image); self.assertIs(result[1],image); self.assertEqual(self.calls,[])
    def test_enabled_stage_delegates_exactly(self):
        node=self.stage.DonutTiledUpscaleStage()
        self.assertEqual(node.check_lazy_status("img",enabled=True,model=None,seed=9),["model"])
        self.assertEqual(node.run_stage("img",model="model",seed=9),("upscaled","debug"))
        self.assertEqual(self.calls[0][1],dict(image="img",model="model",seed=9))
    def test_stage_schema_is_backwards_compatible(self):
        spec=self.stage.DonutTiledUpscaleStage.INPUT_TYPES()
        self.assertTrue(spec["optional"]["enabled"][1]["default"])
        self.assertTrue(spec["required"]["model"][1]["lazy"])
        self.assertEqual(spec["required"]["image"],("IMAGE",))
    def test_grouped_merge_preserves_unaffected_fields(self):
        values=self.merge.DonutModelMergeKrea2Grouped().merge_grouped(ratio_mode="Grouped",body_ratio=.8,fusion_ratio=.2,model1="a",model2="b",**{"tmlp.":.7})[0]
        self.assertEqual(values["blocks.27."],.8); self.assertEqual(values["txtfusion.projector."],.2); self.assertEqual(values["tmlp."],.7)
    def test_per_block_merge_forwards_unchanged(self):
        fields={"blocks.0.":.2,"txtfusion.projector.":.7}
        self.assertEqual(self.merge.DonutModelMergeKrea2Grouped().merge_grouped(**fields)[0],fields)
    def test_prompt_injection_uses_requested_text_seed(self):
        with patch.object(prompt,"wildcard_roots",return_value=()):
            result=self.injection.DonutPromptInjectionRecursive().process_recursive(prompt="face",seed=12)
        self.assertEqual(result,("face "+random.Random(12).choice(["a","b"]),"style preview"))
    def test_conditioning_raw_and_zeroed_are_distinct(self):
        calls=[]
        class Encode:
            def encode(self,clip,text): calls.append((clip,text)); return ([{"text":text}],)
        class Zero:
            def zero_out(self,value): return ([{"zero_of":deepcopy(value)}],)
        core=types.ModuleType("nodes"); core.CLIPTextEncode=Encode; core.ConditioningZeroOut=Zero
        with patch.dict(sys.modules,{"nodes":core}),patch.object(prompt,"wildcard_roots",return_value=()):
            out=prompt.DonutPromptConditioning().encode("clip","face","","negative")["result"]
        self.assertEqual(len(calls),2)  # Reuse identical face/full encodings.
        self.assertEqual(out[0],"face"); self.assertEqual(out[1],"face")
        self.assertNotEqual(out[5],out[6]); self.assertEqual(out[6],[{"text":"negative"}])


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=Path(os.environ.get("DONUT_WORKFLOW_SOURCE", ROOT / "tests" / "fixtures" / "ComfyUI_00016_.json.gz"))
        if not source.is_file(): raise unittest.SkipTest("Set DONUT_WORKFLOW_SOURCE to the original uploaded workflow for the real-graph regression tests")
        opener=gzip.open if source.suffix==".gz" else open
        with opener(source,"rt",encoding="utf-8") as handle: cls.original=json.load(handle)
        cls.migrated,cls.report=migration.migrate(cls.original)
    def nodes(self,w): return {n["id"]:n for g in migration.graphs(w) for n in g["nodes"]}
    def test_all_links_and_subgraph_boundaries_validate(self): self.assertTrue(migration.validate(self.migrated))
    def test_node_count_and_dependencies(self):
        self.assertEqual((self.report["before_nodes"],self.report["after_nodes"]),(68,40))
        packs={n.get("properties",{}).get("cnr_id") for n in self.nodes(self.migrated).values()}
        self.assertFalse(packs & migration.REMOVED_PACKS)
        self.assertTrue(set(self.report["retained_reasons"]) <= packs)
    def test_current_settings_and_bypasses_are_preserved(self):
        before,after=self.nodes(self.original),self.nodes(self.migrated)
        for node_id in (942,468,883,487,881,897,1033,893,1045,64,430,1120,1122,63,759,52,50,993,984,1118,1031,1095):
            self.assertEqual(before[node_id].get("widgets_values_named"),after[node_id].get("widgets_values_named"),str(node_id))
            self.assertEqual(before[node_id]["mode"],after[node_id]["mode"])
        for node_id in (989,983):
            fields=deepcopy(after[node_id]["widgets_values_named"]); enabled=fields.pop("enabled")
            self.assertEqual(fields,before[node_id]["widgets_values_named"])
            self.assertEqual(enabled,before[node_id]["mode"]!=4)
        self.assertEqual(after[891]["mode"],4)
    def test_six_lora_slots_including_disabled_settings(self):
        before,after=self.nodes(self.original),self.nodes(self.migrated)
        rows=json.loads(after[1055]["widgets_values_named"]["slots_json"])
        self.assertEqual(len(rows),6)
        for row in rows:
            node_id,slot=map(int,row["id"].split(":")); old=before[node_id]["widgets_values_named"]
            for key in ("lora_name","model_weight","clip_weight","block_vector","block_preset"):
                self.assertEqual(row[key],old[f"{key}_{slot}"])
            self.assertEqual(row["enabled"],old[f"switch_{slot}"]=="On")
            self.assertTrue(row["inherit_block_vector"])
    def test_sampler_and_text_seed_wiring(self):
        main=self.migrated["definitions"]["subgraphs"][0]
        edges=list(map(migration.unpack,main["links"]))
        nodes=self.nodes(self.migrated)
        expected={993:"seed",989:"seed_upscale_1",983:"seed_upscale_2",984:"seed_face"}
        for node_id,name in expected.items():
            slot=next(i for i,p in enumerate(nodes[node_id]["inputs"]) if p["name"]=="seed")
            edge=next(e for e in edges if e["target_id"]==node_id and e["target_slot"]==slot)
            self.assertEqual(edge["origin_id"],-10); self.assertEqual(main["inputs"][edge["origin_slot"]]["name"],name)
        root_edges=list(map(migration.unpack,self.migrated["links"]))
        for node_id in (53,56,57,891):
            slot=next(i for i,p in enumerate(nodes[node_id]["inputs"]) if p["name"]=="seed")
            edge=next(e for e in root_edges if e["target_id"]==node_id and e["target_slot"]==slot)
            self.assertEqual((edge["origin_id"],edge["origin_slot"]),(195,0))
    def test_source_is_not_mutated_and_migration_is_idempotent(self):
        saved=deepcopy(self.original); output,report=migration.migrate(self.original)
        self.assertEqual(saved,self.original)
        second,_=migration.migrate(output); self.assertEqual(output,second)
    def test_unknown_workflow_fails_closed(self):
        bad=deepcopy(self.original); next(n for n in bad["nodes"] if n["id"]==855)["type"]="OtherLoRA"
        with self.assertRaisesRegex(ValueError,"Unsupported"): migration.migrate(bad)
    def test_validator_detects_broken_links(self):
        bad=deepcopy(self.migrated); bad["links"][0][1]=99999999
        with self.assertRaisesRegex(ValueError,"Dangling"): migration.validate(bad)


if __name__ == "__main__": unittest.main()

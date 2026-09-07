#!/usr/bin/env python3
"""Migrate the supplied ComfyUI_00016 workflow without hand-editing links.

This is deliberately a guarded migration, not a heuristic graph optimiser.
Unknown layouts fail rather than silently dropping settings or functionality.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import gzip
import json
from pathlib import Path
import uuid

BRANCH = "feat/workflow-streamlining"
REMOVED_PACKS = {"cg-use-everywhere", "mikey_nodes", "RES4LYF", "derfuu_comfyui_moddednodes", "ComfyUI_Comfyroll_CustomNodes", "rgthree-comfy"}


def graphs(workflow):
    return [workflow, *workflow.get("definitions", {}).get("subgraphs", [])]


def unpack(link):
    if isinstance(link, list):
        return dict(zip(("id", "origin_id", "origin_slot", "target_id", "target_slot", "type"), link))
    return deepcopy(link)


class Graph:
    def __init__(self, value, root=False):
        self.data, self.root = value, root
        self.nodes = {n["id"]: n for n in value["nodes"]}
        self.edges = [unpack(link) for link in value["links"]]
        self.next_id = max([value.get("last_link_id", 0), value.get("state", {}).get("lastLinkId", 0), *(e["id"] for e in self.edges)]) + 1

    def port(self, node, name, output=False):
        ports = self.data["inputs"] if node == -10 else self.data["outputs"] if node == -20 else self.nodes[node]["outputs" if output else "inputs"]
        matches = [index for index, p in enumerate(ports) if p["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one port {node}:{name}, found {len(matches)}")
        return matches[0]

    def connect(self, source, source_slot, target, target_slot, kind):
        self.edges = [e for e in self.edges if (e["target_id"], e["target_slot"]) != (target, target_slot)]
        self.edges.append(dict(id=self.next_id, origin_id=source, origin_slot=source_slot,
                               target_id=target, target_slot=target_slot, type=kind))
        self.next_id += 1

    def source(self, old_id, old_slot, new_id, new_slot):
        for edge in self.edges:
            if (edge["origin_id"], edge["origin_slot"]) == (old_id, old_slot):
                edge.update(origin_id=new_id, origin_slot=new_slot)

    def remove(self, *ids):
        self.edges = [e for e in self.edges if e["origin_id"] not in ids and e["target_id"] not in ids]
        for node_id in ids:
            self.nodes.pop(node_id)

    def bypass_text(self, node_id):
        incoming = [e for e in self.edges if e["target_id"] == node_id and e["target_slot"] == self.port(node_id, "prompt")]
        if len(incoming) != 1:
            raise ValueError(f"Wildcard {node_id} has no unique prompt source")
        self.source(node_id, 0, incoming[0]["origin_id"], incoming[0]["origin_slot"])
        self.remove(node_id)

    def save(self):
        self.data["nodes"] = list(self.nodes.values())
        for node in self.nodes.values():
            node.get("properties", {}).pop("ue_properties", None)
            for port in node.get("inputs", []): port["link"] = None
            for port in node.get("outputs", []): port["links"] = []
        if not self.root:
            for port in self.data.get("inputs", []) + self.data.get("outputs", []): port["linkIds"] = []
        for edge in self.edges:
            source, target = edge["origin_id"], edge["target_id"]
            if source == -10:
                self.data["inputs"][edge["origin_slot"]]["linkIds"].append(edge["id"])
            else:
                self.nodes[source]["outputs"][edge["origin_slot"]]["links"].append(edge["id"])
            if target == -20:
                self.data["outputs"][edge["target_slot"]]["linkIds"].append(edge["id"])
            else:
                self.nodes[target]["inputs"][edge["target_slot"]]["link"] = edge["id"]
        for order, node in enumerate(self.nodes.values()): node["order"] = order
        self.data["links"] = [[e[k] for k in ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")] for e in self.edges] if self.root else self.edges
        if self.root:
            self.data["last_node_id"] = max(self.nodes)
            self.data["last_link_id"] = self.next_id - 1
        else:
            self.data.setdefault("state", {}).update(lastNodeId=max(self.nodes), lastLinkId=self.next_id - 1)
        extra = self.data.setdefault("extra", {})
        for key in ("ue_links", "links_added_by_ue"): extra.pop(key, None)


def input_port(name, kind, widget=False, optional=False):
    result = dict(name=name, type=kind, link=None)
    if widget: result["widget"] = {"name": name}
    if optional: result["shape"] = 7
    return result


def make_node(old, kind, inputs, outputs, named, values=None, title=None):
    node = deepcopy(old)
    node.update(type=kind, inputs=inputs, outputs=[dict(name=name, type=typ, links=[]) for name, typ in outputs],
                widgets_values=list(named.values()) if values is None else values, widgets_values_named=named)
    node["properties"] = {"cnr_id": "donutnodes", "aux_id": "DonutsDelivery/ComfyUI-DonutNodes", "Node name for S&R": kind}
    if title: node["title"] = title
    return node


def named(node, name):
    try: return node["widgets_values_named"][name]
    except KeyError: raise ValueError(f"Node {node['id']} has no named value {name!r}; re-export the supported source workflow") from None


def validate(workflow):
    errors = []
    subgraphs = {g["id"]: g for g in workflow.get("definitions", {}).get("subgraphs", [])}
    for graph in graphs(workflow):
        nodes = {n["id"]: n for n in graph["nodes"]}
        if len(nodes) != len(graph["nodes"]): errors.append("Duplicate node IDs")
        edges = {e["id"]: e for e in map(unpack, graph["links"])}
        if len(edges) != len(graph["links"]): errors.append("Duplicate link IDs")
        destinations = set()
        for edge in edges.values():
            source, target, out_slot, in_slot = (edge[k] for k in ("origin_id", "target_id", "origin_slot", "target_slot"))
            try:
                out = graph["inputs"][out_slot] if source == -10 else nodes[source]["outputs"][out_slot]
                inp = graph["outputs"][in_slot] if target == -20 else nodes[target]["inputs"][in_slot]
                outgoing = out.get("linkIds", []) if source == -10 else out.get("links", [])
                incoming = inp.get("linkIds", []) if target == -20 else [inp.get("link")]
                if edge["id"] not in (outgoing or []) or edge["id"] not in incoming: errors.append(f"Non-reciprocal link {edge['id']}")
                if (target, in_slot) in destinations: errors.append(f"Multiple drivers on {target}:{in_slot}")
                destinations.add((target, in_slot))
                if out["type"] != inp["type"] and "*" not in (out["type"], inp["type"]):
                    errors.append(f"Type mismatch on {edge['id']}: {out['type']} -> {inp['type']}")
            except (KeyError, IndexError): errors.append(f"Dangling endpoint on link {edge['id']}")
        for node in nodes.values():
            for port in node.get("inputs", []):
                if port.get("link") is not None and port["link"] not in edges: errors.append(f"Stale input link on {node['id']}")
            for port in node.get("outputs", []):
                if any(e not in edges for e in (port.get("links") or [])): errors.append(f"Stale output link on {node['id']}")
            if node["type"] in subgraphs:
                definition = subgraphs[node["type"]]
                names = {p["name"] for p in definition["inputs"]}
                if any(p["name"] not in names for p in node["inputs"]): errors.append("Unknown subgraph instance input")
                for port in definition["inputs"]:
                    instance_ports = [p for p in node["inputs"] if p["name"] == port["name"]]
                    connected = bool(instance_ports and instance_ports[0].get("link") is not None)
                    if port.get("linkIds") and not connected and port["name"] not in node.get("widgets_values_named", {}):
                        errors.append(f"Unresolved subgraph boundary: {port['name']}")
        # Reject data-flow cycles in both root and subgraph scopes.
        visiting, visited = set(), set()
        children = {key: [] for key in nodes}
        for edge in edges.values():
            if edge["origin_id"] in nodes and edge["target_id"] in nodes:
                children[edge["origin_id"]].append(edge["target_id"])
        def walk(key):
            if key in visiting: raise ValueError(f"Cycle at node {key}")
            if key in visited: return
            visiting.add(key)
            for child in children[key]: walk(child)
            visiting.remove(key); visited.add(key)
        try:
            for key in nodes: walk(key)
        except ValueError as error: errors.append(str(error))
    if errors: raise ValueError("Workflow validation failed:\n" + "\n".join(errors))
    return True


def migrate(original):
    if original.get("extra", {}).get("donut_streamlining", {}).get("schema") == 1:
        validate(original)
        return deepcopy(original), {"already_migrated": True}
    workflow = deepcopy(original)
    root = Graph(workflow, True)
    expected = {195: "SeedGenerator", 777: "Seed String", 855: "DonutLoRAStack", 1116: "DonutLoRAStack", 1055: "DonutApplyLoRAStack",
                53: "Wildcard Processor", 56: "Wildcard Processor", 57: "Wildcard Processor", 51: "Anything Everywhere", 1005: "Anything Everywhere"}
    for node_id, kind in expected.items():
        if root.nodes.get(node_id, {}).get("type") != kind: raise ValueError(f"Unsupported source: expected {node_id} / {kind}")
    main_id, merge_id = root.nodes[1014]["type"], root.nodes[1124]["type"]
    definitions = {g["id"]: g for g in workflow["definitions"]["subgraphs"]}
    main = Graph(definitions[main_id])
    merge = Graph(definitions[merge_id])
    report = {"before_nodes": sum(len(g["nodes"]) for g in graphs(original)), "removed_packs": sorted(REMOVED_PACKS),
              "retained_reasons": {"ComfyUI-bleh": "Exact custom ER-SDE preset behaviour", "was-node-suite-comfyui": "Full configured WebP/history/naming/metadata save behaviour",
                                   "comfyui-impact-pack": "SAM/detailer implementation", "comfyui-impact-subpack": "Ultralytics detector provider", "krea2-nag": "Keep adjustable Krea2 NAG, even though alpha is currently zero"}}
    # Materialise the four implicit connections BEFORE discarding UE controllers.
    for connection in workflow["extra"].get("ue_links", []):
        target = int(connection["downstream"]); slot = connection["downstream_slot"]
        if root.nodes[target]["inputs"][slot].get("link") is not None:
            raise ValueError("Refusing to overwrite an explicit link with a UE connection")
        root.connect(int(connection["upstream"]), connection["upstream_slot"], target, slot, connection["type"])
    if len(workflow["extra"].get("ue_links", [])) != 4: raise ValueError("Expected four recorded implicit connections")
    root.remove(51, 1005)

    text_seed = named(root.nodes[195], "seed")
    filename_seed = named(root.nodes[777], "seed")
    seeds = {"text_seed": text_seed, "text_seed_control": "randomize", "sampler_seed": text_seed,
             "sampler_seed_control": "randomize", "filename_seed": filename_seed, "filename_seed_control": "randomize"}
    root.nodes[195] = make_node(root.nodes[195], "DonutSeedPlan", [], [("text", "INT"), ("base", "INT"), ("upscale_1", "INT"), ("upscale_2", "INT"), ("face", "INT"), ("filename", "STRING")], seeds, title="Text seed independent of sampling")
    root.source(777, 1, 195, 5); root.remove(777)
    for edge in root.edges:
        if edge["target_id"] == 1014 and edge["target_slot"] == root.port(1014, "seed"):
            edge["origin_slot"] = 1
    for node_id in (53, 56, 57):
        old = root.nodes[node_id]
        fields = dict(text=named(old, "prompt"), seed=text_seed, control_after_generate="fixed", max_depth=128, missing="error", separator="")
        root.nodes[node_id] = make_node(old, "DonutText", [input_port("seed", "INT", True), input_port("prefix", "STRING", optional=True), input_port("suffix", "STRING", optional=True)], [("text", "STRING")], fields)
    injection = root.nodes[891]
    injection["inputs"].append(input_port("seed", "INT", True))
    root.connect(195, 0, 891, len(injection["inputs"]) - 1, "INT")
    injection["widgets_values_named"].update(seed=text_seed, control_after_generate="fixed", wildcard_depth=128, missing_wildcard="error")
    # Existing order is retained; only the new optional fields are appended.
    injection["widgets_values"][-2:] = [text_seed, "fixed"]
    injection["widgets_values"] += [128, "error"]

    # Unlimited LoRA rows retain six slot identities, including disabled rows.
    stack_nodes = [root.nodes[1116], root.nodes[855]]
    if len({named(n, "model_type") for n in stack_nodes}) != 1 or len({named(n, "civitai_lookup") for n in stack_nodes}) != 1:
        raise ValueError("Mixed stack-wide controls require an explicit migration decision")
    rows = []
    for node in stack_nodes:
        for slot in range(1, 4):
            vector_link = next((p.get("link") for p in node["inputs"] if p["name"] == f"block_vector_{slot}"), None)
            if vector_link is not None and not any(e["id"] == vector_link and e["origin_id"] == 1036 for e in root.edges):
                raise ValueError("Unsupported external per-slot block vector source")
            rows.append(dict(id=f"{node['id']}:{slot}", enabled=named(node, f"switch_{slot}") == "On", lora_name=named(node, f"lora_name_{slot}"),
                             model_weight=named(node, f"model_weight_{slot}"), clip_weight=named(node, f"clip_weight_{slot}"),
                             block_preset=named(node, f"block_preset_{slot}"), block_vector=named(node, f"block_vector_{slot}"),
                             inherit_block_vector=vector_link is not None, lora_hash=node.get("properties", {}).get("lora_hashes", ["", "", ""])[slot - 1]))
    fields = dict(model_type=named(stack_nodes[0], "model_type"), slots_json=json.dumps(rows, separators=(",", ":")),
                  global_block_vector=named(root.nodes[1036], "value"), civitai_lookup=named(stack_nodes[0], "civitai_lookup"),
                  **root.nodes[1055]["widgets_values_named"])
    root.nodes[1055] = make_node(root.nodes[1055], "DonutLoRALoader", [input_port("model", "MODEL"), input_port("clip", "CLIP"), input_port("lora_stack", "LORA_STACK", optional=True)],
                                [("model", "MODEL"), ("clip", "CLIP"), ("lora_stack", "LORA_STACK"), ("show_help", "STRING")], fields, title="LoRAs · add / remove / reorder")
    root.remove(855, 1116, 1036)

    # Preserve all 38 per-block defaults while replacing two primitive fan-outs.
    old_merge = deepcopy(merge.nodes[1119])
    old_merge.update(id=1124, pos=root.nodes[1124]["pos"], inputs=deepcopy(root.nodes[1124]["inputs"]))
    old_merge["widgets_values_named"].update(ratio_mode="Grouped", body_ratio=named(merge.nodes[1123], "value"), fusion_ratio=named(merge.nodes[1121], "value"))
    old_merge["widgets_values"] += ["Grouped", named(merge.nodes[1123], "value"), named(merge.nodes[1121], "value")]
    old_merge["title"] = "Krea2 merge · body / text fusion"
    root.nodes[1124] = old_merge
    workflow["definitions"]["subgraphs"] = [g for g in workflow["definitions"]["subgraphs"] if g["id"] != merge_id]

    instance = root.nodes[1014]
    def expose(name, typ, default, source_slot=None, label=None):
        graph_slot = len(main.data["inputs"])
        main.data["inputs"].append(dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"donut-streamline/{main_id}/{name}")), name=name, type=typ, linkIds=[], label=label or name,
                                               pos=[main.data["inputNode"]["bounding"][0] + main.data["inputNode"]["bounding"][2] - 24,
                                                    main.data["inputNode"]["bounding"][1] + 24 + graph_slot * 20]))
        main.data["inputNode"]["bounding"][3] = max(main.data["inputNode"]["bounding"][3], 48 + (graph_slot + 1) * 20)
        instance["inputs"].append(input_port(name, typ, True))
        if label: instance["inputs"][-1]["label"] = label
        instance["widgets_values_named"][name] = default
        instance["widgets_values"].append(default)
        if source_slot is not None: root.connect(195, source_slot, 1014, len(instance["inputs"]) - 1, typ)
        return graph_slot
    ports = {
        "upscale_1": expose("seed_upscale_1", "INT", text_seed + 2, 2),
        "upscale_2": expose("seed_upscale_2", "INT", text_seed + 3, 3),
        "face": expose("seed_face", "INT", text_seed + 4, 4),
        "text": expose("text_seed", "INT", text_seed, 0),
        "enable_1": expose("upscale_1_enabled", "BOOLEAN", main.nodes[989]["mode"] != 4, label="Enable first upscale"),
        "enable_2": expose("upscale_2_enabled", "BOOLEAN", main.nodes[983]["mode"] != 4, label="Enable second upscale"),
    }
    for node_id in (998, 1001, 1012, 1010, 1007, 1006, 1013, 1011): main.bypass_text(node_id)
    for node_id, name in ((989, "upscale_1"), (983, "upscale_2"), (984, "face")):
        main.connect(-10, ports[name], node_id, main.port(node_id, "seed"), "INT")
    main.remove(992, 986, 982, 1000)

    fields = dict(edit_negative=named(main.nodes[1008], "Text"), separator=named(main.nodes[996], "separator"), text_seed=text_seed)
    for old_id, output in ((1008, 2), (995, 3), (990, 4), (994, 5), (1002, 6)):
        main.source(old_id, 0, 996, output)
    # Face-specific edit text is not replaced with the combined scene prompt.
    for edge in main.edges:
        if edge["target_id"] == 984 and edge["target_slot"] == main.port(984, "edit_prompt"):
            edge.update(origin_id=996, origin_slot=1)
    main.nodes[996] = make_node(main.nodes[996], "DonutPromptConditioning", [input_port("clip", "CLIP"), input_port("face", "STRING"), input_port("scene", "STRING"), input_port("negative", "STRING"), input_port("text_seed", "INT", True)],
                                [("full_text", "STRING"), ("face_text", "STRING"), ("edit_negative", "STRING"), ("positive", "CONDITIONING"), ("face_positive", "CONDITIONING"), ("negative_zeroed", "CONDITIONING"), ("negative_raw", "CONDITIONING")], fields)
    main.edges = [e for e in main.edges if e["target_id"] != 996]
    main.remove(1008, 995, 990, 994, 1002, 997)
    for source_name, target_slot, kind in (("clip", 0, "CLIP"), ("prompt", 1, "STRING"), ("prompt_1", 2, "STRING"), ("text", 3, "STRING")):
        main.connect(-10, main.port(-10, source_name), 996, target_slot, kind)
    main.connect(-10, ports["text"], 996, 4, "INT")
    for node_id, name in ((989, "enable_1"), (983, "enable_2")):
        node = main.nodes[node_id]
        enabled = node["mode"] != 4
        node["mode"] = 0
        node["inputs"].append(input_port("enabled", "BOOLEAN", True, True))
        node["widgets_values"].append(enabled); node["widgets_values_named"]["enabled"] = enabled
        main.connect(-10, ports[name], node_id, len(node["inputs"]) - 1, "BOOLEAN")
    root.remove(1026)
    instance["title"] = "Sampling / upscales / face · stage controls"
    instance["size"] = [350, 1080]
    main.data["name"] = "Sampling · upscales · face"

    # Compact root layout. Model loading, prompt authoring, stages and previews
    # remain distinct; no opaque all-in-one sampling supernode is introduced.
    positions = {1122: (-1900,-1180),1120: (-1900,-1050),1124: (-1550,-1150),1045: (-1900,-890),1042: (-1550,-900),430: (-1900,-700),468: (-1900,-480),942: (-1900,-360),50: (-1900,-210),52: (-1900,-80),487: (-1550,-680),195: (-1550,-490),893: (-1550,-150),1055: (-1200,-1200),53: (-650,-1200),56: (-650,-860),57: (-650,-530),891: (-650,-280),716: (-1900,90),881: (-650,-130),883: (-650,240),1033: (-650,370),897: (-650,470),1014: (-100,-1200),912: (300,-1140),913: (870,-1140),914: (1460,-1140),64: (2030,-1110),759: (300,-1230),63: (590,-1230)}
    for node_id, pos in positions.items(): root.nodes[node_id]["pos"] = list(pos)
    root.nodes[1055]["size"] = [480, 960]
    root.nodes[195]["size"] = [300, 250]
    for node_id in (53,56): root.nodes[node_id]["size"] = [480,300]
    root.nodes[57]["size"] = [480,220]
    root.nodes[487]["flags"]["collapsed"] = True
    workflow["extra"].pop("linearData", None)  # It pointed at deleted/nonexistent graph IDs.
    workflow["extra"]["donut_streamlining"] = {"schema": 1, "required_branch": BRANCH,
        "source_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
        "retained_packs": sorted(report["retained_reasons"])}
    workflow["extra"]["ds"] = {"scale": 0.6, "offset": [2020, 1390]}
    main.save(); root.save()
    workflow["last_node_id"] = max(n["id"] for g in graphs(workflow) for n in g["nodes"])
    workflow["revision"] = workflow.get("revision", 0) + 1
    validate(workflow)
    remaining = {n.get("properties", {}).get("cnr_id") for g in graphs(workflow) for n in g["nodes"]}
    if remaining & REMOVED_PACKS: raise ValueError(f"Unexpected remaining utility dependencies: {remaining & REMOVED_PACKS}")
    report.update(after_nodes=sum(len(g["nodes"]) for g in graphs(workflow)), lora_slots=len(rows),
                  seeds={"text": text_seed, "base": text_seed, "upscale_1": text_seed + 2, "upscale_2": text_seed + 3, "face": text_seed + 4, "filename": filename_seed},
                  structural_validation="passed", live_comfyui_gpu_validation="not performed")
    return workflow, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.source.suffix == ".gz" else open
    with opener(args.source, "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    output, report = migrate(source)
    if args.output.resolve() == args.source.resolve(): raise ValueError("Choose a new output path; the source is never overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()

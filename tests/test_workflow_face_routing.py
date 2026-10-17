import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ("workflows/v5/DonutWF_v5.json", "workflows/v6/DonutWF_v6.json")


def face_branch(path, node_id=1180):
    workflow = json.loads((ROOT / path).read_text())
    for subgraph in workflow["definitions"]["subgraphs"]:
        nodes = {node["id"]: node for node in subgraph["nodes"]}
        if node_id in nodes:
            return nodes, {link["id"]: link for link in subgraph["links"]}
    raise AssertionError(f"{path} has no node {node_id}")


class WorkflowFaceRoutingTests(unittest.TestCase):
    def test_face_detailer_gets_only_the_face_instruction(self):
        for path in WORKFLOWS:
            with self.subTest(path):
                nodes, links = face_branch(path)
                conditioning = nodes[1180]
                names = [output["name"] for output in conditioning["outputs"]]
                inputs = {item["name"]: links[item["link"]] for item in conditioning["inputs"] if item.get("link")}
                # The whole-image prompt is only context for the scene input.
                self.assertEqual(inputs["face"]["origin_id"], -10)
                self.assertEqual(inputs["scene"]["origin_id"], 996)
                detailer_text = [link for link in links.values() if link["origin_id"] == 1180 and link["target_id"] == 984]
                detailer_cond = [link for link in links.values() if link["origin_id"] == 1180 and link["target_id"] == 1118]
                self.assertEqual([names[link["origin_slot"]] for link in detailer_text], ["face_text"])
                self.assertEqual([names[link["origin_slot"]] for link in detailer_cond], ["face_positive"])

    def test_v6_face_prompt_starts_with_the_expanded_style(self):
        nodes, links = face_branch(WORKFLOWS[1], 1179)
        expansion = nodes[1179]
        prefix = next(item for item in expansion["inputs"] if item["name"] == "prefix")
        link = links[prefix["link"]]
        # Style box, expanded by the "Face related only" node, then the face prompt.
        self.assertEqual((link["origin_id"], link["origin_slot"]), (1128, 9))
        self.assertEqual(expansion["widgets_values_named"]["separator"], " ")
        output = next(item for item in nodes[1128]["outputs"] if item["name"] == "text_3")
        self.assertIn(prefix["link"], output["links"])


if __name__ == "__main__":
    unittest.main()

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/donut_dependency_check.js", import.meta.url), "utf8")
  .replace(/^import .*;\n/, "");

function setup() {
  let extension;
  const app = { registerExtension(value) { extension = value; } };
  const elements = [];
  const document = {
    createElement(tag) {
      assert.equal(tag, "textarea");
      const element = { style: {}, setAttribute(name, value) { this[name] = value; } };
      elements.push(element);
      return element;
    },
  };
  vm.runInNewContext(source, { app, document });
  const node = {
    comfyClass: "DonutDependencyCheck", size: [200, 100], widgets: [],
    addDOMWidget(name, type, element, options) {
      const widget = { name, type, element, options };
      this.widgets.push(widget);
      return widget;
    },
    setSize(value) { this.size = value; },
    setDirtyCanvas() { this.dirty = true; },
  };
  return { extension, node, elements };
}

test("only the diagnostic node is modified", () => {
  const { extension, node, elements } = setup();
  node.comfyClass = "DonutApplyLoRAStack";
  extension.nodeCreated(node);
  assert.equal(elements.length, 0);
  assert.equal(node.widgets.length, 0);
});

test("report is selectable read-only text and not serialized into workflows", () => {
  const { extension, node, elements } = setup();
  extension.nodeCreated(node);
  assert.equal(elements[0].readOnly, true);
  assert.equal(node.widgets[0].options.serialize, false);
  assert.equal(node.widgets[0].options.getValue(), "");
  assert.equal(node.size[0], 520);
});

test("nodeCreated is idempotent", () => {
  const { extension, node, elements } = setup();
  extension.nodeCreated(node);
  extension.nodeCreated(node);
  assert.equal(elements.length, 1);
  assert.equal(node.widgets.length, 1);
});

test("reports are displayed literally, not evaluated as HTML", () => {
  const { extension, node, elements } = setup();
  extension.nodeCreated(node);
  node.onExecuted({ text: ["<script>unsafe()</script>", "NumPy failure"] });
  assert.equal(elements[0].value, "<script>unsafe()</script>\nNumPy failure");
  assert.equal(elements[0].innerHTML, undefined);
  assert.equal(node.dirty, true);
});

test("execution preserves earlier callback context and return value", () => {
  const { extension, node } = setup();
  let context, received;
  node.onExecuted = function (message) { context = this; received = message; return 42; };
  extension.nodeCreated(node);
  const message = { text: ["ok"] };
  assert.equal(node.onExecuted(message), 42);
  assert.equal(context, node);
  assert.equal(received, message);
});

test("empty output does not crash the UI", () => {
  const { extension, node, elements } = setup();
  extension.nodeCreated(node);
  node.onExecuted({});
  assert.equal(elements[0].value, "");
});

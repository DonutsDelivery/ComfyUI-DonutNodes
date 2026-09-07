// Regression tests for the actual preset-label mutation callback, without a browser.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("../web/donut_krea2_fusion_control.js", import.meta.url), "utf8")
  .replace(/^import .*;\n/m, "");
const settings = ["tap_method", "tap_profile", "per_layer_weights", "tap_strength",
  "tap_formula", "tap_normalization", "projector_method", "projector_profile",
  "projector_layer_weights", "projector_strength", "projector_formula",
  "projector_normalization", "fusion_method", "fusion_strength"];

function makeNode(preset, mode) {
  let extension;
  const context = vm.createContext({ app: { registerExtension(value) { extension = value; } } });
  vm.runInContext(source, context);
  class Node {
    constructor() {
      this.size = [400, 400];
      this.widgets = [...settings, "compatibility_preset", "ui_mode"].map(name => ({
        name, type: "number", value: 1, computeSize: () => [100, 20],
      }));
      this.widgets.find(w => w.name === "compatibility_preset").value = preset;
      this.widgets.find(w => w.name === "ui_mode").value = mode;
    }
    computeSize() { return [400, 400]; }
    setSize(value) { this.size = value; }
  }
  extension.beforeRegisterNodeDef(Node, { name: "DonutKrea2FusionControl" });
  const node = new Node();
  node.onNodeCreated();
  return node;
}
const widget = (node, name) => node.widgets.find(w => w.name === name);
function edit(node, name, value) {
  const item = widget(node, name);
  item.value = value;
  item.callback?.(value);
}

for (const mode of ["Simple", "Advanced"]) {
  for (const preset of ["UncensorFix", "TeacherFix", "DONUT settings: Krea2 C33 TeacherFix EMA5000"]) {
    test(`${mode}: ${preset} survives strength edits synchronously`, () => {
      const node = makeNode(preset, mode);
      for (const value of [0, .75, 1, 2]) {
        edit(node, "tap_strength", value);
        assert.equal(widget(node, "compatibility_preset").value, preset);
        assert.equal(widget(node, "tap_strength").value, value);
      }
    });
  }
}

test("UncensorFix stays active while other Advanced controls are edited", () => {
  const node = makeNode("UncensorFix", "Advanced");
  for (const name of settings) {
    edit(node, name, name.endsWith("profile") ? "off" : 1);
    assert.equal(widget(node, "compatibility_preset").value, "UncensorFix", name);
  }
});

test("ordinary COPY-style presets still become Custom when edited", () => {
  const node = makeNode("Balanced", "Advanced");
  edit(node, "tap_strength", .75);
  assert.equal(widget(node, "compatibility_preset").value, "Custom");
});

test("explicitly selecting Custom disables UncensorFix rather than restoring it", () => {
  const node = makeNode("UncensorFix", "Advanced");
  edit(node, "compatibility_preset", "Custom");
  edit(node, "tap_strength", .75);
  assert.equal(widget(node, "compatibility_preset").value, "Custom");
});

test("applying a different preset does not keep UncensorFix selected", () => {
  const node = makeNode("UncensorFix", "Advanced");
  const preset = "COPY settings: Krea2FilterBypass 2vector";
  edit(node, "compatibility_preset", preset);
  assert.equal(widget(node, "compatibility_preset").value, preset);
  assert.equal(widget(node, "projector_method").value, "Krea2FilterBypass 2vector diff");
});

// Exercise both real extensions together so a rename cannot leave the preset
// table, legacy-workflow migration and strength callback out of sync.
const modeSource = fs.readFileSync(new URL("../web/donut_krea2_fusion_simple_mode.js", import.meta.url), "utf8")
  .replace(/^import .*;\n/m, "");

function makeCombinedNode(preset, mode, reverse) {
  const extensions = [];
  const tasks = [];
  const context = vm.createContext({
    app: { registerExtension(extension) { extensions.push(extension); } },
    queueMicrotask(fn) { tasks.push(fn); },
  });
  for (const script of reverse ? [modeSource, source] : [source, modeSource]) {
    vm.runInContext(`(() => {\n${script}\n})()`, context);
  }
  class Node {
    constructor() {
      this.size = [400, 400];
      this.widgets = [...settings, "compatibility_preset", "ui_mode"].map(name => ({
        name, type: "number", value: 1, computeSize: () => [100, 20],
      }));
      widget(this, "compatibility_preset").value = preset;
      widget(this, "ui_mode").value = mode;
    }
    computeSize() { return [400, 400]; }
    setSize(value) { this.size = value; }
  }
  for (const extension of extensions) {
    extension.beforeRegisterNodeDef(Node, { name: "DonutKrea2FusionControl" });
  }
  const flush = () => {
    let remaining = 1000;
    while (tasks.length) {
      assert.ok(remaining-- > 0, "callbacks must settle");
      tasks.shift()();
    }
  };
  const node = new Node();
  node.onNodeCreated();
  flush();
  return { node, flush };
}

for (const reverse of [false, true]) {
  for (const mode of ["Simple", "Advanced"]) {
    for (const oldName of ["UncensorFix", "TeacherFix", "DONUT settings: Krea2 C33 TeacherFix EMA5000"]) {
      test(`combined extensions: ${mode}, reverse=${reverse}, saved=${oldName}`, () => {
        const { node, flush } = makeCombinedNode(oldName, mode, reverse);
        node.onConfigure();
        flush();
        assert.equal(widget(node, "compatibility_preset").value, "UncensorFix");
        edit(node, "compatibility_preset", "UncensorFix");
        flush();
        assert.equal(widget(node, "tap_profile").value, "off");
        assert.equal(widget(node, "projector_profile").value, "off");
        assert.equal(widget(node, "fusion_method").value, "Standard Krea2 fusion");
        edit(node, "tap_strength", .75);
        assert.equal(widget(node, "compatibility_preset").value, "UncensorFix");
        flush();
        assert.equal(widget(node, "tap_strength").value, .75);
        assert.equal(widget(node, "compatibility_preset").value, "UncensorFix");
        assert.equal(widget(node, "tap_strength").hidden, false);
        if (mode === "Simple") {
          for (const name of settings.filter(name => name !== "tap_strength")) {
            assert.equal(widget(node, name).hidden, true, name);
          }
        }
        edit(node, "compatibility_preset", "Custom");
        flush();
        edit(node, "tap_strength", .5);
        flush();
        assert.equal(widget(node, "compatibility_preset").value, "Custom");
        edit(node, "compatibility_preset", "Bypass 2");
        flush();
        assert.equal(widget(node, "compatibility_preset").value, "Bypass 2");
        assert.equal(widget(node, "projector_method").value, "Krea2FilterBypass 2vector diff");
      });
    }
  }
}

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

function makeWidget(name, value, values = undefined) {
  return { name, value, options: values ? { values } : {}, callback() {} };
}

function makeFusion(id = 7) {
  const widgets = [
    makeWidget("compatibility_preset", "Balanced", ["Balanced", "Rebalance", "UncensorFix"]),
    makeWidget("tap_method", "Donut 12-tap gains"),
    makeWidget("tap_profile", "classic"),
    makeWidget("tap_strength", 1),
    makeWidget("tap_formula", "scale_around_1"),
    makeWidget("tap_normalization", "tensor_rms"),
    makeWidget("per_layer_weights", ""),
    makeWidget("projector_method", "Donut projector-input gains"),
    makeWidget("projector_profile", "off"),
    makeWidget("projector_strength", 1),
    makeWidget("projector_formula", "scale_around_1"),
    makeWidget("projector_normalization", "none"),
    makeWidget("projector_layer_weights", ""),
    makeWidget("fusion_method", "Standard Krea2 fusion"),
    makeWidget("fusion_strength", 1),
  ];
  return { id, type: "DonutKrea2FusionControl", widgets, setDirtyCanvas() {} };
}

function widget(node, name) {
  return node.widgets.find(item => item.name === name);
}

function load(rootGraph) {
  let extension;
  const queued = [];
  const context = vm.createContext({
    app: { rootGraph, canvas: {}, registerExtension(value) { extension = value; } },
    queueMicrotask(callback) { queued.push(callback); },
  });
  const source = readFileSync(new URL("../web/donut_krea2_fusion_experiments.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "");
  vm.runInContext(source, context);
  return { extension, flush() { queued.splice(0).forEach(callback => callback()); } };
}

test("experimental preset choices and recipes are applied to a Fusion node", () => {
  const fusion = makeFusion();
  const rootGraph = { nodes: [fusion] };
  const { extension, flush } = load(rootGraph);
  extension.afterConfigureGraph(); flush();

  const preset = widget(fusion, "compatibility_preset");
  const names = preset.options.values;
  assert.ok(names.includes("Experiment · NAG-friendly mean"));
  assert.ok(names.includes("Experiment · NAG-friendly static RMS"));
  assert.ok(names.includes("Experiment · NAG-friendly power 0.60"));
  assert.ok(names.includes("Experiment · soft tensor RMS 0.75"));

  preset.value = "Experiment · NAG-friendly mean"; preset.callback(preset.value);
  assert.equal(widget(fusion, "tap_normalization").value, "mean_gain");
  assert.equal(widget(fusion, "tap_strength").value, 1);
  assert.equal(widget(fusion, "tap_formula").value, "scale_around_1");

  preset.value = "Experiment · NAG-friendly static RMS"; preset.callback(preset.value);
  assert.equal(widget(fusion, "tap_normalization").value, "rms_gain");

  preset.value = "Experiment · NAG-friendly power 0.60"; preset.callback(preset.value);
  assert.equal(widget(fusion, "tap_normalization").value, "none");
  assert.equal(widget(fusion, "tap_formula").value, "geometric_power");
  assert.equal(widget(fusion, "tap_strength").value, 0.60);

  preset.value = "Experiment · soft tensor RMS 0.75"; preset.callback(preset.value);
  assert.equal(widget(fusion, "tap_normalization").value, "tensor_rms");
  assert.equal(widget(fusion, "tap_formula").value, "scale_around_1");
  assert.equal(widget(fusion, "tap_strength").value, 0.75);
  assert.equal(preset.value, "Experiment · soft tensor RMS 0.75");
});

test("outer V4 compatibility selector drives the nested Fusion node", () => {
  const fusion = makeFusion(22);
  const outerPreset = makeWidget("compatibility_preset", "Balanced", ["Balanced", "Rebalance"]);
  const stage = { id: 10, widgets: [outerPreset], subgraph: { nodes: [fusion] } };
  const rootGraph = { nodes: [stage] };
  const { extension, flush } = load(rootGraph);
  extension.afterConfigureGraph(); flush();

  outerPreset.value = "Experiment · NAG-friendly power 0.60";
  outerPreset.callback(outerPreset.value);
  assert.equal(widget(fusion, "tap_formula").value, "geometric_power");
  assert.equal(widget(fusion, "tap_strength").value, 0.60);
  assert.equal(widget(fusion, "compatibility_preset").value, outerPreset.value);

  extension.afterConfigureGraph(); flush();
  const values = outerPreset.options.values;
  assert.equal(values.filter(value => value === "Experiment · NAG-friendly power 0.60").length, 1);
});

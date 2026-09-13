import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.events = {};
    const classes = new Set();
    this.style = { setProperty() {} }; this.classList = {
      add: (...names) => names.forEach(name => classes.add(name)),
      toggle: (name, force) => { const value = force === undefined ? !classes.has(name) : force; if (value) classes.add(name); else classes.delete(name); return value; },
      contains: name => classes.has(name),
    };
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(name, callback) { this.events[name] = callback; }
  querySelectorAll(selector) {
    const matches = [];
    const visit = item => {
      for (const child of item.children) {
        const classMatch = selector.startsWith(".") && String(child.className || "").split(/\s+/).includes(selector.slice(1));
        const tagMatch = !selector.startsWith(".") && child.tagName === selector.toUpperCase();
        if (classMatch || tagMatch) matches.push(child);
        visit(child);
      }
    };
    visit(this); return matches;
  }
}

const findControl = (root, title) => {
  const stack = [root];
  while (stack.length) {
    const item = stack.pop();
    if (item["aria-label"] === title) return item;
    stack.push(...item.children);
  }
};

test("workflow panel refreshes dependent Fusion controls after a preset selection", () => {
  let extension, Panel;
  const deferred = [];
  const fusion = {
    widgets: [], graph: { beforeChange() {}, afterChange() {} }, setDirtyCanvas() {},
  };
  const add = (name, value, callback = () => {}) => {
    const widget = { name, value, callback, options: { values: [] } };
    fusion.widgets.push(widget); return widget;
  };
  add("tap_method", "Donut 12-tap gains");
  const profile = add("tap_profile", "classic");
  const normalization = add("tap_normalization", "none");
  const weights = add("per_layer_weights", Array(12).fill("1.0").join(","));
  const method = add("fusion_method", "Standard Krea2 fusion");
  const composition = add("uncensorfix_controls", "Fusion + UncensorFix weights");
  const preset = add("compatibility_preset", "Custom");
  const projectorStrength = add("projector_strength", 1);
  const innerStrength = add("tap_strength", 1, value => { projectorStrength.value = value; });
  preset.options.values = ["Custom", "UncensorFix", "Balanced", "Balanced + Enhancer"];
  profile.options.values = ["off", "classic"];
  method.options.values = ["Standard Krea2 fusion"];

  const nestedGraph = { getNodeById: id => id === 1118 ? fusion : undefined };
  const outerPreset = {name: "compatibility_preset", value: "Custom", options: preset.options};
  const outerStrength = {name: "tap_strength", value: 1};
  const group = { subgraph: nestedGraph, widgets: [outerPreset, outerStrength], graph: fusion.graph, setDirtyCanvas() {} };
  const rootGraph = { getNodeById: id => id === 1014 ? group : undefined };
  class LGraphNode {
    constructor(title) { this.title = title; this.widgets = []; this.properties = {}; this.size = [460, 510]; }
    addDOMWidget(name, type, element, options) { const widget = { name, type, element, options }; this.widgets.push(widget); return widget; }
    setDirtyCanvas() {}
  }
  const context = vm.createContext({
    app: { rootGraph, registerExtension(value) { extension = value; } },
    api: {}, LGraphNode, LiteGraph: { registerNodeType(_name, type) { Panel = type; } },
    document: { activeElement: null, createElement: tag => new Element(tag) },
    IntersectionObserver: class { observe() {} disconnect() {} },
    queueMicrotask(callback) { deferred.push(callback); }, crypto: { randomUUID: () => "test" },
    createLoraService: () => ({}), decodeRows: () => [], moveRow: () => [],
    createLoraToolbar: () => new Element("div"), renderLoraInformation: () => ({}),
    weightControl: (title, get, set) => {
      const element = new Element("input"); element.setAttribute("aria-label", title);
      element.events.input = () => set(Number(element.value));
      return {element, refresh: () => { element.value = get(); }};
    }, vectorControl: () => ({}), promptTools: () => ({}), wildcardLibrary: () => new Element("div"),
    fitModule() {}, fitTextarea() {}, scheduleLayout() {},
  });
  const source = fs.readFileSync(new URL("../web/donut_app_controls.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "");
  vm.runInContext(source, context);
  extension.registerCustomNodes();
  const panel = new Panel();
  panel.properties.donut_app_controls = { groups: [{ title: "Fusion", controls: [
    { path: [1014], widget: "compatibility_preset", title: "Compatibility preset" },
    { path: [1014], widget: "tap_strength", title: "Tap strength" },
    { path: [1014, 1118], widget: "tap_method", title: "Tap method" },
    { path: [1014, 1118], widget: "tap_profile", title: "Tap profile" },
    { path: [1014, 1118], widget: "fusion_method", title: "Fusion method" },
  ] }] };
  panel._donutAppControls.render();

  const selector = findControl(panel._donutAppControls.root, "Compatibility preset");
  selector.value = "UncensorFix";
  selector.events.change();
  deferred.splice(0).forEach(callback => callback());

  assert.equal(findControl(panel._donutAppControls.root, "Tap profile").value, "off");
  assert.equal(findControl(panel._donutAppControls.root, "Fusion method").value, "Standard Krea2 fusion");
  const strength = findControl(panel._donutAppControls.root, "Tap strength");
  strength.value = "0.65";
  strength.events.input();
  assert.equal(outerStrength.value, 0.65);
  assert.equal(innerStrength.value, 0.65);
  assert.equal(projectorStrength.value, 0.65, "outer strength invokes dependent inner callback");

  selector.value = "Balanced";
  selector.events.change();
  deferred.splice(0).forEach(callback => callback());
  assert.equal(profile.value, "classic");
  assert.equal(normalization.value, "tensor_rms");
  assert.equal(weights.value, "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0");
  assert.equal(method.value, "Standard Krea2 fusion");
  assert.equal(composition.value, "Fusion only");

  selector.value = "Balanced + Enhancer";
  selector.events.change();
  deferred.splice(0).forEach(callback => callback());
  assert.equal(profile.value, "classic");
  assert.equal(normalization.value, "tensor_rms");
  assert.equal(weights.value, "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0");
  assert.equal(method.value, "capitan01R Krea2T-Enhancer operation");
  assert.equal(composition.value, "Fusion only");
});

test("App Mode adds a LoRA before its native editor has initialized", () => {
  let extension, Panel;
  const state = { name: "slots_json", value: "[]", callback() {} };
  const target = {
    widgets: [state, { name: "model_type", value: "Auto" }, { name: "civitai_lookup", value: "Off" }],
    graph: { beforeChange() {}, afterChange() {} }, setDirtyCanvas() {},
  };
  const stage = { subgraph: { getNodeById: id => id === 1055 ? target : undefined } };
  const rootGraph = { getNodeById: id => id === 1138 ? stage : undefined };
  class LGraphNode {
    constructor(title) { this.title = title; this.widgets = []; this.properties = {}; this.size = [460, 510]; }
    addDOMWidget(name, type, element, options) { const widget = { name, type, element, options }; this.widgets.push(widget); return widget; }
    setDirtyCanvas() {}
  }
  const context = vm.createContext({
    app: { rootGraph, registerExtension(value) { extension = value; } },
    api: {}, LGraphNode, LiteGraph: { registerNodeType(_name, type) { Panel = type; } },
    document: { activeElement: null, createElement: tag => new Element(tag) },
    IntersectionObserver: class { observe() {} disconnect() {} }, queueMicrotask() {},
    createLoraService: () => ({ catalog: async () => ({ loras: ["None", "example.safetensors"], presets: ["None"] }) }),
    decodeRows: value => JSON.parse(value || "[]"), moveRow: rows => rows,
    createLoraToolbar: () => new Element("div"), renderLoraInformation: () => ({}),
    weightControl: () => ({ element: new Element("input"), refresh() {} }), vectorControl: () => ({}),
    promptTools: () => ({}), wildcardLibrary: () => new Element("div"),
    fitModule() {}, fitTextarea() {}, scheduleLayout() {},
  });
  const source = fs.readFileSync(new URL("../web/donut_app_controls.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "");
  vm.runInContext(source, context);
  extension.registerCustomNodes();
  const panel = new Panel();
  panel.properties.donut_app_controls = { groups: [{ title: "LoRAs", loras: [1138, 1055] }] };
  panel._donutAppControls.render();
  const walk = value => [value, ...value.children.flatMap(walk)];
  const add = walk(panel._donutAppControls.root).find(item => item.textContent === "Add LoRA");
  assert.ok(add);
  assert.doesNotThrow(() => add.onclick());
  const rows = JSON.parse(state.value);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].lora_name, "None");
  assert.ok(rows[0].id);
});

test("prompt variants can duplicate the connected Prompt and an existing row", () => {
  let extension, Panel;
  const state = { name: "prompt_sets_json", value: "[]", callback() {} };
  const index = { name: "prompt_set_index", value: 1 };
  const variant = { widgets: [state, index], graph: { beforeChange() {}, afterChange() {} }, setDirtyCanvas() {} };
  const variantGroup = { subgraph: { getNodeById: id => id === 996 ? variant : undefined } };
  const source = text => ({ widgets: [{ name: "Text", value: text }] });
  const stage = {
    subgraph: {
      getNodeById: id => ({ 1125: source("face"), 1126: source("scene"), 1127: source("negative") }[id]),
    },
  };
  const rootGraph = {
    getNodeById: id => ({ 1014: variantGroup, 1138: stage }[id]),
  };
  class LGraphNode {
    constructor(title) { this.title = title; this.widgets = []; this.properties = {}; this.size = [460, 510]; }
    addDOMWidget(name, type, element, options) { const widget = { name, type, element, options }; this.widgets.push(widget); return widget; }
    setDirtyCanvas() {}
  }
  const context = vm.createContext({
    app: { rootGraph, registerExtension(value) { extension = value; } },
    api: {}, LGraphNode, LiteGraph: { registerNodeType(_name, type) { Panel = type; } },
    document: { activeElement: null, createElement: tag => new Element(tag) },
    IntersectionObserver: class { observe() {} disconnect() {} }, queueMicrotask() {},
    createLoraService: () => ({}), decodeRows: () => [], moveRow: () => [],
    createLoraToolbar: () => new Element("div"), renderLoraInformation: () => ({}),
    weightControl: () => ({ element: new Element("input"), refresh() {} }), vectorControl: () => ({}),
    promptTools: () => ({ element: new Element("div"), refresh() {} }), wildcardLibrary: () => new Element("div"),
    fitModule() {}, fitTextarea() {}, scheduleLayout() {}, crypto: { randomUUID: (() => { let id = 0; return () => `test-${++id}`; })() },
  });
  const sourceCode = fs.readFileSync(new URL("../web/donut_app_controls.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "");
  vm.runInContext(sourceCode, context);
  extension.registerCustomNodes();
  const panel = new Panel();
  panel.properties.donut_app_controls = { groups: [
    { title: "Prompt", shared_prompt_tools: true, controls: [
      { path: [1138, 1125], widget: "Text", title: "Face and general style", ui: { prompt: true } },
      { path: [1138, 1126], widget: "Text", title: "Prompt · subject and scene", ui: { prompt: true } },
      { path: [1138, 1127], widget: "Text", title: "Negative prompt", ui: { prompt: true } },
    ] },
    { title: "Prompts", prompt_sets: [1014, 996], prompt_source_paths: [[1138, 1125], [1138, 1126], [1138, 1127]] },
  ] };
  panel._donutAppControls.render();
  const walk = value => [value, ...value.children.flatMap(walk)];
  const base = walk(panel._donutAppControls.root).find(item => item.tagName === "SECTION");
  assert.ok(base.classList.contains("donut-prompt-active"));
  const duplicatePrompt = walk(panel._donutAppControls.root).find(item => item.textContent === "Duplicate Prompt");
  assert.ok(duplicatePrompt);
  duplicatePrompt.onclick();
  assert.deepEqual(JSON.parse(state.value)[0], { id: "test-1", face: "face", scene: "scene", negative: "negative" });

  index.value = 2;
  panel._donutAppControls.render();
  assert.ok(!walk(panel._donutAppControls.root).find(item => item.tagName === "SECTION").classList.contains("donut-prompt-active"));
  assert.ok(walk(panel._donutAppControls.root).find(item => item.className === "donut-prompt-set-row")
    .classList.contains("donut-prompt-active"));

  const duplicateRow = walk(panel._donutAppControls.root).find(item => item.textContent === "Duplicate");
  assert.ok(duplicateRow);
  duplicateRow.onclick();
  const rows = JSON.parse(state.value);
  assert.equal(rows.length, 2);
  assert.notEqual(rows[1].id, rows[0].id);
  assert.deepEqual({ face: rows[1].face, scene: rows[1].scene, negative: rows[1].negative },
    { face: rows[0].face, scene: rows[0].scene, negative: rows[0].negative });
});

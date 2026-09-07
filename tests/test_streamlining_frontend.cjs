// Dependency-free frontend contract tests, not a real ComfyUI browser session.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.events = {}; this.textContent = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  setAttribute(key, value) { this[key] = value; }
  addEventListener(name, callback) { this.events[name] = callback; }
  fire(name) { this.events[name]?.({preventDefault() {}}); }
  checkValidity() { return Number.isFinite(this.valueAsNumber) && this.valueAsNumber >= -1000 && this.valueAsNumber <= 1000; }
  get valueAsNumber() { return Number(this.value); }
}
function all(root, predicate) { return [root, ...root.children.flatMap(child => all(child, () => true))].filter(predicate); }
let extension;
const context = vm.createContext({ console, URLSearchParams, Math, Date,
  document: {createElement: tag => new Element(tag)},
  app: {registerExtension: value => { extension = value; }, graph: {change() {}}},
  api: {apiURL: value => value},
  ComfyWidgets: {STRING(node, name) { const widget = {name, value: '', inputEl: {}}; node.widgets.push(widget); return {widget}; }},
});
const source = fs.readFileSync(path.join(__dirname, '../web/donut_workflow.js'), 'utf8')
  .replace(/^import .*;\n/gm, '').replace(/export function /g, 'function ');
vm.runInContext(source, context);
const definition = {name: 'DonutLoRALoader', input: {required: {slots_json: ['STRING', {
  donut_loras: ['None', 'a.safetensors', 'b.safetensors'], donut_presets: ['None', 'KREA2-ALL:1,1', 'SDXL-ALL:1,1'],
}]}}};
const rows = count => Array.from({length: count}, (_, index) => ({id: `row-${index}`, enabled: index % 2 === 0,
  lora_name: `lora-${index}`, model_weight: index / 10, clip_weight: index === 0 ? 0 : -index / 10, block_vector: `${index},1`,
  block_preset: 'None', inherit_block_vector: index % 2 !== 0, lora_hash: `HASH${index}`, custom: {keep: index}}));
function makeNode(name = definition.name, initial = '[]') {
  class Node {
    constructor() {
      this.widgets = [{name: 'model_type', value: 'KREA2'}, {name: 'slots_json', value: initial, type: 'customtext'},
        {name: 'global_block_vector', value: '1,1'}]; this.size = [400, 600]; this.parentEvents = [];
    }
    onNodeCreated() { this.parentEvents.push('created'); }
    onConfigure() { this.parentEvents.push('configured'); }
    onExecuted() { this.parentEvents.push('executed'); }
    onSerialize(data) { this.parentEvents.push('serialized'); data.parent = true; }
    addDOMWidget(name, type, root, options) { this.root = root; const w = {name, type, options}; this.widgets.push(w); return w; }
    setDirtyCanvas() {}
    setSize(size) { this.size = size; }
    computeSize() { return this.size; }
  }
  extension.beforeRegisterNodeDef(Node, {...definition, name});
  return new Node();
}
const state = node => node.widgets.find(w => w.name === 'slots_json');
const saved = node => JSON.parse(state(node).value);
const boxes = node => all(node.root, e => e.tag === 'fieldset');
const click = (root, label) => { const b = all(root, e => e.tag === 'button' && e.textContent === label)[0]; assert.ok(b, label); b.fire('click'); };
const input = (root, label) => all(root, e => e.tag === 'label' && e.textContent === `${label} `)[0].children[0];

test('restore all six slots by named state, preserving disabled rows and metadata', () => {
  const node = makeNode(); node.onNodeCreated(); node.onConfigure({widgets_values_named: {slots_json: JSON.stringify(rows(6))}});
  assert.deepEqual(saved(node), rows(6)); assert.equal(boxes(node).length, 6);
  assert.equal(state(node).type, 'converted-widget'); assert.equal(node.widgets.length, 4);
});
test('adding slots is unbounded by the old three-slot contract', () => {
  const node = makeNode('DonutLoRALoader', JSON.stringify(rows(6))); node.onNodeCreated();
  for (let i = 0; i < 5; i++) click(node.root, '+ Add LoRA');
  assert.equal(saved(node).length, 11); assert.equal(new Set(saved(node).map(row => row.id)).size, 11);
  assert.deepEqual(saved(node).slice(0, 6), rows(6));
});
test('reorder and removal move complete rows, not just filenames', () => {
  const node = makeNode('DonutLoRALoader', JSON.stringify(rows(6))); node.onNodeCreated();
  click(boxes(node)[2], '↑'); const expected = rows(6); [expected[1], expected[2]] = [expected[2], expected[1]];
  assert.deepEqual(saved(node), expected); click(boxes(node)[4], 'Remove'); expected.splice(4, 1);
  assert.deepEqual(saved(node), expected);
});
test('named and positional serialization roundtrip with parent hooks intact', () => {
  const original = rows(7); const node = makeNode('DonutDynamicLoRAStack'); node.onNodeCreated();
  node.onConfigure({widgets_values: ['KREA2', JSON.stringify(original), '1,1']});
  const output = {}; node.onSerialize(output); assert.equal(output.parent, true);
  assert.equal(state(node).serializeValue(), JSON.stringify(original));
  const copy = makeNode(); copy.onNodeCreated(); copy.onConfigure(output); assert.deepEqual(saved(copy), original);
  assert.deepEqual(node.parentEvents, ['created', 'configured', 'serialized']);
});
test('malformed loaded JSON is not replaced with an empty stack', () => {
  const node = makeNode(); node.onNodeCreated(); node.onConfigure({widgets_values_named: {slots_json: '[broken'}});
  const output = {}; node.onSerialize(output); assert.equal(output.widgets_values_named.slots_json, '[broken');
  const repair = all(node.root, e => e.tag === 'textarea')[0]; assert.ok(repair);
  repair.value = JSON.stringify(rows(1)); repair.fire('change'); assert.deepEqual(saved(node), rows(1));
});
test('filename edits clear stale hash and late execution metadata cannot restore it', () => {
  const node = makeNode('DonutLoRALoader', JSON.stringify(rows(1))); node.onNodeCreated();
  const filename = input(boxes(node)[0], 'LoRA'); filename.value = 'new.safetensors'; filename.fire('change');
  assert.equal(saved(node)[0].lora_hash, '');
  node.onExecuted({donut_loras: [{id: 'row-0', lora_name: 'lora-0', lora_hash: 'OLDHASH'}]});
  assert.equal(saved(node)[0].lora_hash, '');
  node.onExecuted({donut_loras: [{id: 'row-0', lora_name: 'new.safetensors', lora_hash: 'NEWHASH'}]});
  assert.equal(saved(node)[0].lora_hash, 'NEWHASH');
});
test('block presets edit the actual vector and disable inheritance', () => {
  const node = makeNode('DonutLoRALoader', JSON.stringify(rows(2))); node.onNodeCreated();
  const box = boxes(node)[1]; const select = all(box, e => e.tag === 'select')[0];
  assert.equal(select.children.some(e => e.value.startsWith('SDXL')), false);
  select.value = 'KREA2-ALL:1,1'; select.fire('change');
  assert.equal(saved(node)[1].block_vector, '1,1'); assert.equal(saved(node)[1].inherit_block_vector, false);
});
test('invalid number does not corrupt canonical row state', () => {
  const node = makeNode('DonutLoRALoader', JSON.stringify(rows(1))); node.onNodeCreated();
  const weight = input(boxes(node)[0], 'Model strength'); weight.value = 'NaN'; weight.fire('change');
  assert.equal(saved(node)[0].model_weight, 0); weight.value = '0.75'; weight.fire('change'); assert.equal(saved(node)[0].model_weight, .75);
});
test('seed controls have unique serialized names', () => {
  const node = makeNode('DonutSeedPlan'); node.widgets = ['text_seed', 'control_after_generate', 'sampler_seed',
    'control_after_generate', 'filename_seed', 'control_after_generate'].map(name => ({name})); node.onNodeCreated();
  assert.deepEqual(node.widgets.map(w => w.name), ['text_seed', 'text_seed_control', 'sampler_seed', 'sampler_seed_control', 'filename_seed', 'filename_seed_control']);
});
test('grouped merge hides controls without deleting or changing their values', () => {
  const node = makeNode('DonutModelMergeKrea2'); node.widgets = [
    {name: 'ratio_mode', value: 'Grouped'}, {name: 'blocks.0.', value: .45, type: 'number'},
    {name: 'txtfusion.projector.', value: .35, type: 'number'}, {name: 'tmlp.', value: .95, type: 'number'}];
  node.onNodeCreated(); assert.equal(node.widgets.length, 4); assert.equal(node.widgets[1].type, 'converted-widget');
  assert.equal(node.widgets[3].type, 'number'); node.widgets[0].value = 'Per block'; node.widgets[0].callback();
  assert.equal(node.widgets[1].type, 'number'); assert.equal(node.widgets[1].value, .45);
});
test('prompt preview is read-only, not an extra backend input', () => {
  const node = makeNode('DonutText'); node.onNodeCreated(); node.onExecuted({text: ['resolved prompt']});
  const preview = node.widgets.find(w => w.name === 'resolved_text');
  assert.equal(preview.value, 'resolved prompt'); assert.equal(preview.options.serialize, false); assert.equal(preview.inputEl.readOnly, true);
});

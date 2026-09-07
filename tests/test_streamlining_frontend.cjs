// Contract tests for the real extension against a small native-widget/DOM harness.
// This is not a full ComfyUI renderer or a GPU test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const clean = value => JSON.parse(JSON.stringify(value));
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.events = {}; this.textContent = ''; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this[name] = value; }
    addEventListener(name, fn) { this.events[name] = fn; }
    contains(child) { return this === child || this.children.some(c => c.contains(child)); }
}
const text = root => root.textContent + root.children.map(text).join(' ');
const elements = (root, tag) => [root, ...root.children.flatMap(c => elements(c))].filter(e => !tag || e.tag === tag);
const settle = async () => { for (let i = 0; i < 16; i++) await new Promise(r => setImmediate(r)); };
const rows = count => Array.from({ length: count }, (_, i) => ({ id: `row-${i}`, enabled: i % 2 === 0,
    lora_name: `folder/model-${i}.safetensors`, model_weight: .75, clip_weight: 0,
    block_vector: '0,1,0', block_preset: 'KREA2-TEST:0,1,0', inherit_block_vector: true,
    lora_hash: '', custom: { preserved: i } }));
function setup(handler = undefined) {
    const requests = []; let extension;
    const api = {
        apiURL: value => `/api${value}`,
        async fetchApi(url, options) {
            requests.push(url);
            if (handler) { const result = await handler(url, options); if (result) return result; }
            const value = url === '/models/loras' ? ['folder/a.safetensors', 'folder/b.safetensors']
                : url.startsWith('/object_info') ? { DonutLoRAStack: { input: { required: {
                    lora_name_1: [['None', 'fallback.safetensors']],
                    block_preset_1: [['None', 'KREA2-TEST:0,1,0', 'KREA2-ALL:1,1,1', 'SDXL-ALL:1,1']]
                } } } }
                : url.includes('/analyze?') ? { found: true, supported: true, tensor_count: 8,
                    components: [{ name: 'UNet', modules: 4, groups: [{ name: 'blocks', indices: [3, 4] }] }] }
                : { name: new URL(url, 'http://local').searchParams.get('name'), hash: 'a'.repeat(10), has_collage: true,
                    civitai: { model_id: 123, model_version_id: 456, model_name: 'Test model', version_name: 'v2',
                        base_model: 'Krea2', recommended_weight: .65, trained_words: ['trigger'], creator_username: 'Author',
                        description: '<script>not executable</script>', model_url: 'javascript:bad()' } };
            return { ok: true, status: 200, json: async () => value };
        }
    };
    const app = { registerExtension: x => { extension = x; }, graph: { change() {} } };
    const context = vm.createContext({ console: { warn() {}, log() {} }, app, api, URLSearchParams, URL,
        Math, Date, AbortController, setTimeout, clearTimeout, queueMicrotask,
        document: { createElement: tag => new Element(tag) },
        ComfyWidgets: { STRING(node, name) { const w = node.addWidget('customtext', name, '', () => {}, {}); w.inputEl = {}; return { widget: w }; } }
    });
    for (const file of ['donut_lora_ui.js', 'donut_native_lora.js', 'donut_workflow_repair.js', 'donut_workflow.js']) {
        const source = fs.readFileSync(path.join(__dirname, '../web', file), 'utf8')
            .replace(/^import .*;\n/gm, '').replace(/^export \{.*\} from .*;\n/gm, '').replace(/export function /g, 'function ');
        vm.runInContext(source, context);
    }
    const backendNames = ['model_type', 'slots_json', 'global_block_vector', 'civitai_lookup', 'safe_stack', 'fusion_aware', 'max_fusion_boost', 'safe_limit', 'execution_mode'];
    function make(initial = rows(3), lookup = 'Off', type = 'DonutLoRALoader') {
        class Node {
            constructor() {
                this.widgets = []; this.inputs = []; this.properties = {}; this.size = [360, 900]; this.parent = [];
                const values = ['KREA2', JSON.stringify(initial), '1,1,1', lookup, 'On', 'Use headroom', 2, 1, 'Experimental bypass'];
                backendNames.forEach((name, i) => {
                    const type = name === 'slots_json' ? 'customtext' : name === 'global_block_vector' ? 'text'
                        : ['max_fusion_boost', 'safe_limit'].includes(name) ? 'number' : 'combo';
                    this.addWidget(type, name, values[i], () => {}, type === 'combo' ? { values: [values[i]] } : {});
                });
            }
            addWidget(type, name, value, callback, options) {
                // ComfyUI_frontend v1.51.9 LGraphNode.ts checks this before adding
                // the widget. The old permissive mock missed the production bug.
                if (type === 'combo' && !options?.values) throw new Error("LiteGraph addWidget('combo',...) requires to pass values in options: { values:['red','blue'] }");
                const w = { type, name, value, callback, options }; this.widgets.push(w); return w;
            }
            addDOMWidget(name, type, element, options) { const w = { type, name, element, options }; this.widgets.push(w); return w; }
            removeWidget(widget) { widget.onRemove?.(); this.widgets.splice(this.widgets.indexOf(widget), 1); }
            computeSize() { return [this.size[0], 70 + this.widgets.reduce((sum, w) => sum + (w.hidden ? 0 : w.options?.getMinHeight?.() || 24), 0)]; }
            setSize(value) { this.size = value; }
            setDirtyCanvas() {}
            onConfigure() { this.parent.push('configured'); }
            onSerialize(data) { data.parent = true; }
            onExecuted() { this.parent.push('executed'); }
            onRemoved() { this.parent.push('removed'); }
        }
        extension.beforeRegisterNodeDef(Node, { name: type, input: { required: { slots_json: ['STRING', {}] } } });
        const node = new Node(); node.onNodeCreated(); return node;
    }
    return { make, requests, api, context, get extension() { return extension; } };
}
const find = (node, suffix, id = 'row-0') => node.widgets.find(w => w.name === `donut_row:${id}:${suffix}`);
const change = (widget, value) => { widget.value = value; widget.callback(value); };
const saved = node => JSON.parse(node.widgets.find(w => w.name === 'slots_json').value);
const panel = (node, id) => find(node, 'information', id).element;
const click = (root, label) => {
    const button = elements(root, 'button').find(b => b.textContent === label);
    assert.ok(button, `Visible button ${label}`);
    button.events.click({ preventDefault() {}, stopPropagation() {} });
};
const rowAction = (node, action, id = 'row-0') => {
    const label = { 'Move up': '↑', 'Move down': '↓' }[action] || action;
    const widget = find(node, ['Remove', 'Move up', 'Move down'].includes(action) ? 'actions' : 'information', id);
    click(widget.element, label);
};

test('native combo lists installed files even when unknown STRING metadata was stripped', async () => {
    const { make } = setup(); const node = make(); await settle();
    const picker = find(node, 'lora_name'); assert.equal(picker.type, 'combo');
    assert.deepEqual(clean(picker.options.values), ['None', 'folder/a.safetensors', 'folder/b.safetensors', 'folder/model-0.safetensors']);
    assert.equal(node.widgets.some(w => w.element && elements(w.element).some(e => ['select', 'datalist', 'input'].includes(e.tag))), false);
});
test('catalog falls back to the old loader schema when the core models route is unavailable', async () => {
    const { make } = setup(url => url === '/models/loras' ? { ok: false, status: 404 } : null);
    const node = make(); await settle(); assert.ok(find(node, 'lora_name').options.values.includes('fallback.safetensors'));
});
test('disabled rows retain native picker, enable and actions but hide every dependent parameter', async () => {
    const { make } = setup(); const node = make(); await settle();
    change(find(node, 'advanced'), true); const before = saved(node); const height = node.size[1];
    change(find(node, 'enabled'), false);
    for (const suffix of ['model_weight', 'clip_weight', 'advanced', 'block_preset', 'inherit', 'block_vector', 'information']) assert.equal(find(node, suffix).hidden, true, suffix);
    for (const suffix of ['lora_name', 'enabled', 'actions']) assert.notEqual(find(node, suffix).hidden, true, suffix);
    assert.ok(node.size[1] < height);
    change(find(node, 'enabled'), true); assert.deepEqual(saved(node), before);
    assert.equal(find(node, 'block_vector').value, '0,1,0'); assert.equal(find(node, 'clip_weight').value, 0);
});
test('disabled rows do not trigger local analysis or CivitAI lookup; Off prevents network lookup', async () => {
    const { make, requests } = setup(); make([{ ...rows(1)[0], enabled: false }], 'On'); await settle();
    assert.equal(requests.some(u => /\/(info|analyze)\?/.test(u)), false);
    const node = make(rows(1), 'Off'); await settle();
    assert.ok(requests.some(u => u.includes('/analyze?'))); assert.equal(requests.some(u => u.includes('/info?')), false);
    change(node.widgets.find(w => w.name === 'civitai_lookup'), 'On'); await settle();
    assert.ok(requests.some(u => u.includes('/info?')));
});
test('lookup shows links, detected components, recommendations, triggers and bounded local previews before queueing', async () => {
    const { make } = setup(); const node = make(rows(1), 'On'); await settle();
    const root = panel(node); const content = text(root);
    for (const expected of ['Detected weights', '8 tensors', 'blocks: 3–4', 'Suggested weight: 0.65', 'Test model', 'Triggers: trigger']) assert.ok(content.includes(expected), expected);
    const links = elements(root, 'a'); assert.ok(links.some(a => a.href === 'https://civitai.com/models/123?modelVersionId=456'));
    assert.ok(links.every(a => a.rel === 'noopener noreferrer'));
    assert.equal(elements(root, 'script').length, 0);
    const image = elements(root, 'img')[0]; assert.equal(image.style.height, '112px');
    assert.match(image.src, /^\/api\/donut\/loras\/preview\?/); assert.ok(find(node, 'information').options.getMaxHeight() <= 360);
});
test('preset combo labels contain no long vectors and selecting a preset edits only its intended fields', async () => {
    const { make } = setup(); const node = make(); await settle();
    const preset = find(node, 'block_preset'); assert.ok(preset.options.values.every(v => !v.includes(':') && !v.includes(',')));
    assert.ok(!preset.options.values.includes('SDXL-ALL'));
    change(preset, 'KREA2-ALL'); assert.equal(saved(node)[0].block_preset, 'KREA2-ALL:1,1,1');
    assert.equal(saved(node)[0].block_vector, '1,1,1'); assert.equal(saved(node)[0].inherit_block_vector, false);
});
test('suggested weight is never applied silently and explicit action does not touch zero CLIP strength', async () => {
    const { make } = setup(); const node = make(rows(1), 'On'); await settle();
    assert.equal(saved(node)[0].model_weight, .75); rowAction(node, 'Use suggested model weight');
    assert.equal(saved(node)[0].model_weight, .65); assert.equal(saved(node)[0].clip_weight, 0);
});
test('add, reorder and remove preserve complete rows and have no six-row ceiling', async () => {
    const { make } = setup(); const node = make(rows(6)); await settle();
    for (let i = 0; i < 4; i++) node.widgets.find(w => w.name === '+ Add LoRA').callback();
    assert.equal(saved(node).length, 10);
    rowAction(node, 'Move up', 'row-2'); assert.equal(saved(node)[1].id, 'row-2');
    assert.deepEqual(saved(node)[1].custom, { preserved: 2 });
    rowAction(node, 'Remove', 'row-2'); assert.equal(saved(node).length, 9);
});
test('workflow and API serializers exclude every row/helper widget; only canonical JSON carries row state', async () => {
    const { make } = setup(); const node = make(rows(6)); await settle();
    const output = {}; node.onSerialize(output); assert.equal(output.parent, true);
    assert.equal(output.widgets_values.length, 9);
    assert.equal(Object.keys(output.widgets_values_named).length, 9);
    assert.equal(node.widgets.filter(w => w.name.startsWith('donut_row:')).every(w => w.serialize === false && w.options.serialize === false), true);
    const copy = make([]); copy.onConfigure(output); await settle(); assert.deepEqual(saved(copy), saved(node));
    const positional = make([]); positional.onConfigure({ widgets_values: output.widgets_values }); await settle(); assert.deepEqual(saved(positional), saved(node));
});
test('invalid saved JSON is preserved for repair rather than replaced by an empty stack', async () => {
    const { make } = setup(); const node = make(); node.onConfigure({ widgets_values_named: { slots_json: '[broken' } }); await settle();
    const output = {}; node.onSerialize(output); assert.equal(output.widgets_values_named.slots_json, '[broken');
    assert.equal(node.widgets.find(w => w.name === '+ Add LoRA').disabled, true);
    change(node.widgets.find(w => w.name === 'Repair slots_json'), JSON.stringify(rows(2))); assert.equal(saved(node).length, 2);
});
test('changing filename clears its hash; stale execution metadata cannot attach to the new selection', async () => {
    const { make } = setup(); const node = make([{ ...rows(1)[0], lora_hash: 'a'.repeat(64) }]); await settle();
    change(find(node, 'lora_name'), 'folder/b.safetensors'); assert.equal(saved(node)[0].lora_hash, '');
    node.onExecuted({ donut_loras: [{ id: 'row-0', lora_name: 'folder/model-0.safetensors', lora_hash: 'b'.repeat(64) }] });
    assert.equal(saved(node)[0].lora_hash, '');
    node.onExecuted({ donut_loras: [{ id: 'row-0', lora_name: 'folder/b.safetensors', lora_hash: 'c'.repeat(64) }] });
    assert.equal(saved(node)[0].lora_hash, 'c'.repeat(64));
});
test('metadata responses arriving after a row was replaced cannot change its name or hash', async () => {
    let resolveOld;
    const { make } = setup(url => url.includes('/info?') && url.includes('model-0') ? new Promise(resolve => { resolveOld = resolve; }) : null);
    const node = make(rows(1), 'On'); await settle();
    change(find(node, 'lora_name'), 'folder/b.safetensors'); await settle();
    resolveOld({ ok: true, json: async () => ({ hash: 'b'.repeat(10), civitai: { model_name: 'OLD' } }) }); await settle();
    assert.equal(saved(node)[0].lora_name, 'folder/b.safetensors'); assert.equal(saved(node)[0].lora_hash, 'a'.repeat(10));
    assert.ok(!text(panel(node)).includes('OLD'));
});
test('full stored hashes are not downgraded to ten-character UI lookup prefixes', async () => {
    const { make } = setup(); const node = make([{ ...rows(1)[0], lora_hash: 'b'.repeat(64) }], 'On'); await settle();
    assert.equal(saved(node)[0].lora_hash, 'b'.repeat(64));
});
test('failed catalog requests display retry state and never reset saved names', async () => {
    const { make } = setup(url => /models\/loras|object_info/.test(url) ? { ok: false, status: 503 } : null);
    const node = make(); await settle(); assert.ok(node.widgets.some(w => w.name.includes('LoRA list unavailable')));
    assert.equal(saved(node)[0].lora_name, 'folder/model-0.safetensors');
});
test('network errors are visible and the row provides retry instead of a blank metadata panel', async () => {
    const { make } = setup(url => url.includes('/info?') ? { ok: false, status: 503 } : null);
    const node = make(rows(1), 'On'); await settle(); assert.ok(text(panel(node)).includes('503'));
    assert.ok(text(panel(node)).includes('Retry metadata'));
});
test('metadata calls are deduplicated by filename and limited to two simultaneous requests', async () => {
    let active = 0, peak = 0;
    const { context, api } = setup(async url => {
        if (url.includes('/info?')) {
            active++; peak = Math.max(peak, active); await new Promise(r => setTimeout(r, 2)); active--;
        }
    });
    const service = context.createLoraService(api);
    const promises = Array.from({ length: 10 }, (_, i) => service.details('info', `file-${i % 5}`));
    await Promise.all(promises); assert.equal(peak, 2);
});
test('read-only promoted widget types can be hidden and restored without changing values', () => {
    const { context } = setup(); const widget = { get type() { return 'number'; }, value: 8 };
    context.setHidden(widget, true); assert.equal(widget.hidden, true); assert.equal(widget.value, 8);
    context.setHidden(widget, false); assert.equal(widget.hidden, false); assert.equal(widget.type, 'number');
});
test('removed nodes ignore pending results and leave the refresh registry', async () => {
    const { make, extension } = setup(); const node = make(rows(1), 'On'); node.onRemoved(); await settle();
    const before = saved(node); await extension.refreshComboInNodes(); assert.deepEqual(saved(node), before);
});

test('v1 workflow widget/port repair matches the delivered corrected workflow and is idempotent', { skip: !process.env.DONUT_BROKEN_WORKFLOW || !process.env.DONUT_FIXED_WORKFLOW }, () => {
    const { context } = setup();
    const broken = JSON.parse(fs.readFileSync(process.env.DONUT_BROKEN_WORKFLOW));
    const fixed = JSON.parse(fs.readFileSync(process.env.DONUT_FIXED_WORKFLOW));
    assert.equal(context.repairStreamlinedWorkflow(broken), true);
    const actual = broken.nodes.find(n => n.id === 1014), expected = fixed.nodes.find(n => n.id === 1014);
    assert.equal(actual.inputs.length, 37); assert.equal(actual.widgets_values.length, 25);
    assert.deepEqual(clean(actual.widgets_values), expected.widgets_values);
    assert.deepEqual(clean(actual.widgets_values_named), expected.widgets_values_named);
    assert.deepEqual(clean(actual.inputs.map(p => [p.name, p.link, p.widget?.name])), expected.inputs.map(p => [p.name, p.link, p.widget?.name ?? null]));
    assert.deepEqual(clean(broken.links), fixed.links);
    assert.equal(context.repairStreamlinedWorkflow(broken), false);
});
test('already-repaired workflows retain later user edits rather than reapplying stale named defaults', () => {
    const { context } = setup(); const workflow = { extra: { donut_streamlining: { schema: 2 } }, nodes: [{ widgets_values: [16], widgets_values_named: { steps: 8 } }] };
    assert.equal(context.repairStreamlinedWorkflow(workflow), false); assert.equal(workflow.nodes[0].widgets_values[0], 16);
});

test('nonfinite numeric edits restore the previous displayed and canonical value', async () => {
    const { make } = setup(); const node = make(); await settle();
    change(find(node, 'model_weight'), NaN); assert.equal(find(node, 'model_weight').value, .75);
    assert.equal(saved(node)[0].model_weight, .75);
});
test('text previews are UI-only, collapsed initially, and reveal the resolved text without changing backend values', async () => {
    const { make } = setup(); const node = make([], 'Off', 'DonutText'); await settle();
    const preview = node.widgets.find(w => w.name === 'resolved_text');
    assert.equal(preview.hidden, true); assert.equal(preview.serialize, false); assert.equal(preview.options.serialize, false);
    node.onExecuted({ text: ['resolved'] }); assert.equal(preview.value, 'resolved');
    node.widgets.find(w => w.name === 'Show / hide resolved text').callback(); assert.equal(preview.hidden, false);
});
test('UI lookup errors still expose backend execution preview data when available', async () => {
    const { make } = setup(url => url.includes('/info?') ? { ok: false, status: 503 } : null);
    const node = make(rows(1), 'On'); await settle();
    node.onExecuted({ donut_loras: [{ id: 'row-0', lora_name: 'folder/model-0.safetensors', lora_hash: 'd'.repeat(64),
        text: 'Cached trigger words', image: { filename: 'preview.jpg', type: 'temp', subfolder: '' } }] });
    assert.ok(text(panel(node)).includes('Cached trigger words'));
    assert.match(elements(panel(node), 'img')[0].src, /^\/api\/view\?/);
});
test('seed widgets keep separate serialized control names', () => {
    const { make } = setup(); const node = make([], 'Off', 'DonutSeedPlan');
    node.widgets = ['text_seed', 'control_after_generate', 'sampler_seed', 'control_after_generate', 'filename_seed', 'control_after_generate'].map(name => ({ name }));
    node.onNodeCreated(); assert.deepEqual(node.widgets.map(w => w.name), ['text_seed', 'text_seed_control', 'sampler_seed', 'sampler_seed_control', 'filename_seed', 'filename_seed_control']);
});
test('grouped merge still hides and restores original per-block controls without losing values', () => {
    const { make } = setup(); const node = make([], 'Off', 'DonutModelMergeKrea2');
    node.widgets = [{ name: 'ratio_mode', value: 'Grouped' }, { name: 'blocks.0.', value: .3, type: 'number' }, { name: 'tmlp.', value: .8, type: 'number' }];
    node.onNodeCreated(); assert.equal(node.widgets[1].hidden, true); assert.notEqual(node.widgets[2].hidden, true);
    change(node.widgets[0], 'Per block'); assert.equal(node.widgets[1].hidden, false); assert.equal(node.widgets[1].value, .3);
});

test('Add LoRA immediately creates visible native rows before any catalog response', async () => {
    const { make } = setup(); const node = make([]);
    for (let i = 1; i <= 4; i++) {
        node.widgets.find(w => w.name === '+ Add LoRA').callback();
        const selected = node.widgets.filter(w => w.name.startsWith('donut_row:') && w.name.endsWith(':lora_name'));
        assert.equal(selected.length, i); assert.equal(saved(node).length, i);
        assert.ok(selected.every(w => !w.hidden && Array.isArray(w.options.values) && w.options.values.includes('None')));
    }
    await settle(); assert.equal(saved(node).length, 4);
});
test('prototype getter-only widget types also stay native when hidden', () => {
    const { context } = setup();
    class Widget { get type() { return 'combo'; } }
    const widget = new Widget(); widget.value = 'beta';
    context.setHidden(widget, true); context.setHidden(widget, false);
    assert.equal(widget.type, 'combo'); assert.equal(widget.value, 'beta');
});
test('detach and re-add restores metadata lifecycle and Add LoRA functionality', async () => {
    const { make, requests, extension } = setup(); const node = make(rows(1), 'Off'); await settle();
    node.onRemoved(); node.onAdded();
    change(node.widgets.find(w => w.name === 'civitai_lookup'), 'On'); await settle();
    assert.ok(requests.some(u => u.includes('/info?')));
    assert.ok(text(panel(node)).includes('Test model'));
    node.widgets.find(w => w.name === '+ Add LoRA').callback(); assert.equal(saved(node).length, 2);
    await extension.refreshComboInNodes(); assert.equal(saved(node).length, 2);
});
test('old PNG loader state with a trailing DOM helper reloads without losing rows', async () => {
    const { make } = setup(); const original = rows(6), node = make([]);
    node.onConfigure({ widgets_values: ['KREA2', JSON.stringify(original), '1,1,1', 'Off', 'On', 'Use headroom', 2, 1, 'Experimental bypass', ''],
        widgets_values_named: { slots_json: JSON.stringify(original), donut_lora_editor: '' } });
    await settle(); assert.deepEqual(saved(node), original);
    const out = {}; node.onSerialize(out); assert.equal(out.widgets_values.length, 9);
    assert.equal('donut_lora_editor' in out.widgets_values_named, false);
});

test('every row has visible Remove and reorder buttons before its native picker, even while disabled', async () => {
    const { make } = setup(); const node = make(rows(6)); await settle();
    for (const row of saved(node)) {
        const actions = find(node, 'actions', row.id), picker = find(node, 'lora_name', row.id);
        assert.notEqual(actions.type, 'combo'); assert.ok(!actions.hidden);
        assert.ok(node.widgets.indexOf(actions) < node.widgets.indexOf(picker));
        assert.deepEqual(elements(actions.element, 'button').map(b => b.textContent), ['↑', '↓', 'Remove']);
        assert.ok(elements(actions.element, 'button').find(b => b.textContent === 'Remove').title.includes('keeps the file'));
    }
    assert.equal(node.widgets.some(w => w.type === 'combo' && w.options.values?.includes('Row actions')), false);
    const before = saved(node);
    rowAction(node, 'Remove', 'row-1');
    assert.deepEqual(saved(node), before.filter(row => row.id !== 'row-1'));
});
test('remove the only row, serialize/reload the empty stack, then add a visible row again', async () => {
    const { make } = setup(); const node = make(rows(1)); await settle();
    rowAction(node, 'Remove'); assert.deepEqual(saved(node), []);
    assert.equal(node.widgets.filter(w => w.name.startsWith('donut_row:')).length, 0);
    const out = {}; node.onSerialize(out); const copy = make(); copy.onConfigure(out); await settle();
    assert.deepEqual(saved(copy), []);
    copy.widgets.find(w => w.name === '+ Add LoRA').callback();
    assert.equal(saved(copy).length, 1);
    assert.ok(copy.widgets.some(w => w.name.endsWith(':actions') && !w.hidden));
});
test('remove targets row identity after reorder, not a cached position or filename', async () => {
    const { make } = setup(); const list = rows(3).map(r => ({ ...r, lora_name: 'same.safetensors' }));
    const node = make(list); await settle();
    const oldRemove = elements(find(node, 'actions', 'row-2').element, 'button').find(b => b.textContent === 'Remove');
    rowAction(node, 'Move up', 'row-2');
    oldRemove.events.click({ preventDefault() {}, stopPropagation() {} });
    assert.deepEqual(saved(node).map(r => r.id), ['row-0', 'row-1']);
    oldRemove.events.click({ preventDefault() {}, stopPropagation() {} }); // Deleted-row callback is a no-op.
    assert.equal(saved(node).length, 2);
});
test('first/last reorder arrows are disabled and do not modify canonical state', async () => {
    const { make } = setup(); const node = make(rows(2)); await settle(); const before = saved(node);
    assert.equal(elements(find(node, 'actions').element, 'button')[0].disabled, true);
    assert.equal(elements(find(node, 'actions', 'row-1').element, 'button')[1].disabled, true);
    rowAction(node, 'Move up'); rowAction(node, 'Move down', 'row-1');
    assert.deepEqual(saved(node), before);
});
test('row changes bracket graph undo hooks; restoring the snapshot restores all removed fields', async () => {
    const { make } = setup(); const node = make(rows(3)); await settle(); let snapshot, after = 0;
    const before = saved(node);
    node.graph = { beforeChange() { snapshot = {}; node.onSerialize(snapshot); }, afterChange() { after++; } };
    rowAction(node, 'Remove', 'row-1'); assert.equal(after, 1);
    assert.deepEqual(JSON.parse(snapshot.widgets_values_named.slots_json), before);
    node.onConfigure(snapshot); await settle(); assert.deepEqual(saved(node), before);
});
test('late metadata cannot restore a removed LoRA or reinsert its information panel', async () => {
    let resolve;
    const { make } = setup(url => url.includes('/info?') ? new Promise(r => { resolve = r; }) : null);
    const node = make(rows(1), 'On'); await settle(); rowAction(node, 'Remove');
    resolve({ ok: true, json: async () => ({ hash: 'b'.repeat(10), civitai: { model_name: 'Deleted' } }) });
    await settle(); assert.deepEqual(saved(node), []);
    assert.equal(node.widgets.filter(w => w.name.startsWith('donut_row:')).length, 0);
});
test('long block lists compact losslessly without hiding gaps', () => {
    const { context } = setup();
    assert.equal(context.compactLoraIndices(Array.from({ length: 28 }, (_, i) => i)), '0–27');
    assert.equal(context.compactLoraIndices([7, 1, 0, 2, 4, 7, 6]), '0–2, 4, 6–7');
    assert.equal(context.compactLoraIndices([]), '');
});
test('metadata hierarchy defaults to a compact block summary and separate expanded CivitAI card', async () => {
    const { make } = setup(); const node = make(rows(1), 'On'); await settle();
    const root = panel(node), sections = elements(root, 'details');
    assert.equal(sections[0].open, false); assert.equal(sections[1].open, true); assert.equal(sections[2].open, false);
    assert.equal(elements(root, 'img')[0].style.objectFit, 'contain');
    const hashElement = elements(sections[2]).find(e => e.textContent.startsWith('Hash:'));
    assert.ok(hashElement); assert.equal(hashElement.tag, 'div');
    assert.equal(elements(root, 'a').find(e => e.textContent.includes('CivitAI')).style.display, 'block');
});
test('expanded metadata sections survive strength updates and a catalog rerender', async () => {
    const { make } = setup(); const node = make(rows(1), 'On'); await settle();
    const weights = elements(panel(node), 'details')[0]; weights.open = true; weights.events.toggle();
    change(find(node, 'model_weight'), .9); assert.equal(elements(panel(node), 'details')[0].open, true);
    await node._donutNativeLoras.refresh(); assert.equal(elements(panel(node), 'details')[0].open, true);
});
test('toolbar click stops propagation so Remove does not start a canvas drag', async () => {
    const { make } = setup(); const node = make(rows(1)); await settle(); let stopped = false;
    const remove = elements(find(node, 'actions').element, 'button')[2];
    remove.events.click({ preventDefault() {}, stopPropagation() { stopped = true; } });
    assert.equal(stopped, true); assert.deepEqual(saved(node), []);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

test('control and disclosure remeasurement preserves configured panel width', () => {
    const handlers = {};
    const sandbox = {
        app: {},
        document: {createElement: () => ({}), head: {append() {}}, addEventListener() {}},
        requestAnimationFrame: () => 1,
        ResizeObserver: class {observe() {} disconnect() {}},
    };
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(new URL('../web/donut_layout.js', import.meta.url), 'utf8')
        .replace(/^import .*;\n/m, '').replaceAll('export function', 'function'), sandbox);
    const node = {properties: {}, size: [960, 600], computeSize: (out = [0, 0]) => { out[0] = 220; out[1] = 500; return out; }};
    const root = {style: {}, isConnected: true, offsetWidth: 936, offsetHeight: 500,
        scrollHeight: 500, addEventListener: (name, handler) => handlers[name] = handler,
        querySelectorAll: () => []};
    const dom = {options: {}};
    sandbox.fitModule(node, dom, root);
    assert.equal(dom.options.selectOn.length, 0);
    node.size = dom.computeSize(); // Comfy widget callback remeasurement.
    assert.equal(node.size[0], 960);
    root.scrollHeight = 800;
    handlers.toggle();
    node.size = dom.computeSize();
    assert.equal(node.size[0], 960);
    assert.equal(node.size[1], 800);
    assert.equal(dom.computeLayoutSize().minWidth, 960);
    // Selecting a title bar can leave a cached minimum on the DOM widget.
    // The overlay must still use the full node width.
    dom.width = 220;
    assert.equal(dom.width, 960);
    node.size[0] = 1200;
    assert.equal(dom.width, 1200);
    node.size[0] = 960;
    // Dragging uses the node's sizing API. Legacy LiteGraph ignores the
    // widget's returned width and starts from its ordinary 220px minimum.
    node.size = node.computeSize();
    assert.equal(node.size[0], 960);
    const output = new Float32Array([220, 600]);
    // Existing consumers may provide and reuse a size buffer.
    assert.equal(node.computeSize(output), output);
    assert.equal(output[0], 960);
    assert.equal(output[1], 500);
});

test('all panels import one shared layout manager', () => {
    const dir = new URL('../web/', import.meta.url);
    const imports = fs.readdirSync(dir).filter(name => name.endsWith('.js')).flatMap(name =>
        [...fs.readFileSync(new URL(name, dir), 'utf8').matchAll(/from ["'](\.\/donut_layout\.js[^"']*)["']/g)].map(match => match[1]));
    assert.ok(imports.length >= 6);
    assert.equal(new Set(imports).size, 1, `Multiple layout managers: ${imports}`);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../web/donut_canvas_wheel.js', import.meta.url), 'utf8');
const {bindPanelWheel} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
function fixture() {
    const calls = [];
    const graph = {};
    const canvas = {graph, canvas: {}, processMouseWheel(event) {calls.push(event); event.preventDefault();}};
    const container = {contains: element => element === canvas.canvas};
    class Root extends EventTarget {
        closest(selector) {return selector === '.graph-canvas-container' ? this.container : this.dialog;}
        removeEventListener(name, fn) {super.removeEventListener(name, fn, {capture: true});}
    }
    const root = new Root();
    root.container = container;
    const node = {graph};
    const cleanup = bindPanelWheel(root, node, () => canvas);
    const emit = (prevented = false) => {
        const event = new Event('wheel', {cancelable: true, bubbles: true});
        Object.assign(event, {deltaX: 7, deltaY: -120, deltaMode: 1, clientX: 250, clientY: 180,
            ctrlKey: true, metaKey: false, shiftKey: false, altKey: true});
        if (prevented) event.preventDefault();
        root.dispatchEvent(event);
        return event;
    };
    return {calls, canvas, root, node, cleanup, emit};
}

test('forwards the original event once and stops control/ancestor wheel handlers', () => {
    const f = fixture();
    let downstream = 0;
    f.root.addEventListener('wheel', () => downstream++);
    const event = f.emit();
    assert.deepEqual(f.calls, [event]);
    assert.equal(event.defaultPrevented, true);
    assert.equal(downstream, 0);
    assert.equal(event.deltaMode, 1);
    assert.equal(event.altKey, true);
    assert.equal(event.clientX, 250);
});
test('does not repeat a wheel event already forwarded by the frontend', () => {
    const f = fixture(); f.emit(true); assert.equal(f.calls.length, 0);
});
for (const [name, change] of [
    ['App Mode', f => {f.root.container = null;}],
    ['a different graph', f => {f.node.graph = {};}],
    ['a removed node', f => {f.node.graph = null;}],
    ['a missing graph', f => {f.canvas.graph = null;}],
    ['a different canvas container', f => {f.root.container = {contains: () => false};}],
    ['a dialog', f => {f.root.dialog = {};}],
    ['an unavailable canvas handler', f => {f.canvas.processMouseWheel = null;}],
]) test(`leaves ${name} alone`, () => {
    const f = fixture(); change(f);
    assert.equal(f.emit().defaultPrevented, false);
    assert.equal(f.calls.length, 0);
});
test('a disabled or out-of-viewport canvas can decline the event', () => {
    const f = fixture(); f.canvas.processMouseWheel = () => {};
    let downstream = 0; f.root.addEventListener('wheel', () => downstream++);
    assert.equal(f.emit().defaultPrevented, false);
    assert.equal(downstream, 1);
});
test('cleanup removes the wheel handler', () => {
    const f = fixture(); f.cleanup(); f.emit(); assert.equal(f.calls.length, 0);
});
test('layout binds once, removes on removal, and rebinds on re-add', () => {
    let binds = 0, removals = 0;
    const context = {
        app: {}, bindPanelWheel: () => {binds++; return () => removals++;},
        document: {createElement: () => ({}), head: {append() {}}, addEventListener() {}},
        requestAnimationFrame: () => 1,
        ResizeObserver: class {observe() {} disconnect() {}},
    };
    vm.createContext(context);
    vm.runInContext(fs.readFileSync(new URL('../web/donut_layout.js', import.meta.url), 'utf8')
        .replace(/^import .*;\n/mg, '').replaceAll('export function', 'function'), context);
    const node = {size: [500, 500], computeSize: () => [500, 500]};
    const root = {style: {}, addEventListener() {}};
    context.fitModule(node, {options: {}}, root);
    node.onAdded(); node.onAdded();
    assert.equal(binds, 1);
    node.onRemoved();
    assert.equal(removals, 1);
    node.onAdded();
    assert.equal(binds, 2);
});

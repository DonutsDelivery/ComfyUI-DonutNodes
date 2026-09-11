import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

test('result keeps the expanded prompt for its run and selected stage', () => {
    let extension, Node;
    const listeners = {};
    const el = () => ({children: [], append(...items) {this.children.push(...items);}, setAttribute() {}});
    const sandbox = {URLSearchParams, document: {createElement:el},
        app: {registerExtension: value => extension=value},
        api: {apiURL: x=>x, addEventListener:(name, fn)=>listeners[name]=fn},
        createProgress: () => ({element: {style: {}}, attach() {}, detach() {}}),
        fitModule() {},
        LGraphNode: class {addDOMWidget(name,type,root) {this.root=root; return {options:{}};}},
        LiteGraph: {registerNodeType:(name, cls)=>Node=cls}};
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(new URL('../web/donut_latest_preview.js', import.meta.url),'utf8').replace(/^import .*;\n/gm,''),sandbox);
    extension.registerCustomNodes();
    const node = new Node(); node.onAdded();
    const emit = detail => listeners.executed({detail});
    emit({prompt_id:'a',output:{donut_final_prompt:['woman holding a donut']}});
    emit({prompt_id:'a',node:'912',output:{images:[{filename:'a.png'}]}});
    const text = node.root.children[5].children[1];
    assert.equal(text.textContent,'woman holding a donut');
    emit({prompt_id:'b',output:{donut_final_prompt:['woman holding a sign']}});
    assert.equal(text.textContent,'woman holding a donut');
    emit({prompt_id:'b',node:'914',output:{images:[{filename:'b.png'}]}});
    assert.equal(text.textContent,'woman holding a sign');
    node.root.children[1].value='912'; node.root.children[1].onchange();
    assert.equal(text.textContent,'woman holding a donut');
});

test('empty execution events do not interrupt the workflow UI', () => {
    let extension, Node;
    const listeners = {};
    const el = () => ({children: [], append(...items) {this.children.push(...items);}, setAttribute() {}});
    const sandbox = {URLSearchParams, document: {createElement:el},
        app: {registerExtension: value => extension=value},
        api: {apiURL: x=>x, addEventListener:(name, fn)=>listeners[name]=fn},
        createProgress: () => ({element: {style: {}}, attach() {}, detach() {}}),
        fitModule() {},
        LGraphNode: class {addDOMWidget(name,type,root) {this.root=root; return {options:{}};}},
        LiteGraph: {registerNodeType:(name, cls)=>Node=cls}};
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(new URL('../web/donut_latest_preview.js', import.meta.url),'utf8').replace(/^import .*;\n/gm,''),sandbox);
    extension.registerCustomNodes();
    const node = new Node(); node.onAdded();
    assert.doesNotThrow(() => listeners.executed({}));
    assert.doesNotThrow(() => listeners.executed());
});

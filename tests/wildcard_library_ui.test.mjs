import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/donut_wildcards.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "").replace(/export function/g, "function");
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.value = ""; this.disabled = false; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute() {}
    focus() {}
}
const response = data => ({ok:true, json:async () => data});
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; };
const tick = () => new Promise(resolve => setImmediate(resolve));
async function setup(fetchApi) {
    const context = {api:{fetchApi}, document:{createElement:tag => new Element(tag)}, fitTextarea() {}, scheduleLayout() {}};
    const root = vm.runInNewContext(source + "\nwildcardLibrary()", context);
    await tick();
    const [, select, fresh, nameLabel, textLabel, save, status] = root.children;
    return {select, fresh, name:nameLabel.children[1], text:textLabel.children[1], save, status};
}
const catalog = () => response({names:["haircolor"], directory:"/user/wildcards"});

test("pending file load cannot overwrite a newly created wildcard or its status", async () => {
    const load = deferred();
    const ui = await setup(path => path.includes("?name=") ? load.promise : Promise.resolve(catalog()));
    ui.select.value = "haircolor";
    const pending = ui.select.onchange();
    assert.equal(ui.save.disabled, true);
    ui.fresh.onclick();
    ui.name.value = "fornames"; ui.text.value = "Alice\nEmma";
    load.resolve(response({name:"haircolor", text:"red"}));
    await pending;
    assert.equal(ui.name.value, "fornames");
    assert.equal(ui.text.value, "Alice\nEmma");
    assert.equal(ui.save.disabled, false);
    assert.equal(ui.status.textContent, "");
});

test("save locks the editor and confirms the actual path only after the write", async () => {
    const write = deferred(); const posts = [];
    const ui = await setup((path, options) => {
        if (options.method === "POST") { posts.push(JSON.parse(options.body)); return write.promise; }
        return Promise.resolve(catalog());
    });
    ui.name.value = "fornames"; ui.text.value = "Alice\nEmma";
    const pending = ui.save.onclick();
    assert.equal(ui.save.disabled, true);
    assert.equal(ui.name.disabled, true);
    assert.equal(ui.fresh.disabled, true);
    assert.match(ui.status.textContent, /Saving fornames/);
    await ui.save.onclick();
    assert.equal(posts.length, 1);
    assert.deepEqual(posts[0], {name:"fornames", text:"Alice\nEmma"});
    write.resolve(response({name:"fornames", path:"/user/wildcards/fornames.txt"}));
    await pending;
    assert.equal(ui.save.disabled, false);
    assert.match(ui.status.textContent, /Saved fornames\*/);
    assert.match(ui.status.textContent, /\/user\/wildcards\/fornames\.txt/);
});

test("failed save keeps draft and unlocks editor", async () => {
    const ui = await setup((_path, options) => Promise.resolve(options.method === "POST"
        ? {ok:false, text:async () => "Disk write failed"} : catalog()));
    ui.name.value = "fornames"; ui.text.value = "Alice";
    await ui.save.onclick();
    assert.equal(ui.name.value, "fornames");
    assert.equal(ui.text.value, "Alice");
    assert.equal(ui.save.disabled, false);
    assert.match(ui.status.textContent, /Disk write failed/);
});

test("successful write is not reported as failed if catalog refresh fails", async () => {
    let lists = 0;
    const ui = await setup((_path, options) => Promise.resolve(options.method === "POST"
        ? response({name:"fornames", path:"/user/wildcards/fornames.txt"})
        : ++lists === 1 ? catalog() : {ok:false, text:async () => "Catalog unavailable"}));
    ui.name.value = "fornames"; ui.text.value = "Alice";
    await ui.save.onclick();
    assert.match(ui.status.textContent, /Saved fornames\*/);
    assert.match(ui.status.textContent, /Catalog unavailable/);
    assert.equal(ui.save.disabled, false);
});

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

test("existing V4 Generate panel gains SDA controls beside Turbo mode", () => {
    let extension;
    const queued = [];
    const sampler = { id:77, type:"DonutSampler", widgets: [
        {name:"turbo_mode", value:true},
        {name:"sda_enabled", value:false},
        {name:"sda_strength", value:1},
    ] };
    const subgraph = { nodes:[sampler], getNodeById: id => id === 77 ? sampler : undefined };
    const stage = { id:10, widgets:[{name:"turbo_mode", value:true}], subgraph };
    let renders = 0;
    const panel = {
        properties: { donut_app_controls: { groups: [{ title:"Sampling", controls:[
            {path:[10], widget:"turbo_mode", title:"Turbo mode"},
            {path:[10], widget:"steps", title:"Steps"},
        ] }] } },
        _donutAppControls: { render() { renders += 1; } },
    };
    const rootGraph = {
        nodes:[stage, panel],
        getNodeById: id => id === 10 ? stage : undefined,
    };
    const scope = vm.createContext({
        app: { rootGraph, registerExtension(value) { extension = value; } },
        queueMicrotask(callback) { queued.push(callback); },
    });
    const source = readFileSync(new URL("../web/donut_krea2_sda_controls.js", import.meta.url), "utf8")
        .replace(/^import .*;\n/gm, "");
    vm.runInContext(source, scope);
    extension.afterConfigureGraph();
    queued.splice(0).forEach(callback => callback());

    const controls = panel.properties.donut_app_controls.groups[0].controls;
    assert.deepEqual(Array.from(controls, item => item.widget), [
        "turbo_mode", "sda_enabled", "sda_strength", "steps",
    ]);
    assert.deepEqual(Array.from(controls[1].path), [10, 77]);
    assert.deepEqual(Array.from(controls[2].path), [10, 77]);
    assert.equal(controls[1].title, "SDA diversity");
    assert.equal(controls[2].title, "SDA strength");
    assert.equal(renders, 1);

    extension.afterConfigureGraph();
    queued.splice(0).forEach(callback => callback());
    assert.equal(controls.filter(item => item.widget === "sda_enabled").length, 1, "control injection is idempotent");
});

test("older DonutSampler definitions without SDA remain untouched", () => {
    let extension;
    const queued = [];
    const sampler = { id:77, type:"DonutSampler", widgets: [{name:"turbo_mode", value:true}] };
    const subgraph = { nodes:[sampler], getNodeById:() => sampler };
    const stage = { id:10, widgets:[{name:"turbo_mode", value:true}], subgraph };
    const group = { title:"Sampling", controls:[{path:[10], widget:"turbo_mode", title:"Turbo mode"}] };
    const panel = { properties:{donut_app_controls:{groups:[group]}}, _donutAppControls:{render(){ throw new Error("should not render"); }} };
    const scope = vm.createContext({
        app: { rootGraph:{nodes:[stage,panel], getNodeById:id => id === 10 ? stage : undefined}, registerExtension(value) { extension = value; } },
        queueMicrotask(callback) { queued.push(callback); },
    });
    const source = readFileSync(new URL("../web/donut_krea2_sda_controls.js", import.meta.url), "utf8")
        .replace(/^import .*;\n/gm, "");
    vm.runInContext(source, scope);
    extension.afterConfigureGraph();
    queued.splice(0).forEach(callback => callback());
    assert.equal(group.controls.length, 1);
});

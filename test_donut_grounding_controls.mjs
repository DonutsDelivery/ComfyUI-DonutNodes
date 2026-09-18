import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("./web/donut_grounding_controls.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/, "");
function graph(nodes) {
    return { nodes, getNodeById(id) { return nodes.find(node => node.id === id); } };
}
function setup({ direct = false, available = true } = {}) {
    const sampler = { id: 3, comfyClass: "DonutSampler",
        widgets: available ? [{ name: "grounding_schedule" }] : [] };
    const outer = { id: 2, subgraph: graph([sampler]) };
    const controls = [{ path: direct ? [3] : [2], widget: "turbo_mode", title: "Turbo" }];
    let renders = 0, extension;
    const panel = { id: 1, properties: { donut_app_controls: { groups: [{ controls }] } },
        _donutAppControls: { render() { renders++; } } };
    const app = { rootGraph: graph([panel, direct ? sampler : outer]),
        registerExtension(value) { extension = value; } };
    vm.runInNewContext(source, { app, queueMicrotask: callback => callback() });
    return { app, controls, run: () => extension.afterConfigureGraph(), renders: () => renders };
}

for (const direct of [false, true]) {
    const state = setup({ direct });
    state.run();
    assert.deepEqual(state.controls.map(item => item.widget),
        ["turbo_mode", "grounding_schedule", "grounding_start_px", "grounding_end_px"]);
    assert.deepEqual(Array.from(state.controls[1].path), direct ? [3] : [2, 3]);
    assert.equal(state.controls[2].weights.step, 64);
    state.run();
    assert.equal(state.controls.length, 4, "graph reload must not duplicate controls");
    assert.equal(state.renders(), 1);
}
const unavailable = setup({ available: false });
unavailable.run();
assert.equal(unavailable.controls.length, 1);
assert.equal(unavailable.renders(), 0);
const empty = setup();
empty.app.rootGraph = undefined;
empty.run();
console.log("Grounding controls: nested/direct binding, idempotence, missing widget and missing graph passed.");

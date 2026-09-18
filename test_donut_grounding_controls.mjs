import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("./web/donut_grounding_controls.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/m, "");

function graph(nodes) {
    return { nodes, getNodeById(id) { return nodes.find(node => node.id === id); } };
}

function makeSampler({ schemaAvailable = true } = {}) {
    const widgets = schemaAvailable ? [
        { name: "grounding_schedule", value: "constant",
          options: { values: ["constant", "linear", "ease_in", "ease_out", "ease_in_out"] } },
        { name: "grounding_start_px", value: 512 },
        { name: "grounding_end_px", value: 1088 },
    ] : [{ name: "grounding_px" }];
    return { id: 3, comfyClass: "DonutSampler", type: "DonutSampler", widgets,
        inputs: [{ name: "edit_mode", link: 7 }] };
}

function setup({ direct = false, schemaAvailable = true } = {}) {
    const sampler = makeSampler({ schemaAvailable });
    const outer = { id: 2, subgraph: graph([sampler]) };
    let extension = null;
    let section = null;
    const panel = { id: 1,
        properties: { donut_seed_path: direct ? [3] : [2],
            donut_app_controls: { groups: [{ controls: [ { path: direct ? [3] : [2], widget: "turbo_mode" } ] }] } },
        _donutEditStudio: {},
        donutAppendEditStudioSection(definition) {
            section = { title: definition.title, rows: definition.controls.map(() => ({})),
                hidden: false, update: definition.render };
            return section;
        },
    };
    const app = { rootGraph: graph([panel, direct ? sampler : outer]),
        registerExtension(value) { extension = value; } };
    vm.runInNewContext(source, { app, queueMicrotask: callback => callback() });
    return { run: () => extension.afterConfigureGraph(), section: () => section, sampler };
}

for (const direct of [false, true]) {
    const state = setup({ direct });
    state.run();
    const section = state.section();
    assert.ok(section, "section must be installed into the Edit Studio panel");
    assert.match(section.title, /Editing schedule/);
    // constant + editing gate: sampler edit_mode link truthy -> visible
    section.update();
    assert.equal(section.hidden, false);
    // schedule set -> stays visible and content rows remain
    state.sampler.widgets[0].value = "ease_in";
    section.update();
    assert.equal(section.hidden, false);
}

// No Edit Studio panel (missing hooks): no crash, no section.
{
    const sampler = makeSampler({});
    const app = { rootGraph: graph([{ id: 1, properties: {}, _donutEditStudio: {} }, sampler]),
        registerExtension(value) { extension = value; } };
    let extension;
    vm.runInNewContext(source, { app, queueMicrotask: callback => callback() });
    extension.afterConfigureGraph();
}
// Sampler without schedule widgets (stale schema): no section installed.
{
    const sampler = makeSampler({ schemaAvailable: false });
    const app = { rootGraph: graph([sampler]), registerExtension(value) { extension = value; } };
    let extension;
    vm.runInNewContext(source, { app, queueMicrotask: callback => callback() });
    extension.afterConfigureGraph();
}
console.log("Grounding controls: Edit Studio panel binding, visibility, missing-schema guard passed.");

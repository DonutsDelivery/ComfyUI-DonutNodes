import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("./web/donut_grounding_controls.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/m, "");
const flush = async () => new Promise(resolve => setImmediate(resolve));

function graph(nodes) {
    return { nodes, getNodeById(id) { return nodes.find(node => node.id === id); } };
}

function makeSampler({ schemaAvailable = true, nagValues = {} } = {}) {
    const widgets = schemaAvailable ? [
        { name: "grounding_schedule", value: "constant",
          options: { values: ["constant", "linear", "ease_in", "ease_out", "ease_in_out"] } },
        { name: "grounding_start_px", value: 512 },
        { name: "grounding_end_px", value: 1088 },
        { name: "nag_alpha", value: 0.15 },
        { name: "nag_alpha_schedule", value: Object.hasOwn(nagValues, "schedule") ? nagValues.schedule : "constant",
          options: { values: ["constant", "linear", "ease_in", "ease_out", "ease_in_out"] } },
        { name: "nag_alpha_start", value: Object.hasOwn(nagValues, "start") ? nagValues.start : 0.15 },
        { name: "nag_alpha_end", value: Object.hasOwn(nagValues, "end") ? nagValues.end : 0.15 },
        { name: "txtfusion_internal_guard", value: false },
        { name: "txtfusion_reference_checkpoint", value: "None" },
    ] : [{ name: "grounding_px" }];
    return { id: 3, comfyClass: "DonutSampler", type: "DonutSampler", widgets,
        inputs: [{ name: "edit_mode", link: 7 }] };
}

async function run({ nodes } = {}) {
    let section = null;
    const panel = { id: 1, properties: {},
        _donutEditStudio: {},
        donutAppendEditStudioSection(definition) {
            section = { title: definition.title, hidden: false, update: definition.render };
            return section;
        } };
    let extension = null;
    const app = { rootGraph: graph([panel, ...nodes]), registerExtension(value) { extension = value; } };
    const context = { app, queueMicrotask: callback => callback(),
        globalThis: { requestAnimationFrame(callback) { setImmediate(callback); } } };
    vm.runInNewContext(source, context);
    extension.afterConfigureGraph();
    await flush();
    return { section: () => section, repair: context.repairLegacyNagWidgetValues };
}

// Nested path as in V4 (sampler inside a subgraph).
{
    const sampler = makeSampler({});
    const state = await run({ nodes: [{ id: 2, subgraph: graph([sampler]) }] });
    const section = state.section();
    assert.ok(section, "section installed via retry pairing");
    assert.match(section.title, /Editing schedule/);
    section.update();
    assert.equal(section.hidden, false, "edit_mode linked -> visible");
    sampler.widgets[0].value = "linear";
    section.update();
    assert.equal(section.hidden, false);
}

// Direct root-level sampler also pairs.
{
    const sampler = makeSampler({});
    assert.ok((await run({ nodes: [sampler] })).section(), "direct sampler also pairs");
}

// An earlier build saved three new NAG controls into positions occupied by
// the legacy TextFusion widgets. A bad curve marks the full schedule triple as
// stale and repairs it from the existing static alpha; legacy controls and
// valid manual schedules remain untouched.
{
    const sampler = makeSampler({ nagValues: { schedule: 0.25, start: 0, end: null } });
    let changed = 0;
    for (const widget of sampler.widgets) widget.callback = () => changed++;
    const state = await run({ nodes: [{ id: 2, subgraph: graph([sampler]) }] });
    const value = name => sampler.widgets.find(widget => widget.name === name).value;
    assert.equal(value("nag_alpha_schedule"), "constant");
    assert.equal(value("nag_alpha_start"), 0.15);
    assert.equal(value("nag_alpha_end"), 0.15);
    assert.equal(value("txtfusion_internal_guard"), false);
    assert.equal(value("txtfusion_reference_checkpoint"), "None");
    assert.equal(changed, 3);
    assert.equal(state.repair(sampler), false, "repair is idempotent");
}
{
    const sampler = makeSampler({ nagValues: { schedule: "ease_in", start: 0.1, end: 0.8 } });
    await run({ nodes: [sampler] });
    assert.equal(sampler.widgets.find(widget => widget.name === "nag_alpha_schedule").value, "ease_in");
    assert.equal(sampler.widgets.find(widget => widget.name === "nag_alpha_start").value, 0.1);
    assert.equal(sampler.widgets.find(widget => widget.name === "nag_alpha_end").value, 0.8);
}

// Editing off (no edit_mode link, constant) hides the section.
{
    const sampler = makeSampler({});
    sampler.inputs[0].link = null;
    const state = await run({ nodes: [sampler] });
    assert.ok(state.section(), "installed even when editing is off");
    state.section().update();
    assert.equal(state.section().hidden, true, "constant without edit link hides");
}

// Stale schema (no schedule widgets) installs nothing.
{
    const sampler = { id: 3, type: "DonutSampler", widgets: [{ name: "grounding_px" }],
        inputs: [{ name: "edit_mode", link: 7 }] };
    assert.equal((await run({ nodes: [sampler] })).section(), null);
}
console.log("Grounding controls: panel pairing, invalid legacy NAG value repair, and valid value preservation passed.");

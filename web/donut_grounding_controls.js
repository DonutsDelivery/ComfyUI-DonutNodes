import { app } from "../../scripts/app.js";

// Find the nested DonutSampler that owns the grounding_schedule widget.
// V4 nests it inside the Generate subgraph; scan every descendant graph so the
// binding works regardless of which panel anchors the probe.
function visitGraph(graph, visitor) {
    for (const node of graph?.nodes || []) {
        const result = visitor(node);
        if (result) return result;
        const nested = visitGraph(node.subgraph, visitor);
        if (nested) return nested;
    }
    return null;
}

function hasScheduleWidgets(node) {
    const names = new Set((node?.widgets || []).map(widget => widget.name));
    return names.has("grounding_schedule") && names.has("grounding_start_px") && names.has("grounding_end_px");
}

function findGroundedSampler() {
    return visitGraph(app.rootGraph, node => {
        const type = node.comfyClass || node.type || node.properties?.["Node name for S&R"];
        return type === "DonutSampler" && hasScheduleWidgets(node) ? node : null;
    });
}

function editingActive(sampler) {
    // truthy when an edit_mode link exists OR the schedule is dynamic; the
    // dynamic case already implies Edit Mode, and constant outside edit mode
    // is a no-op the section may hide.
    const schedule = sampler.widgets.find(widget => widget.name === "grounding_schedule")?.value;
    const dynamic = !!schedule && schedule !== "constant";
    if (dynamic) return true;
    const input = sampler.inputs?.find(entry => entry.name === "edit_mode");
    return !!(input && input.link != null);
}

function installGroundingIntoEditPanel(panel, sampler) {
    if (panel._donutGroundingSection) return;
    const scheduleWidget = sampler.widgets.find(widget => widget.name === "grounding_schedule");
    const startWidget = sampler.widgets.find(widget => widget.name === "grounding_start_px");
    const endWidget = sampler.widgets.find(widget => widget.name === "grounding_end_px");
    const choices = scheduleWidget.options?.values || ["constant", "linear", "ease_in", "ease_out", "ease_in_out"];
    const section = panel.donutAppendEditStudioSection({
        title: "Editing schedule (experimental)",
        controls: [
            { name: "grounding_schedule", kind: "select", title: "Grounding schedule",
              choices, get: () => scheduleWidget.value,
              set: value => { scheduleWidget.value = value; scheduleWidget.callback?.(value); } },
            { name: "grounding_start_px", kind: "number", title: "Start grounding px",
              step: 64, min: 0, max: 4096, get: () => startWidget.value,
              set: value => { startWidget.value = value; startWidget.callback?.(value); } },
            { name: "grounding_end_px", kind: "number", title: "End grounding px",
              step: 64, min: 0, max: 4096, get: () => endWidget.value,
              set: value => { endWidget.value = value; endWidget.callback?.(value); } },
        ],
        render() {
            if ("hidden" in section) section.hidden = !editingActive(sampler);
        },
    });
    if (section) {
        panel._donutGroundingSection = section;
        panel._donutEditStudio?.render?.();
    }
}

function pairing() {
    // Pair every grounded sampler with the first Edit Studio panel that has no
    // grounded section yet. Multiple samplers reuse the first panel's section.
    const panels = (app.rootGraph?.nodes || []).filter(node => node._donutEditStudio && node.donutAppendEditStudioSection);
    if (!panels.length) return false;
    let installed = false;
    for (const panel of panels) {
        if (panel._donutGroundingSection) continue;
        const sampler = findGroundedSampler();
        if (!sampler) return installed;
        installGroundingIntoEditPanel(panel, sampler);
        installed = true;
    }
    return installed;
}

function purgeStalePanelEntries() {
    // An early build injected the schedule controls into app-control panels
    // (beside Turbo) and app-control configs are saved with the workflow, so
    // those stale entries render in panel 6 even after the injection stops.
    // Scrub them on every configure; Edit Studio owns these settings now.
    const visit = graph => {
        for (const panel of graph?.nodes || []) {
            const config = panel.properties?.donut_app_controls;
            if (!config?.groups) continue;
            for (const group of config.groups) {
                const controls = group.controls;
                if (Array.isArray(controls)) {
                    const filtered = controls.filter(item => !String(item.widget || "").startsWith("grounding_"));
                    if (filtered.length !== controls.length) group.controls = filtered;
                }
            }
        }
        for (const node of graph?.nodes || []) if (node.subgraph) visit(node.subgraph);
    };
    visit(app.rootGraph);
}

function retryUntilRendered(attempts) {
    if (!attempts) return;
    purgeStalePanelEntries();
    if (pairing()) return;
    const next = () => retryUntilRendered(attempts - 1);
    (globalThis.requestAnimationFrame || (callback => setTimeout(callback, 350)))(next);
}

app.registerExtension({
    name: "Donut.GroundingScheduleControls",
    afterConfigureGraph() {
        queueMicrotask(() => retryUntilRendered(20));
    },
    nodeCreated(node) {
        if (node.properties?.["Node name for S&R"] === "DonutEditStudio") {
            (globalThis.requestAnimationFrame || (callback => setTimeout(callback, 200)))(pairing);
        }
    },
});

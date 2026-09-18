import { app } from "../../scripts/app.js";

function resolvePath(path) {
    let graph = app.rootGraph, node;
    for (const id of path || []) {
        node = graph?.getNodeById(id);
        graph = node?.subgraph;
    }
    return node;
}

function findGroundedSampler(startPath) {
    const target = resolvePath(startPath);
    if (target?.widgets?.some(widget => widget.name === "grounding_schedule")) {
        return target;
    }
    const visit = (graph, prefix) => {
        for (const current of graph?.nodes || []) {
            const type = current.comfyClass || current.type || current.properties?.["Node name for S&R"];
            if (type === "DonutSampler" && current.widgets?.some(widget => widget.name === "grounding_schedule")) {
                return current;
            }
            const nested = visit(current.subgraph, []);
            if (nested) return nested;
        }
    };
    return visit(target?.subgraph, []);
}

function editingActive(sampler) {
    const input = sampler.inputs?.find(entry => entry.name === "edit_mode");
    if (!input || input.link == null) return false;
    return Boolean(sampler.properties?.["_donut_grounding_edit_hint_" + input.link] ?? true) &&
        input.link != null;
}

function installGroundingIntoEditPanel(panel) {
    if (panel._donutGroundingSection || !panel._donutEditStudio || !panel.donutAppendEditStudioSection) return;
    const probePaths = [panel.properties?.donut_seed_path];
    for (const group of panel.properties?.donut_app_controls?.groups || []) {
        for (const control of group.controls || []) {
            if (control.widget) probePaths.push(control.path);
        }
    }
    let sampler = null;
    for (const path of probePaths) {
        if (!Array.isArray(path)) continue;
        const found = findGroundedSampler(path);
        if (found) { sampler = found; break; }
    }
    if (!sampler) return;
    const scheduleWidget = sampler.widgets.find(widget => widget.name === "grounding_schedule");
    const startWidget = sampler.widgets.find(widget => widget.name === "grounding_start_px");
    const endWidget = sampler.widgets.find(widget => widget.name === "grounding_end_px");
    if (!scheduleWidget || !startWidget || !endWidget) return;

    const scheduleSelectOptions = scheduleWidget.options?.values
        || scheduleWidget.options?.choices
        || ["constant", "linear", "ease_in", "ease_out", "ease_in_out"];
    const section = panel.donutAppendEditStudioSection({
        title: "Editing schedule (experimental · Krea2 grounding over steps)",
        controls: [
            { name: "grounding_schedule", kind: "select",
              title: "Grounding schedule",
              choices: scheduleSelectOptions,
              get: () => scheduleWidget.value,
              set: value => { scheduleWidget.value = value; scheduleWidget.callback?.(value); } },
            { name: "grounding_start_px", kind: "number", title: "Start grounding px",
              step: 64, min: 0, max: 4096,
              get: () => startWidget.value,
              set: value => { startWidget.value = value; startWidget.callback?.(value); } },
            { name: "grounding_end_px", kind: "number", title: "End grounding px",
              step: 64, min: 0, max: 4096,
              get: () => endWidget.value,
              set: value => { endWidget.value = value; endWidget.callback?.(value); } },
        ],
        render() {
            // Visible only while editing is enabled, so non-editing users never
            // see controls the sampler would ignore (dynamic grounding is
            // Edit-Mode-only by design).
            const dynamic = !!scheduleWidget.value && scheduleWidget.value !== "constant";
            const show = dynamic || editingActive(sampler);
            if ("hidden" in section) section.hidden = !show;
        },
    });
    panel._donutGroundingSection = section;
}

function installGroundingControls() {
    for (const panel of app.rootGraph?.nodes || []) {
        try { installGroundingIntoEditPanel(panel); } catch (error) { /* panel not ready; retried on next configure */ }
    }
}

app.registerExtension({
    name: "Donut.GroundingScheduleControls",
    afterConfigureGraph() {
        // Bind directly to the nested sampler's widgets, like V4's SDA
        // controls, but surface them inside the Edit Studio panel so every
        // editing-related setting lives in one place.
        queueMicrotask(installGroundingControls);
    },
});

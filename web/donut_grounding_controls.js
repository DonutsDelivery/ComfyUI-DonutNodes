import { app } from "../../scripts/app.js";

function resolve(path) {
    let graph = app.rootGraph, node;
    for (const id of path || []) {
        node = graph?.getNodeById(id);
        graph = node?.subgraph;
    }
    return node;
}

function samplerPath(path) {
    const hasSchedule = node => node?.widgets?.some(widget => widget.name === "grounding_schedule");
    const target = resolve(path);
    if (hasSchedule(target)) return [...path];
    const visit = (graph, prefix) => {
        for (const node of graph?.nodes || []) {
            const nested = [...prefix, node.id];
            const type = node.comfyClass || node.type || node.properties?.["Node name for S&R"];
            if (type === "DonutSampler" && hasSchedule(node)) return nested;
            const found = visit(node.subgraph, nested);
            if (found) return found;
        }
    };
    return visit(target?.subgraph, path);
}

function installGroundingControls() {
    for (const panel of app.rootGraph?.nodes || []) {
        const config = panel.properties?.donut_app_controls;
        if (!config?.groups) continue;
        let changed = false;
        for (const group of config.groups) {
            const controls = group.controls;
            if (!Array.isArray(controls) || controls.some(item => item.widget === "grounding_schedule")) continue;
            const anchor = controls.findIndex(item => item.widget === "turbo_mode");
            if (anchor < 0) continue;
            const path = samplerPath(controls[anchor].path);
            if (!path) continue;
            controls.splice(anchor + 1, 0,
                { path, widget: "grounding_schedule", title: "Edit grounding schedule" },
                { path: [...path], widget: "grounding_start_px", title: "Start grounding px",
                  weights: { min: 0, max: 4096, step: 64 } },
                { path: [...path], widget: "grounding_end_px", title: "End grounding px",
                  weights: { min: 0, max: 4096, step: 64 } },
            );
            changed = true;
        }
        if (changed) panel._donutAppControls?.render();
    }
}

app.registerExtension({
    name: "Donut.GroundingScheduleControls",
    afterConfigureGraph() {
        // Bind directly to the nested sampler, like V4's SDA controls. Existing
        // workflow JSON and Edit Studio's grounding socket stay unchanged.
        queueMicrotask(installGroundingControls);
    },
});

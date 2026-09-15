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
    const target = resolve(path);
    if (target?.widgets?.some(widget => widget.name === "sda_enabled")) return [...path];

    const visit = (graph, prefix) => {
        for (const current of graph?.nodes || []) {
            const nestedPath = [...prefix, current.id];
            const type = current.comfyClass || current.type || current.properties?.["Node name for S&R"];
            if (type === "DonutSampler" && current.widgets?.some(widget => widget.name === "sda_enabled")) {
                return nestedPath;
            }
            const nested = visit(current.subgraph, nestedPath);
            if (nested) return nested;
        }
    };
    return visit(target?.subgraph, path);
}

function installSdaControls() {
    for (const panel of app.rootGraph?.nodes || []) {
        const config = panel.properties?.donut_app_controls;
        if (!config?.groups) continue;
        let changed = false;
        for (const group of config.groups) {
            const controls = group.controls;
            if (!Array.isArray(controls)) continue;
            const turboIndex = controls.findIndex(item => item.widget === "turbo_mode");
            if (turboIndex < 0 || controls.some(item => item.widget === "sda_enabled")) continue;
            const turbo = controls[turboIndex];
            const path = samplerPath(turbo.path);
            if (!path) continue;

            controls.splice(turboIndex + 1, 0,
                { path, widget: "sda_enabled", title: "SDA diversity" },
                { path: [...path], widget: "sda_strength", title: "SDA strength",
                  weights: { min: 0, max: 2, step: 0.05 } },
            );
            changed = true;
        }
        if (changed) panel._donutAppControls?.render();
    }
}

app.registerExtension({
    name: "Donut.Krea2SDAControls",
    afterConfigureGraph() {
        // Existing V4 JSONs do not need to be rewritten just to expose the new
        // sampler widgets. The saved Turbo control points at the outer Generate
        // subgraph, so locate its DonutSampler descendant and bind SDA directly.
        queueMicrotask(installSdaControls);
    },
});

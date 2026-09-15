import { app } from "../../scripts/app.js";

function resolve(path) {
    let graph = app.rootGraph, node;
    for (const id of path || []) {
        node = graph?.getNodeById(id);
        graph = node?.subgraph;
    }
    return node;
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
            const sampler = resolve(turbo.path);
            if (!sampler?.widgets?.some(widget => widget.name === "sda_enabled")) continue;

            controls.splice(turboIndex + 1, 0,
                { path: [...turbo.path], widget: "sda_enabled", title: "SDA diversity" },
                { path: [...turbo.path], widget: "sda_strength", title: "SDA strength",
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
        // sampler widgets. Inject controls beside Turbo mode after all graph
        // subgraphs and frontend-only panels have finished restoring.
        queueMicrotask(installSdaControls);
    },
});

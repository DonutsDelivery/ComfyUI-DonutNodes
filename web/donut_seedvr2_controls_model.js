// Pure, idempotent panel upgrade. Keep the existing stages and every graph link.
// Paths are discovered from the panel's actual DonutTiledUpscale controls, not
// hard-coded V4 node IDs; copied/remapped modules keep working.
export function addSeedVR2Controls(rootGraph) {
    const entries = [];
    function walk(graph, path = [], ancestors = new Set()) {
        if (!graph || ancestors.has(graph)) return;
        const next = new Set(ancestors).add(graph);
        for (const node of graph.nodes || graph._nodes || []) {
            const nodePath = [...path, node.id];
            entries.push({node, path:nodePath});
            if (node.subgraph) walk(node.subgraph, nodePath, next);
        }
    }
    walk(rootGraph);
    const key = path => JSON.stringify(path.map(String));
    const byPath = new Map(entries.map(entry => [key(entry.path), entry]));
    const isStage = node => (node?.type === "DonutTiledUpscale" || node?.properties?.["Node name for S&R"] === "DonutTiledUpscale")
        && node.widgets?.some(widget => widget.name === "upscale_engine");
    const changed = [];
    for (const {node:panel} of entries) {
        const groups = panel.properties?.donut_app_controls?.groups;
        if (!Array.isArray(groups)) continue;
        const candidates = new Map();
        for (const group of groups) for (const control of group.controls || []) {
            if (!Array.isArray(control.path)) continue;
            let entry = byPath.get(key(control.path));
            if (!entry && control.fallback_type === "DonutTiledUpscale") {
                const parent = key(control.path.slice(0, -1));
                const matches = entries.filter(candidate => isStage(candidate.node) && key(candidate.path.slice(0, -1)) === parent);
                if (matches.length === 1) entry = matches[0];
            }
            if (!entry || !isStage(entry.node)) continue;
            const id = key(entry.path);
            if (!candidates.has(id)) candidates.set(id, {entry, group});
        }
        let updated = false;
        for (const [id, {entry, group}] of candidates) {
            if (groups.some(item => item.controls?.some(control => control.widget === "upscale_engine" && Array.isArray(control.path) && key(control.path) === id))) continue;
            const field = (widget, title, extra = {}) => ({path:[...entry.path], widget, title, fallback_type:"DonutTiledUpscale", ...extra});
            const title = group.title || entry.node.title || "Upscale";
            const engine = {
                title:`${title} · engine`,
                advanced:true,
                description:"Donut preserves the existing upscale + diffusion recipe. SeedVR2 shares the stage seed, scale and resize filter, but uses its own model, VAE and sampling settings.",
                controls:[field("upscale_engine", "Upscale engine")],
                donut_seedvr2:true,
            };
            const settings = {
                title:`${title} · SeedVR2`, advanced:true, donut_seedvr2:true,
                visible_when:{path:[...entry.path], widget:"upscale_engine", value:"SeedVR2"},
                description:"Native ComfyUI SeedVR2. Full-canvas diffusion; VAE tiling does not tile the diffusion model. No Krea LoRAs, NAG or reference conditioning are applied here.",
                controls:[
                    field("seedvr2_model_name", "SeedVR2 diffusion model"),
                    field("seedvr2_vae_name", "SeedVR2 VAE"),
                    field("seedvr2_steps", "SeedVR2 steps"),
                    field("seedvr2_denoise", "SeedVR2 denoise", {weights:{min:0.01, max:1, step:0.01}}),
                    field("seedvr2_color_correction", "SeedVR2 color correction"),
                    field("seedvr2_vae_tile_size", "SeedVR2 VAE tile size"),
                ],
            };
            groups.splice(groups.indexOf(group), 0, engine, settings);
            updated = true;
        }
        if (updated) changed.push(panel);
    }
    return changed;
}

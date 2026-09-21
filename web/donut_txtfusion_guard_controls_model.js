// Presentation-only binding. Do not copy/reset backend widget values.
export const GUARD_CONTROLS = [
    ['txtfusion_internal_guard', 'Checkpoint-referenced txtfusion guard · base pass'],
    ['txtfusion_reference_checkpoint', 'Guard reference · same diffusion checkpoint'],
];
const MARK = 'donut_txtfusion_internal_guard';
const key = path => JSON.stringify((path || []).map(String));
const type = node => node.comfyClass || node.properties?.['Node name for S&R'] || node.type;
const groups = panel => panel.properties?.donut_app_controls?.groups || [];
function entries(root) {
    const definitions = new Map((root?.definitions?.subgraphs || []).map(g => [String(g.id), g]));
    const result = [];
    function walk(graph, path = [], ancestors = new Set()) {
        if (!graph || ancestors.has(graph)) return;
        const visited = new Set(ancestors).add(graph);
        for (const node of graph.nodes || graph._nodes || []) {
            const at = [...path, node.id];
            result.push({node, path:at});
            walk(node.subgraph || definitions.get(String(node.type)), at, visited);
        }
    }
    walk(root);
    return result;
}
function role(panel) {
    const explicit = panel.properties?.donut_panel_role;
    if (explicit) return explicit;
    const title = String(panel.title || '').toLowerCase();
    if (/seed.*guidance/.test(title)) return 'guidance';
    if (/generate.*finish|generation/.test(title)) return 'generate';
    return null;
}
export function addTxtfusionGuardControls(root) {
    const all = entries(root);
    const panels = all.map(e => e.node).filter(n => Array.isArray(n.properties?.donut_app_controls?.groups));
    const changed = [];
    for (const panel of panels.filter(n => role(n) === 'guidance')) {
        const family = panel.properties.donut_app_controls.seed_path;
        if (!family?.length) continue;
        const owners = panels.filter(n => role(n) === 'generate'
            && key(n.properties.donut_app_controls.seed_path) === key(family));
        if (owners.length !== 1) continue;
        const controls = groups(owners[0]).flatMap(g => g.controls || []);
        const candidates = all.filter(e => type(e.node) === 'DonutSampler' && controls.some(c =>
            c.path?.length && c.path.length <= e.path.length
            && c.path.every((id, index) => String(id) === String(e.path[index]))));
        // Never guess between two first-pass samplers in an A/B workflow.
        if (candidates.length !== 1) continue;
        const target = candidates[0];
        const config = panel.properties.donut_app_controls;
        const before = JSON.stringify(config.groups);
        const previous = config.groups.find(g => g[MARK]);
        if (previous) {
            // Only our own tagged group's bindings are repaired on reload.
            previous.controls = GUARD_CONTROLS.map(([widget, title]) => ({path:[...target.path], widget, title}));
        } else {
            // Avoid adding duplicates to manually extended panels.
            const existing = new Set(config.groups.flatMap(g => g.controls || []).map(c => `${key(c.path)}:${c.widget}`));
            const missing = GUARD_CONTROLS.filter(([w]) => !existing.has(`${key(target.path)}:${w}`));
            if (missing.length) config.groups.push({
                [MARK]:true, title:'Experimental · checkpoint txtfusion guard',
                advanced:true, donut_category_fixed:true, donut_category_rank:12,
                description:'Base DonutSampler only. Default Off. Select the same diffusion checkpoint as the loaded model. Does not affect upscale. Floating-point txtfusion only; image benefit unverified.',
                controls:missing.map(([widget, title]) => ({path:[...target.path], widget, title})),
            });
        }
        if (JSON.stringify(config.groups) !== before) changed.push(panel);
    }
    return changed;
}

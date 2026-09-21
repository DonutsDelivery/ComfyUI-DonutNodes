// Model-wide control: bind to the real Fusion Control, not promoted sampler/NAG widgets.
const MARK = 'donut_txtfusion_model_rms_guard';
const pathKey = path => JSON.stringify((path || []).map(String));
const nodeType = n => n.comfyClass || n.properties?.['Node name for S&R'] || n.type;
function entries(root) {
    const definitions = new Map((root?.definitions?.subgraphs || []).map(g => [String(g.id), g]));
    const result = [];
    function walk(graph, path = [], parents = new Set()) {
        if (!graph || parents.has(graph)) return;
        const next = new Set(parents).add(graph);
        for (const node of graph.nodes || graph._nodes || []) {
            const at = [...path, node.id];
            result.push({node, path: at});
            walk(node.subgraph || definitions.get(String(node.type)), at, next);
        }
    }
    walk(root); return result;
}
export function addModelTxtfusionGuardControls(root) {
    const all = entries(root), changed = [];
    for (const {node: panel} of all) {
        const cfg = panel.properties?.donut_app_controls;
        const isModels = panel.properties?.donut_panel_role === 'models'
            || /(?:^|[ ·])Models$/i.test(panel.title || '');
        if (!isModels || !Array.isArray(cfg?.groups)) continue;
        const anchors = cfg.groups.filter(g => !g[MARK]).flatMap(g => g.controls || [])
            .filter(c => ['uncensorfix_controls', 'tap_method', 'compatibility_preset'].includes(c.widget));
        const candidates = all.filter(e => nodeType(e.node) === 'DonutKrea2FusionControl'
            && anchors.some(c => c.path?.length && c.path.length <= e.path.length
                && c.path.every((id, i) => String(id) === String(e.path[i]))));
        // One MODEL panel must have one explicit Fusion owner. Do not wire an
        // unrelated A/B branch or another panel family's model by numeric ID.
        if (candidates.length !== 1) continue;
        const target = candidates[0], before = JSON.stringify(cfg.groups);
        const others = cfg.groups.filter(g => !g[MARK]).flatMap(g => g.controls || []);
        if (others.some(c => c.widget === 'txtfusion_rms_guard' && pathKey(c.path) === pathKey(target.path))) continue;
        const previous = cfg.groups.find(g => g[MARK]);
        const group = {
            [MARK]: true, title: 'Txtfusion RMS guard', advanced: false,
            donut_category_fixed: true, donut_category_rank: 70.75,
            description: 'Experimental. All connected model stages, with NAG on or off. Rebalance stays unchanged. Extra reference memory and computation; image benefit unverified.',
            controls: [{path: [...target.path], widget: 'txtfusion_rms_guard', title: 'Normalize txtfusion changes · all stages'}],
        };
        if (previous) Object.assign(previous, group);
        else cfg.groups.push(group);
        if (JSON.stringify(cfg.groups) !== before) changed.push(panel);
    }
    return changed;
}

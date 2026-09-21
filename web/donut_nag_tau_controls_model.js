export const NO_TAU_WIDGET = 'nag_disable_tau_clipping';
const MARK = 'donut_nag_disable_tau_clipping';
const key = path => JSON.stringify((path || []).map(String));

export function addNagNoTauControls(root) {
    const changed = [];
    const definitions = new Map((root?.definitions?.subgraphs || []).map(g => [String(g.id), g]));
    const panels = [];
    function walk(graph, ancestors = new Set()) {
        if (!graph || ancestors.has(graph)) return;
        const visited = new Set(ancestors).add(graph);
        for (const node of graph.nodes || graph._nodes || []) {
            if (Array.isArray(node.properties?.donut_app_controls?.groups)) panels.push(node);
            walk(node.subgraph || definitions.get(String(node.type)), visited);
        }
    }
    walk(root);

    for (const panel of panels) {
        const groups = panel.properties.donut_app_controls.groups;
        const before = JSON.stringify(groups);
        for (const group of groups) {
            const controls = group.controls || [];
            const tau = controls.find(control => control.widget === 'nag_tau');
            if (!tau?.path?.length) continue;
            const existing = controls.find(control =>
                control.widget === NO_TAU_WIDGET && key(control.path) === key(tau.path));
            if (existing) {
                existing.title = 'Disable tau clipping · experiment';
                existing[MARK] = true;
                continue;
            }
            const index = controls.indexOf(tau);
            controls.splice(index + 1, 0, {
                [MARK]: true,
                path: [...tau.path],
                widget: NO_TAU_WIDGET,
                title: 'Disable tau clipping · experiment',
            });
            group.controls = controls;
        }
        if (JSON.stringify(groups) !== before) changed.push(panel);
    }
    return changed;
}

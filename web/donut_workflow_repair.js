// Repair only the v1 migration from PR47, before ComfyUI can shift positional
// widget values into different promoted controls. v2 saves are left untouched.
export function repairStreamlinedWorkflow(workflow) {
    if (workflow.extra?.donut_streamlining?.schema !== 1) return false;
    const definitions = new Map((workflow.definitions?.subgraphs || []).map(g => [g.id, g]));
    const changes = [];
    for (const node of workflow.nodes || []) {
        const graph = definitions.get(node.type);
        if (!graph?.nodes.some(n => n.type === "DonutPromptConditioning")) continue;
        const nodes = new Map(graph.nodes.map(n => [n.id, n]));
        const links = new Map(graph.links.map(link => [link.id, link]));
        const saved = new Map(node.inputs.map(p => [p.name, p]));
        const named = node.widgets_values_named;
        if (!named) throw new Error("Donut v1 repair needs the original named widget values. Load the corrected workflow file.");
        const order = [];
        const inputs = graph.inputs.map(port => {
            const input = { ...(saved.get(port.name) || {}), name: port.name, type: port.type, link: saved.get(port.name)?.link ?? null };
            if (port.label) input.label = port.label;
            if (port.shape) input.shape = port.shape;
            const promoted = (port.linkIds || []).some(id => {
                const link = links.get(id);
                if (!link || link.origin_id !== -10) return false;
                return !!nodes.get(link.target_id)?.inputs?.[link.target_slot]?.widget;
            });
            if (promoted) {
                if (!(port.name in named)) throw new Error(`Donut v1 repair: missing value for ${port.name}`);
                input.widget = { name: port.name }; order.push(port.name);
            } else delete input.widget;
            return input;
        });
        const remaps = (workflow.links || []).filter(link => link[3] === node.id).map(link => {
            const oldName = node.inputs[link[4]]?.name;
            const slot = inputs.findIndex(port => port.name === oldName);
            if (slot < 0) throw new Error(`Donut v1 repair: unknown input ${oldName}`);
            return [link, slot];
        });
        changes.push({ node, inputs, order, named, remaps });
    }
    if (!changes.length) return false;
    // Commit changes only after every instance is validated.
    for (const { node, inputs, order, named, remaps } of changes) {
        node.inputs = inputs;
        node.widgets_values = order.map(name => named[name]);
        node.widgets_values_named = Object.fromEntries(order.map(name => [name, named[name]]));
        node.properties = { ...node.properties, donut_widget_order: order, donut_stage_controls: true };
        for (const [link, slot] of remaps) link[4] = slot;
    }
    workflow.extra.donut_streamlining.schema = 2;
    workflow.extra.donut_streamlining.required_branch = "main";
    return true;
}

// Normalize tagged Donut workflows at BOTH import and export. Some frontend /
// extension combinations save only exposed sockets on a subgraph instance,
// while its definition and promoted value array still contain every socket.
// Never reset parameters or infer a cable's meaning from its numerical index.
const schemas = new Set([1, 2, 3]);
const clone = value => JSON.parse(JSON.stringify(value));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const key = value => String(value);
const fields = ["id", "origin_id", "origin_slot", "target_id", "target_slot", "type"];
const edgeData = edge => Array.isArray(edge) ? Object.fromEntries(fields.map((f, i) => [f, edge[i]])) : edge;
const compatible = (a, b) => a === b || a === "*" || b === "*";
const fail = message => { throw new Error(`Donut workflow reload: ${message}. Original data was not changed.`); };
function indexed(items, field, label) {
    const result = new Map();
    for (const item of items || []) {
        const id = item?.[field];
        if (id == null || result.has(key(id))) fail(`ambiguous ${label}`);
        result.set(key(id), item);
    }
    return result;
}
function scalar(value, type) {
    // Comfy seed widgets support integers beyond JS's lossless range. Keep
    // the parsed number as-is; this validator must not round or replace it.
    if (type === "INT") return Number.isInteger(value);
    if (type === "FLOAT") return typeof value === "number" && Number.isFinite(value);
    if (type === "BOOLEAN") return typeof value === "boolean";
    if (type === "STRING") return typeof value === "string";
    return typeof value === "string" || typeof value === "boolean" || (typeof value === "number" && Number.isFinite(value));
}

export function repairStreamlinedWorkflow(workflow) {
    const tag = workflow?.extra?.donut_streamlining;
    if (!schemas.has(tag?.schema)) return false;
    const definitions = indexed(workflow.definitions?.subgraphs, "id", "subgraph IDs");
    if (!definitions.size) return false;
    const contexts = new Map();
    for (const graph of [workflow, ...definitions.values()]) {
        contexts.set(graph, {
            nodes: indexed(graph.nodes, "id", "node IDs"),
            links: indexed((graph.links || []).map(edgeData), "id", "link IDs")
        });
    }
    const memo = new Map();
    function promoted(graph, port, visiting = new Set()) {
        const token = `${graph.id}/${port.id ?? port.name}`;
        if (visiting.has(token)) fail("recursive promoted input");
        if (memo.has(token)) return memo.get(token);
        const next = new Set(visiting); next.add(token);
        const context = contexts.get(graph);
        let found = false;
        for (const id of port.linkIds || []) {
            const edge = context.links.get(key(id));
            if (!edge || edge.origin_id !== -10 || graph.inputs[edge.origin_slot]?.name !== port.name) fail(`invalid internal link for ${port.name}`);
            if (edge.target_id === -20) continue;
            const target = context.nodes.get(key(edge.target_id));
            const input = target?.inputs?.[edge.target_slot];
            if (!input || key(input.link) !== key(edge.id)) fail(`invalid promoted target for ${port.name}`);
            const nested = definitions.get(target.type);
            if (nested) {
                const inner = nested.inputs.find(p => p.name === input.name);
                if (!inner) fail(`unknown nested input ${input.name}`);
                found = promoted(nested, inner, next) || found;
            } else found = !!input.widget || found;
        }
        memo.set(token, found);
        return found;
    }
    const changes = [];
    let instances = 0;
    for (const [scope] of contexts) {
        for (const node of scope.nodes || []) {
            const graph = definitions.get(node.type);
            if (!graph) continue;
            instances++;
            const oldInputs = node.inputs || [], oldOutputs = node.outputs || [];
            const savedIn = indexed(oldInputs, "name", `input names on ${node.id}`);
            const savedOut = indexed(oldOutputs, "name", `output names on ${node.id}`);
            const defIn = indexed(graph.inputs, "name", `definition inputs on ${node.id}`);
            const defOut = indexed(graph.outputs, "name", `definition outputs on ${node.id}`);
            for (const name of savedIn.keys()) if (!defIn.has(name)) fail(`unknown input ${node.id}:${name}`);
            for (const name of savedOut.keys()) if (!defOut.has(name)) fail(`unknown output ${node.id}:${name}`);
            const order = graph.inputs.filter(p => promoted(graph, p)).map(p => p.name);
            const values = node.widgets_values || [];
            const storedOrder = node.properties?.donut_widget_order;
            const oldOrder = oldInputs.filter(p => p.widget).map(p => p.name);
            // Named values win. For positional-only files, use an explicit saved
            // order, or the socket-widget order if its length is unambiguous.
            const fallbackOrder = Array.isArray(storedOrder) && storedOrder.length === values.length ? storedOrder
                : oldOrder.length === values.length ? oldOrder
                : tag.schema !== 1 && order.length === values.length ? order : [];
            const positional = Object.fromEntries(fallbackOrder.map((name, i) => [name, values[i]]));
            if (tag.schema !== 1) {
                for (const name of order) {
                    if (Object.hasOwn(positional, name) && Object.hasOwn(node.widgets_values_named || {}, name)
                        && !same(positional[name], node.widgets_values_named[name])) fail(`conflicting saved values for ${node.id}:${name}`);
                }
            }
            const named = { ...positional, ...(node.widgets_values_named || {}) };
            if (tag.schema === 1 && graph.nodes.some(n => n.type === "DonutPromptConditioning")) {
                for (const name of ["prompt", "prompt_1", "text"]) if (!order.includes(name)) delete named[name];
            }
            const inputs = graph.inputs.map(port => {
                const saved = savedIn.get(port.name);
                if (saved && !compatible(saved.type, port.type)) fail(`input type changed on ${node.id}:${port.name}`);
                const input = { ...clone(saved || {}), name: port.name, type: port.type, link: saved?.link ?? null };
                if (port.label !== undefined && input.label === undefined) input.label = port.label;
                if (port.shape !== undefined && input.shape === undefined) input.shape = port.shape;
                if (order.includes(port.name)) {
                    // A connected promoted widget gets its execution value
                    // from its cable. Some serializers omit its unused cache
                    // or store null. Preserve that absence, never invent a seed.
                    // The cable is validated below before any changes commit.
                    const missingConnectedCache = input.link != null && named[port.name] == null;
                    if (!missingConnectedCache && (!Object.hasOwn(named, port.name) || !scalar(named[port.name], port.type))) fail(`missing or invalid saved value for ${node.id}:${port.name}`);
                    input.widget = { name: port.name };
                } else delete input.widget;
                return input;
            });
            const outputs = graph.outputs.map(port => {
                const saved = savedOut.get(port.name);
                if (saved && !compatible(saved.type, port.type)) fail(`output type changed on ${node.id}:${port.name}`);
                return saved ? clone(saved) : { name: port.name, type: port.type, links: [] };
            });
            const remaps = [], seenInputs = new Set(), seenOutputs = new Set();
            for (const raw of scope.links || []) {
                const edge = edgeData(raw);
                if (key(edge.target_id) === key(node.id)) {
                    const old = oldInputs[edge.target_slot];
                    const index = inputs.findIndex(p => p.name === old?.name);
                    if (!old || index < 0 || key(old.link) !== key(edge.id) || seenInputs.has(old.name)) fail(`ambiguous destination on link ${edge.id}`);
                    if (!compatible(edge.type, inputs[index].type)) fail(`incompatible destination on link ${edge.id}`);
                    seenInputs.add(old.name);
                    if (index !== edge.target_slot) remaps.push({ raw, field: "target_slot", index });
                }
                if (key(edge.origin_id) === key(node.id)) {
                    const old = oldOutputs[edge.origin_slot];
                    const index = outputs.findIndex(p => p.name === old?.name);
                    if (!old || index < 0 || !(old.links || []).some(id => key(id) === key(edge.id))) fail(`ambiguous origin on link ${edge.id}`);
                    if (!compatible(edge.type, outputs[index].type)) fail(`incompatible origin on link ${edge.id}`);
                    seenOutputs.add(key(edge.id));
                    if (index !== edge.origin_slot) remaps.push({ raw, field: "origin_slot", index });
                }
            }
            for (const port of oldInputs) if (port.link != null && !seenInputs.has(port.name)) fail(`stale input link ${node.id}:${port.name}`);
            for (const port of oldOutputs) for (const id of port.links || []) if (!seenOutputs.has(key(id))) fail(`stale output link ${node.id}:${port.name}`);
            const props = { ...node.properties, donut_widget_order: order };
            if (graph.nodes.some(n => n.type === "DonutPromptConditioning")) props.donut_stage_controls = true;
            const nextValues = order.map(name => named[name]);
            // Keep unknown named metadata; only the positional array is restricted
            // to actual promoted controls. No authoring text/settings are reset.
            if (!same(inputs, oldInputs) || !same(outputs, oldOutputs) || !same(values, nextValues) || !same(props, node.properties) || !same(named, node.widgets_values_named) || remaps.length) {
                changes.push({ node, inputs, outputs, named, nextValues, props, remaps });
            }
        }
    }
    if (!instances) return false;
    // No mutation until ALL scopes and instances have passed validation.
    for (const c of changes) {
        Object.assign(c.node, { inputs: c.inputs, outputs: c.outputs, widgets_values: c.nextValues,
            widgets_values_named: c.named, properties: c.props });
        for (const { raw, field, index } of c.remaps) {
            if (Array.isArray(raw)) raw[fields.indexOf(field)] = index;
            else raw[field] = index;
        }
    }
    const upgraded = tag.schema !== 3 || tag.required_branch !== "main";
    tag.schema = 3; tag.required_branch = "main";
    return upgraded || changes.length > 0;
}

const guarded = new WeakMap();
export function installWorkflowSerializationGuard(graph) {
    if (!graph || typeof graph.serialize !== "function") return false;
    if (guarded.get(graph) === graph.serialize) return false;
    const original = graph.serialize;
    const wrapper = function(...args) {
        const result = original.apply(this, args);
        // Work on the detached serialized data, never reorder live socket arrays
        // without their live links. Covers JSON saves and PNG workflow metadata.
        repairStreamlinedWorkflow(result);
        return result;
    };
    graph.serialize = wrapper;
    guarded.set(graph, wrapper);
    return true;
}

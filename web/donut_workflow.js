import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// One STRING is the canonical state. UI rows are never independent positional
// widgets, so deleting/reordering a slot cannot shift another slot's values.
export function decodeRows(value) {
    const rows = JSON.parse(value || "[]");
    if (!Array.isArray(rows) || rows.some(row => !row || typeof row !== "object" || Array.isArray(row))) {
        throw new Error("LoRA slots must be an array of objects");
    }
    return rows;
}
export function moveRow(rows, from, to) {
    const result = rows.slice();
    if (from < 0 || from >= rows.length || to < 0 || to >= rows.length) return result;
    const [row] = result.splice(from, 1);
    result.splice(to, 0, row);
    return result;
}
const element = (tag, text) => {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    return el;
};
function hideWidget(widget) {
    if (!widget._donutOriginal) widget._donutOriginal = { type: widget.type, computeSize: widget.computeSize };
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
}
function restoreWidget(widget) {
    if (widget._donutOriginal) Object.assign(widget, widget._donutOriginal);
}
function dirty(node) { node.setDirtyCanvas?.(true, true); app.graph?.change?.(); }

function installRows(node, definition) {
    const state = node.widgets.find(w => w.name === "slots_json");
    if (!state) return;
    hideWidget(state);
    const options = definition.input.required.slots_json[1];
    const root = element("div");
    Object.assign(root.style, { overflow: "auto", padding: "8px", boxSizing: "border-box", font: "12px sans-serif" });
    const listId = `donut-loras-${globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)}`;
    let rows = [], metadata = new Map();
    state.serializeValue = () => state.value;
    const commit = () => { state.value = JSON.stringify(rows); dirty(node); };
    const field = (parent, label, value, type, changed) => {
        const wrapper = element("label", `${label} `);
        Object.assign(wrapper.style, { display: "block", margin: "4px 0" });
        const input = element("input"); input.type = type;
        if (type === "checkbox") input.checked = !!value;
        else input.value = value ?? "";
        if (type === "number") { input.step = "0.01"; input.min = "-1000"; input.max = "1000"; }
        else if (type !== "checkbox") input.style.width = "95%";
        input.addEventListener("change", () => {
            if (type === "number" && (!Number.isFinite(input.valueAsNumber) || !input.checkValidity())) return;
            changed(type === "checkbox" ? input.checked : type === "number" ? input.valueAsNumber : input.value);
            commit();
        });
        wrapper.append(input); parent.append(wrapper); return input;
    };
    const button = (parent, text, fn) => {
        const b = element("button", text); b.type = "button";
        b.addEventListener("click", event => { event.preventDefault(); fn(); }); parent.append(b); return b;
    };
    function render() {
        root.replaceChildren();
        const choices = element("datalist"); choices.id = listId;
        for (const name of options.donut_loras || ["None"]) { const o = element("option"); o.value = name; choices.append(o); }
        root.append(choices);
        const header = element("div", `${rows.length} LoRA slot${rows.length === 1 ? "" : "s"} `);
        button(header, "+ Add LoRA", () => {
            rows.push({ id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`, enabled: true,
                lora_name: "None", model_weight: 1, clip_weight: 1, block_vector: "", block_preset: "None",
                inherit_block_vector: false, lora_hash: "" });
            commit(); render();
        });
        root.append(header);
        rows.forEach((row, index) => {
            const box = element("fieldset"); box.append(element("legend", `LoRA ${index + 1}`));
            const nameInput = field(box, "LoRA", row.lora_name, "text", value => {
                if (value !== row.lora_name) { row.lora_hash = ""; metadata.delete(String(row.id)); }
                row.lora_name = value;
            });
            nameInput.setAttribute("list", listId);
            field(box, "Enabled", row.enabled, "checkbox", value => { row.enabled = value; });
            field(box, "Model strength", row.model_weight, "number", value => { row.model_weight = value; });
            field(box, "CLIP / text-fusion strength", row.clip_weight, "number", value => { row.clip_weight = value; });
            const advanced = element("details"); advanced.append(element("summary", "Block weights / metadata"));
            const inherit = field(advanced, "Inherit global vector", row.inherit_block_vector, "checkbox", value => {
                row.inherit_block_vector = value; vector.disabled = value;
            });
            const presets = element("select");
            const modelType = node.widgets.find(w => w.name === "model_type")?.value || "Auto";
            for (const value of options.donut_presets || ["None"]) {
                if (modelType !== "Auto" && value !== "None" && !value.startsWith(`${modelType}-`)) continue;
                const o = element("option", value.split(":")[0]); o.value = value; presets.append(o);
            }
            presets.value = row.block_preset || "None";
            presets.addEventListener("change", () => {
                row.block_preset = presets.value;
                row.block_vector = presets.value.includes(":") ? presets.value.slice(presets.value.indexOf(":") + 1) : "";
                row.inherit_block_vector = false; inherit.checked = false;
                vector.disabled = false; vector.value = row.block_vector; commit();
            });
            advanced.append(presets);
            const vector = field(advanced, "Block vector", row.block_vector, "text", value => { row.block_vector = value; });
            vector.disabled = !!row.inherit_block_vector;
            if (row.lora_hash) advanced.append(element("small", `Hash: ${row.lora_hash}`));
            const info = metadata.get(String(row.id));
            if (info?.text) { const pre = element("pre", info.text); pre.style.whiteSpace = "pre-wrap"; advanced.append(pre); }
            if (info?.image?.filename) {
                const image = element("img"); image.style.maxWidth = "100%"; image.loading = "lazy";
                image.src = api.apiURL(`/view?${new URLSearchParams(info.image)}`); advanced.append(image);
            }
            box.append(advanced);
            button(box, "↑", () => { rows = moveRow(rows, index, index - 1); commit(); render(); }).disabled = index === 0;
            button(box, "↓", () => { rows = moveRow(rows, index, index + 1); commit(); render(); }).disabled = index === rows.length - 1;
            button(box, "Remove", () => { rows.splice(index, 1); commit(); render(); });
            root.append(box);
        });
    }
    function restore(data) {
        const index = node.widgets.indexOf(state);
        const value = data?.widgets_values_named?.slots_json ?? data?.widgets_values?.[index] ?? state.value;
        state.value = value;
        try { rows = decodeRows(value); render(); }
        catch (error) {
            root.replaceChildren(element("p", `Invalid LoRA state: ${error.message}`));
            // Never replace malformed/unknown saved state with an empty stack.
            const edit = element("textarea"); edit.value = value; edit.style.width = "95%";
            edit.addEventListener("change", () => { state.value = edit.value; restore(); dirty(node); }); root.append(edit);
        }
    }
    const dom = node.addDOMWidget("donut_lora_editor", "div", root, { serialize: false });
    dom.computeSize = () => [420, Math.min(850, Math.max(260, rows.length * 190 + 50))];
    const configure = node.onConfigure;
    node.onConfigure = function(data) { const r = configure?.apply(this, arguments); restore(data); return r; };
    const serialize = node.onSerialize;
    node.onSerialize = function(data) {
        serialize?.apply(this, arguments);
        (data.widgets_values_named ||= {}).slots_json = state.value;
    };
    const executed = node.onExecuted;
    node.onExecuted = function(message) {
        executed?.apply(this, arguments);
        metadata = new Map((message.donut_loras || []).map(item => [String(item.id), item]));
        let hashChanged = false;
        for (const row of rows) {
            const item = metadata.get(String(row.id));
            // The user may edit a slot while a queued run is executing.
            if (item?.lora_name === row.lora_name && item.lora_hash && item.lora_hash !== row.lora_hash) {
                row.lora_hash = item.lora_hash; hashChanged = true;
            }
        }
        if (hashChanged) commit();
        render();
    };
    const modelWidget = node.widgets.find(w => w.name === "model_type");
    if (modelWidget) { const cb = modelWidget.callback; modelWidget.callback = function() { cb?.apply(this, arguments); render(); }; }
    restore();
    node.setSize?.([Math.max(440, node.size?.[0] || 0), Math.max(600, node.size?.[1] || 0)]);
}

app.registerExtension({
    name: "Donut.WorkflowStreamlining",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData.name;
        if (!["DonutDynamicLoRAStack", "DonutLoRALoader", "DonutText", "DonutPromptConditioning", "DonutSeedPlan", "DonutModelMergeKrea2"].includes(name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            if (["DonutDynamicLoRAStack", "DonutLoRALoader"].includes(name)) installRows(this, nodeData);
            if (["DonutText", "DonutPromptConditioning"].includes(name)) {
                const preview = ComfyWidgets.STRING(this, "resolved_text", ["STRING", { multiline: true }], app).widget;
                preview.options ||= {}; preview.options.serialize = false;
                if (preview.inputEl) preview.inputEl.readOnly = true;
                const executed = this.onExecuted;
                this.onExecuted = function(message) { executed?.apply(this, arguments); preview.value = (message.text || []).join("\n"); };
            }
            if (name === "DonutSeedPlan") {
                for (const seedName of ["text_seed", "sampler_seed", "filename_seed"]) {
                    const index = this.widgets.findIndex(w => w.name === seedName);
                    const control = this.widgets[index + 1];
                    if (index >= 0 && control?.name?.includes("control_after_generate")) control.name = `${seedName}_control`;
                }
            }
            if (name === "DonutModelMergeKrea2") {
                const mode = this.widgets.find(w => w.name === "ratio_mode");
                const update = () => {
                    if (!mode) return;
                    for (const widget of this.widgets) {
                        if (widget.name === "first." || widget.name === "last." || widget.name.startsWith("blocks.") || widget.name.startsWith("txtfusion.")) {
                            if (mode.value === "Grouped") hideWidget(widget); else restoreWidget(widget);
                        }
                    }
                    this.setSize?.([this.size[0], this.computeSize()[1]]); this.setDirtyCanvas?.(true, true);
                };
                if (mode) { const cb = mode.callback; mode.callback = function() { cb?.apply(this, arguments); update(); }; }
                const configure = this.onConfigure;
                this.onConfigure = function() { const r = configure?.apply(this, arguments); update(); return r; };
                update();
            }
            return result;
        };
    },
});

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createLoraToolbar, renderLoraInformation } from "./donut_lora_ui.js";
import { createLoraService, decodeRows, moveRow } from "./donut_native_lora.js";

import { weightControl, vectorControl } from "./donut_weight_controls.js";
import { promptTools, wildcardLibrary } from "./donut_wildcards.js";
import { fitModule, fitTextarea, scheduleLayout } from "./donut_layout.js?v=15";

const service = createLoraService(api);
const FUSION_PRESET = (tapMethod, tapProfile, tapNormalization, projectorMethod, fusionMethod) => ({
    tap_method: tapMethod, tap_profile: tapProfile, tap_strength: 1,
    tap_formula: "scale_around_1", tap_normalization: tapNormalization,
    projector_method: projectorMethod, projector_profile: "off", projector_strength: 1,
    projector_formula: "scale_around_1", projector_normalization: "none",
    fusion_method: fusionMethod, fusion_strength: 1,
});
const FUSION_PRESETS = {
    "Bypass 2": FUSION_PRESET("Donut 12-tap gains", "off", "tensor_rms", "Krea2FilterBypass 2vector diff", "Standard Krea2 fusion"),
    "Bypass 3": FUSION_PRESET("Donut 12-tap gains", "off", "tensor_rms", "Krea2FilterBypass 3vector diff", "Standard Krea2 fusion"),
    Rebalance: FUSION_PRESET("nova452 Rebalance operation", "classic", "none", "Donut projector-input gains", "Standard Krea2 fusion"),
    Enhancer: FUSION_PRESET("Donut 12-tap gains", "off", "tensor_rms", "Donut projector-input gains", "capitan01R Krea2T-Enhancer operation"),
    "Rebalance + Enhancer": FUSION_PRESET("nova452 Rebalance operation", "classic", "none", "Donut projector-input gains", "capitan01R Krea2T-Enhancer operation"),
    "Rebalance + Bypass 2": FUSION_PRESET("nova452 Rebalance operation", "classic", "none", "Krea2FilterBypass 2vector diff", "Standard Krea2 fusion"),
    "Rebalance + Bypass 3": FUSION_PRESET("nova452 Rebalance operation", "classic", "none", "Krea2FilterBypass 3vector diff", "Standard Krea2 fusion"),
    Balanced: FUSION_PRESET("Donut 12-tap gains", "classic", "tensor_rms", "Donut projector-input gains", "Standard Krea2 fusion"),
    "Balanced + Enhancer": FUSION_PRESET("Donut 12-tap gains", "classic", "tensor_rms", "Donut projector-input gains", "capitan01R Krea2T-Enhancer operation"),
    UncensorFix: FUSION_PRESET("Donut 12-tap gains", "off", "tensor_rms", "Donut projector-input gains", "Standard Krea2 fusion"),
};
for (const [legacy, current] of Object.entries({
    "COPY settings: Krea2FilterBypass 2vector": "Bypass 2",
    "COPY settings: Krea2FilterBypass 3vector": "Bypass 3",
    "COPY settings: nova452 ConditioningKrea2Rebalance profile @ tap strength 1": "Rebalance",
    "COPY settings: capitan01R Krea2T-Enhancer defaults": "Enhancer",
    "HYBRID settings: Rebalance + Krea2T-Enhancer": "Rebalance + Enhancer",
    "HYBRID settings: Rebalance + Krea2FilterBypass 2vector": "Rebalance + Bypass 2",
    "HYBRID settings: Rebalance + Krea2FilterBypass 3vector": "Rebalance + Bypass 3",
    "DONUT settings: RMS-balanced classic": "Balanced",
    "DONUT settings: RMS-balanced classic + Krea2T-Enhancer": "Balanced + Enhancer",
    TeacherFix: "UncensorFix",
    "DONUT settings: Krea2 C33 TeacherFix EMA5000": "UncensorFix",
})) FUSION_PRESETS[legacy] = FUSION_PRESETS[current];

function applyFusionPreset(node, preset) {
    const values = preset === "Off" ? FUSION_PRESETS.UncensorFix : FUSION_PRESETS[preset];
    if (!values || !node.widgets?.some(widget => widget.name === "tap_method")) return;
    node._donutApplyingKrea2Preset = true;
    try {
        for (const [name, value] of Object.entries(values)) {
            const widget = node.widgets.find(item => item.name === name);
            if (!widget) continue;
            widget.value = value;
            widget.callback?.(value, app.canvas, node);
        }
    } finally {
        node._donutApplyingKrea2Preset = false;
    }
    // Converted/linked widgets may have no profile callback installed.
    const neutral = Array(12).fill("1.0").join(",");
    for (const [name, value] of Object.entries({
        per_layer_weights: values.tap_profile === "classic"
            ? "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0" : neutral,
        projector_layer_weights: neutral,
        compatibility_preset: preset,
        uncensorfix_controls: ["UncensorFix", "TeacherFix", "DONUT settings: Krea2 C33 TeacherFix EMA5000"].includes(preset)
            ? "Fusion + UncensorFix weights" : "Fusion only",
    })) {
        const widget = node.widgets.find(item => item.name === name);
        if (widget) widget.value = value;
    }
}
const element = (tag, text) => {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    return result;
};
function resolve(path) {
    let graph = app.rootGraph, node;
    for (const id of path) {
        node = graph?.getNodeById(id);
        graph = node?.subgraph;
    }
    return node;
}
function commitWidget(node, widget, value) {
    node.graph.beforeChange();
    widget.value = value;
    widget.callback?.(value, app.canvas, node);
    if (widget.name === "compatibility_preset") applyFusionPreset(node, value);
    node.graph.afterChange();
    node.setDirtyCanvas(true, true);
}
function install(node, appOnly = false) {
    if (node._donutAppControls) return;
    const root = element("div");
    root.className = "donut-app-controls" + (appOnly ? " donut-app-only" : " donut-section-controls");
    const refreshers = [];
    function commit(target, widget, value) {
        commitWidget(target, widget, value);
        if (!["compatibility_preset", "tap_strength"].includes(widget.name)) return;
        // v4 exposes the preset on the outer subgraph, while the advanced
        // controls address its inner Fusion node. The outer widget does not
        // run the inner node's preset callback. Resolve the actual Fusion
        // controls from this panel's configuration, including saved workflows.
        const controls = node.properties?.donut_app_controls?.groups?.flatMap(group => group.controls || []) || [];
        const source = controls.find(item => item.widget === widget.name && resolve(item.path) === target);
        if (!source) return;
        const destinations = new Set(controls.filter(item => item.widget === "tap_method"
            && item.path.length > source.path.length
            && source.path.every((id, index) => id === item.path[index]))
            .map(item => resolve(item.path)).filter(Boolean));
        for (const destination of destinations) {
            destination.graph.beforeChange();
            if (widget.name === "compatibility_preset") applyFusionPreset(destination, value);
            else {
                const innerStrength = destination.widgets?.find(item => item.name === "tap_strength");
                if (innerStrength) {
                    innerStrength.value = value;
                    innerStrength.callback?.(value, app.canvas, destination);
                }
            }
            destination.graph.afterChange();
            destination.setDirtyCanvas(true, true);
        }
        if (widget.name === "compatibility_preset" && destinations.size && FUSION_PRESETS[value]) {
            const strength = target.widgets?.find(item => item.name === "tap_strength");
            if (strength) commitWidget(target, strength, 1);
        }
    }
    // Preset widgets change several sibling widgets through their callbacks.
    // The workflow panel owns separate HTML inputs for those siblings, so it
    // must refresh them immediately instead of waiting for its visibility
    // observer/polling interval to run.
    function refreshControls() {
        refreshers.forEach(refresh => refresh());
    }
    function field(parent, path, name, title, choices, weightOptions, ui, modeControl) {
        const target = resolve(path);
        if (modeControl === "bypass") {
            if (!target) {
                parent.append(element("p", `${title}: control unavailable`));
                return;
            }
            const label = element("label"), caption = element("span", title), input = element("input");
            input.type = "checkbox";
            input.setAttribute("aria-label", title);
            const refresh = () => { input.checked = target.mode !== 2 && target.mode !== 4; };
            refresh(); refreshers.push(refresh);
            input.addEventListener("change", () => {
                target.graph.beforeChange();
                // ComfyUI serializes mode 4 as bypass. Mode 0 makes the
                // existing node active again without changing its connections.
                target.mode = input.checked ? 0 : 4;
                target.graph.afterChange();
                target.setDirtyCanvas(true, true);
            });
            label.append(caption, input); parent.append(label);
            return;
        }
        const widget = target?.widgets?.find(w => w.name === name);
        // Migrate the shared generation control once. Subsequent selections,
        // including Fixed, are preserved across refreshes and workflow saves.
        if (widget && name === "fixed" && title === "After generation" && !target.properties?.donut_randomize_default_v1) {
            target.properties ||= {};
            widget.value = "randomize";
            target.properties.donut_randomize_default_v1 = true;
        }
        if (!widget) {
            parent.append(element("p", `${title}: control unavailable`));
            return;
        }
        if (/vector|layer_weights/.test(name) && typeof widget.value === "string") {
            const control = vectorControl(title, () => widget.value, value => commit(target, widget, value), weightOptions);
            parent.append(control.element); refreshers.push(control.refresh); return;
        }
        if (typeof widget.value === "number" && (/ratio|weight|strength|cfg|denoise|alpha|scale|^blocks\.|^txtfusion\.|^first\.|^last\.|^tmlp\.|^txtmlp\.|^tproj\./.test(name))) {
            const min = Math.max(widget.options?.min ?? -2, -2), max = Math.min(widget.options?.max ?? 2, /cfg/.test(name) ? 20 : 2);
            const control = weightControl(title, () => widget.value, value => commit(target, widget, value), {min,max,...weightOptions});
            parent.append(control.element); refreshers.push(control.refresh); return;
        }
        const label = element("label"), caption = element("span", title);
        let values = choices || widget.options?.values;
        if (typeof values === "function") values = values();
        const composition = name === "uncensorfix_controls";
        if (composition) values = ["Fusion only", "Fusion + UncensorFix weights"];
        const boolean = typeof widget.value === "boolean";
        const numeric = typeof widget.value === "number";
        const multiline = widget.options?.multiline || (typeof widget.value === "string" && !values && /text|prompt|instruction/i.test(name));
        const input = element(Array.isArray(values) ? "select" : multiline ? "textarea" : "input");
        if (Array.isArray(values)) for (const value of values) {
            const option = element("option", String(value)); option.value = value; input.append(option);
        }
        if (input.tagName === "INPUT") input.type = boolean ? "checkbox" : numeric ? "number" : "text";
        if (numeric) {
            input.step = widget.options?.precision === 0 || /seed/.test(name) ? "1" : "any";
            if (widget.options?.min !== undefined) input.min = widget.options.min;
            if (widget.options?.max !== undefined) input.max = widget.options.max;
        }
        input.setAttribute("aria-label", title);
        if (ui?.prompt) { input.classList.add("donut-long-prompt"); input.style.minHeight = `${ui.height || 300}px`; }
        const refresh = () => {
            if (document.activeElement === input) return;
            if (boolean) input.checked = widget.value;
            else if (composition) {
                // Keep legacy serialization/behavior until the user edits it,
                // but never offer obsolete modes as new choices.
                const presetControl = node.properties?.donut_app_controls?.groups
                    ?.flatMap(group => group.controls || [])
                    .find(item => item.widget === "compatibility_preset"
                        && item.path.every((id, index) => path[index] === id));
                const presetNode = presetControl ? resolve(presetControl.path) : target;
                const preset = presetNode?.widgets?.find(item => item.name === "compatibility_preset")?.value;
                const legacyActive = ["UncensorFix", "TeacherFix", "DONUT settings: Krea2 C33 TeacherFix EMA5000"].includes(preset);
                input.value = widget.value === "Fusion + LoRA" || widget.value === "Fusion + UncensorFix weights"
                    || (legacyActive && ["LoRA only", "LoRA + fusion controls"].includes(widget.value))
                    ? "Fusion + UncensorFix weights" : "Fusion only";
            }
            else input.value = widget.value ?? "";
            if (input.tagName === "TEXTAREA") fitTextarea(input);
        };
        refresh(); refreshers.push(refresh);
        input.addEventListener(Array.isArray(values) || boolean ? "change" : "input", () => {
            if (numeric && (!input.value || !Number.isFinite(Number(input.value)))) { refresh(); return; }
            commit(target, widget, boolean ? input.checked : numeric ? Number(input.value) : input.value);
            refreshControls();
            // A few native callbacks defer dependent-widget updates. Refresh
            // once more after that microtask so the panel always mirrors the
            // node's final values.
            queueMicrotask(refreshControls);
        });
        label.append(caption, input); parent.append(label);
        if (name === "uncensorfix_controls" && target.widgets?.some(item => item.name === "uncensorfix_strength")) {
            field(parent, path, "uncensorfix_strength", "UncensorFix weight strength");
        }
        if (ui?.prompt) {
            const tools = promptTools(input, value => commit(target, widget, value), () => {
                const seedNode = resolve(node.properties?.donut_app_controls?.seed_path || []);
                return seedNode?.widgets?.find(w => w.name === "seed")?.value ?? 0;
            });
            parent.append(tools.element); refreshers.push(tools.refresh);
        }
    }
    function loras(parent, path) {
        const target = resolve(path), state = target?.widgets?.find(w => w.name === "slots_json");
        if (!state) return;
        const list = element("div"), status = element("p"), add = element("button", "Add LoRA");
        add.type = "button";
        let catalog = ["None"], presets = ["None"], last;
        const information = new Map();
        function save(rows, rebuild = true) { commit(target, state, JSON.stringify(rows)); target._donutNativeLoras.restore(); last = state.value; if (rebuild) render(); }
        function render() {
            last = state.value; list.replaceChildren();
            let rows;
            try { rows = decodeRows(state.value); } catch { status.textContent = "Invalid LoRA configuration. Repair slots_json in the workflow."; return; }
            if (!rows.length) list.append(element("p", "No LoRAs in this stack. Add a LoRA to begin."));
            rows.forEach((row, index) => {
                const box = element("fieldset"); box.className = "donut-lora-row";
                const key = String(row.id ?? index + 1);
                const action = name => {
                    const current = decodeRows(state.value);
                    const at = current.findIndex((item, i) => String(item.id ?? i + 1) === key);
                    if (at < 0) return;
                    if (name === "Remove") { current.splice(at, 1); information.delete(key); save(current); }
                    else if (name === "Move up" || name === "Move down") save(moveRow(current, at, at + (name === "Move up" ? -1 : 1)));
                    else if (name === "Retry metadata") loadInfo(true);
                    else if (name === "Use suggested model weight") {
                        const weight = information.get(key)?.info?.civitai?.recommended_weight;
                        if (typeof weight === "number" && Number.isFinite(weight)) { current[at].model_weight = weight; save(current); }
                    }
                };
                const toolbar = createLoraToolbar(index, rows.length, action);
                toolbar.classList.add("donut-lora-toolbar"); box.append(toolbar);
                const edit = (key, title, type) => {
                    const label = element("label"), input = element(type === "select" ? "select" : "input");
                    if (type === "select") for (const name of [...new Set([...catalog, row.lora_name])].filter(Boolean)) {
                        const option = element("option", name); option.value = name; input.append(option);
                    }
                    else input.type = type;
                    if (type === "checkbox") input.checked = key === "enabled" ? row[key] !== false : !!row[key];
                    else input.value = row[key] ?? (type === "number" ? 1 : "None");
                    if (type === "number") input.step = "any";
                    input.setAttribute("aria-label", `${title} ${index + 1}`);
                    input[type === "number" ? "oninput" : "onchange"] = () => {
                        row[key] = type === "checkbox" ? input.checked : type === "number" ? Number(input.value) : input.value;
                        if (key === "lora_name") delete row.lora_hash;
                        save(rows, type !== "number");
                    };
                    label.append(element("span", title), input); box.append(label);
                };
                edit("enabled", "Enabled", "checkbox"); edit("lora_name", "Installed LoRA", "select");
                for (const [key, title] of [["model_weight", "Model strength"], ["clip_weight", "Text strength"]]) {
                    const control = weightControl(`${title} ${index + 1}`, () => row[key] ?? 1, value => { row[key] = value; save(rows, false); });
                    box.append(control.element);
                }
                const presetLabel = element("label"), preset = element("select");
                const family = target.widgets.find(w => w.name === "model_type")?.value;
                for (const value of [...new Set([...presets.filter(p => p === "None" || family === "Auto" || p.startsWith(`${family}-`)), row.block_preset || "None"])]) {
                    const option = element("option", value.split(":")[0]); option.value = value; preset.append(option);
                }
                preset.value = row.block_preset || "None"; preset.setAttribute("aria-label", `Block preset ${index + 1}`);
                preset.onchange = () => {
                    row.block_preset = preset.value;
                    row.block_vector = preset.value.includes(":") ? preset.value.slice(preset.value.indexOf(":") + 1) : "";
                    row.inherit_block_vector = false; save(rows);
                };
                presetLabel.append(element("span", "Block preset"), preset); box.append(presetLabel);
                edit("inherit_block_vector", "Use global block weights", "checkbox");
                if (!row.inherit_block_vector) {
                    const weights = vectorControl(`LoRA ${index + 1} block weights`, () => row.block_vector, value => {
                        row.block_vector = value; row.block_preset = "None"; save(rows, false);
                    });
                    box.append(weights.element);
                }
                const more = element("details"), info = element("div");
                more.className = "donut-lora-information";
                more.append(element("summary", "LoRA information · model compatibility, triggers and metadata"), info);
                let entry = information.get(key);
                if (!entry || entry.name !== row.lora_name) { entry = {name:row.lora_name, view:{}}; information.set(key, entry); }
                function paintInfo() {
                    if (!info.isConnected || information.get(key) !== entry) return;
                    renderLoraInformation(info, row, entry, {api,lookupOn:target.widgets.find(w => w.name === "civitai_lookup")?.value === "On",view:entry.view,onAction:action,onResize:scheduleLayout});
                    scheduleLayout();
                }
                async function loadInfo(force = false) {
                    if (!row.lora_name || row.lora_name === "None") { info.textContent = "Choose an installed LoRA first."; return; }
                    const lookup = target.widgets.find(w => w.name === "civitai_lookup")?.value === "On";
                    await Promise.all([['analyze','analysis'], ...(lookup ? [['info','info']] : [])].map(async ([kind, field]) => {
                        if (!force && (entry[field] || entry[field + "Loading"])) return;
                        entry[field + "Loading"] = true; delete entry[field + "Error"];
                        try { entry[field] = await service.details(kind, row.lora_name, force); }
                        catch (error) { entry[field + "Error"] = error.message; }
                        finally { entry[field + "Loading"] = false; paintInfo(); }
                    }));
                    paintInfo();
                }
                more.ontoggle = () => { if (more.open) { paintInfo(); void loadInfo(); } scheduleLayout(); };
                box.append(more);
                list.append(box);
            });
        }
        add.onclick = () => save([...decodeRows(state.value), {id:crypto.randomUUID(), enabled:true, lora_name:"None", model_weight:1, clip_weight:0, inherit_block_vector:true}]);
        const refreshList = element("button", "Refresh installed LoRAs"); refreshList.type = "button";
        refreshList.onclick = async () => {
            refreshList.disabled = true;
            try { const data = await service.catalog(true); catalog = data.loras; presets = data.presets; status.textContent = "Installed LoRAs refreshed."; render(); }
            catch (error) { status.textContent = error.message; }
            finally { refreshList.disabled = false; }
        };
        parent.append(add, refreshList, status, list); render();
        service.catalog().then(data => { catalog = data.loras; presets = data.presets; render(); }).catch(error => { status.textContent = error.message; });
        refreshers.push(() => { if (last !== state.value && !list.contains(document.activeElement)) render(); });
    }
    function render() {
        root.replaceChildren(); refreshers.length = 0;
        root.classList.toggle("donut-section-columns", node.properties?.donut_columns === "sections");
        root.style.setProperty("--donut-accent", node.properties?.panel_color || "#b99cff");
        const config = node.properties?.donut_app_controls;
        if (!config) return;
        if (appOnly) root.append(element("h2", "Create"), element("p", "Everyday controls, with advanced settings grouped by purpose."));
        else { root.append(element("h2", node.title)); dom.label = node.title; }
        for (const group of config.groups) {
            const section = element(group.advanced ? "details" : "section");
            section.append(element(group.advanced ? "summary" : "h3", group.title));
            if (!group.advanced && node.properties?.donut_columns === "fields" && !group.controls?.some(item => item.ui?.prompt || item.weights?.vertical)) section.classList.add("donut-field-columns");
            if (group.color) section.style.setProperty("--donut-accent", group.color);
            if (group.description) section.append(element("p", group.description));
            let bank;
            for (const item of group.controls || []) {
                if (item.weights?.vertical && !bank) { bank = element("div"); bank.className = "donut-weight-bank"; section.append(bank); }
                field(item.weights?.vertical ? bank : section, item.path, item.widget, item.title, item.choices, item.weights, item.ui, item.mode);
            }
            if (group.loras) loras(section, group.loras);
            if (group.wildcard_library) section.append(wildcardLibrary());
            if (group.visible_when) {
                const condition = group.visible_when;
                const refresh = () => {
                    const target = resolve(condition.path);
                    section.hidden = target?.widgets?.find(w => w.name === condition.widget)?.value !== condition.value;
                };
                refresh(); refreshers.push(refresh);
            }
            root.append(section);
        }
        root.querySelectorAll('textarea').forEach(fitTextarea); scheduleLayout();
    }
    const dom = node.addDOMWidget("workflow_controls", "custom", root, {serialize:false, hideOnZoom:false, getValue:() => "", setValue:() => {}});
    dom.label = appOnly ? "Creation controls" : node.title;
    dom.serialize = false; dom.options.serialize = false;
    if (appOnly) {
        // App Mode mounts the element directly. Canvas rendering must allocate
        // no space and must never paint the combined app panel on this node.
        dom.isVisible = () => false;
        dom.computeSize = () => [0, -4];
        dom.computeLayoutSize = () => ({minHeight:0, maxHeight:0, minWidth:0});
        dom.options.getMinHeight = () => 0;
        dom.options.getMaxHeight = () => 0;
        dom.options.margin = 0;
    } else {
        dom.computeSize = () => [440, node.properties?.panel_height || 460];
        dom.computeLayoutSize = () => ({minHeight:node.properties?.panel_height || 460, maxHeight:node.properties?.panel_height || 460, minWidth:380});
        dom.options.getMinHeight = dom.options.getMaxHeight = () => node.properties?.panel_height || 460;
    }
    let timer;
    const observer = new IntersectionObserver(entries => {
        clearInterval(timer);
        if (entries.some(entry => entry.isIntersecting)) {
            refreshers.forEach(refresh => refresh());
            timer = setInterval(() => refreshers.forEach(refresh => refresh()), 500);
        }
    });
    observer.observe(root);
    const added = node.onAdded;
    node.onAdded = function() { const result = added?.apply(this, arguments); observer.observe(root); return result; };
    const removed = node.onRemoved;
    node.onRemoved = function() { clearInterval(timer); observer.disconnect(); return removed?.apply(this, arguments); };
    node._donutAppControls = {render, dom, root};
    if (!appOnly) fitModule(node, dom, root);
}
app.registerExtension({
    name:"Donut.AppControls",
    setup() {
        const style = element("style");
        style.textContent = `.graph-canvas-container .donut-app-only{display:none!important}.graph-canvas-container [data-testid="node-widget"]:has(.donut-app-only){display:none!important}.donut-section-controls{border-radius:10px;border-top:5px solid var(--donut-accent)}.donut-section-controls section:first-child{border-top:0;padding-top:0}.donut-app-controls{box-sizing:border-box;width:100%;height:auto;overflow:visible;padding:18px;color:var(--input-text,#eee);background:var(--comfy-menu-bg,#242424);font:15px/1.45 system-ui;container-type:inline-size}.donut-app-controls *{box-sizing:border-box}.donut-app-controls h2{margin:0 0 6px}.donut-app-controls h3{margin:0 0 12px;color:var(--donut-accent)}.donut-app-controls p{opacity:.75;margin:6px 0 14px}.donut-app-controls section,.donut-app-controls details{border-top:1px solid #ffffff25;padding:16px 0}.donut-app-controls summary{font-weight:650;cursor:pointer;padding:4px 0 12px}.donut-app-controls label{display:flex;align-items:center;gap:12px;justify-content:space-between;margin:9px 0}.donut-app-controls label:has(textarea){display:block}.donut-app-controls input,.donut-app-controls select,.donut-app-controls textarea,.donut-app-controls button{font:inherit;color:inherit;border:1px solid #ffffff35;background:var(--comfy-input-bg,#181818);border-radius:6px;padding:7px;min-width:0}.donut-app-controls input:not([type=checkbox]),.donut-app-controls select{width:55%}.donut-app-controls textarea{overflow:hidden;display:block;width:100%;min-height:90px;resize:vertical;margin-top:6px}.donut-app-controls button{cursor:pointer;margin:5px 6px 0 0}.donut-app-controls button:disabled{opacity:.35}.donut-app-controls fieldset{border:1px solid #ffffff25;border-radius:8px;margin:8px 0;padding:10px}@container (max-width:420px){.donut-app-controls label{display:block}.donut-app-controls input:not([type=checkbox]),.donut-app-controls select{display:block;width:100%;margin-top:5px}.donut-app-controls input[type=checkbox]{float:right}}.donut-app-controls input:focus,.donut-app-controls textarea:focus,.donut-app-controls select:focus{outline:2px solid var(--donut-accent)}.donut-app-controls .donut-weight-row{display:block}.donut-app-controls .donut-weight-value{display:flex;align-items:center;gap:8px;margin-top:5px}.donut-app-controls .donut-weight-value input[type=range]{flex:1;width:0;accent-color:var(--donut-accent);padding:0;min-width:65px}.donut-app-controls .donut-weight-value input[type=number]{width:82px;flex:none;margin:0}.donut-app-controls .donut-vector-control{margin:16px 0;border-top:1px solid #ffffff20;padding-top:8px}.donut-app-controls .donut-vector-grid{display:grid;grid-template-columns:1fr;gap:0 14px}.donut-app-controls [hidden]{display:none!important}@container(min-width:600px){.donut-app-controls .donut-vector-grid{grid-template-columns:1fr 1fr}}`;
        style.textContent += `
.donut-app-controls h2{color:#f5f7fb;font-size:25px;font-weight:750;line-height:1.25;margin:0 0 22px;padding-bottom:14px;border-bottom:2px solid var(--donut-accent)}
.donut-app-controls{min-height:0;flex:none!important;scrollbar-width:thin;scrollbar-color:var(--donut-accent) #20252d}
.donut-app-controls .donut-weight-bank{display:flex;align-items:stretch;gap:5px;overflow:visible;width:max-content;padding:10px 3px 16px;scrollbar-color:var(--donut-accent) #20252d;scrollbar-width:auto}
.donut-app-controls .donut-vector-grid{display:flex;flex-wrap:nowrap;gap:3px}
.donut-app-controls .donut-weight-row.donut-weight-vertical{display:flex;flex-direction:column;align-items:center;flex:0 0 34px;width:34px;margin:0;gap:6px}
.donut-app-controls .donut-weight-vertical>span{height:27px;font-size:12px;line-height:27px;max-width:34px;overflow:hidden;white-space:nowrap;text-align:center;color:#edf1f7}
.donut-app-controls .donut-weight-vertical .donut-weight-value{display:flex;flex-direction:column;gap:9px;margin:0;width:34px}
.donut-app-controls .donut-weight-vertical .donut-weight-value input[type=range]{writing-mode:vertical-lr;direction:rtl;appearance:auto;width:24px;min-width:0;height:180px;flex:none;display:block;margin:0}
.donut-app-controls .donut-weight-vertical .donut-weight-value input[type=number]{width:34px;min-width:0;display:block;padding:5px 2px;font-size:12px;text-align:center;appearance:textfield}
.donut-app-controls .donut-weight-vertical input::-webkit-inner-spin-button{appearance:none;margin:0}
.donut-app-controls textarea.donut-long-prompt{font-size:18px;line-height:1.65;padding:14px;resize:vertical}
:is(.donut-app-controls,.donut-edit-studio) .donut-prompt-tools{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 20px}
:is(.donut-app-controls,.donut-edit-studio) .donut-prompt-tools select{width:auto;flex:1;min-width:160px}
:is(.donut-app-controls,.donut-edit-studio) .donut-prompt-tools button{margin:0;font-size:13px}
:is(.donut-app-controls,.donut-edit-studio) .donut-prompt-tools p{flex-basis:100%;font-size:13px;margin:0}
:is(.donut-app-controls,.donut-edit-studio) textarea.donut-prompt-preview{flex-basis:100%;min-height:180px;background:#172b24;color:#d4f5de;font-size:15px}
.donut-app-controls .donut-wildcard-library>select{width:65%}
.donut-app-controls .donut-wildcard-library textarea{min-height:320px;font-size:16px;line-height:1.6}
.donut-app-controls .donut-wildcard-library>p:last-child{font-size:12px;overflow-wrap:anywhere}
.lg-node:has(.donut-section-controls) .lg-node-header,.lg-node:has(.donut-edit-studio) .lg-node-header,.lg-node:has(.donut-reference-studio) .lg-node-header{color:#f5f7fb!important;font-weight:750}
.lg-node:has(.donut-section-controls) [data-testid=node-title],.lg-node:has(.donut-edit-studio) [data-testid=node-title],.lg-node:has(.donut-reference-studio) [data-testid=node-title]{color:#f5f7fb!important}
.donut-edit-studio textarea{min-height:240px;font-size:16px;line-height:1.6}
.donut-edit-studio,.donut-reference-studio{flex:none!important;min-height:0}
`;
        style.textContent += `
.donut-section-columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(260px,100%),1fr));column-gap:20px;align-items:start}
.donut-section-columns>h2,.donut-section-columns>details{grid-column:1/-1}
.donut-section-columns>section{min-width:0}
@container(min-width:560px){
.donut-field-columns{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);column-gap:20px;align-items:start}
.donut-field-columns>h3,.donut-field-columns>summary,.donut-field-columns>.donut-vector-control,.donut-field-columns>.donut-weight-bank{grid-column:1/-1}
.donut-field-columns>label{display:flex;flex-direction:column;align-items:stretch;gap:6px}
.donut-field-columns>label input:not([type=checkbox]),.donut-field-columns>label select{width:100%}
.donut-app-controls fieldset{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);column-gap:20px}
.donut-app-controls fieldset>.donut-vector-control,.donut-app-controls fieldset>label:has(select[aria-label^="Installed LoRA"]){grid-column:1/-1}
}
`;
        style.textContent += `.donut-app-controls .donut-lora-toolbar,.donut-app-controls .donut-lora-information{grid-column:1/-1;min-width:0}.donut-app-controls .donut-lora-toolbar button{width:auto;min-width:32px}.donut-app-controls .donut-lora-information{font-size:13px}.donut-app-controls .donut-lora-information input[type=number]{min-height:0}`;
        document.head.append(style);
    },
    registerCustomNodes() {
        class DonutWorkflowPanel extends LGraphNode {
            constructor() {
                super("Donut · Section controls");
                this.isVirtualNode = true;
                this.serialize_widgets = false;
                this.properties = {donut_app_controls:{groups:[]}, panel_height:460};
                install(this);
                this.size = [460, 510];
            }
        }
        DonutWorkflowPanel.title = "Donut Section Controls";
        DonutWorkflowPanel.category = "donut/interface";
        LiteGraph.registerNodeType("DonutWorkflowPanel", DonutWorkflowPanel);
    },
    beforeRegisterNodeDef(nodeType, definition) {
        if (definition.name !== "DonutEditStudio") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() { const result = created?.apply(this, arguments); install(this, true); return result; };
    },
    afterConfigureGraph() {
        for (const node of app.rootGraph.nodes) node._donutAppControls?.render();
        scheduleLayout();
    },
    async refreshComboInNodes() {
        await service.catalog(true);
        for (const node of app.rootGraph.nodes) node._donutAppControls?.render();
    },
});

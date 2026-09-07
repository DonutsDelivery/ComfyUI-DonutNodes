import { createLoraToolbar, renderLoraInformation } from "./donut_lora_ui.js";

// Native ComfyUI controls; slots_json remains the ONLY serialized row state.
// No datalist, HTML select or hand-built dropdown menu.
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
    result.splice(to, 0, result.splice(from, 1)[0]);
    return result;
}
const unique = values => [...new Set(values.filter(v => typeof v === "string" && v.length))];
const titleOf = value => String(value || "None").split(":")[0];
const hasFile = row => !!row.lora_name && row.lora_name !== "None";
const enabled = row => row.enabled !== false;
const hexHash = value => typeof value === "string" && /^[a-f\d]{10,128}$/i.test(value);

// Shared per-extension service: no requests per row just to list the same files.
// Two in-flight metadata requests avoid flooding a local server on workflow load.
export function createLoraService(api) {
    let catalog, catalogJob, active = 0;
    const waiters = [], jobs = new Map();
    async function json(path) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 120_000);
        try {
            const response = await api.fetchApi(path, { cache: "no-store", signal: controller.signal });
            if (!response.ok) {
                let reason = "Request failed";
                try { const data = await response.json(); if (typeof data.error === "string") reason = data.error; } catch { /* Non-JSON error response. */ }
                throw new Error(`${reason} (${response.status})`);
            }
            return await response.json();
        } finally { clearTimeout(timer); }
    }
    async function limited(path) {
        if (active >= 2) await new Promise(resolve => waiters.push(resolve));
        else active++;
        try { return await json(path); }
        finally {
            const next = waiters.shift();
            if (next) next(); else active--;
        }
    }
    return {
        async catalog(force = false) {
            if (catalogJob) return catalogJob;
            if (catalog && !force) return catalog;
            catalogJob = (async () => {
                const [files, schema] = await Promise.allSettled([
                    json("/models/loras"), json("/object_info/DonutLoRAStack")
                ]);
                const required = schema.status === "fulfilled"
                    ? schema.value?.DonutLoRAStack?.input?.required : undefined;
                const names = files.status === "fulfilled" && Array.isArray(files.value)
                    ? files.value : required?.lora_name_1?.[0];
                if (!Array.isArray(names)) throw new Error("Cannot list installed LoRAs. Use Refresh installed LoRAs to retry.");
                catalog = { loras: unique(["None", ...names]),
                    presets: unique(["None", ...(required?.block_preset_1?.[0] || catalog?.presets || [])]) };
                return catalog;
            })();
            try { return await catalogJob; } finally { catalogJob = undefined; }
        },
        details(kind, name, force = false) {
            const key = `${kind}:${name}`;
            if (force) jobs.delete(key);
            if (!jobs.has(key)) {
                const job = limited(`/donut/loras/${kind}?name=${encodeURIComponent(name)}`);
                jobs.set(key, job);
                job.catch(() => { if (jobs.get(key) === job) jobs.delete(key); });
            }
            return jobs.get(key);
        }
    };
}

export function setHidden(widget, hide) {
    if (!widget._donutVisible) widget._donutVisible = {
        type: widget.type, computeSize: widget.computeSize, computeLayoutSize: widget.computeLayoutSize
    };
    widget.hidden = !!hide;
    if (hide) {
        // Keep the native widget class/type intact, including getter-only
        // promoted views. Hidden controls are still valid serialized controls.
        widget.computeSize = () => [0, -4]; // Legacy canvas accounts for a 4px gap.
        widget.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    } else {
        widget.computeSize = widget._donutVisible.computeSize;
        widget.computeLayoutSize = widget._donutVisible.computeLayoutSize;
    }
}

export function installNativeLoras(node, definition, { app, api, service }) {
    if (node._donutNativeLoras) return;
    const state = node.widgets?.find(w => w.name === "slots_json");
    if (!state) return;
    // Capture backend widgets BEFORE adding UI-only controls. Never address the
    // JSON by its index in the changing live widget array.
    const backend = node.widgets.slice();
    const stateIndex = backend.indexOf(state);
    const options = definition.input?.required?.slots_json?.[1] || {};
    let catalog = { loras: unique(["None", ...(options.donut_loras || [])]),
        presets: unique(["None", ...(options.donut_presets || [])]) };
    let rows = [], ui = [], disposed = false, invalid = false, listError = "";
    let generation = 0, catalogLoading = true;
    const panels = new Map(), details = new Map(), expanded = new Set(), panelViews = new Map();
    const get = name => backend.find(w => w.name === name)?.value;
    const lookupOn = () => get("civitai_lookup") === "On";
    const dirty = () => { node.setDirtyCanvas?.(true, true); app.graph?.change?.(); };
    const fit = () => {
        const size = node.computeSize?.();
        if (size && Number.isFinite(size[1])) node.setSize?.([Math.max(320, node.size?.[0] || 320), size[1]]);
        node.setDirtyCanvas?.(true, true);
    };
    const commit = () => { state.value = JSON.stringify(rows); dirty(); };
    setHidden(state, true);
    state.serializeValue = () => state.value;
    const make = (type, name, value, callback, options = {}) => {
        const widget = node.addWidget(type, name, value, callback, { ...options, serialize: false });
        widget.serialize = false; // Exclude from workflow arrays as well as API inputs.
        widget.options.serialize = false;
        ui.push(widget);
        return widget;
    };
    const el = (tag, text) => {
        const element = document.createElement(tag);
        if (text !== undefined) element.textContent = String(text);
        return element;
    };
    function clearUI() {
        for (const panel of panels.values()) panel.dispose?.();
        for (const widget of ui) {
            if (node.removeWidget) node.removeWidget(widget);
            else { widget.onRemove?.(); const i = node.widgets.indexOf(widget); if (i >= 0) node.widgets.splice(i, 1); }
        }
        ui = []; panels.clear();
    }
    function current(row, epoch) {
        return !disposed && epoch === generation && rows.includes(row) && enabled(row) && hasFile(row);
    }
    function validInfo(row) {
        const info = details.get(String(row.id));
        return info?.name === row.lora_name ? info : undefined;
    }
    function changeRows(change) {
        const graph = node.graph || app.graph;
        graph?.beforeChange?.();
        try { change(); commit(); render(); }
        finally { graph?.afterChange?.(); }
    }
    function rowAction(row, action) {
        // Resolve identity at click time. Reorder/refresh must never make a stale
        // index remove the neighbour, including two slots with the same filename.
        const index = rows.indexOf(row), key = String(row.id);
        if (index < 0 || disposed || invalid) return;
        if (action === "Retry metadata") { requestDetails(row, true); return; }
        if (action === "Move up" || action === "Move down") {
            const to = index + (action === "Move up" ? -1 : 1);
            if (to < 0 || to >= rows.length) return;
            changeRows(() => { rows = moveRow(rows, index, to); });
        } else if (action === "Remove") {
            changeRows(() => { rows.splice(index, 1); details.delete(key); expanded.delete(key); panelViews.delete(key); });
        } else if (action === "Use suggested model weight") {
            const weight = validInfo(row)?.info?.civitai?.recommended_weight;
            if (typeof weight !== "number" || !Number.isFinite(weight) || Math.abs(weight) > 1000) return;
            changeRows(() => { row.model_weight = weight; }); // CLIP/text fusion stays unchanged.
        }
    }
    function updatePanel(row) {
        const panel = panels.get(String(row.id));
        if (!panel) return;
        panel.observer?.disconnect();
        const result = renderLoraInformation(panel.root, row, validInfo(row) || {}, {
            api, lookupOn: lookupOn(), view: panel.view,
            onAction: action => rowAction(row, action), onResize: () => panel.measure()
        });
        panel.content = result.content;
        panel.height = result.preferredHeight;
        panel.observer?.observe(panel.content);
        setHidden(panel.widget, !enabled(row) || !hasFile(row));
        panel.measure(); fit();
    }
    function requestDetails(row, force = false) {
        if (!enabled(row) || !hasFile(row) || disposed) return;
        const epoch = generation, name = row.lora_name;
        const entry = validInfo(row) || { name };
        details.set(String(row.id), entry);
        const accept = () => current(row, epoch) && row.lora_name === name && validInfo(row) === entry;
        const load = (kind, target) => {
            if (!force && (entry[target] || entry[`${target}Loading`] || entry[`${target}Error`])) return;
            entry[`${target}Loading`] = true; delete entry[`${target}Error`];
            service.details(kind, name, force).then(result => {
                if (!accept()) return;
                entry[target] = result;
                // Never replace a full hash by the shorter prefix returned by UI lookup.
                if (target === "info" && hexHash(result.hash) && !row.lora_hash) { row.lora_hash = result.hash; commit(); }
            }, error => { if (accept()) entry[`${target}Error`] = error.message; })
                .finally(() => { entry[`${target}Loading`] = false; if (accept()) updatePanel(row); });
        };
        load("analyze", "analysis");
        if (lookupOn()) load("info", "info");
        updatePanel(row);
    }
    function render() {
        clearUI();
        const add = make("button", "+ Add LoRA", null, () => {
            if (invalid || disposed) return;
            changeRows(() => rows.push({ id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`, enabled: true,
                lora_name: "None", model_weight: 1, clip_weight: 1, block_preset: "None", block_vector: "", inherit_block_vector: false, lora_hash: "" }));
        });
        add.disabled = invalid;
        make("button", listError || (catalogLoading ? "Loading installed LoRAs…" : `Refresh installed LoRAs (${catalog.loras.filter(n => n !== "None").length})`), null, () => { void refresh(true); });
        if (invalid) {
            const editor = make("text", "Repair slots_json", state.value, value => { state.value = value; restore(); dirty(); });
            editor.tooltip = "Malformed saved JSON is retained. Correct it here; no rows have been discarded.";
            fit(); return;
        }
        rows.forEach((row, index) => {
            const key = String(row.id);
            const prefix = `donut_row:${key}:`;
            const toolbar = createLoraToolbar(index, rows.length, action => rowAction(row, action));
            const actions = node.addDOMWidget(prefix + "actions", "div", toolbar, {
                serialize: false, margin: 0, getMinHeight: () => 43, getMaxHeight: () => 43, getHeight: () => 43
            });
            actions.serialize = false; ui.push(actions);
            actions.computeSize = () => [280, 43];
            const name = make("combo", prefix + "lora_name", row.lora_name || "None", value => {
                if (value === row.lora_name) return;
                row.lora_name = value; row.lora_hash = ""; details.delete(key);
                commit(); render();
            }, { values: unique([...catalog.loras, row.lora_name]) });
            // addWidget validates combos synchronously. The initial array is
            // required even though the live getter below supplies later updates.
            name.label = `LoRA ${index + 1}`;
            // Getter uses fresh catalog without replacing the selected/saved name.
            Object.defineProperty(name.options, "values", { configurable: true,
                get: () => unique([...catalog.loras, row.lora_name]), set: () => {} });
            const dependent = [];
            const on = make("toggle", prefix + "enabled", enabled(row), value => {
                row.enabled = !!value; commit(); visibility();
                if (enabled(row)) requestDetails(row);
            }, { on: "Enabled", off: "Disabled" });
            on.label = "Enabled";
            for (const [field, label] of [["model_weight", "Model strength"], ["clip_weight", "CLIP / text-fusion strength"]]) {
                const weight = make("number", prefix + field, row[field] ?? 1, value => {
                    if (!Number.isFinite(value) || Math.abs(value) > 1000) { weight.value = row[field] ?? 1; return; }
                    row[field] = value; commit(); updatePanel(row);
                }, { min: -1000, max: 1000, step: 0.1, precision: 2 });
                weight.label = label; dependent.push(weight);
            }
            const advanced = make("toggle", prefix + "advanced", expanded.has(key), value => {
                if (value) expanded.add(key); else expanded.delete(key);
                visibility();
            }, { on: "Shown", off: "Hidden" });
            advanced.label = "Block weights"; dependent.push(advanced);
            const preset = make("combo", prefix + "block_preset", titleOf(row.block_preset), value => {
                const raw = catalog.presets.find(p => titleOf(p) === value);
                if (!raw) return;
                row.block_preset = raw;
                row.block_vector = raw.includes(":") ? raw.slice(raw.indexOf(":") + 1) : "";
                row.inherit_block_vector = false;
                vector.value = row.block_vector; inherit.value = false; vector.disabled = false; commit();
            }, { values: unique([
                ...catalog.presets.filter(p => p === "None" || get("model_type") === "Auto" || p.startsWith(`${get("model_type")}-`)).map(titleOf),
                titleOf(row.block_preset)
            ]) });
            preset.label = "Block preset";
            Object.defineProperty(preset.options, "values", { configurable: true, get: () => unique([
                ...catalog.presets.filter(p => p === "None" || get("model_type") === "Auto" || p.startsWith(`${get("model_type")}-`)).map(titleOf),
                titleOf(row.block_preset)
            ]), set: () => {} });
            const inherit = make("toggle", prefix + "inherit", !!row.inherit_block_vector, value => {
                row.inherit_block_vector = !!value; vector.disabled = !!value; commit();
            });
            inherit.label = "Inherit global vector";
            const vector = make("text", prefix + "block_vector", row.block_vector || "", value => { row.block_vector = value; commit(); });
            vector.label = "Block vector"; vector.disabled = !!row.inherit_block_vector;
            const root = el("div");
            root.setAttribute("aria-label", `LoRA ${index + 1} information`);
            if (!panelViews.has(key)) panelViews.set(key, {});
            const panel = { root, height: 125, view: panelViews.get(key), content: null, frame: null };
            const dom = node.addDOMWidget(prefix + "information", "div", root, {
                serialize: false, margin: 0,
                getMinHeight: () => panel.height, getMaxHeight: () => panel.height, getHeight: () => panel.height
            });
            dom.serialize = false; ui.push(dom); panel.widget = dom; panels.set(key, panel);
            dom.computeSize = () => [280, panel.height];
            const measure = () => {
                panel.frame = null;
                if (disposed || panels.get(key) !== panel || !panel.content?.isConnected) return;
                const natural = panel.content.scrollHeight;
                if (!natural) return; // Hidden/collapsed nodes cannot be measured yet.
                const height = Math.max(56, Math.min(360, Math.ceil(natural) + 10));
                if (height !== panel.height) { panel.height = height; fit(); }
            };
            panel.measure = () => {
                if (panel.frame != null) return;
                if (typeof requestAnimationFrame === "function") panel.frame = requestAnimationFrame(measure);
                else measure();
            };
            if (typeof ResizeObserver === "function") panel.observer = new ResizeObserver(panel.measure);
            panel.dispose = () => {
                panel.observer?.disconnect();
                if (panel.frame != null && typeof cancelAnimationFrame === "function") cancelAnimationFrame(panel.frame);
                panel.frame = null;
            };
            const removedPanel = dom.onRemove;
            dom.onRemove = function() { panel.dispose(); removedPanel?.apply(this, arguments); };
            function visibility() {
                dependent.forEach(widget => setHidden(widget, !enabled(row)));
                [preset, inherit, vector].forEach(widget => setHidden(widget, !enabled(row) || !expanded.has(key)));
                setHidden(dom, !enabled(row) || !hasFile(row));
                fit();
            }
            updatePanel(row); visibility(); requestDetails(row);
        });
        fit();
    }
    async function refresh(force = false) {
        catalogLoading = true;
        try { catalog = await service.catalog(force); listError = ""; }
        catch (error) { listError = "LoRA list unavailable · click to retry"; console.warn("[Donut LoRA]", error); }
        catalogLoading = false;
        if (!disposed) render();
    }
    function restore(data) {
        generation++;
        const value = data?.widgets_values_named?.slots_json ?? data?.widgets_values?.[stateIndex] ?? state.value;
        state.value = value;
        try {
            rows = decodeRows(value);
            const ids = new Set();
            rows.forEach((row, index) => {
                row.id = String(row.id ?? index + 1);
                if (!row.id || ids.has(row.id)) throw new Error("Duplicate/empty row ID");
                ids.add(row.id);
                if (row.enabled !== undefined && typeof row.enabled !== "boolean") throw new Error("Enabled must be boolean");
            });
            invalid = false;
        } catch (error) { invalid = true; rows = []; console.warn("[Donut LoRA] Saved row JSON needs repair:", error.message); }
        // Pending results from a previous configure must not attach to new rows.
        for (const value of details.values()) { value.analysisLoading = false; value.infoLoading = false; }
        render();
    }
    const configured = node.onConfigure;
    node.onConfigure = function(data) { const result = configured?.apply(this, arguments); restore(data); return result; };
    const serialized = node.onSerialize;
    node.onSerialize = function(data) {
        serialized?.apply(this, arguments);
        data.widgets_values = backend.map(widget => widget.value);
        data.widgets_values_named = Object.fromEntries(backend.map(widget => [widget.name, widget.value]));
    };
    const executed = node.onExecuted;
    node.onExecuted = function(message) {
        executed?.apply(this, arguments);
        for (const result of message.donut_loras || []) {
            const row = rows.find(row => String(row.id) === String(result.id) && row.lora_name === result.lora_name);
            if (!row) continue;
            const entry = validInfo(row) || { name: row.lora_name };
            entry.execution = result; details.set(String(row.id), entry);
            if (hexHash(result.lora_hash) && result.lora_hash !== row.lora_hash) { row.lora_hash = result.lora_hash; commit(); }
            updatePanel(row);
        }
    };
    for (const name of ["civitai_lookup", "model_type"]) {
        const widget = backend.find(w => w.name === name);
        if (widget) {
            const callback = widget.callback;
            widget.callback = function(value) { callback?.apply(this, arguments); if (value !== undefined) widget.value = value; render(); };
        }
    }
    const removed = node.onRemoved, added = node.onAdded;
    node.onRemoved = function() { disposed = true; generation++; return removed?.apply(this, arguments); };
    node.onAdded = function() {
        const result = added?.apply(this, arguments);
        // Moving a node into/out of a subgraph can detach and re-add the same
        // instance. It must not remain permanently disposed after that move.
        if (disposed) {
            disposed = false; generation++;
            for (const entry of details.values()) { entry.analysisLoading = false; entry.infoLoading = false; }
            render();
        }
        return result;
    };
    node._donutNativeLoras = { refresh, get rows() { return rows; } };
    restore(); void refresh();
}

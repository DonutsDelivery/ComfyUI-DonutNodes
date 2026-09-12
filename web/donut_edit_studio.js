import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ASPECT_RATIOS, targetDimensions, cropBox, imageLocation } from "./donut_edit_geometry.js";
import { promptTools } from "./donut_wildcards.js";
import { fitModule, fitTextarea } from "./donut_layout.js?v=15";

const studios = new Set();
let activeStudio = null;
const CSS = `
.donut-edit-studio { --de-bg:#14191f; --de-card:#1b222a; --de-line:#34404d; --de-text:#e8edf3; --de-dim:#9caebb; --de-accent:#92e4c7;
    box-sizing:border-box; width:100%; height:auto; overflow:visible; padding:16px; background:var(--de-bg); color:var(--de-text); border-radius:12px;
    font:13px/1.45 Inter,system-ui,sans-serif; scrollbar-width:thin; color-scheme:dark; container-type:inline-size; }
.donut-edit-studio * { box-sizing:border-box; }
.donut-edit-studio button,.donut-edit-studio input,.donut-edit-studio select,.donut-edit-studio textarea { font:inherit; color:inherit; }
.donut-edit-studio button { cursor:pointer; border:1px solid var(--de-line); border-radius:7px; background:#242e38; padding:6px 10px; }
.donut-edit-studio button:hover { background:#33424f; border-color:#79929f; }
.donut-edit-studio button:focus-visible,.donut-edit-studio input:focus-visible,.donut-edit-studio select:focus-visible,.donut-edit-studio textarea:focus-visible { outline:2px solid var(--de-accent); outline-offset:2px; }
.donut-edit-studio button:disabled { cursor:wait; opacity:.5; }
.donut-edit-studio .de-top { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px; }
.donut-edit-studio .de-eyebrow { font-size:11px; font-weight:700; letter-spacing:.14em; color:var(--de-accent); margin-bottom:3px; }
.donut-edit-studio .de-title { font-size:23px; font-weight:650; letter-spacing:-.5px; }
.donut-edit-studio .de-toggle { display:inline-flex; align-items:center; gap:7px; white-space:nowrap; cursor:pointer; }
.donut-edit-studio input[type=checkbox] { appearance:none; width:30px; height:17px; border:1px solid #586977; border-radius:10px; background:#26303a; margin:0; cursor:pointer; flex-shrink:0; }
.donut-edit-studio input[type=checkbox]::before { content:""; display:block; width:11px; height:11px; border-radius:50%; background:#abb9c2; margin:2px; transition:transform .12s; }
.donut-edit-studio input[type=checkbox]:checked { background:#306552; border-color:var(--de-accent); }
.donut-edit-studio input[type=checkbox]:checked::before { transform:translateX(12px); background:var(--de-accent); }
.donut-edit-studio .de-toggle-main { padding:9px 11px; border-radius:20px; border:1px solid var(--de-line); background:#1c2630; }
.donut-edit-studio .de-refs { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:12px; }
.donut-edit-studio .de-ref { min-width:0; border:1px solid var(--de-line); border-radius:10px; overflow:hidden; background:var(--de-card); }
.donut-edit-studio .de-ref.de-active { border-color:var(--de-accent); box-shadow:0 0 0 1px #92e4c722; }
.donut-edit-studio .de-ref-head { padding:10px; display:flex; align-items:center; gap:7px; }
.donut-edit-studio .de-letter { display:grid; place-items:center; width:24px; height:24px; border-radius:6px; background:#344553; font-weight:750; }
.donut-edit-studio .de-ref-name { font-weight:600; flex:1; }
.donut-edit-studio .de-ref-head .de-toggle { gap:4px; font-size:11px; color:var(--de-dim); }
.donut-edit-studio .de-stage { position:relative; height:207px; margin:0 9px; border-radius:6px; background:#11161b; overflow:hidden; cursor:grab; outline:none; }
.donut-edit-studio .de-stage:focus-visible { box-shadow:inset 0 0 0 2px var(--de-accent); }
.donut-edit-studio .de-stage.de-dragover { box-shadow:inset 0 0 0 2px var(--de-accent); background:#203e35; }
.donut-edit-studio .de-stage canvas { display:block; width:100%; height:100%; touch-action:none; }
.donut-edit-studio .de-empty { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; flex-direction:column; padding:20px; text-align:center; gap:8px; color:var(--de-dim); pointer-events:none; }
.donut-edit-studio .de-empty-mark { font-size:30px; line-height:1; color:var(--de-accent); }
.donut-edit-studio .de-ref-meta { color:var(--de-dim); min-height:30px; padding:6px 10px 3px; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.donut-edit-studio .de-ref-actions { display:flex; gap:5px; padding:3px 9px 10px; }
.donut-edit-studio .de-ref-actions button { font-size:12px; padding:5px 8px; }
.donut-edit-studio .de-ref-actions button.de-awaiting-paste { border-color:var(--de-accent); background:#285044; }
.donut-edit-studio .de-ref-actions .de-clear { margin-left:auto; color:var(--de-dim); }
.donut-edit-studio .de-caption { color:var(--de-dim); display:flex; align-items:center; justify-content:space-between; gap:8px; font-size:12px; margin:9px 1px 15px; }
.donut-edit-studio .de-crop-key { display:inline-block; width:10px; height:10px; border:2px solid #ff6c75; margin-right:5px; vertical-align:-1px; }
.donut-edit-studio .de-subhead { font-size:13px; font-weight:650; margin-bottom:8px; }
.donut-edit-studio textarea { overflow:hidden; display:block; width:100%; min-height:68px; resize:vertical; background:#10171d; border:1px solid var(--de-line); border-radius:7px; padding:9px; line-height:1.5; }
.donut-edit-studio textarea::placeholder { color:#778995; }
.donut-edit-studio .de-section { margin-top:14px; border-top:1px solid #29343f; padding-top:13px; }
.donut-edit-studio .de-row { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:10px; }
.donut-edit-studio .de-field { display:flex; flex-direction:column; gap:5px; min-width:0; color:var(--de-dim); font-size:12px; }
.donut-edit-studio .de-field select,.donut-edit-studio .de-field input[type=number] { width:100%; min-width:0; background:#212b35; border:1px solid var(--de-line); border-radius:6px; padding:7px; color:var(--de-text); }
.donut-edit-studio .de-output { display:flex; align-items:baseline; justify-content:space-between; gap:9px; padding:9px 11px; border-radius:7px; background:#1d2e2c; border:1px solid #35594e; }
.donut-edit-studio .de-output strong { font-size:19px; letter-spacing:.2px; }
.donut-edit-studio .de-output span { font-size:12px; color:#aed1c5; }
.donut-edit-studio .de-help { color:var(--de-dim); font-size:12px; margin-top:7px; }
.donut-edit-studio .de-ground { display:flex; align-items:center; gap:12px; }
.donut-edit-studio .de-ground input[type=range] { flex:1; width:100%; accent-color:var(--de-accent); }
.donut-edit-studio .de-ground input[type=number] { width:76px; background:#212b35; border:1px solid var(--de-line); border-radius:6px; padding:6px; }
.donut-edit-studio .de-scale-labels { display:flex; justify-content:space-between; color:var(--de-dim); font-size:11px; margin-top:2px; padding-right:89px; }
.donut-edit-studio details { margin-top:12px; }
.donut-edit-studio summary { color:var(--de-dim); cursor:pointer; padding:3px 0; }
.donut-edit-studio details .de-row { margin:10px 0 0; grid-template-columns:minmax(0,3fr) minmax(70px,1fr); }
.donut-edit-studio .de-status { margin-top:11px; color:var(--de-accent); font-size:12px; min-height:17px; }
.donut-edit-studio .de-status.de-error { color:#ffa4a9; }
@container (max-width:480px) {
    .donut-edit-studio .de-refs,.donut-edit-studio .de-row { grid-template-columns:1fr; }
    .donut-edit-studio .de-top,.donut-edit-studio .de-output { flex-wrap:wrap; }
}
.donut-edit-studio [hidden] { display:none !important; }
`;

function element(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined) value.textContent = text;
    return value;
}
function button(text, label, action) {
    const value = element("button", "", text);
    value.type = "button"; value.title = label; value.setAttribute("aria-label", label);
    value.addEventListener("click", action);
    return value;
}
function textTarget(target) { return target?.closest?.("input,textarea,select,[contenteditable=true]"); }
function clipboardImage(data) {
    const files = [...(data?.files || [])];
    const direct = files.find(file => file?.type?.startsWith("image/"));
    if (direct) return direct;
    for (const item of [...(data?.items || [])]) {
        if (!item?.type?.startsWith("image/")) continue;
        const file = item.getAsFile?.();
        if (file) return file;
    }
    return null;
}

export function installEditStudio(node, definition) {
    if (node._donutEditStudio) return;
    const inputNames = new Set(Object.keys({...definition.input.required, ...definition.input.optional}));
    const backend = new Map(node.widgets.filter(widget => inputNames.has(widget.name)).map(widget => [widget.name, widget]));
    const root = element("div", "donut-edit-studio");
    root.setAttribute("aria-label", "Donut Edit Studio");
    const get = name => backend.get(name)?.value;
    const values = () => Object.fromEntries([...backend].map(([name, widget]) => [name, widget.value]));
    let disposed = false, activeSlot = "a", dragging = false;
    const slots = {}, controls = new Map();
    const commitValues = changes => {
        const changed = Object.entries(changes).filter(([name, value]) => backend.has(name) && backend.get(name).value !== value);
        if (!changed.length) return;
        if (!dragging) node.graph?.beforeChange?.();
        for (const [name, value] of changed) backend.get(name).value = value;
        if (!dragging) node.graph?.afterChange?.();
        node.setDirtyCanvas?.(true, true);
    };
    const commit = (name, value) => commitValues({[name]:value});
    function status(message, error = false) {
        statusLine.textContent = message; statusLine.classList.toggle("de-error", error);
    }
    function control(name, tag, options = {}) {
        const input = element(tag);
        input.setAttribute("aria-label", options.label || name);
        if (tag === "select") for (const option of options.choices || []) {
            const item = element("option", "", option); item.value = option; input.append(item);
        }
        else Object.assign(input, options);
        const read = () => input.type === "checkbox" ? input.checked : input.type === "number" || input.type === "range" ? Number(input.value) : input.value;
        const update = event => {
            if ((input.type === "number" || input.type === "range") && (!input.value || !Number.isFinite(Number(input.value)))) return;
            let value = read();
            if (typeof value === "number") {
                const clamped = Math.min(Number(input.max || Infinity), Math.max(Number(input.min || -Infinity), value));
                if (event.type === "input" && value !== clamped) return;
                value = clamped;
            }
            commit(name, value); render();
        };
        input.addEventListener(tag === "textarea" || ["number", "range"].includes(input.type) ? "input" : "change", update);
        if (input.type === "number") input.addEventListener("change", update);
        const entries = controls.get(name) || []; entries.push(input); controls.set(name, entries);
        return input;
    }
    function toggle(name, title, className = "") {
        const label = element("label", `de-toggle ${className}`);
        label.append(control(name, "input", {type:"checkbox", label:title}), element("span", "", title));
        return label;
    }
    function field(label, input) {
        const wrap = element("label", "de-field"); wrap.append(element("span", "", label), input); return wrap;
    }
    function activate(key, claim = true) {
        activeSlot = key;
        for (const [name, slot] of Object.entries(slots)) slot.card.classList.toggle("de-active", name === key);
        if (claim) activeStudio = studio;
    }
    function outputSize() {
        const image = get("enabled") && slots.a?.image;
        const b = get("enabled") && slots.b?.image;
        return targetDimensions(values(), image ? [image.naturalWidth, image.naturalHeight] : undefined, b ? [b.naturalWidth, b.naturalHeight] : undefined);
    }
    function boxFor(key) {
        const image = slots[key].image;
        return cropBox(image.naturalWidth, image.naturalHeight, ...outputSize(),
            Number(get(`crop_${key}_x`)), Number(get(`crop_${key}_y`)),
            key === "a" && get("enabled") && get("resolution_mode") === "Reference A · crop only");
    }
    function draw(key) {
        const slot = slots[key], canvas = slot.canvas, image = slot.image;
        // DOM widgets are scaled with the graph. clientWidth is the unscaled drawing size.
        const width = canvas.clientWidth || 260, height = canvas.clientHeight || 207, dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
        const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
        if (!image) return;
        const scale = Math.min((width - 16) / image.naturalWidth, (height - 16) / image.naturalHeight);
        const iw = image.naturalWidth * scale, ih = image.naturalHeight * scale;
        const ox = (width - iw) / 2, oy = (height - ih) / 2;
        slot.layout = {width, height, scale, ox, oy};
        ctx.drawImage(image, ox, oy, iw, ih);
        const [x1, y1, x2, y2] = boxFor(key), x = ox + x1 * scale, y = oy + y1 * scale, w = (x2 - x1) * scale, h = (y2 - y1) * scale;
        ctx.fillStyle = "rgba(6,10,14,.68)";
        ctx.fillRect(ox, oy, iw, y - oy); ctx.fillRect(ox, y + h, iw, oy + ih - y - h);
        ctx.fillRect(ox, y, x - ox, h); ctx.fillRect(x + w, y, ox + iw - x - w, h);
        ctx.strokeStyle = "#ff6c75"; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h);
        ctx.strokeStyle = "rgba(255,255,255,.25)"; ctx.lineWidth = .5;
        for (const third of [1 / 3, 2 / 3]) {
            ctx.beginPath(); ctx.moveTo(x + w * third, y); ctx.lineTo(x + w * third, y + h); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(x, y + h * third); ctx.lineTo(x + w, y + h * third); ctx.stroke();
        }
        slot.meta.textContent = `${image.naturalWidth} × ${image.naturalHeight}  →  crop ${x2 - x1} × ${y2 - y1}`;
        slot.meta.title = get(`image_${key}`);
    }
    function loadPreview(key) {
        const slot = slots[key], path = String(get(`image_${key}`) || "");
        if (slot.path === path) return;
        slot.path = path; slot.image = null; slot.layout = null;
        const epoch = ++slot.epoch;
        slot.empty.hidden = false; slot.meta.textContent = path ? "Loading saved reference…" : "No image selected";
        slot.emptyText.textContent = "Drop an image or click here, then Ctrl+V";
        if (!path) { draw(key); return; }
        const image = new Image();
        image.onload = () => {
            if (disposed || epoch !== slot.epoch) return;
            slot.image = image; slot.empty.hidden = true; render();
        };
        image.onerror = () => {
            if (disposed || epoch !== slot.epoch) return;
            slot.emptyText.textContent = "Saved image unavailable. Upload or paste a replacement.";
            slot.meta.textContent = "Missing image · safe to leave empty when editing is off"; draw(key);
        };
        image.src = api.apiURL(path.startsWith("donutref:")
            ? `/donut/edit-studio/reference/${encodeURIComponent(path.slice(9))}`
            : `/view?${new URLSearchParams(imageLocation(path))}`);
    }
    async function upload(key, file) {
        if (!file?.type?.startsWith("image/")) { status("Choose an image file.", true); return; }
        const slot = slots[key], epoch = ++slot.uploadEpoch;
        slot.pasteButton?.classList.remove("de-awaiting-paste");
        slot.uploading = true; slot.actions.forEach(action => { action.disabled = true; });
        status(`Saving reference ${key.toUpperCase()}…`);
        try {
            const data = new FormData();
            data.append("image", file, file.name || `reference-${key}.png`);
            const response = await api.fetchApi("/donut/edit-studio/reference", {method:"POST", body:data});
            if (!response.ok) throw new Error(`Upload failed (${response.status})`);
            const saved = await response.json();
            if (disposed || epoch !== slot.uploadEpoch) return;
            commitValues({[`image_${key}`]:saved.reference, [`crop_${key}_x`]:.5, [`crop_${key}_y`]:.5,
                enabled:true, ...(key === "b" ? {use_reference_b:true} : {})});
            slot.path = null; activate(key); render();
            status(`Reference ${key.toUpperCase()} saved. Save the workflow to keep this selection.`);
        } catch (error) { if (!disposed && epoch === slot.uploadEpoch) status(error.message, true); }
        finally { if (!disposed && epoch === slot.uploadEpoch) { slot.uploading = false; slot.actions.forEach(action => { action.disabled = false; }); } }
    }
    async function pasteFromButton(key) {
        activate(key);
        const pasteButton = slots[key].pasteButton;
        try {
            if (typeof navigator.clipboard?.read !== "function") {
                throw new DOMException("Clipboard image reads are unavailable", "NotSupportedError");
            }
            const entries = await navigator.clipboard.read();
            for (const item of entries) {
                const type = item.types.find(type => type.startsWith("image/"));
                if (type) { await upload(key, await item.getType(type)); return; }
            }
            status("The clipboard has no image. Copy an image, then paste again.", true);
        } catch (error) {
            slots[key].stage.focus();
            pasteButton.classList.add("de-awaiting-paste");
            status(`Reference ${key.toUpperCase()} selected · browser blocked clipboard access, press Ctrl+V now.`, true);
            console.debug("[Donut Edit Studio] Clipboard button fallback:", error);
        }
    }
    function paste(event) {
        const file = clipboardImage(event.clipboardData);
        if (!file) return false;
        event.preventDefault(); event.stopImmediatePropagation(); upload(activeSlot, file); return true;
    }

    const top = element("div", "de-top"), title = element("div");
    title.append(element("div", "de-eyebrow", "DONUT / KREA 2"), element("div", "de-title", "Edit Studio"));
    top.append(title, toggle("enabled", "Editing", "de-toggle-main")); root.append(top);
    const references = element("div", "de-refs"); root.append(references);
    for (const key of ["a", "b"]) {
        const card = element("div", "de-ref"), head = element("div", "de-ref-head"), stage = element("div", "de-stage");
        head.append(element("span", "de-letter", key.toUpperCase()), element("span", "de-ref-name", key === "a" ? "Base / scene" : "Subject / identity"));
        if (key === "b") head.append(toggle("use_reference_b", "Use B"));
        stage.tabIndex = 0; stage.setAttribute("role", "group"); stage.setAttribute("aria-label", `Reference ${key.toUpperCase()} crop preview. Paste an image with Ctrl+V.`);
        const canvas = element("canvas"), empty = element("div", "de-empty"), emptyText = element("div", "", "Drop an image or click here, then Ctrl+V");
        empty.append(element("span", "de-empty-mark", "+"), emptyText); stage.append(canvas, empty);
        const meta = element("div", "de-ref-meta"), actions = element("div", "de-ref-actions"), fileInput = element("input");
        fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.hidden = true;
        fileInput.addEventListener("change", () => { if (fileInput.files[0]) upload(key, fileInput.files[0]); fileInput.value = ""; });
        const uploadButton = button("Upload", `Upload reference ${key.toUpperCase()}`, () => { activate(key); fileInput.click(); });
        const pasteButton = button("Paste", `Paste reference ${key.toUpperCase()} from clipboard`, () => pasteFromButton(key));
        const center = button("Center", `Center the crop for reference ${key.toUpperCase()}`, () => { commitValues({[`crop_${key}_x`]:.5, [`crop_${key}_y`]:.5}); render(); });
        const clear = button("×", `Clear reference ${key.toUpperCase()}`, () => {
            ++slots[key].uploadEpoch; commitValues({[`image_${key}`]:"", ...(key === "b" ? {use_reference_b:false} : {})}); render();
        }); clear.className = "de-clear";
        actions.append(uploadButton, pasteButton, center, clear); card.append(head, stage, meta, actions, fileInput); references.append(card);
        slots[key] = {card, stage, canvas, empty, emptyText, meta, image:null, path:null, epoch:0, uploadEpoch:0,
            pasteButton, actions:[uploadButton, pasteButton, clear]};
        stage.addEventListener("focus", () => activate(key));
        stage.addEventListener("dragover", event => { event.preventDefault(); event.stopPropagation(); stage.classList.add("de-dragover"); });
        stage.addEventListener("dragleave", () => stage.classList.remove("de-dragover"));
        stage.addEventListener("drop", event => {
            event.preventDefault(); event.stopPropagation(); stage.classList.remove("de-dragover"); activate(key);
            const file = clipboardImage(event.dataTransfer); if (file) upload(key, file);
        });
        let drag;
        function moveCrop(event) {
            const slot = slots[key]; if (!slot.image || !slot.layout) return;
            const rect = canvas.getBoundingClientRect(), {width, height, scale, ox, oy} = slot.layout;
            const [x1, y1, x2, y2] = boxFor(key), cw = x2 - x1, ch = y2 - y1;
            const x = ((event.clientX - rect.left) * width / rect.width - ox) / scale;
            const y = ((event.clientY - rect.top) * height / rect.height - oy) / scale;
            const dx = slot.image.naturalWidth - cw, dy = slot.image.naturalHeight - ch;
            if (dx > 0) commit(`crop_${key}_x`, Math.max(0, Math.min(1, (x - drag.x) / dx)));
            if (dy > 0) commit(`crop_${key}_y`, Math.max(0, Math.min(1, (y - drag.y) / dy)));
            draw(key);
        }
        canvas.addEventListener("pointerdown", event => {
            if (event.button !== 0) return;
            event.stopPropagation(); activate(key); stage.focus();
            const slot = slots[key]; if (!slot.image || !slot.layout) return;
            const rect = canvas.getBoundingClientRect(), {width, height, scale, ox, oy} = slot.layout;
            const x = ((event.clientX - rect.left) * width / rect.width - ox) / scale, y = ((event.clientY - rect.top) * height / rect.height - oy) / scale;
            const [x1, y1, x2, y2] = boxFor(key), inside = x >= x1 && x <= x2 && y >= y1 && y <= y2;
            drag = {x:inside ? x - x1 : (x2 - x1) / 2, y:inside ? y - y1 : (y2 - y1) / 2};
            node.graph?.beforeChange?.(); dragging = true;
            canvas.setPointerCapture(event.pointerId); moveCrop(event);
        });
        canvas.addEventListener("pointermove", event => { if (drag) moveCrop(event); });
        const endDrag = () => { if (dragging) { dragging = false; node.graph?.afterChange?.(); } drag = null; };
        canvas.addEventListener("pointerup", endDrag); canvas.addEventListener("pointercancel", endDrag);
    }
    const caption = element("div", "de-caption"), cropCaption = element("span");
    cropCaption.append(element("span", "de-crop-key"), document.createTextNode("Red frame = kept area · drag to reposition"));
    caption.append(cropCaption); root.append(caption);
    const promptSection = element("div"), sharedPromptHelp = element("p", "de-help", "Editing uses the subject, scene and style from Prompts.");
    promptSection.append(element("div", "de-subhead", "Edit instruction · what should change?"));
    const instruction = control("prompt", "textarea", {label:"Edit instruction", placeholder:"e.g. Place the person from B into the scene in A. Preserve their face and clothing.", rows:3});
    const tools = promptTools(instruction, value => commit("prompt", value), () => {
        let graph = app.rootGraph, source;
        for (const id of node.properties?.donut_seed_path || []) { source = graph?.getNodeById(id); graph = source?.subgraph; }
        return source?.widgets?.find(w => w.name === "seed")?.value ?? 0;
    });
    promptSection.append(instruction, tools.element);
    root.append(promptSection, sharedPromptHelp);
    const sizeSection = element("div", "de-section");
    sizeSection.append(element("div", "de-subhead", "Output size"));
    const sizingRow = element("div", "de-row");
    sizingRow.append(field("Size from", control("resolution_mode", "select", {label:"Output sizing mode", choices:definition.input.required.resolution_mode[0]})),
        field("Pixel grid", control("multiple", "select", {label:"Pixel grid", choices:["16", "32", "64"]})));
    const presetRow = element("div", "de-row"), customRow = element("div", "de-row");
    const aspectField = field("Aspect ratio", control("aspect_ratio", "select", {label:"Output aspect ratio", choices:["Auto · Reference A", "Auto · Reference B", ...Object.keys(ASPECT_RATIOS)]}));
    const megapixelField = field("Megapixels", control("megapixels", "input", {type:"number", min:"0.1", max:"16", step:"0.1", label:"Output megapixels"}));
    presetRow.append(aspectField, megapixelField);
    customRow.append(field("Width", control("width", "input", {type:"number", min:"32", max:"16384", step:"32", label:"Custom output width"})),
        field("Height", control("height", "input", {type:"number", min:"32", max:"16384", step:"32", label:"Custom output height"})));
    const output = element("div", "de-output"), dimensions = element("strong"), outputMeta = element("span"); output.append(dimensions, outputMeta);
    const sizeHelp = element("div", "de-help"); sizeSection.append(sizingRow, presetRow, customRow, output, sizeHelp); root.append(sizeSection);
    const grounding = element("div", "de-section"), groundingRow = element("div", "de-ground");
    grounding.append(element("div", "de-subhead", "Image grounding"));
    groundingRow.append(control("grounding_px", "input", {type:"range", min:"0", max:"4096", step:"64", label:"Image grounding resolution"}),
        control("grounding_px", "input", {type:"number", min:"0", max:"4096", step:"64", label:"Grounding pixels"}));
    const scaleLabels = element("div", "de-scale-labels"); scaleLabels.append(element("span", "", "More edit freedom"), element("span", "", "More likeness"));
    grounding.append(groundingRow, scaleLabels); root.append(grounding);
    const advanced = element("details"), advancedRow = element("div", "de-row"); advanced.append(element("summary", "", "Identity edit LoRA"));
    advancedRow.append(field("LoRA", control("lora_name", "select", {label:"Identity edit LoRA", choices:definition.input.required.lora_name[1].donut_loras})),
        field("Strength", control("lora_strength", "input", {type:"number", min:"-20", max:"20", step:"0.05", label:"Identity edit LoRA strength"})));
    advanced.append(advancedRow); root.append(advanced);
    const statusLine = element("div", "de-status"); statusLine.setAttribute("role", "status"); statusLine.setAttribute("aria-live", "polite"); root.append(statusLine);

    function render() {
        promptSection.hidden = !!node.properties?.donut_shared_prompt;
        sharedPromptHelp.hidden = !node.properties?.donut_shared_prompt;
        if (disposed) return;
        for (const [name, inputs] of controls) for (const input of inputs) {
            const value = get(name);
            if (input.type === "checkbox") input.checked = Boolean(value);
            else {
                if (input.tagName === "SELECT" && value && ![...input.options].some(option => option.value === String(value))) {
                    const option = element("option", "", String(value)); option.value = value; input.append(option);
                }
                if (input.value !== String(value ?? "")) input.value = value ?? "";
            }
        }
        const mode = get("resolution_mode"), editing = get("enabled"), refMode = mode.startsWith("Reference A");
        customRow.hidden = mode !== "Custom";
        presetRow.hidden = mode === "Custom" || (editing && mode === "Reference A · crop only");
        aspectField.hidden = editing && refMode;
        megapixelField.hidden = editing && mode === "Reference A · crop only";
        for (const key of ["a", "b"]) { loadPreview(key); draw(key); }
        const [width, height] = outputSize(); dimensions.textContent = `${width} × ${height}`;
        outputMeta.textContent = `${(width * height / (1024 * 1024)).toFixed(2)} MP · /${get("multiple")} grid`;
        sizeHelp.textContent = !editing && refMode ? "Editing is off. Generation uses the preset size; reference sizing resumes when enabled."
            : mode === "Reference A · crop only" ? "Trims A to the grid without resizing. Images smaller than the grid are enlarged."
            : mode === "Custom" ? "Dimensions snap to the selected grid. The previews show the resulting crop."
            : "Preserves image proportions, then crops to the selected output shape.";
        if (!statusLine.classList.contains("de-error") && !slots.a.uploading && !slots.b.uploading) {
            statusLine.textContent = !editing ? "Editing off · images are optional; generation controls remain active."
                : !get("image_a") ? "Add a base image to A to start editing."
                : get("use_reference_b") && !get("image_b") ? "Add the subject image to B, or turn Use B off."
                : get("use_reference_b") ? "Two references · A sets the scene; B supplies subject identity."
                : "One reference · enable B to combine two images.";
        }
    }

    for (const widget of backend.values()) {
        widget.hidden = true; widget.computeSize = () => [0, -4];
        widget.options ||= {}; widget.options.hidden = true;
        widget.computeLayoutSize = () => ({minHeight:0, maxHeight:0, minWidth:0});
    }
    const dom = node.addDOMWidget("edit_studio", "custom", root, {serialize:false, hideOnZoom:false, getValue:() => "", setValue:() => {}});
    dom.label = "References and output size";
    dom.options.serialize = false;
    node.properties.panel_min_width = Math.max(640, node.properties.panel_min_width || 0);
    fitModule(node, dom, root);
    node.setSize([Math.max(640, node.size[0]), Math.max(1070, node.size[1])]);
    const studio = {node, root, render, paste, upload, activate, get activeSlot() {return activeSlot;}, slots,
        refreshLoras(choices) {
            for (const input of controls.get("lora_name") || []) {
                input.replaceChildren(...choices.map(name => {
                    const option = element("option", "", name); option.value = name; return option;
                }));
            }
            render();
        },
    };
    node._donutEditStudio = studio; studios.add(studio);
    const configured = node.onConfigure, removed = node.onRemoved, added = node.onAdded;
    node.onConfigure = function() { const result = configured?.apply(this, arguments); render(); root.querySelectorAll('textarea').forEach(fitTextarea); return result; };
    node.onAdded = function() { const result = added?.apply(this, arguments); disposed = false; studios.add(studio); observer.observe(root); render(); return result; };
    const observer = new ResizeObserver(() => { if (!disposed) for (const key of ["a", "b"]) draw(key); }); observer.observe(root);
    node.onRemoved = function() { disposed = true; studios.delete(studio); observer.disconnect(); return removed?.apply(this, arguments); };
    root.addEventListener("paste", event => { if (!textTarget(event.target)) paste(event); }, true);
    activate("a", false); render();
}

app.registerExtension({
    name:"Donut.EditStudio",
    setup() {
        if (!document.getElementById("donut-edit-studio-style")) {
            const style = element("style"); style.id = "donut-edit-studio-style"; style.textContent = CSS; document.head.append(style);
        }
        window.addEventListener("paste", event => {
            if (textTarget(event.target)) return;
            const containing = [...studios].find(studio => studio.root.contains?.(event.target));
            if (containing) { containing.paste(event); return; }
            if (activeStudio && studios.has(activeStudio)
                && activeStudio.root.contains?.(document.activeElement)) {
                activeStudio.paste(event); return;
            }
            const selected = Object.values(app.canvas?.selected_nodes || {});
            if (selected.length !== 1) return;
            const studio = selected[0]._donutEditStudio;
            if (studio && studios.has(studio)) studio.paste(event);
        }, true);
    },
    async beforeRegisterNodeDef(nodeType, definition) {
        if (definition.name !== "DonutEditStudio") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments); installEditStudio(this, definition); return result;
        };
    },
    refreshComboInNodes(definitions) {
        const choices = definitions.DonutEditStudio?.input.required.lora_name[1].donut_loras;
        if (choices) for (const studio of studios) studio.refreshLoras(choices);
    },
});

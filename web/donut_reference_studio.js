import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fitModule } from "./donut_layout.js?v=15";
import { clipboardImage, readClipboardImage } from "./donut_clipboard.js?v=1";

const referenceStudios = new Set();
let activeReferenceStudio = null;

const el = (tag, text) => {
    const item = document.createElement(tag);
    if (text !== undefined) item.textContent = text;
    return item;
};
function install(node) {
    const root = el("div"), status = el("p"), previews = [], slots = {};
    root.className = "donut-edit-studio donut-reference-studio";
    root.tabIndex = 0;
    const top = el("div"), heading = el("div"), eyebrow = el("div", "DONUT / KREA 2"), title = el("div", "Reference guidance");
    top.className = "de-top"; eyebrow.className = "de-eyebrow"; title.className = "de-title";
    heading.append(eyebrow, title); top.append(heading); root.append(top);
    const widget = name => node.widgets.find(w => w.name === name);
    const get = name => widget(name)?.value;
    function set(name, value) {
        node.graph.beforeChange(); widget(name).value = value;
        if (name === "image_a" || name === "image_b") {
            const crop = widget(`crop_data_${name.slice(-1)}`);
            if (crop) crop.value = "";
        }
        node.graph.afterChange(); node.setDirtyCanvas(true, true); refresh();
    }
    function toggle(name, title) {
        const label = el("label", title), input = el("input");
        input.type = "checkbox"; input.setAttribute("aria-label", title);
        input.onchange = () => set(name, input.checked);
        label.className = "de-toggle"; label.prepend(input);
        return input;
    }
    const enabled = toggle("enabled", "Use reference guidance");
    enabled.parentElement.classList.add("de-toggle-main"); top.append(enabled.parentElement);
    const second = toggle("use_reference_b", "Use B");
    const grid = el("div"); grid.className = "de-refs"; status.className = "de-status";
    const help = el("p", "Uses each full image. Describe what to borrow in the main prompt."); help.className = "de-help";
    root.append(grid, help, status);
    let disposed = false, activeSlot = "a", studio;
    function activate(key) {
        activeSlot = key; activeReferenceStudio = studio;
        for (const [slotKey, slot] of Object.entries(slots)) slot.card.classList.toggle("de-selected", slotKey === key);
    }
    for (const [name, title] of [["image_a", "Reference A"], ["image_b", "Reference B"]]) {
        const box = el("section"), img = el("img"), empty = el("p", "Drop or paste an image here"), file = el("input"), clear = el("button", "Clear");
        box.tabIndex = 0; box.setAttribute("aria-label", `${title} image slot`);
        box.className = "de-ref";
        const head = el("div"), letter = el("span", name === "image_a" ? "A" : "B"), label = el("span", name === "image_a" ? "Primary reference" : "Second reference");
        head.className = "de-ref-head"; letter.className = "de-letter"; label.className = "de-ref-name";
        head.append(letter, label); if (name === "image_b") head.append(second.parentElement);
        const stage = el("div"), actions = el("div"), uploadButton = el("button", "Upload"), pasteButton = el("button", "Paste"), meta = el("div", "No image selected");
        stage.className = "de-stage"; stage.tabIndex = 0; empty.className = "de-empty";
        empty.textContent = "Drop an image or click here, then Ctrl+V";
        actions.className = "de-ref-actions"; meta.className = "de-ref-meta"; clear.className = "de-clear";
        file.hidden = true; uploadButton.type = pasteButton.type = "button";
        uploadButton.setAttribute("aria-label", `Upload ${title}`); pasteButton.setAttribute("aria-label", `Paste ${title} from clipboard`);
        uploadButton.onclick = () => file.click();
        pasteButton.onclick = async () => {
            activate(name.slice(-1));
            try {
                const image = await readClipboardImage();
                if (image) { await upload(image); return; }
                status.textContent = "Copy an image, then paste it here.";
            } catch {
                pasteButton.classList.add("de-awaiting-paste");
                status.textContent = `Reference ${name.slice(-1).toUpperCase()} selected · click Paste again after allowing clipboard access, or press Ctrl+V.`;
                stage.focus();
            }
        };
        stage.onclick = () => { activate(name.slice(-1)); stage.focus(); };
        stage.onfocus = () => activate(name.slice(-1));
        img.onload = () => { meta.textContent = `${img.naturalWidth} × ${img.naturalHeight} · Full image`; };
        stage.append(img, empty); actions.append(uploadButton, pasteButton, clear, file);
        box.append(head, stage, meta, actions);
        img.alt = title; file.type = "file"; file.accept = "image/*"; file.setAttribute("aria-label", `Upload ${title}`); clear.type = "button";
        let revision = 0, source;
        async function upload(image) {
            if (!image) return;
            pasteButton.classList.remove("de-awaiting-paste");
            const current = ++revision;
            status.textContent = `Saving ${title}…`;
            try {
                const form = new FormData(); form.append("image", image, image.name || "clipboard.png");
                const response = await api.fetchApi("/donut/edit-studio/reference", {method:"POST", body:form});
                if (!response.ok) throw new Error(await response.text());
                const data = await response.json();
                if (disposed || current !== revision) return;
                set(name, data.reference); status.textContent = `${title} saved.`;
            } catch (error) { if (!disposed && current === revision) status.textContent = error.message; }
        }
        file.onchange = () => { upload(file.files[0]); file.value = ""; };
        clear.onclick = () => { ++revision; set(name, ""); };
        box.ondragover = event => { event.preventDefault(); event.stopPropagation(); };
        box.ondrop = event => { event.preventDefault(); event.stopPropagation(); upload([...event.dataTransfer.files].find(f => f.type.startsWith("image/"))); };
        function paste(event) {
            const image = clipboardImage(event.clipboardData);
            if (image) { event.preventDefault(); event.stopImmediatePropagation(); upload(image); }
            return Boolean(image);
        }
        box.onpaste = paste;
        slots[name.slice(-1)] = {card:box, image:img, stage, meta, upload, paste};
        previews.push(() => {
            box.classList.toggle("de-active", name === "image_a" || !!get("use_reference_b"));
            const value = get(name);
            if (source !== value) {
                source = value;
                if (value?.startsWith("donutref:")) img.src = api.apiURL(`/donut/edit-studio/reference/${encodeURIComponent(value.slice(9))}`);
                else if (value) img.src = api.apiURL(`/view?filename=${encodeURIComponent(value)}&type=input`);
                else { img.removeAttribute("src"); meta.textContent = "No image selected"; }
            }
            img.hidden = !value; empty.hidden = !!value;
        });
        grid.append(box);
    }
    function refresh() {
        enabled.checked = !!get("enabled"); second.checked = !!get("use_reference_b");
        previews.forEach(refresh => refresh());
        const input = node.inputs?.find(i => i.name === "edit_active");
        const link = input?.link != null ? node.graph?.links[input.link] : null;
        const editNode = link ? node.graph.getNodeById(link.origin_id) : null;
        const editing = editNode?.widgets?.find(w => w.name === "enabled")?.value ?? get("edit_active");
        if (editing && get("enabled")) status.textContent = "Paused while Editing is on.";
        else if (status.textContent === "Paused while Editing is on.") status.textContent = "";
    }
    for (const w of node.widgets) {
        w.hidden = true; w.computeSize = () => [0, -4]; w.computeLayoutSize = () => ({minHeight:0, maxHeight:0});
        w.options ||= {}; w.options.hidden = true;
    }
    const dom = node.addDOMWidget("reference_guidance", "custom", root, {serialize:false, hideOnZoom:false, getValue:() => "", setValue:() => {}});
    dom.serialize = false; dom.label = "Reference guidance";
    dom.computeSize = () => [500, 650]; dom.computeLayoutSize = () => ({minHeight:650, minWidth:380});
    fitModule(node, dom, root);
    let timer;
    const observer = new IntersectionObserver(entries => {
        clearInterval(timer);
        if (entries.some(entry => entry.isIntersecting)) { refresh(); timer = setInterval(refresh, 500); }
    });
    observer.observe(root);
    const added = node.onAdded, removed = node.onRemoved, configured = node.onConfigure;
    studio = {node,root,slots,refresh,activate,
        awaitingPaste:() => Boolean(root.querySelector(".de-awaiting-paste")),
        paste(event) { return slots[activeSlot]?.paste(event) || false; }};
    referenceStudios.add(studio); activate("a");
    node.onAdded = function() { disposed = false; referenceStudios.add(studio); observer.observe(root); return added?.apply(this, arguments); };
    node.onRemoved = function() { disposed = true; referenceStudios.delete(studio); clearInterval(timer); observer.disconnect(); return removed?.apply(this, arguments); };
    node.onConfigure = function() { const result = configured?.apply(this, arguments); refresh(); return result; };
    node.size = [540, 720]; node._donutReferenceStudio = studio; refresh();
}
app.registerExtension({
    name:"Donut.ReferenceStudio",
    setup() {
        const style = el("style");
        style.textContent = `.donut-reference-studio{--de-accent:#92e4c7}.donut-reference-studio .de-stage img{display:block;width:100%;height:100%;object-fit:contain}.donut-reference-studio [hidden]{display:none!important}.donut-reference-studio .de-ref:not(.de-active){opacity:.65}.donut-reference-studio .de-ref.de-selected{border-color:var(--de-accent);box-shadow:0 0 0 1px #92e4c722}`;
        document.head.append(style);
        window.addEventListener("paste", event => {
            if (event.target?.closest?.("input,textarea,select,[contenteditable=true]")) return;
            const containing=[...referenceStudios].find(item=>item.root.contains?.(event.target));
            if (containing) {containing.paste(event);return;}
            if (activeReferenceStudio && referenceStudios.has(activeReferenceStudio)
                    && activeReferenceStudio.awaitingPaste?.()) {
                activeReferenceStudio.paste(event);return;
            }
            if (activeReferenceStudio && referenceStudios.has(activeReferenceStudio)
                    && activeReferenceStudio.root.contains?.(document.activeElement)) {
                activeReferenceStudio.paste(event);return;
            }
            const selected=Object.values(app.canvas?.selected_nodes || {});
            if(selected.length===1 && selected[0]._donutReferenceStudio) selected[0]._donutReferenceStudio.paste(event);
        }, true);
    },
    beforeRegisterNodeDef(type, definition) {
        if (definition.name !== "DonutReferenceStudio") return;
        const created = type.prototype.onNodeCreated;
        type.prototype.onNodeCreated = function() { const result = created?.apply(this, arguments); install(this); return result; };
    },
});

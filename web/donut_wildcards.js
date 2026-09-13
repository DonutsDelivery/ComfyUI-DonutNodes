import { api } from "../../scripts/api.js";
import { fitTextarea, scheduleLayout } from "./donut_layout.js?v=15";
const el = (tag, text) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node; };
let catalog;
async function request(path, data) {
    const response = await api.fetchApi(path, data === undefined ? {cache:"no-store"} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)});
    if (!response.ok) throw new Error(await response.text());
    return response.json();
}
async function names() {
    if (!catalog) catalog = request("/donut/wildcards").catch(error => { catalog = undefined; throw error; });
    return catalog;
}
async function populate(select, placeholder = "Choose a wildcard…") {
    const data = await names();
    const value = select.value;
    select.replaceChildren(); const empty = el("option", placeholder); empty.value = ""; select.append(empty);
    for (const name of data.names) { const option = el("option", `${name}*`); option.value = name; select.append(option); }
    select.value = data.names.includes(value) ? value : "";
    return data;
}
export function promptTools(textarea, commit, getSeed) {
    const fields = Array.isArray(textarea)
        ? textarea.map(item => ({textarea: item.textarea || item, commit: item.commit || commit, label: item.label}))
        : [{textarea, commit, label: textarea.getAttribute("aria-label")}];
    let active = fields[0];
    const root = el("div"), target = el("select"), select = el("select"), insert = el("button", "Insert wildcard"), preview = el("button", "Preview expanded"), status = el("p"), result = el("textarea");
    root.className = "donut-prompt-tools";
    let previewText, previewSeed, expanded = false, revision = 0;
    if (fields.length > 1) {
        const first = el("option", "Insert into…"); first.value = ""; target.append(first);
        fields.forEach((field, index) => {
            const option = el("option", field.label || `Prompt ${index + 1}`); option.value = String(index); target.append(option);
        });
        target.value = "0";
        target.setAttribute("aria-label", "Prompt target for wildcard");
        target.onchange = () => { if (target.value !== "") active = fields[Number(target.value)]; collapse(); };
    } else target.hidden = true;
    const collapse = () => {
        expanded = false; ++revision; result.hidden = true;
        preview.textContent = "Preview expanded";
        preview.setAttribute("aria-expanded", "false");
        status.textContent = ""; scheduleLayout();
    };
    const refresh = () => { if (expanded && (previewText !== active.textarea.value || previewSeed !== getSeed())) collapse(); };
    fields.forEach(field => field.textarea.addEventListener("input", collapse));
    preview.setAttribute("aria-expanded", "false");
    select.setAttribute("aria-label", "Wildcard to insert");
    insert.type = preview.type = "button"; result.readOnly = true; result.hidden = true;
    result.className = "donut-prompt-preview"; result.setAttribute("aria-label", "Expanded prompt preview");
    const update = () => populate(select).catch(error => { status.textContent = error.message; });
    select.onfocus = update; update();
    insert.onclick = () => {
        if (!select.value) { status.textContent = "Choose a wildcard to insert."; return; }
        const input = active.textarea, start = input.selectionStart, end = input.selectionEnd;
        const before = input.value.slice(0, start), after = input.value.slice(end);
        const token = `${before && !/\s$/.test(before) ? " " : ""}${select.value}*${after && !/^\s/.test(after) ? " " : ""}`;
        input.value = before + token + after; active.commit(input.value); input.focus();
        collapse();
        input.setSelectionRange(before.length + token.length, before.length + token.length);
        status.textContent = "Token inserted. Fixed seed keeps its choice repeatable.";
    };
    preview.onclick = async () => {
        if (expanded) { collapse(); return; }
        expanded = true;
        const current = ++revision;
        const text = active.textarea.value, seed = getSeed();
        previewText = text; previewSeed = seed;
        preview.textContent = "Hide preview";
        preview.setAttribute("aria-expanded", "true");
        status.textContent = "Expanding…";
        try {
            const data = await request("/donut/wildcards/preview", {text, seed});
            if (current !== revision) return;
            if (text !== active.textarea.value || seed !== getSeed()) { collapse(); return; }
            previewText = text; previewSeed = seed;
            result.value = data.text; result.hidden = false; status.textContent = `Preview for seed ${seed}.`;
            fitTextarea(result); scheduleLayout();
        } catch (error) {
            if (current !== revision) return;
            collapse(); status.textContent = error.message;
        }
    };
    root.append(target, select, insert, preview, status, result);
    return {element:root, refresh};
}
export function wildcardLibrary() {
    const root = el("div"), select = el("select"), fresh = el("button", "New wildcard"), name = el("input"), text = el("textarea"), save = el("button", "Save wildcard"), status = el("p"), location = el("p");
    root.className = "donut-wildcard-library";
    select.setAttribute("aria-label", "Saved wildcards"); name.setAttribute("aria-label", "Wildcard name"); text.setAttribute("aria-label", "Wildcard options");
    name.placeholder = "haircolor"; text.placeholder = "One choice per line\nblack hair\nblonde hair\nred hair";
    fresh.type = save.type = "button";
    const nameLabel = el("label"), textLabel = el("label");
    nameLabel.append(el("span", "Name · invoke with name*"), name);
    textLabel.append(el("span", "Options · one per line"), text);
    let revision = 0;
    select.onchange = async () => {
        if (!select.value) return;
        const current = ++revision;
        try {
            const data = await request(`/donut/wildcards/file?name=${encodeURIComponent(select.value)}`);
            if (current !== revision) return;
            name.value = data.name; text.value = data.text; status.textContent = ""; fitTextarea(text);
        } catch (error) { status.textContent = error.message; }
    };
    fresh.onclick = () => { ++revision; select.value = ""; name.value = text.value = status.textContent = ""; name.focus(); };
    save.onclick = async () => {
        try {
            const data = await request("/donut/wildcards/file", {name:name.value.trim(), text:text.value});
            catalog = undefined; await populate(select); select.value = data.name;
            status.textContent = `Saved ${data.name}*. Ready to insert into any prompt.`;
        } catch (error) { status.textContent = error.message; }
    };
    populate(select).then(data => { location.textContent = `Saved as .txt files in ${data.directory}`; }).catch(error => { status.textContent = error.message; });
    root.append(el("p", "Create reusable choices here. Insert a token in a prompt, or type haircolor*. Choices resolve when you generate; the shared seed controls the result."), select, fresh, nameLabel, textLabel, save, status, location);
    return root;
}

// Presentation only. No model loading, network requests or serialized row state.
// Filename/preset selectors and numeric controls remain native ComfyUI widgets.
function loraElement(tag, text, style = {}) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    Object.assign(element.style, style);
    return element;
}
function loraButton(text, title, callback) {
    const button = loraElement("button", text, {
        font: "inherit", lineHeight: "20px", minHeight: "26px", padding: "1px 9px",
        border: "1px solid var(--border-color, #595959)", borderRadius: "4px",
        background: "var(--comfy-input-bg, #292929)", color: "var(--input-text, #ddd)",
        cursor: "pointer", flexShrink: "0"
    });
    button.type = "button"; button.title = title;
    button.setAttribute("aria-label", title);
    button.addEventListener("click", event => {
        event.preventDefault(); event.stopPropagation();
        if (!button.disabled) callback();
    });
    return button;
}

export function createLoraToolbar(index, count, onAction) {
    const root = loraElement("div", undefined, {
        boxSizing: "border-box", width: "100%", padding: "6px 12px 2px",
        font: "12px sans-serif", color: "var(--input-text, #ddd)"
    });
    root.setAttribute("aria-label", `LoRA ${index + 1} actions`);
    const bar = loraElement("div", undefined, {
        display: "flex", alignItems: "center", gap: "5px", paddingTop: "6px",
        borderTop: "1px solid var(--border-color, #595959)"
    });
    bar.append(loraElement("strong", `LoRA ${index + 1}`, { flex: "1", minWidth: "0" }));
    const up = loraButton("↑", `Move LoRA ${index + 1} up`, () => onAction("Move up"));
    const down = loraButton("↓", `Move LoRA ${index + 1} down`, () => onAction("Move down"));
    up.disabled = index === 0; down.disabled = index === count - 1;
    for (const button of [up, down]) if (button.disabled) {
        button.style.opacity = "0.35"; button.style.cursor = "default";
    }
    const remove = loraButton("Remove", `Remove LoRA ${index + 1} from this stack (keeps the file)`, () => onAction("Remove"));
    bar.append(up, down, remove); root.append(bar);
    return root;
}

// Runs are lossless: [0,1,2,4,6,7] -> "0–2, 4, 6–7", not "0–7".
export function compactLoraIndices(indices) {
    const values = [...new Set((indices || []).filter(Number.isInteger))].sort((a, b) => a - b);
    const ranges = [];
    for (let i = 0; i < values.length; i++) {
        const first = values[i];
        let last = first;
        while (i + 1 < values.length && values[i + 1] === last + 1) last = values[++i];
        ranges.push(first === last ? String(first) : `${first}–${last}`);
    }
    return ranges.join(", ");
}

export function renderLoraInformation(root, row, data, { api, lookupOn, view, onAction, onResize }) {
    root.replaceChildren();
    Object.assign(root.style, {
        boxSizing: "border-box", width: "100%", padding: "0 12px 8px",
        overflow: "auto", font: "12px/1.4 sans-serif", color: "var(--input-text, #ddd)"
    });
    const card = loraElement("div", undefined, {
        boxSizing: "border-box", padding: "8px 10px", display: "grid", gap: "8px",
        border: "1px solid var(--border-color, #595959)", borderRadius: "5px",
        background: "var(--comfy-input-bg, #292929)", minWidth: "0", overflowWrap: "anywhere"
    });
    root.append(card);
    const muted = { opacity: "0.75", fontSize: "11px" };
    const section = (key, title, defaultOpen = false) => {
        const box = loraElement("details"); box.open = view[key] ?? defaultOpen;
        const summary = loraElement("summary", title, { cursor: "pointer", fontWeight: "600" });
        box.append(summary);
        const body = loraElement("div", undefined, { paddingTop: "7px", display: "grid", gap: "6px", minWidth: "0" });
        box.append(body);
        box.addEventListener("toggle", () => {
            // Ignore events from a panel replaced by an async metadata update.
            if (!root.contains(box)) return;
            view[key] = box.open; onResize();
        });
        return { box, body };
    };
    const analysis = data.analysis;
    const count = analysis?.components?.length || 0;
    const detected = section("weightsOpen", analysis?.supported
        ? `Detected weights · ${analysis.tensor_count} tensors · ${count} component${count === 1 ? "" : "s"}`
        : "Detected weights");
    if (analysis?.supported) {
        for (const component of analysis.components || []) {
            const group = loraElement("div", undefined, { display: "grid", gap: "2px" });
            group.append(loraElement("strong", `${component.name} · ${component.modules} modules`));
            for (const block of component.groups || []) {
                const line = loraElement("div", `${block.name}: ${compactLoraIndices(block.indices)}`);
                line.title = (block.indices || []).join(", "); group.append(line);
            }
            if (component.ungrouped_names?.length) group.append(loraElement("div", component.ungrouped_names.join(", "), muted));
            detected.body.append(group);
        }
    } else detected.body.append(loraElement("div", analysis?.error || data.analysisError || "Inspecting local file…", muted));
    // Keep errors visible even while the long block list is collapsed.
    if (data.analysisError || analysis?.error) detected.box.open = true;
    card.append(detected.box);

    const info = data.info?.civitai;
    const civitai = section("civitaiOpen", lookupOn ? "CivitAI" : "CivitAI · automatic lookup off", true);
    card.append(civitai.box);
    const overview = loraElement("div", undefined, {
        display: "grid", gridTemplateColumns: "minmax(0, 1fr)", gap: "10px", alignItems: "start"
    });
    const text = loraElement("div", undefined, { display: "grid", gap: "5px", minWidth: "0" });
    overview.append(text); civitai.body.append(overview);
    if (info) {
        text.append(loraElement("strong", info.model_name || row.lora_name));
        text.append(loraElement("div", [info.version_name, info.base_model].filter(Boolean).join(" · "), muted));
        if (typeof info.recommended_weight === "number" && Number.isFinite(info.recommended_weight)) {
            const weight = loraElement("div", `Suggested weight: ${info.recommended_weight}`);
            weight.title = "CivitAI example/cache hint, not an optimum. Never applied automatically.";
            text.append(weight);
        }
    } else text.append(loraElement("div", data.info?.error || data.infoError || (lookupOn
        ? "Looking up hash and previews…" : "Enable CivitAI lookup to fetch metadata automatically."), muted));

    const isHash = value => typeof value === "string" && /^[a-f\d]{10,128}$/i.test(value);
    const hash = isHash(row.lora_hash) ? row.lora_hash : data.info?.hash;
    if (info || (lookupOn && isHash(hash))) {
        const id = Number(info?.model_id), version = Number(info?.model_version_id);
        const link = loraElement("a", info ? "Open on CivitAI ↗" : "Search hash on CivitAI ↗", { display: "block" });
        link.href = Number.isSafeInteger(id) && id > 0 ? `https://civitai.com/models/${id}${Number.isSafeInteger(version) && version > 0 ? "?modelVersionId=" + version : ""}`
            : `https://civitai.com/search/models?query=${encodeURIComponent(isHash(hash) ? hash : row.lora_name)}`;
        link.target = "_blank"; link.rel = "noopener noreferrer";
        text.append(link);
    }
    let previewURL;
    if (isHash(data.info?.hash) && (data.info.has_collage || data.info.preview_count > 0)) {
        // The local preview cache is indexed by the lookup prefix, not a saved full hash.
        previewURL = api.apiURL(`/donut/loras/preview?${new URLSearchParams({ hash: data.info.hash, type: data.info.has_collage ? "collage" : "0" })}`);
    } else if (data.execution?.image?.filename) {
        const image = data.execution.image;
        previewURL = api.apiURL(`/view?${new URLSearchParams({ filename: image.filename, subfolder: image.subfolder || "", type: image.type || "temp" })}`);
    }
    if (previewURL) {
        overview.style.gridTemplateColumns = "minmax(0, 1fr) 92px";
        const link = loraElement("a", undefined, { display: "block", width: "92px" });
        link.href = previewURL; link.target = "_blank"; link.rel = "noopener noreferrer";
        const image = loraElement("img", undefined, {
            display: "block", width: "92px", height: "112px", maxWidth: "100%", objectFit: "contain", borderRadius: "3px"
        });
        image.src = previewURL; image.alt = "LoRA preview (click to open)"; image.loading = "lazy";
        image.addEventListener("error", () => {
            link.replaceChildren(loraElement("span", "Preview unavailable", muted)); onResize();
        });
        link.append(image); overview.append(link);
    } else if (info) text.append(loraElement("div", "No cached preview image available.", muted));
    if (info?.trained_words?.length) civitai.body.append(loraElement("div", `Triggers: ${info.trained_words.join(", ")}`));
    const more = section("moreOpen", "More details");
    if (info?.creator_username) more.body.append(loraElement("div", `Author: ${info.creator_username}`));
    if (isHash(hash)) more.body.append(loraElement("div", `Hash: ${hash}`, { fontSize: "11px", overflowWrap: "anywhere" }));
    more.body.append(loraElement("div", `Your strengths: model ${row.model_weight} · CLIP/text fusion ${row.clip_weight}`));
    if (info?.description) more.body.append(loraElement("div", String(info.description).replace(/<[^>]*>/g, "")));
    if (!info && data.execution?.text) more.body.append(loraElement("div", data.execution.text));
    if (typeof info?.recommended_weight === "number" && Number.isFinite(info.recommended_weight)) {
        more.body.append(loraElement("small", "Suggested weight is an example/cache hint, not a measured optimum."));
        more.body.append(loraButton("Use suggested model weight", "Use the suggested model weight; leave CLIP strength unchanged", () => onAction("Use suggested model weight")));
    }
    more.body.append(loraButton("Retry metadata", "Retry metadata (CivitAI requests require lookup On)", () => onAction("Retry metadata")));
    civitai.body.append(more.box);
    if (data.infoError || data.info?.error) {
        // A failure is actionable without opening More details.
        more.box.open = true;
        text.append(loraElement("small", "Retry metadata below to try again."));
    }
    // The caller observes this natural-height child, not its constrained parent.
    // Short panels shrink; long descriptions scroll within the measured cap.
    return { content: card, preferredHeight: previewURL ? 220 : info ? 178 : 125 };
}

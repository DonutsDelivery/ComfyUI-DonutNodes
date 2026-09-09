const el = (tag, text) => {
    const item = document.createElement(tag);
    if (text !== undefined) item.textContent = text;
    return item;
};

export function numericVector(value) {
    const parts = String(value ?? "").split(",").map(part => part.trim());
    if (!parts.length || parts.some(part => !part || !Number.isFinite(Number(part)))) return null;
    return parts.map(Number);
}

export function weightControl(title, get, set, {min = -2, max = 2, step = 0.01, vertical = false, caption: shortCaption} = {}) {
    const row = el("label"), caption = el("span", title), controls = el("div");
    row.className = vertical ? "donut-weight-row donut-weight-vertical" : "donut-weight-row"; controls.className = "donut-weight-value";
    caption.textContent = shortCaption ?? title; caption.title = title;
    const slider = el("input"), number = el("input");
    slider.type = "range"; number.type = "number";
    slider.step = String(step); number.step = "any";
    slider.setAttribute("aria-label", `${title} slider`); number.setAttribute("aria-label", title);
    const refresh = () => {
        const value = Number(get());
        slider.min = String(Math.min(min, value)); slider.max = String(Math.max(max, value));
        if (document.activeElement !== slider) slider.value = value;
        if (document.activeElement !== number) number.value = value;
    };
    slider.oninput = () => { const value = Number(slider.value); number.value = value; set(value); };
    number.oninput = () => { if (number.value && Number.isFinite(Number(number.value))) { set(Number(number.value)); refresh(); } };
    controls.append(slider, number); row.append(caption, controls); refresh();
    return {element:row, refresh};
}

export function vectorControl(title, get, set, {labels = [], count = 29, min = -2, max = 2} = {}) {
    const root = el("div"), grid = el("div"), bank = el("div"), raw = el("details"), text = el("textarea");
    root.className = "donut-vector-control"; grid.className = "donut-vector-grid";
    root.append(el("h4", title)); bank.className = "donut-weight-bank";
    let allValue = 1;
    const toolbar = el("div"), all = weightControl(`${title} · set all`, () => allValue, value => {
        allValue = value;
        const values = numericVector(get()); if (values) { set(values.map(() => value).join(",")); refresh(); }
    }, {min, max, vertical:true, caption:"ALL"});
    toolbar.append(all.element);
    const initialize = el("button", "Create weight sliders"); initialize.type = "button";
    initialize.onclick = () => { set(Array(count).fill(1).join(",")); refresh(); };
    const description = el("p", "Drag vertically to adjust. ALL changes every weight.");
    raw.append(el("summary", "Advanced · weight text"));
    text.setAttribute("aria-label", `${title} text`); raw.append(text);
    text.onchange = () => { set(text.value); refresh(); };
    bank.append(toolbar, grid); root.append(description, bank, initialize, raw);
    let previous, controls = [], length;
    function refresh() {
        const value = String(get() ?? "");
        if (document.activeElement !== text) text.value = value;
        const values = numericVector(value);
        initialize.hidden = !!values; toolbar.hidden = !values;
        if (!values) { grid.replaceChildren(); controls = []; length = undefined; previous = value; return; }
        if (length !== values.length) {
            grid.replaceChildren(); controls = values.map((_, index) => {
                const control = weightControl(`${title} · ${labels[index] || `Weight ${index + 1}`}`, () => numericVector(get())?.[index] ?? 1, weight => {
                    const current = numericVector(get());
                    if (current) { current[index] = weight; set(current.join(",")); }
                }, {min,max,vertical:true,caption:labels[index] || String(index + 1)});
                grid.append(control.element); return control;
            });
            length = values.length;
        }
        if (value !== previous) controls.forEach(control => control.refresh());
        previous = value;
    }
    refresh();
    return {element:root, refresh};
}

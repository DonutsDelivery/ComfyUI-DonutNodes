const WEIGHT_GROUP = /\(([^():]+):(-?\d*\.?\d+)\)/g;

function roundWeight(value) {
    return Math.round(value * 100) / 100;
}

function formatWeight(value) {
    const rounded = Math.min(10, Math.max(-10, roundWeight(value)));
    return Object.is(rounded, -0) ? "0" : String(rounded);
}

function groups(text) {
    return [...String(text).matchAll(WEIGHT_GROUP)].map(match => ({
        from: match.index,
        to: match.index + match[0].length,
        phrase: match[1],
        weight: Number(match[2]),
    }));
}

function enclosing(text, start, end) {
    return groups(text).find(group => start >= group.from && end <= group.to
        && (start !== end || (start > group.from && start < group.to)));
}

function wordBounds(text, index) {
    if (!text) return null;
    let start = Math.max(0, Math.min(index, text.length));
    if (start > 0 && (start === text.length || /\s/.test(text[start] || "")) && !/\s/.test(text[start - 1])) start -= 1;
    while (start > 0 && !/\s/.test(text[start - 1])) start -= 1;
    let end = start;
    while (end < text.length && !/\s/.test(text[end])) end += 1;
    if (start === end || /[():]/.test(text.slice(start, end))) return null;
    return [start, end];
}

// Shift+Up/Down changes a marked phrase by 0.05, starting at 1.00.
// Returning to 1.00 removes the (phrase:weight) markup.
export function adjustPromptWeight(text, start, end, delta) {
    if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(delta) || delta === 0) return null;
    const source = String(text);
    const group = enclosing(source, start, end);
    if (group) {
        const next = Math.min(10, Math.max(-10, roundWeight(group.weight + delta)));
        if (next === group.weight) return null;
        const replacement = next === 1 ? group.phrase : `(${group.phrase}:${formatWeight(next)})`;
        return {
            text: source.slice(0, group.from) + replacement + source.slice(group.to),
            start: group.from,
            end: group.from + replacement.length,
        };
    }
    let from = start;
    let to = end;
    if (from === to) {
        const word = wordBounds(source, from);
        if (!word) return null;
        [from, to] = word;
    }
    const selected = source.slice(from, to);
    if (!selected.trim() || /[():]/.test(selected)) return null;
    const wrapped = `(${selected}:${formatWeight(1 + delta)})`;
    return { text: source.slice(0, from) + wrapped + source.slice(to), start: from, end: from + wrapped.length };
}

export function promptWeightTarget(element) {
    if (!element || element.tagName !== "TEXTAREA" || element.readOnly) return false;
    if (element.classList.contains("donut-prompt-preview") || element.closest(".donut-wildcard-library")) return false;
    if (element.classList.contains("donut-long-prompt") || element.closest(".donut-prompt-set-row")) return true;
    return element.closest(".donut-edit-studio") && /instruction|prompt/i.test(element.getAttribute("aria-label") || "");
}

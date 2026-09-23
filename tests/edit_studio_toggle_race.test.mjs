import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

const categoriesSource = readFileSync(new URL("../web/donut_panel_categories.js", import.meta.url), "utf8");
const editStudioSource = readFileSync(new URL("../web/donut_edit_studio.js", import.meta.url), "utf8");

// Scheduler double of the DOM dispatch model that caused the bug: each event
// dispatch runs to completion and drains microtasks (the browser's microtask
// checkpoint), while setTimeout tasks wait in the task queue until every
// dispatch has finished.
function makeEnvironment(scheduleRerender) {
    const microtasks = [], tasks = [];
    const state = {checkboxChecked: true, widgetValue: true, renders: 0};
    const queueMicrotask = task => microtasks.push(task);
    const setTimeout = task => tasks.push(task);
    function render() {
        state.renders += 1;
        state.checkboxChecked = Boolean(state.widgetValue); // render() writeback
    }
    function dispatchInput() {
        state.checkboxChecked = !state.checkboxChecked;     // click flips OFF
        scheduleRerender({state, queueMicrotask, setTimeout, render});
        for (const task of microtasks.splice(0)) task();    // checkpoint
    }
    function dispatchChange() {
        state.widgetValue = state.checkboxChecked;          // commitValues
        for (const task of microtasks.splice(0)) task();
    }
    return {
        state,
        click: () => { dispatchInput(); dispatchChange(); },
        settle: () => { while (tasks.length) tasks.shift()(); },
    };
}

const microtaskRerender = ({state, queueMicrotask, render}) =>
    queueMicrotask(() => render());  // the old listener body: render between input and change
const taskRerender = ({setTimeout, render}) =>
    setTimeout(render);              // the shipped listener body: render after change

function runScenario(scheduleRerender) {
    const env = makeEnvironment(scheduleRerender);
    env.click();
    env.settle();
    return env.state.widgetValue;
}

test("a microtask re-render between input and change cancels the toggle (documents the bug)", () => {
    assert.equal(runScenario(microtaskRerender), true, "stale render wrote the old value back before commit");
});

test("the shipped macrotask re-render preserves the toggle", () => {
    assert.equal(runScenario(taskRerender), false);
});

test("panel-root Edit Studio re-render is deferred to a task, not a microtask", () => {
    assert.doesNotMatch(categoriesSource, /queueMicrotask\(\(\) => \{\s*\n\s*for \(const \{node:entry\}/);
    assert.match(categoriesSource, /const change = \(\) => setTimeout\(\(\) => \{\s*\n\s*for \(const \{node:entry\} of graphEntries\(rootGraph\(\)\)\) entry\._donutEditStudio\?\.render\(\);/);
});

test("the refresh() dedupe microtask is untouched", () => {
    assert.match(categoriesSource, /let pending = false;/);
    assert.match(categoriesSource, /queueMicrotask\(\(\) => \{\s*\n\s*pending = false;/);
});

test("clearing reference A exits edit mode so text-to-image stays reachable", () => {
    const clear = editStudioSource.match(/const clear = button\("×"[\s\S]*?\); clear\.className/);
    assert.ok(clear, "clear button not found");
    assert.match(clear[0], /key === "b" \? \{use_reference_b:false\} : \{mask_data:"", inpaint_enabled:false, enabled:false\}/);
});

import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/donut_edit_studio.js", import.meta.url), "utf8");
const referenceSource = readFileSync(new URL("../web/donut_reference_studio.js", import.meta.url), "utf8");
const clipboardSource = readFileSync(new URL("../web/donut_clipboard.js", import.meta.url), "utf8");
const scope = vm.createContext({DOMException});
vm.runInContext(clipboardSource.replace(/^export /gm, ""), scope);

test("clipboard images are accepted from the files collection", () => {
    const image = {type: "image/png", name: "clipboard.png"};
    assert.equal(scope.clipboardImage({files: [image], items: []}), image);
});

test("clipboard images fall back to item.getAsFile", () => {
    const image = {type: "image/jpeg", name: "clipboard.jpeg"};
    const item = {type: "image/jpeg", getAsFile: () => image};
    assert.equal(scope.clipboardImage({files: [], items: [item]}), image);
});

test("non-image clipboard entries are ignored", () => {
    const text = {type: "text/plain", getAsFile: () => ({type: "text/plain"})};
    assert.equal(scope.clipboardImage({files: [], items: [text]}), null);
    assert.equal(scope.clipboardImage(undefined), null);
});

test("both panels use the same async clipboard image reader", async () => {
    const image = {type:"image/png"};
    const clipboard = {read:async()=>[
        {types:["text/plain"]},
        {types:["image/png"],getType:async type => type === "image/png" ? image : null},
    ]};
    assert.equal(await scope.readClipboardImage(clipboard), image);
    assert.equal(await scope.readClipboardImage({read:async()=>[{types:["text/plain"]}]}), null);
    await assert.rejects(()=>scope.readClipboardImage({}),/unavailable/);
    assert.match(source,/readClipboardImage\(\)/);
    assert.match(referenceSource,/readClipboardImage\(\)/);
});

test("Edit Studio captures paste before ComfyUI document handlers", () => {
    assert.match(source, /window\.addEventListener\("paste",[\s\S]*?}, true\);/);
    assert.match(source, /activeStudio\.root\.contains\?\.\(document\.activeElement\)/);
});

test("Paste button arms the selected slot when browser clipboard access is blocked", () => {
    assert.match(source, /pasteButton\.classList\.add\("de-awaiting-paste"\)/);
    assert.match(source, /activeStudio\.awaitingPaste\?\.\(\)/);
    assert.match(source, /click Paste again after allowing clipboard access, or press Ctrl\+V/);
});

test("Reference Guidance supports global Ctrl+V and routes it to the selected A or B slot", () => {
    assert.match(referenceSource,/window\.addEventListener\("paste",[\s\S]*?}, true\);/);
    assert.match(referenceSource,/activeReferenceStudio\.root\.contains\?\.\(document\.activeElement\)/);
    assert.match(referenceSource,/activeSlot = key/);
    assert.match(referenceSource,/activeReferenceStudio\.awaitingPaste\?\.\(\)/);
    assert.match(referenceSource,/stopImmediatePropagation\(\)/);
});

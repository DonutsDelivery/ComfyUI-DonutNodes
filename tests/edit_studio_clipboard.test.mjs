import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/donut_edit_studio.js", import.meta.url), "utf8");
const helper = source.slice(
    source.indexOf("function clipboardImage"),
    source.indexOf("\n\nexport function installEditStudio"),
);
const scope = vm.createContext({});
vm.runInContext(helper, scope);

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

test("Edit Studio captures paste before ComfyUI document handlers", () => {
    assert.match(source, /window\.addEventListener\("paste",[\s\S]*?}, true\);/);
    assert.match(source, /activeStudio\.root\.contains\?\.\(document\.activeElement\)/);
});

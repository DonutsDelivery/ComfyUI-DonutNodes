import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/donut_edit_studio.js", import.meta.url), "utf8");
const helper = source.slice(
    source.indexOf("function normalizeEditStudioValue"),
    source.indexOf("\n\nexport function installEditStudio"),
);
const scope = vm.createContext({Object, Math, Number});
vm.runInContext(helper, scope);

test("malformed inpaint metadata is normalized to backend widget types", () => {
    const backend = new Map([
        ["inpaint_enabled", {value: ""}],
        ["mask_data", {value: null}],
        ["mask_feather", {value: "not-a-number"}],
    ]);
    scope.normalizeEditStudioInpaintWidgets(backend, {
        widgets_values_named: {inpaint_enabled: "", mask_data: "", mask_feather: ""},
    });
    assert.equal(backend.get("inpaint_enabled").value, false);
    assert.equal(backend.get("mask_data").value, "");
    assert.equal(backend.get("mask_feather").value, 8);
});

test("valid inpaint metadata is preserved and numeric feather values are bounded", () => {
    const backend = new Map([
        ["inpaint_enabled", {value: false}],
        ["mask_data", {value: "mask"}],
        ["mask_feather", {value: 8}],
    ]);
    scope.normalizeEditStudioInpaintWidgets(backend, {
        widgets_values_named: {inpaint_enabled: "true", mask_data: 123, mask_feather: 999},
    });
    assert.equal(backend.get("inpaint_enabled").value, true);
    assert.equal(backend.get("mask_data").value, "123");
    assert.equal(backend.get("mask_feather").value, 128);
});

test("Edit Studio serialization excludes the non-serializable DOM widget", () => {
    assert.match(source, /data\.widgets_values = \[\.\.\.backend\.values\(\)\]\.map/);
    assert.match(source, /data\.widgets_values_named = Object\.fromEntries\(\[\.\.\.backend\]/);
    assert.match(source, /addDOMWidget\("edit_studio", "custom", root, \{serialize:false/);
    assert.match(source, /dom\.serialize = false/);
});

test("Edit Studio normalizes mask feather immediately before API queue serialization", () => {
    assert.match(source, /const maskFeather = backend\.get\("mask_feather"\)/);
    assert.match(source, /maskFeather\.beforeQueued = function\(\.\.\.args\)/);
    assert.match(source, /const result = queued\?\.apply\(this, args\);\n\s+normalizeEditStudioInpaintWidgets\(backend\);/);
    assert.match(source, /maskFeather\.serializeValue = function\(\.\.\.args\)/);
    assert.match(source, /normalizeEditStudioValue\("mask_feather", value\)/);
    assert.match(source, /api\.addEventListener\("promptQueueing"/);
    assert.match(source, /studio\.prepareForQueue\?\.\(\)/);
});

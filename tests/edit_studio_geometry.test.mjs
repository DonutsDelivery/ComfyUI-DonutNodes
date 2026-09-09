import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const scope = vm.createContext({});
vm.runInContext(readFileSync(new URL("../web/donut_edit_geometry.js", import.meta.url), "utf8").replaceAll("export ", ""), scope);
const plain = value => JSON.parse(JSON.stringify(value));
const settings = {aspect_ratio:"4:3 Standard", megapixels:1, width:1000, height:770, multiple:"64"};

test("crop-only grids never round the image up", () => {
    assert.deepEqual(plain(scope.targetDimensions({...settings, resolution_mode:"Reference A · crop only"}, [1001,777])), [960,768]);
    assert.deepEqual(plain(scope.targetDimensions({...settings, multiple:"32", resolution_mode:"Reference A · crop only"}, [1001,777])), [992,768]);
    assert.deepEqual(plain(scope.cropBox(1001,777,960,768,.5,.5,true)), [21,5,981,773]);
});
test("custom and megapixel sizes match the backend's grid calculations", () => {
    assert.deepEqual(plain(scope.targetDimensions({...settings, resolution_mode:"Custom"})), [1024,768]);
    assert.deepEqual(plain(scope.targetDimensions({...settings, multiple:"32", resolution_mode:"Reference A · megapixels"}, [1000,2000])), [736,1440]);
    assert.deepEqual(plain(scope.targetDimensions({...settings, resolution_mode:"Preset"})), [1152,896]);
});
test("crop positions reach each edge independently", () => {
    assert.deepEqual(plain(scope.cropBox(120,60,64,64,0,.5)), [0,0,60,60]);
    assert.deepEqual(plain(scope.cropBox(120,60,64,64,1,.5)), [60,0,120,60]);
    assert.deepEqual(plain(scope.cropBox(60,120,64,64,.5,1)), [0,60,60,120]);
});
test("saved annotated input paths retain nested folders and spaces", () => {
    assert.deepEqual(plain(scope.imageLocation("donut/references/My image.png [input]")), {filename:"My image.png", subfolder:"donut/references", type:"input"});
    assert.deepEqual(plain(scope.imageLocation("image.png")), {filename:"image.png", subfolder:"", type:"input"});
});

test("16-pixel grid retains dimensions not divisible by 32", () => {
    assert.deepEqual(plain(scope.targetDimensions({...settings, multiple:"16", width:1008, height:784, resolution_mode:"Custom"})), [1008,784]);
});

test("auto aspect selects the closest preset independently from A or B", () => {
    const base = {...settings, resolution_mode:"Preset", multiple:"16"};
    assert.deepEqual(plain(scope.targetDimensions({...base, aspect_ratio:"Auto · Reference A"}, [1900,1080], [1000,1500])), plain(scope.targetDimensions({...base, aspect_ratio:"16:9 Wide"})));
    assert.deepEqual(plain(scope.targetDimensions({...base, aspect_ratio:"Auto · Reference B"}, [1900,1080], [1000,1500])), plain(scope.targetDimensions({...base, aspect_ratio:"2:3 Portrait"})));
});

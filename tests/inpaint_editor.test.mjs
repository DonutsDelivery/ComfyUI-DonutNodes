import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/donut_inpaint_editor.js", import.meta.url), "utf8");
const scope = vm.createContext({document: undefined});
vm.runInContext(source.replace(/^export /gm, ""), scope);
const {readMask, drawMask, maskInverted} = scope;

test("inversion persists only for its source image and defaults off for older masks", () => {
    const data = JSON.stringify({version:1, image:'A', inverted:true, strokes:[]});
    assert.equal(maskInverted(data, 'A'), true);
    assert.equal(maskInverted(data, 'B'), false);
    assert.equal(maskInverted('{"version":1,"image":"A"}', 'A'), false);
    assert.equal(maskInverted('null', 'A'), false);
});

test("inverted preview complements painted alpha", () => {
    const data = new Uint8ClampedArray([146,228,199,255,0,0,0,0]);
    const ctx = {clearRect(){}, getImageData(){return {data}}, putImageData(pixels){assert.equal(pixels.data, data)}};
    drawMask(ctx, 2, 1, [], true);
    assert.equal(data[3], 0);
    assert.equal(data[7], 255);
});

test("rectangles render identically in both drag directions", () => {
    const calls = [];
    const ctx = {clearRect(){},fillRect(...args){calls.push(args)}};
    for (const points of [[[.2,.3],[.8,.7]], [[.8,.7],[.2,.3]]]) {
        drawMask(ctx, 101, 101, [{size:.08, shape:'rectangle', points}]);
    }
    assert.deepEqual(calls[0], calls[1]);
    assert.deepEqual(calls[0].map(Math.round), [20,30,60,40]);
});

test("saved selections survive JSON serialization and belong only to their source image", () => {
    const strokes = [{size:.08, erase:false, points:[[.3,.4],[.6,.7]]}];
    const value = JSON.stringify({version:1, image:"A", strokes});
    const restored = JSON.parse(JSON.stringify(readMask(value, "A")));
    assert.deepEqual(restored, strokes);
    assert.deepEqual(readMask(value, "B").length, 0);
});

test("invalid saved masks cannot crash the preview", () => {
    for (const value of ["", "null", "bad json", JSON.stringify({version:1,image:"A",strokes:[null]}),
        JSON.stringify({version:1,image:"A",strokes:[{size:1,points:[[null,1]]}]})]) {
        assert.equal(readMask(value, "A").length, 0);
    }
});

test("paint and eraser strokes use the same source geometry at different preview sizes", () => {
    function draw(width,height) {
        const calls=[];
        const ctx={clearRect(){},beginPath(){},moveTo(...p){calls.push(p)},lineTo(...p){calls.push(p)},
            stroke(){calls.push(this.globalCompositeOperation)},arc(){},fill(){}};
        drawMask(ctx,width,height,[{size:.1,points:[[0,0],[1,1]]},{size:.05,erase:true,points:[[.5,.5]]}]);
        assert.equal(ctx.globalCompositeOperation,"source-over");
        return calls;
    }
    assert.deepEqual(draw(101,201),[[0,0],[100,200],"source-over",[50,100],"destination-out"]);
    assert.deepEqual(draw(201,401),[[0,0],[200,400],"source-over",[100,200],"destination-out"]);
});

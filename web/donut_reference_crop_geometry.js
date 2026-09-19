// Pixel/normalized-coordinate geometry shared by both reference-card editors.
// Keep the half-up rounding and output-sizing rules in sync with Python.
export const LEGACY = 'Legacy output-linked';
export const INDEPENDENT = 'Independent crops';
export const ASPECTS = {'1:1':[1,1], '2:3':[2,3], '3:2':[3,2], '3:4':[3,4],
    '4:3':[4,3], '9:16':[9,16], '16:9':[16,9], '21:9':[21,9]};
export const round = value => Math.floor(value + .5);
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
export function parseCrop(value, image, size) {
    if (!value) return {version:1, image, source_size:[...size], bounds:[0,0,1,1], aspect:'Original'};
    const data = typeof value === 'string' ? JSON.parse(value) : value;
    if (data?.version !== 1 || data.image !== image || !Array.isArray(data.source_size)
        || data.source_size.length !== 2 || data.source_size.some((v,i) => v !== size[i]))
        throw new Error('This crop belongs to a different image. Reset or select the crop again.');
    const b = data.bounds;
    if (!Array.isArray(b) || b.length !== 4 || !b.every(v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1)
        || b[0] >= b[2] || b[1] >= b[3]) throw new Error('Invalid saved crop coordinates.');
    if (!['Original','Free',...Object.keys(ASPECTS)].includes(data.aspect || 'Free')) throw new Error('Invalid crop aspect.');
    const pixels = b.map((v,i) => round(v * size[i % 2]));
    if (pixels[2] <= pixels[0] || pixels[3] <= pixels[1]) throw new Error('Crop is smaller than one source pixel.');
    return {...data, bounds:[...b], source_size:[...size], aspect:data.aspect || 'Free'};
}
export function cropBox(value, image, size) {
    return parseCrop(value,image,size).bounds.map((v,i) => round(v * size[i % 2]));
}
export function ratioFor(aspect, size) {
    if (aspect === 'Free') return null;
    if (aspect === 'Original') return size[0]/size[1];
    const ratio = ASPECTS[aspect];
    if (!ratio) throw new Error('Unknown crop aspect.');
    return ratio[0]/ratio[1];
}
export function aspectCrop(bounds, aspect, size) {
    const ratio = ratioFor(aspect,size);
    if (ratio === null) return [...bounds];
    const q = ratio * size[1]/size[0], [x1,y1,x2,y2] = bounds;
    const w = Math.min(x2-x1,(y2-y1)*q), h=w/q;
    const cx=(x1+x2)/2, cy=(y1+y2)/2;
    return [cx-w/2,cy-h/2,cx+w/2,cy+h/2];
}
export function moveCrop(bounds, dx, dy) {
    const [x1,y1,x2,y2] = bounds;
    dx=clamp(dx,-x1,1-x2); dy=clamp(dy,-y1,1-y2);
    return [x1+dx,y1+dy,x2+dx,y2+dy];
}
export function resizeCrop(bounds, corner, point, aspect, size) {
    const left=corner.includes('w'), top=corner.includes('n');
    const anchor=[bounds[left?2:0],bounds[top?3:1]];
    const sx=left?-1:1, sy=top?-1:1;
    const maxW=left?anchor[0]:1-anchor[0], maxH=top?anchor[1]:1-anchor[1];
    let w=clamp((point[0]-anchor[0])*sx,Math.min(1/size[0],maxW),maxW);
    let h=clamp((point[1]-anchor[1])*sy,Math.min(1/size[1],maxH),maxH);
    const ratio=ratioFor(aspect,size);
    if (ratio !== null) { const q=ratio*size[1]/size[0]; w=Math.min(Math.max(w,h*q),maxW,maxH*q); h=w/q; }
    const x=anchor[0]+sx*w,y=anchor[1]+sy*h;
    return [Math.min(x,anchor[0]),Math.min(y,anchor[1]),Math.max(x,anchor[0]),Math.max(y,anchor[1])];
}
export function subjectBounds(rgba, width, height, padding=4) {
    if (!Number.isFinite(padding) || padding < 0 || padding > 50) throw new Error('Padding must be 0–50%.');
    if (rgba.length !== width*height*4) throw new Error('Mask dimensions do not match the image.');
    let x1=width,y1=height,x2=-1,y2=-1;
    for (let y=0;y<height;y++) for (let x=0;x<width;x++) {
        // Stored masks are grayscale, not an alpha mask. Ignore tiny residual
        // foreground probabilities; a source-pixel padding margin retains edges.
        if (rgba[(y*width+x)*4] <= 16) continue;
        x1=Math.min(x1,x);y1=Math.min(y1,y);x2=Math.max(x2,x);y2=Math.max(y2,y);
    }
    if (x2 < 0) throw new Error('No foreground in the saved mask. Refine the subject selection first.');
    x2++;y2++;
    const px=(x2-x1)*padding/100,py=(y2-y1)*padding/100;
    return [Math.max(0,x1-px)/width,Math.max(0,y1-py)/height,Math.min(width,x2+px)/width,Math.min(height,y2+py)/height];
}
export function fitGeometry(source, canvas) {
    const scale=Math.min(canvas[0]/source[0],canvas[1]/source[1]);
    const content=source.map((v,i)=>Math.min(canvas[i],Math.max(1,round(v*scale))));
    return {canvas:[...canvas],content,offset:canvas.map((v,i)=>Math.floor((v-content[i])/2))};
}
export function outputDimensions(values, aSize, aBox, bSize, bBox) {
    if ((values.geometry_mode || LEGACY) === LEGACY || !values.enabled) return null;
    if (values.geometry_mode !== INDEPENDENT) throw new Error('Unknown reference geometry mode.');
    const grid=Number(values.multiple ?? 64);
    if (![16,32,64].includes(grid)) throw new Error('Unsupported output pixel grid.');
    const follow=values.output_canvas || 'Follow A crop';
    if (!['Follow A crop','Independent output'].includes(follow)) throw new Error('Unknown output canvas mode.');
    if (!aSize || !aBox) throw new Error('Reference A is required for editing.');
    const a=[aBox[2]-aBox[0],aBox[3]-aBox[1]],b=bBox?[bBox[2]-bBox[0],bBox[3]-bBox[1]]:bSize;
    let ratio;
    if (follow === 'Follow A crop') {
        if (values.resolution_mode === 'Reference A · crop only') return a.map(v=>Math.max(grid,Math.floor(v/grid)*grid));
        ratio=a;
    } else if (values.resolution_mode === 'Custom') {
        const dims=[Number(values.width),Number(values.height)];
        if (dims.some(v=>!Number.isFinite(v)||v<16||v>16384)) throw new Error('Invalid custom output dimensions.');
        return dims.map(v=>Math.max(grid,round(v/grid)*grid));
    } else {
        const aspect=values.aspect_ratio || '4:3 Standard';
        ratio=aspect==='Auto · Reference A'?a:aspect==='Auto · Reference B'?(b||[4,3]):ASPECTS[aspect.split(' ')[0]];
        if (!ratio) throw new Error('Unknown output aspect ratio.');
    }
    const budget=Number(values.megapixels ?? 1);
    if (!Number.isFinite(budget)||budget<.1||budget>16) throw new Error('Output megapixels must be between 0.1 and 16.');
    const scale=Math.sqrt(budget*1024*1024/(ratio[0]*ratio[1]));
    const dims=ratio.map(v=>Math.max(grid,round(v*scale/grid)*grid));
    if (Math.max(...dims)>16384) throw new Error('Crop too narrow for this pixel budget.');
    return dims;
}

export function sizingVisibility(values) {
    if (!values.enabled || values.geometry_mode !== INDEPENDENT) return null;
    const follow=(values.output_canvas || 'Follow A crop')==='Follow A crop';
    const custom=!follow && values.resolution_mode==='Custom';
    const native=follow && values.resolution_mode==='Reference A · crop only';
    return {width:!custom,height:!custom,aspect_ratio:follow||custom,megapixels:custom||native,
        resolution_mode:false,multiple:false};
}

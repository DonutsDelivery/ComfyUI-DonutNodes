// Masks are saved as source-image coordinates, so crop/output changes stay aligned.
export function readMask(value, image) {
    try {
        const data = JSON.parse(value);
        if (data.version !== 1 || data.image !== image || !Array.isArray(data.strokes)) return [];
        return data.strokes.every(stroke => Number.isFinite(stroke?.size) && stroke.size > 0 && stroke.size <= 1
            && (stroke.shape === undefined || stroke.shape === "rectangle")
            && Array.isArray(stroke.points) && stroke.points.length > 0
            && (stroke.shape !== "rectangle" || stroke.points.length === 2)
            && stroke.points.every(point => Array.isArray(point) && point.length === 2
                && point.every(v => Number.isFinite(v) && v >= 0 && v <= 1))) ? data.strokes : [];
    } catch { return []; }
}

export function maskInverted(value, image) {
    try {
        const data = JSON.parse(value);
        return data.version === 1 && data.image === image && data.inverted === true;
    } catch { return false; }
}

export function readOutpaint(value, image) {
    try {
        const doc = JSON.parse(value), p = doc.outpaint;
        if (doc.version !== 1 || doc.image !== image || !p) return null;
        return [['scale',.1,1],['x',0,1],['y',0,1],['overlap',0,128]].every(
            ([key,min,max]) => Number.isFinite(p[key]) && p[key] >= min && p[key] <= max) ? {...p} : null;
    } catch { return null; }
}

export function outpaintRect(source, output, placement) {
    const [width,height] = output, ratio = Math.min(width/source[0],height/source[1])*placement.scale;
    const w=Math.max(1,Math.round(source[0]*ratio)), h=Math.max(1,Math.round(source[1]*ratio));
    return [Math.round((width-w)*placement.x),Math.round((height-h)*placement.y),w,h];
}

export function outpaintPresetSize(source, currentCanvas, direction, multiple = 32) {
    if (!Array.isArray(source) || source.length !== 2 || source.some(v => !Number.isFinite(v) || v <= 0)
            || !Array.isArray(currentCanvas) || currentCanvas.length !== 2 || currentCanvas.some(v => !Number.isFinite(v) || v <= 0)
            || ![16,32,64].includes(Number(multiple)) || !["horizontal","vertical"].includes(direction)) {
        throw new Error("Invalid outpaint canvas preset.");
    }
    const area = currentCanvas[0] * currentCanvas[1];
    const ratio = source[0] / source[1] * (direction === "horizontal" ? 2 : .5);
    let width = Math.min(16384, Math.max(multiple, Math.round(Math.sqrt(area * ratio) / multiple) * multiple));
    let height = Math.min(16384, Math.max(multiple, Math.round(Math.sqrt(area / ratio) / multiple) * multiple));
    while (width * height > area && (width > multiple || height > multiple)) {
        const narrower = width > multiple ? [width - multiple, height] : null;
        const shorter = height > multiple ? [width, height - multiple] : null;
        const error = size => Math.abs(Math.log((size[0] / size[1]) / ratio));
        [width,height] = !shorter || narrower && error(narrower) <= error(shorter) ? narrower : shorter;
    }
    return [width,height];
}

export function drawOutpaintBase(ctx, image, rect, width, height) {
    const [x,y,w,h]=rect, sw=image.naturalWidth, sh=image.naturalHeight;
    const xs=[[0,1,0,x],[0,sw,x,w],[sw-1,1,x+w,width-x-w]];
    const ys=[[0,1,0,y],[0,sh,y,h],[sh-1,1,y+h,height-y-h]];
    for(const [sx,a,dx,b] of xs) for(const [sy,c,dy,d] of ys)
        if(b>0 && d>0) ctx.drawImage(image,sx,sy,a,c,dx,dy,b,d);
}

function outsideRect(ctx,width,height,rect) {
    const [x,y,w,h]=rect;
    ctx.fillRect(0,0,width,y); ctx.fillRect(0,y+h,width,height-y-h);
    ctx.fillRect(0,y,x,h); ctx.fillRect(x+w,y,width-x-w,h);
}

export function drawMask(ctx, width, height, strokes, inverted = false, placement = null) {
    ctx.clearRect(0, 0, width, height);
    if (placement) {
        ctx.fillStyle = "#92e4c7";
        const [x,y,w,h]=placement.rect, o=Math.min(placement.overlap,Math.floor((w-1)/2),Math.floor((h-1)/2));
        const l=x+(x>0?o:0), t=y+(y>0?o:0), r=x+w-(x+w<width?o:0), b=y+h-(y+h<height?o:0);
        outsideRect(ctx,width,height,[l,t,r-l,b-t]);
    }
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (const stroke of strokes) {
        if (!stroke.points?.length) continue;
        ctx.globalCompositeOperation = stroke.erase ? "destination-out" : "source-over";
        ctx.fillStyle = ctx.strokeStyle = "#92e4c7";
        if (stroke.shape === "rectangle" && stroke.points.length === 2) {
            const [[x1, y1], [x2, y2]] = stroke.points;
            ctx.fillRect(Math.min(x1, x2) * (width - 1), Math.min(y1, y2) * (height - 1),
                Math.abs(x2 - x1) * (width - 1), Math.abs(y2 - y1) * (height - 1));
            continue;
        }
        ctx.lineWidth = stroke.size * Math.min(width, height);
        ctx.beginPath();
        stroke.points.forEach(([x, y], i) => {
            if (i) ctx.lineTo(x * (width - 1), y * (height - 1));
            else ctx.moveTo(x * (width - 1), y * (height - 1));
        });
        ctx.stroke();
        const [x, y] = stroke.points[0];
        ctx.beginPath(); ctx.arc(x * (width - 1), y * (height - 1), ctx.lineWidth / 2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
    if (inverted) {
        const pixels = ctx.getImageData(0, 0, width, height);
        for (let i = 0; i < pixels.data.length; i += 4) {
            pixels.data[i] = 146; pixels.data[i + 1] = 228; pixels.data[i + 2] = 199;
            pixels.data[i + 3] = 255 - pixels.data[i + 3];
        }
        ctx.putImageData(pixels, 0, 0);
    }
    if (placement) { ctx.fillStyle="#92e4c7"; outsideRect(ctx,width,height,placement.rect); }

}

export function selectionAction(selected) {
    return selected
        ? {label:"Use selection", message:"Selection ready · describe the change in Prompts.", enabled:true}
        : {label:"Clear & turn off", message:"No area selected · apply to turn selected-area editing off.", enabled:false};
}

export function openInpaintEditor({image, imageName, value, crop, outputSize, canvasSize = outputSize, pixelGrid = 32, feather = 8, onApply}) {
    let strokes = structuredClone(readMask(value, imageName)), current = null, tool = "brush", pointer = null;
    let inverted = maskInverted(value, imageName);
    let placement = readOutpaint(value, imageName);
    const originalCrop = [...crop], normalStrokes = structuredClone(strokes);
    let normalInverted = placement ? false : inverted, outpaintStrokes = placement ? strokes : [], outpaintInverted = placement ? inverted : false;
    if (placement) normalStrokes.length = 0;
    const originalCanvas = [...canvasSize];
    let workingCanvas = [...canvasSize], lastPlacement = placement, moving = null;
    if (placement) tool="move";
    const dialog = document.createElement("dialog");
    dialog.className = "donut-mask-dialog";
    dialog.setAttribute("aria-label", "Paint and outpaint");
    const style = document.createElement("style");
    style.textContent = `
        .donut-mask-dialog { box-sizing:border-box; width:min(1100px,95vw); height:min(94vh,1100px); max-height:94vh; padding:22px; border:1px solid #526678; border-radius:14px; background:#14191f; color:#e8edf3; font:14px/1.5 system-ui; color-scheme:dark; }
        .donut-mask-dialog[open] { display:flex; flex-direction:column; overflow:hidden; }
        .donut-mask-dialog > :not(.dm-canvas-wrap) { flex-shrink:0; }
        .donut-mask-dialog::backdrop { background:#000b; }
        .donut-mask-dialog h2 { margin:0 0 5px; font-size:23px; }
        .donut-mask-dialog p { color:#9caebb; margin:0 0 14px; }
        .donut-mask-dialog .dm-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:7px 0; }
        .donut-mask-dialog button { font:inherit; background:#242e38; color:inherit; border:1px solid #526678; border-radius:7px; padding:6px 10px; cursor:pointer; }
        .donut-mask-dialog button[aria-pressed=true],.donut-mask-dialog .dm-apply { background:#285044; border-color:#92e4c7; }
        .donut-mask-dialog button:disabled { opacity:.4; cursor:default; }
        .donut-mask-dialog label { display:flex; align-items:center; gap:8px; }
        .donut-mask-dialog input { accent-color:#92e4c7; }
        .donut-mask-dialog canvas { display:block; max-width:100%; max-height:100%; width:auto; height:auto; margin:auto; cursor:crosshair; touch-action:none; background:#080c10; }
        .donut-mask-dialog .dm-canvas-wrap { position:relative; flex:1; min-height:0; display:flex; align-items:center; justify-content:center; }
        .donut-mask-dialog .dm-cursor { position:absolute; pointer-events:none; border:1.5px solid white; box-shadow:0 0 0 1px #000b; border-radius:50%; box-sizing:border-box; transform:translate(-50%,-50%); }
        .donut-mask-dialog .dm-footer { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-top:15px; }
        .donut-mask-dialog .dm-message { color:#9caebb; }
    `;
    const title = document.createElement("h2"); title.textContent = "Paint & outpaint";
    const help = document.createElement("p"); help.textContent = "Green = edit area · amber = soft seam preview. Outside stays from A. Red = output crop.";
    const toolbar = document.createElement("div"); toolbar.className = "dm-toolbar";
    function action(text, callback) {
        const b = document.createElement("button"); b.type = "button"; b.textContent = text;
        b.addEventListener("click", callback); return b;
    }
    const brush = action("Brush", () => { tool = "brush"; render(); });
    const eraser = action("Erase", () => { tool = "erase"; render(); });
    const rectangle = action("Rectangle", () => { tool = "rectangle"; render(); });
    const size = document.createElement("input"); size.type = "range"; size.min = "1"; size.max = "30"; size.value = "8";
    size.setAttribute("aria-label", "Brush size");
    const sizeLabel = document.createElement("label"); sizeLabel.append("Brush size", size);
    const sizeValue = document.createElement("output"); sizeLabel.append(sizeValue);
    const undo = action("Undo", () => { strokes.pop(); render(); });
    const clear = action("Clear selection", () => { strokes = []; inverted = false; render(); });
    const invert = action("Invert selection", () => { inverted = !inverted; render(); });
    toolbar.append(brush, eraser, rectangle, sizeLabel, undo, clear, invert);
    const seamToolbar = document.createElement("div"); seamToolbar.className = "dm-toolbar";
    const seam = document.createElement("input"); seam.type = "range"; seam.min = "0"; seam.max = "128"; seam.step = "1";
    seam.value = String(Math.max(0, Math.min(128, Number(feather) || 0)));
    seam.setAttribute("aria-label", "Seam width in output pixels");
    const seamValue = document.createElement("output");
    const seamLabel = document.createElement("label"); seamLabel.append("Seam width", seam, seamValue);
    const seamHelp = document.createElement("span"); seamHelp.textContent = "Blends inward · 0 = hard edge";
    seamToolbar.append(seamLabel, seamHelp);
    seam.addEventListener("input", () => render());
    const canvasBar=document.createElement("div"); canvasBar.className="dm-toolbar";
    canvasBar.append(document.createTextNode("Canvas preset"));
    const placementBar=document.createElement("div"); placementBar.className="dm-toolbar";
    const modeBar=document.createElement("div"); modeBar.className="dm-toolbar";
    const mode=document.createElement("input"); mode.type="checkbox"; mode.checked=!!placement;
    const modeLabel=document.createElement("label"); modeLabel.append(mode,"Outpaint · place A inside the canvas");
    modeBar.append(modeLabel);
    const move=action("Move A",()=>{tool="move";render();}); placementBar.append(move);
    const placementInputs={};
    for (const [key,label,min,max,step] of [['scale','Image size',10,100,1],['x','Horizontal',0,100,1],['y','Vertical',0,100,1],['overlap','Overlap',0,128,1]]) {
        const input=document.createElement("input"); input.type="range"; input.min=min;input.max=max;input.step=step;
        input.setAttribute("aria-label",label); input.style.width="100px";
        const labelNode=document.createElement("label"),readout=document.createElement("output");
        labelNode.append(label,input,readout);placementBar.append(labelNode);placementInputs[key]={input,readout};
        input.addEventListener("input",()=>{placement[key]=Number(input.value)/(key==='overlap'?1:100);render();});
    }
    for(const [name,x,y] of [['Left',0,.5],['Center',.5,.5],['Right',1,.5],['Top',.5,0],['Bottom',.5,1]])
        placementBar.append(action(name,()=>{placement.x=x;placement.y=y;render();}));
    function setOutpaint(active) {
        if(active && !placement){normalStrokes.splice(0,normalStrokes.length,...strokes);normalInverted=inverted;
            strokes=outpaintStrokes;inverted=outpaintInverted;placement=lastPlacement || {scale:.65,x:0,y:.5,overlap:16};tool="move";
        } else if(!active && placement){lastPlacement=placement;outpaintStrokes=strokes;outpaintInverted=inverted;strokes=[...normalStrokes];inverted=normalInverted;placement=null;tool="brush";workingCanvas=[...originalCanvas];}
        mode.checked=active;
        configureCanvas();render();
    }
    mode.addEventListener("change",()=>setOutpaint(mode.checked));
    function usePreset(direction, x, y) {
        setOutpaint(true);
        workingCanvas = direction === "current" ? [...originalCanvas]
            : outpaintPresetSize([image.naturalWidth,image.naturalHeight], originalCanvas, direction, Number(pixelGrid));
        placement = {...placement, scale:direction === "current" ? placement.scale : 1, x, y};
        configureCanvas();render();
    }
    for (const [label,direction,x,y] of [
        ["Keep current","current",.5,.5], ["Add right →","horizontal",0,.5],
        ["← Add left","horizontal",1,.5], ["Add below ↓","vertical",.5,0],
        ["↑ Add above","vertical",.5,1],
    ]) {
        const preset=action(label,()=>usePreset(direction,x,y));
        preset.setAttribute("aria-label",`${label} canvas preset`); canvasBar.append(preset);
    }
    const canvas = document.createElement("canvas");
    let scale = Math.min(1, 1400 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.round(image.naturalWidth * scale); canvas.height = Math.round(image.naturalHeight * scale);
    canvas.setAttribute("aria-label", "Image A. Drag to paint the edit selection.");
    const canvasWrap = document.createElement("div"); canvasWrap.className = "dm-canvas-wrap";
    const cursor = document.createElement("div"); cursor.className = "dm-cursor"; cursor.hidden = true;
    cursor.setAttribute("aria-hidden", "true"); canvasWrap.append(canvas, cursor);
    function updateCursor() {
        sizeValue.textContent = `${size.value}%`;
        cursor.hidden = !pointer || tool === "rectangle" || tool === "move";
        if (cursor.hidden) return;
        const rect = canvas.getBoundingClientRect(), wrap = canvasWrap.getBoundingClientRect();
        const diameter = Number(size.value) / 100 * Math.min(canvas.width, canvas.height) * rect.width / canvas.width;
        cursor.style.width = cursor.style.height = `${diameter}px`;
        cursor.style.left = `${pointer[0] - wrap.left}px`; cursor.style.top = `${pointer[1] - wrap.top}px`;
        cursor.style.borderStyle = tool === "erase" ? "dashed" : "solid";
    }
    size.addEventListener("input", updateCursor);
    const mask = document.createElement("canvas"); mask.width = canvas.width; mask.height = canvas.height;
    // Preview in output-crop space: feather is measured in output pixels,
    // independent of source size, editor zoom, or the chosen output aspect.
    let [outputWidth, outputHeight] = outputSize || [crop[2] - crop[0], crop[3] - crop[1]];
    let previewScale = Math.min(1, 1000 / Math.max(outputWidth, outputHeight));
    const croppedMask = document.createElement("canvas");
    croppedMask.width = Math.max(1, Math.round(outputWidth * previewScale));
    croppedMask.height = Math.max(1, Math.round(outputHeight * previewScale));
    const blurredMask = document.createElement("canvas"); blurredMask.width = croppedMask.width; blurredMask.height = croppedMask.height;
    const paddedMask = document.createElement("canvas");
    const footer = document.createElement("div"); footer.className = "dm-footer";
    const message = document.createElement("span"); message.className = "dm-message"; message.setAttribute("role", "status");
    const buttons = document.createElement("div"); buttons.className = "dm-toolbar";
    const cancel = action("Cancel", () => dialog.close());
    const apply = action("Use selection", () => {
        const active = selectionAction(hasSelection()).enabled;
        const saved = active ? JSON.stringify({version:1, image:imageName, strokes, inverted, ...(placement ? {outpaint:placement} : {})}) : "";
        onApply(saved, Number(seam.value), active, active && placement ? [...workingCanvas] : null); dialog.close();
    }); apply.className = "dm-apply";
    buttons.append(cancel, apply); footer.append(message, buttons);
    dialog.append(style, title, help, modeBar, canvasBar, placementBar, toolbar, seamToolbar, canvasWrap, footer); document.body.append(dialog);
    function configureCanvas() {
        const dimensions=placement ? workingCanvas : [image.naturalWidth,image.naturalHeight];
        crop=placement ? [0,0,...workingCanvas] : [...originalCrop];
        scale=Math.min(1,1400/Math.max(...dimensions));
        canvas.width=Math.round(dimensions[0]*scale);canvas.height=Math.round(dimensions[1]*scale);
        mask.width=canvas.width;mask.height=canvas.height;
        [outputWidth,outputHeight]=placement ? workingCanvas : (outputSize || [crop[2]-crop[0],crop[3]-crop[1]]);
        previewScale=Math.min(1,1000/Math.max(outputWidth,outputHeight));
        croppedMask.width=Math.max(1,Math.round(outputWidth*previewScale));croppedMask.height=Math.max(1,Math.round(outputHeight*previewScale));
        blurredMask.width=croppedMask.width;blurredMask.height=croppedMask.height;
    }
    function previewPlacement() {
        return placement ? {rect:outpaintRect([image.naturalWidth,image.naturalHeight],workingCanvas,placement).map(v=>v*scale),overlap:placement.overlap*scale} : null;
    }
    function hasSelection() {
        const pixels = mask.getContext("2d").getImageData(0, 0, mask.width, mask.height).data;
        const [x1,y1,x2,y2]=crop;
        for (let y=Math.max(0,Math.floor(y1*scale)); y<Math.min(mask.height,Math.ceil(y2*scale)); y++) {
            for (let x=Math.max(0,Math.floor(x1*scale)); x<Math.min(mask.width,Math.ceil(x2*scale)); x++) {
                if (pixels[(y*mask.width+x)*4+3]) return true;
            }
        }
        return false;
    }
    function render() {
        const ctx = canvas.getContext("2d"), m = mask.getContext("2d");
        const positioned=previewPlacement();
        if(positioned) drawOutpaintBase(ctx,image,positioned.rect,canvas.width,canvas.height);
        else ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        drawMask(m, mask.width, mask.height, strokes, inverted, positioned);
        placementBar.hidden=!placement;
        placementBar.style.display=placement?'flex':'none';
        move.setAttribute('aria-pressed',String(tool==='move'));
        if(placement) for(const [key,{input,readout}] of Object.entries(placementInputs)) {
            input.value=String(placement[key]*(key==='overlap'?1:100));readout.textContent=`${Math.round(Number(input.value))}${key==='overlap'?' px':'%'}`;
        }
        invert.setAttribute("aria-pressed", String(inverted));
        help.textContent = inverted
            ? "Inverted: painted strokes protect A; green changes. Amber = soft seam preview · red = output crop."
            : "Green = edit area · amber = soft seam preview. Outside stays from A. Red = output crop.";
        if(placement) help.textContent=`${workingCanvas[0]} × ${workingCanvas[1]} · ${(workingCanvas[0]*workingCanvas[1]/(1024*1024)).toFixed(2)} MP total. Move or shrink A; green fills automatically. Overlap allows changes along A’s edges.`;
        ctx.globalAlpha = .5; ctx.drawImage(mask, 0, 0); ctx.globalAlpha = 1;
        const [x1, y1, x2, y2] = crop;
        seamValue.textContent = `${seam.value} px`;
        if (Number(seam.value) > 0) {
            const c = croppedMask.getContext("2d"), b = blurredMask.getContext("2d");
            c.clearRect(0, 0, croppedMask.width, croppedMask.height);
            c.drawImage(mask, x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale,
                0, 0, croppedMask.width, croppedMask.height);
            const radius = Number(seam.value) * previewScale, pad = Math.ceil(radius * 3);
            const w = croppedMask.width, h = croppedMask.height;
            paddedMask.width = w + pad * 2; paddedMask.height = h + pad * 2;
            const p = paddedMask.getContext("2d");
            // Extend boundary pixels so touching the crop edge does not invent
            // a seam against transparent pixels outside the output image.
            const xs = [[0, 1, 0, pad], [0, w, pad, w], [w - 1, 1, pad + w, pad]];
            const ys = [[0, 1, 0, pad], [0, h, pad, h], [h - 1, 1, pad + h, pad]];
            for (const [sx, sw, dx, dw] of xs) for (const [sy, sh, dy, dh] of ys) {
                p.drawImage(croppedMask, sx, sy, sw, sh, dx, dy, dw, dh);
            }
            b.clearRect(0, 0, blurredMask.width, blurredMask.height);
            b.filter = `blur(${radius}px)`;
            b.drawImage(paddedMask, -pad, -pad); b.filter = "none";
            const hard = c.getImageData(0, 0, croppedMask.width, croppedMask.height);
            const soft = b.getImageData(0, 0, croppedMask.width, croppedMask.height);
            // Show how much blending reduces the painted mask, clipped inward
            // just like the backend's min(hard mask, Gaussian-blurred mask).
            const placedRect=placement && outpaintRect([image.naturalWidth,image.naturalHeight],workingCanvas,placement);
            for (let i = 0; i < hard.data.length; i += 4) {
                const blend = Math.max(0, hard.data[i + 3] - soft.data[i + 3]);
                hard.data[i] = 255; hard.data[i + 1] = 190; hard.data[i + 2] = 65;
                hard.data[i + 3] = Math.min(230, blend * 3);
                if (placement) {
                    const [rx,ry,rw,rh]=placedRect;
                    const px=(i/4%croppedMask.width)/previewScale,py=Math.floor(i/4/croppedMask.width)/previewScale;
                    if(px<rx || px>=rx+rw || py<ry || py>=ry+rh) hard.data[i+3]=0;
                }
            }
            c.putImageData(hard, 0, 0);
            ctx.drawImage(croppedMask, x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
        }
        ctx.strokeStyle = "#ff6c75"; ctx.lineWidth = 2;
        ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
        if(positioned) {ctx.strokeStyle='#92e4c7';ctx.setLineDash([6,4]);ctx.strokeRect(...positioned.rect);ctx.setLineDash([]);}
        brush.setAttribute("aria-pressed", String(tool === "brush")); eraser.setAttribute("aria-pressed", String(tool === "erase"));
        rectangle.setAttribute("aria-pressed", String(tool === "rectangle"));
        size.disabled = tool === "rectangle" || tool === "move"; updateCursor();
        undo.disabled = !strokes.length; clear.disabled = !strokes.length && !inverted;
        const action = selectionAction(hasSelection());
        apply.textContent = action.label;
        apply.disabled = false;
        message.textContent = action.message;
    }
    function point(event) {
        const rect = canvas.getBoundingClientRect();
        return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
    }
    canvas.addEventListener("pointerdown", event => {
        if (event.button !== 0) return;
        event.preventDefault(); canvas.setPointerCapture(event.pointerId);
        pointer = [event.clientX, event.clientY];
        if(tool==='move' && placement){moving={point:point(event),x:placement.x,y:placement.y};return;}
        current = {erase:tool === "erase", size:Number(size.value) / 100, points:[point(event)]};
        if (tool === "rectangle") { current.shape = "rectangle"; current.points.push(point(event)); }
        strokes.push(current); render();
    });
    canvas.addEventListener("pointermove", event => {
        pointer = [event.clientX, event.clientY]; updateCursor();
        if(moving && placement){
            const p=point(event),[,,w,h]=outpaintRect([image.naturalWidth,image.naturalHeight],workingCanvas,placement);
            placement.x=Math.max(0,Math.min(1,moving.x+(p[0]-moving.point[0])*workingCanvas[0]/Math.max(1,workingCanvas[0]-w)));
            placement.y=Math.max(0,Math.min(1,moving.y+(p[1]-moving.point[1])*workingCanvas[1]/Math.max(1,workingCanvas[1]-h)));
            render();return;
        }
        if (current) {
            if (current.shape === "rectangle") current.points[1] = point(event);
            else current.points.push(point(event));
            render();
        }
    });
    canvas.addEventListener("pointerenter", event => { pointer = [event.clientX, event.clientY]; updateCursor(); });
    canvas.addEventListener("pointerleave", () => { pointer = null; updateCursor(); });
    canvas.addEventListener("pointerup", () => { current = null; moving = null; });
    canvas.addEventListener("pointercancel", () => { current = null; moving = null; });
    dialog.addEventListener("close", () => dialog.remove(), {once:true});
    configureCanvas(); dialog.showModal(); render();
    return () => { dialog.close(); dialog.remove(); };
}

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

export function drawMask(ctx, width, height, strokes, inverted = false) {
    ctx.clearRect(0, 0, width, height);
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
}

export function openInpaintEditor({image, imageName, value, crop, outputSize, feather = 8, onApply}) {
    let strokes = structuredClone(readMask(value, imageName)), current = null, tool = "brush", pointer = null;
    let inverted = maskInverted(value, imageName);
    const dialog = document.createElement("dialog");
    dialog.className = "donut-mask-dialog";
    dialog.setAttribute("aria-label", "Paint the area to edit");
    const style = document.createElement("style");
    style.textContent = `
        .donut-mask-dialog { box-sizing:border-box; width:min(1100px,95vw); height:min(94vh,1100px); max-height:94vh; padding:22px; border:1px solid #526678; border-radius:14px; background:#14191f; color:#e8edf3; font:14px/1.5 system-ui; color-scheme:dark; }
        .donut-mask-dialog[open] { display:flex; flex-direction:column; overflow:hidden; }
        .donut-mask-dialog > :not(.dm-canvas-wrap) { flex-shrink:0; }
        .donut-mask-dialog::backdrop { background:#000b; }
        .donut-mask-dialog h2 { margin:0 0 5px; font-size:23px; }
        .donut-mask-dialog p { color:#9caebb; margin:0 0 14px; }
        .donut-mask-dialog .dm-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:12px 0; }
        .donut-mask-dialog button { font:inherit; background:#242e38; color:inherit; border:1px solid #526678; border-radius:7px; padding:7px 14px; cursor:pointer; }
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
    const title = document.createElement("h2"); title.textContent = "Paint the area to edit";
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
    const clear = action("Clear", () => { strokes = []; inverted = false; render(); });
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
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 1400 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.round(image.naturalWidth * scale); canvas.height = Math.round(image.naturalHeight * scale);
    canvas.setAttribute("aria-label", "Image A. Drag to paint the edit selection.");
    const canvasWrap = document.createElement("div"); canvasWrap.className = "dm-canvas-wrap";
    const cursor = document.createElement("div"); cursor.className = "dm-cursor"; cursor.hidden = true;
    cursor.setAttribute("aria-hidden", "true"); canvasWrap.append(canvas, cursor);
    function updateCursor() {
        sizeValue.textContent = `${size.value}%`;
        cursor.hidden = !pointer || tool === "rectangle";
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
    const [outputWidth, outputHeight] = outputSize || [crop[2] - crop[0], crop[3] - crop[1]];
    const previewScale = Math.min(1, 1000 / Math.max(outputWidth, outputHeight));
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
        onApply(JSON.stringify({version:1, image:imageName, strokes, inverted}), Number(seam.value)); dialog.close();
    }); apply.className = "dm-apply";
    buttons.append(cancel, apply); footer.append(message, buttons);
    dialog.append(style, title, help, toolbar, seamToolbar, canvasWrap, footer); document.body.append(dialog);
    function render() {
        const ctx = canvas.getContext("2d"), m = mask.getContext("2d");
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        drawMask(m, mask.width, mask.height, strokes, inverted);
        invert.setAttribute("aria-pressed", String(inverted));
        help.textContent = inverted
            ? "Inverted: painted strokes protect A; green changes. Amber = soft seam preview · red = output crop."
            : "Green = edit area · amber = soft seam preview. Outside stays from A. Red = output crop.";
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
            for (let i = 0; i < hard.data.length; i += 4) {
                const blend = Math.max(0, hard.data[i + 3] - soft.data[i + 3]);
                hard.data[i] = 255; hard.data[i + 1] = 190; hard.data[i + 2] = 65;
                hard.data[i + 3] = Math.min(230, blend * 3);
            }
            c.putImageData(hard, 0, 0);
            ctx.drawImage(croppedMask, x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
        }
        ctx.strokeStyle = "#ff6c75"; ctx.lineWidth = 2;
        ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
        brush.setAttribute("aria-pressed", String(tool === "brush")); eraser.setAttribute("aria-pressed", String(tool === "erase"));
        rectangle.setAttribute("aria-pressed", String(tool === "rectangle"));
        size.disabled = tool === "rectangle"; updateCursor();
        undo.disabled = clear.disabled = !strokes.length;
        // Check painted pixels, including erasures, inside the actual output crop.
        const pixels = m.getImageData(0, 0, mask.width, mask.height).data;
        let selected = false;
        for (let y = Math.max(0, Math.floor(y1 * scale)); y < Math.min(mask.height, Math.ceil(y2 * scale)) && !selected; y++) {
            for (let x = Math.max(0, Math.floor(x1 * scale)); x < Math.min(mask.width, Math.ceil(x2 * scale)); x++) {
                if (pixels[(y * mask.width + x) * 4 + 3]) { selected = true; break; }
            }
        }
        apply.disabled = !selected;
        message.textContent = selected ? "Selection ready · describe the change in Prompts." : "Paint inside the red frame to select an area.";
    }
    function point(event) {
        const rect = canvas.getBoundingClientRect();
        return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
    }
    canvas.addEventListener("pointerdown", event => {
        if (event.button !== 0) return;
        event.preventDefault(); canvas.setPointerCapture(event.pointerId);
        pointer = [event.clientX, event.clientY];
        current = {erase:tool === "erase", size:Number(size.value) / 100, points:[point(event)]};
        if (tool === "rectangle") { current.shape = "rectangle"; current.points.push(point(event)); }
        strokes.push(current); render();
    });
    canvas.addEventListener("pointermove", event => {
        pointer = [event.clientX, event.clientY]; updateCursor();
        if (current) {
            if (current.shape === "rectangle") current.points[1] = point(event);
            else current.points.push(point(event));
            render();
        }
    });
    canvas.addEventListener("pointerenter", event => { pointer = [event.clientX, event.clientY]; updateCursor(); });
    canvas.addEventListener("pointerleave", () => { pointer = null; updateCursor(); });
    canvas.addEventListener("pointerup", () => { current = null; });
    canvas.addEventListener("pointercancel", () => { current = null; });
    dialog.addEventListener("close", () => dialog.remove(), {once:true});
    dialog.showModal(); render();
    return () => { dialog.close(); dialog.remove(); };
}

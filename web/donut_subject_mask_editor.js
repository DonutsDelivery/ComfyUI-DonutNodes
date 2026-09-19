// B's raster-mask editor. Keep operations in source-normalized coordinates;
// display a reduced preview, but apply them to the full-resolution mask on save.
export function openSubjectMaskEditor({image, maskImage = null, onApply}) {
    const dialog = document.createElement('dialog');
    dialog.className = 'donut-subject-mask-editor';
    dialog.setAttribute('aria-label', 'Refine Reference B subject mask');
    const style = document.createElement('style');
    style.textContent = `.donut-subject-mask-editor {width:min(1100px,95vw);height:min(900px,92vh);box-sizing:border-box;padding:20px;border:1px solid #536676;border-radius:12px;background:#14191f;color:#e8edf3;font:14px/1.4 system-ui;color-scheme:dark}
        .donut-subject-mask-editor[open]{display:flex;flex-direction:column;gap:10px}.donut-subject-mask-editor::backdrop{background:#000b}
        .donut-subject-mask-editor h2,.donut-subject-mask-editor p{margin:0}.donut-subject-mask-editor .ds-toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
        .donut-subject-mask-editor button{font:inherit;padding:6px 12px;background:#25313b;border:1px solid #536676;color:inherit;border-radius:6px;cursor:pointer}
        .donut-subject-mask-editor button[aria-pressed=true]{border-color:#92e4c7;background:#285044}.donut-subject-mask-editor button:disabled{opacity:.5;cursor:wait}
        .donut-subject-mask-editor .ds-view{flex:1;min-height:0;display:flex;align-items:center;justify-content:center}
        .donut-subject-mask-editor canvas{max-width:100%;max-height:100%;width:auto;height:auto;touch-action:none;cursor:crosshair}
        .donut-subject-mask-editor .ds-status{min-height:20px}.donut-subject-mask-editor :focus-visible{outline:2px solid #92e4c7;outline-offset:2px}`;
    const title = document.createElement('h2'); title.textContent = 'Refine subject mask · B';
    const help = document.createElement('p'); help.textContent = 'Paint adds the subject; Erase removes background. White keeps, black removes. The original reference is unchanged.';
    const toolbar = document.createElement('div'); toolbar.className = 'ds-toolbar';
    const footer = document.createElement('div'); footer.className = 'ds-toolbar';
    const status = document.createElement('div'); status.className = 'ds-status'; status.setAttribute('role', 'status');
    const view = document.createElement('div'); view.className = 'ds-view';
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 1200 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale)); canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    canvas.setAttribute('aria-label', 'Reference B. Drag to paint or erase the subject mask.'); view.append(canvas);
    const mask = document.createElement('canvas'); mask.width = canvas.width; mask.height = canvas.height;
    const foreground = document.createElement('canvas'); foreground.width = canvas.width; foreground.height = canvas.height;
    let actions = [], current = null, tool = 'paint', showMask = false, showSource = !maskImage, closed = false;
    function button(text, callback, parent = toolbar) {
        const value = document.createElement('button'); value.type = 'button'; value.textContent = text;
        value.addEventListener('click', callback); parent.append(value); return value;
    }
    const paint = button('Paint', () => {tool = 'paint'; render();});
    const erase = button('Erase', () => {tool = 'erase'; render();});
    const rectangle = button('Rectangle', () => {tool = 'rectangle'; render();});
    const size = document.createElement('input'); size.type = 'range'; size.min = '0.2'; size.max = '20'; size.step = '0.1'; size.value = '3';
    size.setAttribute('aria-label', 'Subject mask brush size');
    const sizeLabel = document.createElement('label'); sizeLabel.append('Brush size ', size); toolbar.append(sizeLabel);
    const undo = button('Undo', () => {actions.pop(); render();});
    button('Invert', () => {actions.push({type:'invert'}); render();});
    button('Clear', () => {actions.push({type:'clear'}); render();});
    const sourceToggle = button('Show source', () => {showSource = !showSource; showMask = false; render();});
    const toggle = button('Show mask', () => {showMask = !showMask; render();});
    button('Cancel', () => dialog.close(), footer);
    const apply = button('Use subject mask', async () => {
        apply.disabled = true; status.textContent = 'Saving full-resolution subject mask…';
        try {
            const output = document.createElement('canvas'); output.width = image.naturalWidth; output.height = image.naturalHeight;
            drawMask(output.getContext('2d'), output.width, output.height);
            const blob = await new Promise(resolve => output.toBlob(resolve, 'image/png'));
            if (!blob) throw new Error('The browser could not encode the mask.');
            if (closed) return;
            await onApply(blob);
            if (!closed) dialog.close();
        } catch (error) { if (!closed) status.textContent = error.message; }
        finally { apply.disabled = false; }
    }, footer);
    function drawMask(ctx, width, height) {
        ctx.globalCompositeOperation = 'source-over'; ctx.fillStyle = '#000'; ctx.fillRect(0, 0, width, height);
        if (maskImage) ctx.drawImage(maskImage, 0, 0, width, height);
        ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        for (const action of [...actions, ...(current ? [current] : [])]) {
            if (action.type === 'clear') {ctx.fillStyle = '#000'; ctx.fillRect(0, 0, width, height); continue;}
            if (action.type === 'invert') {
                const data = ctx.getImageData(0, 0, width, height);
                for (let i = 0; i < data.data.length; i += 4) data.data[i] = data.data[i+1] = data.data[i+2] = 255 - data.data[i];
                ctx.putImageData(data, 0, 0); continue;
            }
            ctx.fillStyle = ctx.strokeStyle = action.type === 'erase' ? '#000' : '#fff';
            const points = action.points.map(([x,y]) => [x * width, y * height]);
            if (action.type === 'rectangle') {
                const [a,b = a] = points; ctx.fillRect(Math.min(a[0],b[0]), Math.min(a[1],b[1]), Math.abs(a[0]-b[0]), Math.abs(a[1]-b[1])); continue;
            }
            const radius = action.size * Math.min(width, height) / 2;
            ctx.lineWidth = radius * 2; ctx.beginPath();
            points.forEach(([x,y], index) => index ? ctx.lineTo(x,y) : ctx.moveTo(x,y)); ctx.stroke();
            ctx.beginPath(); ctx.arc(points[0][0], points[0][1], radius, 0, Math.PI * 2); ctx.fill();
        }
    }
    function render() {
        if (closed) return;
        paint.setAttribute('aria-pressed', String(tool === 'paint')); erase.setAttribute('aria-pressed', String(tool === 'erase'));
        rectangle.setAttribute('aria-pressed', String(tool === 'rectangle')); toggle.setAttribute('aria-pressed', String(showMask));
        sourceToggle.setAttribute('aria-pressed', String(showSource && !showMask));
        undo.disabled = !actions.length;
        const m = mask.getContext('2d'), ctx = canvas.getContext('2d'); drawMask(m, mask.width, mask.height);
        if (showMask) {ctx.drawImage(mask, 0, 0); return;}
        const alpha = m.getImageData(0,0,mask.width,mask.height);
        for (let i=0;i<alpha.data.length;i+=4) {alpha.data[i+3]=alpha.data[i]; alpha.data[i]=alpha.data[i+1]=alpha.data[i+2]=255;}
        m.putImageData(alpha,0,0);
        if (showSource) {
            // Keep the original visible while painting from an empty mask.
            ctx.drawImage(image,0,0,canvas.width,canvas.height);
            ctx.globalAlpha=.35; ctx.drawImage(mask,0,0); ctx.globalAlpha=1; return;
        }
        const f = foreground.getContext('2d'); f.globalCompositeOperation='source-over'; f.clearRect(0,0,canvas.width,canvas.height);
        f.drawImage(image,0,0,canvas.width,canvas.height); f.globalCompositeOperation='destination-in'; f.drawImage(mask,0,0);
        ctx.fillStyle='#808080'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.drawImage(foreground,0,0);
    }
    function point(event) {
        const rect = canvas.getBoundingClientRect();
        return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)), Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];
    }
    canvas.addEventListener('pointerdown', event => {
        if (event.button !== 0 || apply.disabled) return;
        event.preventDefault(); event.stopPropagation(); canvas.setPointerCapture(event.pointerId);
        current = {type:tool, size:Number(size.value)/100, points:[point(event)]}; render();
    });
    canvas.addEventListener('pointermove', event => {
        if (!current) return;
        if (current.type === 'rectangle') current.points[1] = point(event); else current.points.push(point(event)); render();
    });
    canvas.addEventListener('pointerup', () => {if (current) {actions.push(current); current=null; render();}});
    canvas.addEventListener('pointercancel', () => {current=null; render();});
    dialog.append(style,title,help,toolbar,view,status,footer); document.body.append(dialog);
    dialog.addEventListener('close', () => {closed=true; dialog.remove();}, {once:true});
    dialog.showModal(); render();
    return () => {if (!closed) dialog.close();};
}

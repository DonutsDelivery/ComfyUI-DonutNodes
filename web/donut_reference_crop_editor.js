import {ASPECTS,parseCrop,cropBox,aspectCrop,moveCrop,resizeCrop} from './donut_reference_crop_geometry.js';

// Shared editor for Edit Studio and Reference Guidance. Only Apply commits.
export function openReferenceCropEditor({image,imageName,value,onApply}) {
    const size=[image.naturalWidth,image.naturalHeight];
    let state;
    try {state=parseCrop(value,imageName,size);} catch {state=parseCrop('',imageName,size);}
    let drag=null;
    const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
    const dialog=el('dialog');dialog.className='donut-crop-dialog';dialog.setAttribute('aria-label','Crop reference image');
    const style=el('style');style.textContent=`
.donut-crop-dialog{box-sizing:border-box;width:min(1050px,96vw);max-height:95vh;padding:20px;border:1px solid #566673;border-radius:12px;background:#171c23;color:#eee;font:14px/1.5 system-ui;color-scheme:dark}
.donut-crop-dialog[open]{display:flex;flex-direction:column;gap:12px}
.donut-crop-dialog::backdrop{background:#000b}
.donut-crop-dialog h2,.donut-crop-dialog p{margin:0}
.donut-crop-dialog .dc-toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
.donut-crop-dialog select,.donut-crop-dialog button{font:inherit;background:#27313c;color:inherit;border:1px solid #697b88;border-radius:5px;padding:7px}
.donut-crop-dialog .dc-view{min-height:0;overflow:hidden;display:flex;justify-content:center}
.donut-crop-dialog canvas{max-width:100%;max-height:62vh;object-fit:contain;touch-action:none;cursor:crosshair}
.donut-crop-dialog [role=status]{color:#c5d6e2}
.donut-crop-dialog .dc-footer{display:flex;justify-content:space-between;gap:12px}
`;
    const title=el('h2','Crop this reference'),help=el('p','Drag inside to move. Drag a corner to resize. Output size will not change this selection.');
    const toolbar=el('div');toolbar.className='dc-toolbar';
    const select=el('select');select.setAttribute('aria-label','Crop aspect ratio');
    for(const value of ['Original','Free',...Object.keys(ASPECTS)]) {const o=el('option',value);o.value=value;select.append(o);}
    select.value=state.aspect;
    function action(text,fn){const b=el('button',text);b.type='button';b.onclick=fn;return b;}
    select.onchange=()=>{state.bounds=aspectCrop(state.bounds,select.value,size);state.aspect=select.value;draw();};
    const reset=action('Reset to full image',()=>{state=parseCrop('',imageName,size);select.value=state.aspect;draw();});
    toolbar.append(el('span','Aspect'),select,reset);
    const canvas=el('canvas'),view=el('div');view.className='dc-view';
    const scale=Math.min(1,1400/Math.max(...size));canvas.width=Math.max(1,Math.round(size[0]*scale));canvas.height=Math.max(1,Math.round(size[1]*scale));
    canvas.setAttribute('aria-label','Reference crop preview. Drag selection or corners.');view.append(canvas);
    const status=el('output');status.setAttribute('role','status');
    const footer=el('div');footer.className='dc-footer';
    const close=()=>{if(dialog.open)dialog.close();dialog.remove();};
    const cancel=action('Cancel',close);
    const apply=action('Use crop',()=>{
        try {parseCrop(state,imageName,size);onApply(JSON.stringify(state));close();}
        catch(error){status.textContent=error.message;}
    });
    footer.append(cancel,apply);dialog.append(style,title,help,toolbar,view,status,footer);document.body.append(dialog);
    function draw(){
        const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
        ctx.clearRect(0,0,w,h);ctx.drawImage(image,0,0,w,h);
        const [x1,y1,x2,y2]=state.bounds.map((v,i)=>v*(i%2?h:w));
        ctx.fillStyle='#0009';ctx.fillRect(0,0,w,y1);ctx.fillRect(0,y2,w,h-y2);ctx.fillRect(0,y1,x1,y2-y1);ctx.fillRect(x2,y1,w-x2,y2-y1);
        ctx.strokeStyle='#ff7580';ctx.lineWidth=2;ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.fillStyle='#fff';
        for(const [x,y] of [[x1,y1],[x2,y1],[x1,y2],[x2,y2]]) ctx.fillRect(x-5,y-5,10,10);
        try {const b=cropBox(state,imageName,size);status.textContent=`Source crop: ${b[2]-b[0]} × ${b[3]-b[1]} px · source ${size[0]} × ${size[1]} px`;apply.disabled=false;}
        catch(error){status.textContent=error.message;apply.disabled=true;}
    }
    function point(event){const r=canvas.getBoundingClientRect();return [(event.clientX-r.left)/r.width,(event.clientY-r.top)/r.height];}
    canvas.onpointerdown=event=>{
        if(event.button!==0)return;
        event.preventDefault();const p=point(event),r=canvas.getBoundingClientRect(),[x1,y1,x2,y2]=state.bounds;
        let corner=null;
        for(const [name,x,y] of [['nw',x1,y1],['ne',x2,y1],['sw',x1,y2],['se',x2,y2]]) {
            if(Math.hypot((p[0]-x)*r.width,(p[1]-y)*r.height)<16){corner=name;break;}
        }
        if(!corner && !(p[0]>=x1&&p[0]<=x2&&p[1]>=y1&&p[1]<=y2))return;
        drag={point:p,bounds:[...state.bounds],corner};canvas.setPointerCapture(event.pointerId);
    };
    canvas.onpointermove=event=>{
        if(!drag)return;const p=point(event);
        state.bounds=drag.corner?resizeCrop(drag.bounds,drag.corner,p,state.aspect,size):moveCrop(drag.bounds,p[0]-drag.point[0],p[1]-drag.point[1]);draw();
    };
    canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
    dialog.addEventListener('close',()=>dialog.remove(),{once:true});dialog.showModal();draw();
    return close;
}

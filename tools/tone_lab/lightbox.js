"use strict";
const lb = {
  index:-1, view:'wipe', split:0.5, peek:false, spaceHeld:false,
  scale:1, cx:0, cy:0, fit:true, cache:null, job:0, frame:0, busy:false,
  opener:null, home:null, oldOverflow:'', oldScrollX:0, oldScrollY:0, enteredFullscreen:false, pointers:new Map(), gesture:null,
};
const lbDialog = $('lightbox');
const lbViewport = $('lbViewport');

const MAX_PREVIEW_PIXELS = 16000000;
const MAX_PREVIEW_SIDE = 8192;

function lbOpen(){ return lbDialog.open && lb.index >= 0 && !!images[lb.index]; }
function lbItem(){ return images[lb.index]; }
function lbGeometry(){
  const rect=lbViewport.getBoundingClientRect();
  return {width:rect.width,height:rect.height,paneWidth:lb.view==='side'?rect.width/2:rect.width,rect};
}
function lbFitScale(){
  const item=lbItem(), box=lbGeometry();
  if(!item) return 1;
  return Math.max(0.00001,Math.min((box.paneWidth-24)/item.width,(box.height-24)/item.height,1/(window.devicePixelRatio||1)));
}
function lbClampCenter(){
  if(!lbOpen()) return;
  const item=lbItem(),box=lbGeometry();
  const halfW=box.paneWidth/(2*lb.scale),halfH=box.height/(2*lb.scale);
  lb.cx=halfW*2>=item.width?item.width/2:clamp(lb.cx,halfW,item.width-halfW);
  lb.cy=halfH*2>=item.height?item.height/2:clamp(lb.cy,halfH,item.height-halfH);
}
function lbFit(){
  if(!lbOpen()) return;
  lb.fit=true;lb.scale=lbFitScale();lb.cx=lbItem().width/2;lb.cy=lbItem().height/2;
  lbDrawSoon();
}
function lbZoomLimits(){
  return [Math.min(lbFitScale()/4,0.01),8/(window.devicePixelRatio||1)];
}
function lbImagePoint(clientX,clientY){
  const {rect,paneWidth}=lbGeometry();
  let x=clientX-rect.left;
  if(lb.view==='side' && x>=paneWidth) x-=paneWidth;
  return {x:lb.cx+(x-paneWidth/2)/lb.scale,y:lb.cy+(clientY-rect.top-rect.height/2)/lb.scale,
          localX:x,localY:clientY-rect.top};
}
function lbZoomTo(scale,clientX,clientY){
  if(!lbOpen()) return;
  const box=lbGeometry();
  if(clientX===undefined){clientX=box.rect.left+box.paneWidth/2;clientY=box.rect.top+box.height/2;}
  const point=lbImagePoint(clientX,clientY),[min,max]=lbZoomLimits();
  lb.scale=clamp(scale,min,max);lb.fit=false;
  lb.cx=point.x-(point.localX-box.paneWidth/2)/lb.scale;
  lb.cy=point.y-(point.localY-box.height/2)/lb.scale;
  lbClampCenter();lbDrawSoon();
}
function lbActual(){
  // At 100%, one original source pixel occupies one device pixel.
  lbZoomTo(1/(window.devicePixelRatio||1));
}
function lbSetView(view){
  lb.view=view;
  lbDialog.querySelectorAll('[data-lb-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lbView===view)));
  if(lb.fit) lbFit(); else {lbClampCenter();lbDrawSoon();}
}
function lbSetControls(show){
  $('lbPanel').hidden=!show;
  $('lbBody').classList.toggle('controls-hidden',!show);
  $('lbSettings').setAttribute('aria-expanded',String(show));
  if(lb.fit) lbFit(); else {lbClampCenter();lbDrawSoon();}
}
function lbSetPeek(value){lb.peek=value;lbDrawSoon();}
function lbSetBusy(value,message='Updating corrected preview…'){
  lb.busy=value;
  $('lbBusy').hidden=!value;
  $('lbBusy').textContent=message;
  lbViewport.setAttribute('aria-busy',String(value));
  lbDialog.querySelectorAll('[data-lb-rate]').forEach(b=>b.disabled=value||optimizerBusy||!!manualPreview);
  $('saveTarget').disabled=value||optimizerBusy;$('acceptOriginal').disabled=value||optimizerBusy;
}
function lbDiscardCache(){
  lb.job++;
  if(lb.cache){lb.cache.original.width=1;lb.cache.corrected.width=1;lb.cache=null;}
}
function openLightbox(index,opener=null){
  if(!images[index]) return;
  if(!lbDialog.open){
    lb.opener=opener||document.activeElement;
    lb.home=true;
    lb.oldOverflow=document.body.style.overflow;
    lb.oldScrollX=window.scrollX;lb.oldScrollY=window.scrollY;

    document.body.style.overflow='hidden';
    lbDialog.showModal();
    lbSetControls(window.innerWidth>900);
  }
  lbShowImage(index);
  lbViewport.focus({preventScroll:true});updateOptimizerStatus();
}
function lbShowImage(index){
  if(!images[index]) return;
  lbDiscardCache();lb.index=index;lb.peek=false;lb.spaceHeld=false;manualPreview=null;
  lb.pointers.clear();lb.gesture=null;
  const item=lbItem();
  $('lbName').textContent=item.file.name;$('lbName').title=item.key;
  $('lbSize').textContent=`${item.width} × ${item.height} · full-size comparison`;
  $('lbCounter').textContent=`${index+1} / ${images.length}`;
  $('lbPrev').disabled=index===0;$('lbNext').disabled=index===images.length-1;
  lbSetBusy(true,'Preparing full-size preview…');lbSyncRating();updateSidebar();lbUpdateMetrics();lbFit();
  const token=lb.job;
  requestAnimationFrame(async()=>{
    if(!lbOpen()||lb.job!==token) return;
    try{
      const scale=Math.min(1,MAX_PREVIEW_SIDE/item.width,MAX_PREVIEW_SIDE/item.height,
        Math.sqrt(MAX_PREVIEW_PIXELS/(item.width*item.height)));
      const w=Math.max(1,Math.floor(item.width*scale)),h=Math.max(1,Math.floor(item.height*scale));
      const original=document.createElement('canvas');original.width=w;original.height=h;
      const ctx=original.getContext('2d',{willReadFrequently:true});
      if(!ctx) throw new Error('Canvas is unavailable in this browser.');
      const decoded=await imageFromUrl(item.url);
      if(!lbOpen()||lb.job!==token)return;
      ctx.drawImage(decoded,0,0,w,h);
      const input=ctx.getImageData(0,0,w,h).data;
      const corrected=document.createElement('canvas');corrected.width=w;corrected.height=h;
      lb.cache={original,corrected,input,output:ctx.createImageData(w,h),key:null,pendingKey:null,ready:false,
        width:w,height:h,reduced:scale<1};
      $('lbSize').textContent=`${item.width} × ${item.height} · `+(scale<1
        ?`preview limited to ${w} × ${h} (16 MP / 8192 px memory limit)`
        :'native-resolution preview');
      lbRequestCorrection();
    }catch(error){
      lbSetBusy(true,'Cannot prepare this image: '+error.message);
      console.warn('Lightbox preparation failed',error);
    }
  });
}
function lbNavigate(delta){
  const next=lb.index+delta;
  if(next>=0&&next<images.length) lbShowImage(next);
}
function lbClose(){
  if(lbDialog.open){lbDialog.close();lbCleanup();}
}
function lbCleanup(){
  if(lb.home===null && lb.index<0) return;
  if(lb.enteredFullscreen && document.fullscreenElement) document.exitFullscreen().catch(()=>{});
  lb.enteredFullscreen=false;
  lbDiscardCache();lb.index=-1;manualPreview=null;lb.peek=false;lb.spaceHeld=false;lb.pointers.clear();lb.gesture=null;
  lb.home=null;
  document.body.style.overflow=lb.oldOverflow;
  window.scrollTo(lb.oldScrollX,lb.oldScrollY);
  if(lb.opener?.isConnected) lb.opener.focus({preventScroll:true});
  lb.opener=null;updateAll();
}
function lbUpdateMetrics(){
  if(!lbOpen())return;
  const item=lbItem(),p=previewTone(item),source=manualPreview?'MANUAL TARGET':modelLabel();
  $('lbMetrics').textContent=`${source} · gamma ${fmt(E.slider(p.gamma),1)} (exp ${fmt(p.gamma,3)}) · gain ${fmt((p.gain-1)*100,1)}% · mean ${fmt(item.stats.mean)} · contrast σ ${fmt(item.stats.std)} · p95 ${fmt(item.stats.p95)}`;
}
function lbSyncRating(){
  const item=lbItem();
  lbDialog.querySelectorAll('[data-lb-rate]').forEach(b=>{
    const active=!!item && item.rating===b.dataset.lbRate;
    b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));
  });
}
function nextUnratedIndex(from){
  if(!images.length) return -1;
  for(let step=1;step<=images.length;step++){
    const i=(from+step)%images.length;
    if(!images[i].rating) return i;
  }
  return -1;
}

function lbRequestCorrection(){
  if(!lbOpen()) return;
  lbUpdateMetrics();
  const cache=lb.cache;if(!cache) return;
  const item=lbItem(),gamma=previewTone(item).gamma,gain=previewTone(item).gain;
  const key=gamma+'|'+gain;
  if(cache.key===key){
    if(cache.pendingKey!==null){lb.job++;cache.pendingKey=null;}
    lbSetBusy(false);lbDrawSoon();return;
  }
  if(cache.pendingKey===key) return;
  cache.pendingKey=key;
  const token=++lb.job;
  lbSetBusy(true);
  // The 256-entry LUT matches Uint8ClampedArray rounding in the original gallery.
  const lut=new Uint8ClampedArray(256);
  for(let x=0;x<256;x++) lut[x]=clamp(Math.pow(x/255,gamma)*gain,0,1)*255;
  const src=cache.input,dst=cache.output.data;
  const chunk=262144*4;
  (async()=>{
    try{
      // Yield between chunks; a new image or adjustment invalidates this job.
      for(let start=0;start<src.length;start+=chunk){
        if(!lbOpen()||lb.job!==token||lb.cache!==cache) return;
        const end=Math.min(src.length,start+chunk);
        for(let p=start;p<end;p+=4){
          dst[p]=lut[src[p]];dst[p+1]=lut[src[p+1]];dst[p+2]=lut[src[p+2]];dst[p+3]=src[p+3];
        }
        if(end<src.length) await new Promise(resolve=>setTimeout(resolve,0));
      }
      if(!lbOpen()||lb.job!==token||lb.cache!==cache) return;
      cache.corrected.getContext('2d').putImageData(cache.output,0,0);
      cache.key=key;cache.pendingKey=null;cache.ready=true;
      lbSetBusy(false);lbDrawSoon();
    }catch(error){
      if(lb.job===token){cache.pendingKey=null;lbSetBusy(true,'Correction failed: '+error.message);}
      console.warn('Lightbox correction failed',error);
    }
  })();
}
function lbDrawSoon(){
  if(!lbOpen()||lb.frame) return;
  lb.frame=requestAnimationFrame(()=>{lb.frame=0;if(lbOpen()) lbDraw();});
}
function lbDraw(){
  const item=lbItem(),box=lbGeometry();
  if(!item || box.width<1 || box.height<1) return;
  const canvas=$('lbCanvas'),dpr=window.devicePixelRatio||1;
  const width=Math.round(box.width*dpr),height=Math.round(box.height*dpr);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  const ctx=canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,box.width,box.height);
  ctx.fillStyle='#111315';ctx.fillRect(0,0,box.width,box.height);
  ctx.imageSmoothingEnabled=lb.scale*dpr<1;ctx.imageSmoothingQuality='high';
  const original=lb.cache?.original||item.img;
  const corrected=lb.cache?.ready?lb.cache.corrected:original;
  function draw(source,paneX,paneWidth){
    ctx.save();ctx.beginPath();ctx.rect(paneX,0,paneWidth,box.height);ctx.clip();
    ctx.drawImage(source,paneX+paneWidth/2-lb.cx*lb.scale,box.height/2-lb.cy*lb.scale,
      item.width*lb.scale,item.height*lb.scale);ctx.restore();
  }
  const left=$('lbLeftLabel'),right=$('lbRightLabel'),wipe=$('lbWipe');
  left.hidden=false;right.hidden=false;wipe.hidden=true;
  if(lb.view==='side'){
    draw(original,0,box.paneWidth);draw(lb.peek?original:corrected,box.paneWidth,box.paneWidth);
    ctx.fillStyle='#444b54';ctx.fillRect(box.paneWidth-0.5,0,1,box.height);
    left.textContent='Original';right.textContent=lb.peek?'Original · held':(manualPreview?'Manual target':'Model output');
  }else if(lb.peek||lb.view==='original'){
    draw(original,0,box.width);left.textContent=lb.peek?'Original · held':'Original';right.hidden=true;
  }else if(lb.view==='corrected'){
    draw(corrected,0,box.width);left.hidden=true;right.textContent=manualPreview?'Manual target':'Model output';
  }else{
    draw(original,0,box.width);
    ctx.save();ctx.beginPath();ctx.rect(box.width*lb.split,0,box.width*(1-lb.split),box.height);ctx.clip();
    draw(corrected,0,box.width);ctx.restore();
    left.textContent='Original';right.textContent=manualPreview?'Manual target':'Model output';wipe.hidden=false;
    wipe.style.left=`${lb.split*100}%`;wipe.setAttribute('aria-valuenow',String(Math.round(lb.split*100)));
  }
  $('lbZoom').textContent=`${Math.round(lb.scale*dpr*100)}%`;
  $('lbFit').setAttribute('aria-pressed',String(lb.fit));
}
function lbBeginGesture(){
  const pts=[...lb.pointers.values()];
  if(pts.length===1){lb.gesture={kind:'pan',x:pts[0].x,y:pts[0].y,cx:lb.cx,cy:lb.cy};}
  else if(pts.length>=2){
    const midX=(pts[0].x+pts[1].x)/2,midY=(pts[0].y+pts[1].y)/2;
    lb.gesture={kind:'pinch',scale:lb.scale,dist:Math.hypot(pts[1].x-pts[0].x,pts[1].y-pts[0].y)||1,
      anchor:lbImagePoint(midX,midY)};
  }else lb.gesture=null;
}
function lbToggleFullscreen(){
  if(document.fullscreenElement){document.exitFullscreen().catch(()=>{});}
  else if(document.documentElement.requestFullscreen){
    // Dialogs cannot themselves be fullscreen elements. Fullscreen the page;
    // the modal dialog remains in the browser's top layer above it.
    document.documentElement.requestFullscreen().then(()=>{lb.enteredFullscreen=true;}).catch(()=>{
      $('lbFullscreen').textContent='Fullscreen unavailable';
      $('lbFullscreen').title='This browser did not allow fullscreen. The full-window lightbox still works.';
    });
  }
}
function setupLightbox(){
  lbDialog.addEventListener('close',()=>{if(!lbDialog.open)lbCleanup();});
  lbDialog.addEventListener('cancel',e=>{e.preventDefault();lbClose();});
  $('lbClose').addEventListener('click',lbClose);
  $('lbPrev').addEventListener('click',()=>lbNavigate(-1));
  $('lbNext').addEventListener('click',()=>lbNavigate(1));
  $('lbFit').addEventListener('click',lbFit);
  $('lbActual').addEventListener('click',lbActual);
  $('lbZoomIn').addEventListener('click',()=>lbZoomTo(lb.scale*1.25));
  $('lbZoomOut').addEventListener('click',()=>lbZoomTo(lb.scale/1.25));
  $('lbSettings').addEventListener('click',()=>lbSetControls($('lbPanel').hidden));
  $('lbFullscreen').addEventListener('click',lbToggleFullscreen);
  if(!document.documentElement.requestFullscreen){$('lbFullscreen').disabled=true;}
  document.addEventListener('fullscreenchange',()=>{
    if(!document.fullscreenElement) lb.enteredFullscreen=false;
    $('lbFullscreen').textContent=document.fullscreenElement?'Exit fullscreen':'Fullscreen';
    if(lb.fit) lbFit();else lbDrawSoon();
  });
  lbDialog.querySelectorAll('[data-lb-view]').forEach(b=>b.addEventListener('click',()=>lbSetView(b.dataset.lbView)));
  lbDialog.querySelectorAll('[data-lb-rate]').forEach(b=>b.addEventListener('click',()=>{
    if(!lb.busy&&!manualPreview) setImageRating(lb.index,b.dataset.lbRate);
  }));
  const peek=$('lbPeek');
  peek.addEventListener('pointerdown',e=>{e.preventDefault();peek.setPointerCapture(e.pointerId);lbSetPeek(true);});
  ['pointerup','pointercancel','lostpointercapture'].forEach(type=>peek.addEventListener(type,()=>lbSetPeek(false)));
  const wipe=$('lbWipe');
  function moveWipe(e){
    const rect=lbViewport.getBoundingClientRect();lb.split=clamp((e.clientX-rect.left)/rect.width,0,1);lbDrawSoon();
  }
  wipe.addEventListener('pointerdown',e=>{e.preventDefault();e.stopPropagation();wipe.setPointerCapture(e.pointerId);moveWipe(e);});
  wipe.addEventListener('pointermove',e=>{if(wipe.hasPointerCapture(e.pointerId)){e.stopPropagation();moveWipe(e);}});
  wipe.addEventListener('pointerup',e=>{e.stopPropagation();if(wipe.hasPointerCapture(e.pointerId)) wipe.releasePointerCapture(e.pointerId);});
  wipe.addEventListener('keydown',e=>{
    if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){
      e.preventDefault();e.stopPropagation();
      lb.split=e.key==='Home'?0:e.key==='End'?1:clamp(lb.split+(e.key==='ArrowRight'?1:-1)*(e.shiftKey?.05:.01),0,1);
      lbDrawSoon();
    }
  });
  lbViewport.addEventListener('wheel',e=>{
    e.preventDefault();const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?lbGeometry().height:1);
    lbZoomTo(lb.scale*Math.exp(clamp(-delta*.002,-.6,.6)),e.clientX,e.clientY);
  },{passive:false});
  lbViewport.addEventListener('dblclick',e=>{
    if(e.target.closest('#lbWipe')) return;
    if(lb.fit) lbZoomTo(1/(window.devicePixelRatio||1),e.clientX,e.clientY);else lbFit();
  });
  lbViewport.addEventListener('pointerdown',e=>{
    if(e.target.closest('#lbWipe')||e.button>0) return;
    e.preventDefault();lbViewport.focus({preventScroll:true});
    lbViewport.setPointerCapture(e.pointerId);lb.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
    lbViewport.classList.add('dragging');lbBeginGesture();
  });
  lbViewport.addEventListener('pointermove',e=>{
    if(!lb.pointers.has(e.pointerId)) return;
    lb.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});const g=lb.gesture;if(!g) return;
    if(g.kind==='pan'&&lb.pointers.size===1){
      if(Math.abs(e.clientX-g.x)+Math.abs(e.clientY-g.y)<2) return;
      lb.fit=false;lb.cx=g.cx-(e.clientX-g.x)/lb.scale;lb.cy=g.cy-(e.clientY-g.y)/lb.scale;
    }else if(g.kind==='pinch'&&lb.pointers.size>=2){
      const [p1,p2]=[...lb.pointers.values()],box=lbGeometry(),[min,max]=lbZoomLimits();
      lb.fit=false;lb.scale=clamp(g.scale*Math.hypot(p2.x-p1.x,p2.y-p1.y)/g.dist,min,max);
      const midX=(p1.x+p2.x)/2,midY=(p1.y+p2.y)/2;
      let localX=midX-box.rect.left;if(lb.view==='side'&&localX>=box.paneWidth) localX-=box.paneWidth;
      lb.cx=g.anchor.x-(localX-box.paneWidth/2)/lb.scale;
      lb.cy=g.anchor.y-(midY-box.rect.top-box.height/2)/lb.scale;
    }
    lbClampCenter();lbDrawSoon();
  });
  function releasePointer(e){
    lb.pointers.delete(e.pointerId);lbBeginGesture();
    if(!lb.pointers.size) lbViewport.classList.remove('dragging');
  }
  ['pointerup','pointercancel','lostpointercapture'].forEach(type=>lbViewport.addEventListener(type,releasePointer));
  new ResizeObserver(()=>{if(!lbOpen())return;if(lb.fit)lbFit();else{lbClampCenter();lbDrawSoon();}}).observe(lbViewport);
  document.addEventListener('keydown',e=>{
    if(!lbOpen()||$('roundReport').open||optimizerBusy) return;
    // Keep text, select, and range editing native. The divider owns its arrows.
    if(e.target.closest('input,select,textarea,[contenteditable="true"]')) return;
    if(e.target.closest('#lbWipe')&&['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
    if(e.ctrlKey||e.metaKey||e.altKey) return;
    const key=e.key.toLowerCase();
    if(e.code==='Space'){
      e.preventDefault();e.stopPropagation();lb.spaceHeld=true;lbSetPeek(true);return;
    }
    const actions={
      arrowleft:()=>lbNavigate(-1),arrowright:()=>lbNavigate(1),
      '0':lbFit,'1':lbActual,'+':()=>lbZoomTo(lb.scale*1.25),'=':()=>lbZoomTo(lb.scale*1.25),
      '-':()=>lbZoomTo(lb.scale/1.25),'w':()=>lbSetView('wipe'),'s':()=>lbSetView('side'),
      'o':()=>lbSetView('original'),'c':()=>lbSetView('corrected'),
      'f':lbToggleFullscreen,'a':()=>lbSetControls($('lbPanel').hidden),
      '2':()=>!lb.busy&&!manualPreview&&setImageRating(lb.index,'too_dark'),
      '3':()=>!lb.busy&&!manualPreview&&setImageRating(lb.index,'good'),
      '4':()=>!lb.busy&&!manualPreview&&setImageRating(lb.index,'too_light'),
      '5':()=>!lb.busy&&!manualPreview&&setImageRating(lb.index,'skip'),
    };
    if(actions[key]){e.preventDefault();if(!e.repeat||['+','=','-'].includes(key))actions[key]();}
  });
  document.addEventListener('keyup',e=>{
    if(e.code==='Space'&&lb.spaceHeld){e.preventDefault();e.stopPropagation();lb.spaceHeld=false;lbSetPeek(false);}
  });
  window.addEventListener('blur',()=>{lb.spaceHeld=false;lbSetPeek(false);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){lb.spaceHeld=false;lbSetPeek(false);}});
}

// Source-visible test hooks; the predictor itself accepts features and model only.
window.ToneLabEngine=E;
window.ToneLabDebug={getState:()=>state,getImages:()=>images,predictItem:i=>modelTone(images[i]),previewItem:i=>previewTone(images[i]),train:trainModel,save:saveNow,loadFiles,importFile,exportSession,exportModel};
setupLightbox();setupApp();
(async()=>{await storageInit();controlsFromState();renderGallery();updateOptimizerStatus();status('Ready. Choose images; shared weights and saved examples are restored when available.');})();

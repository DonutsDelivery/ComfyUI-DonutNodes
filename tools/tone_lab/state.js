"use strict";
const E=buildFeatureEngine();
const $=id=>document.getElementById(id),clamp=E.clip,fmt=(n,d=3)=>Number.isFinite(n)?n.toFixed(d):'—';
const exponentToSlider=E.slider;
const META=E.analyze(new Uint8ClampedArray([0,0,0,255]),1,1);
const DEFAULT_SETTINGS={epochs:400,regularization:.01,step:.025,roundSize:20,autoAdvance:true,autoTrain:false};
const fresh=()=>({version:4,tool:'Donut Tone Lab Feature Learner',schema:E.SCHEMA,model:null,legacyBaseline:null,records:{},groups:{},legacy:[],round:1,votes:{},reports:[],settings:{...DEFAULT_SETTINGS},undo:null,importedModelUnverified:false});
let state=fresh(),images=[],manualPreview=null,optimizerBusy=false,loading=false,worker=null,trainToken=0,loadGeneration=0,modelToken=1,autoTimer=null,saveTimer=null,galleryObserver=null,storage=null,storageOK=false,lastStatus='Ready.',lastVoteTime=0;
const visibleCards=new Set();
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function warn(s){$('storageWarning').hidden=false;$('storageWarning').textContent=s;}
function status(s){lastStatus=s;$('status').textContent=s;updateOptimizerStatus();}
async function storageInit(){
 try{storage=await new Promise((resolve,reject)=>{const req=indexedDB.open('DonutToneLabFeatureV4',1);req.onupgradeneeded=()=>req.result.createObjectStore('state');req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);});
 const stored=await new Promise((resolve,reject)=>{const tx=storage.transaction('state','readonly'),r=tx.objectStore('state').get('current');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});if(stored)state=validateSession(stored);storageOK=true;
 }catch(e){try{const s=localStorage.getItem('donutFeatureLabV4');if(s)state=validateSession(JSON.parse(s));storageOK=true;warn('IndexedDB is unavailable. Browser storage is limited; export a session regularly.');}catch(err){warn('Browser storage is unavailable. Your work is held in this tab only. Export a session before closing.');}}
}
function saveSoon(){clearTimeout(saveTimer);saveTimer=setTimeout(saveNow,350);}
async function saveNow(){
 clearTimeout(saveTimer);
 try{const snap=structuredClone(state);if(storage)await new Promise((resolve,reject)=>{const tx=storage.transaction('state','readwrite');tx.objectStore('state').put(snap,'current');tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);});else localStorage.setItem('donutFeatureLabV4',JSON.stringify(snap));storageOK=true;}
 catch(e){storageOK=false;warn('Could not save locally: '+e.message+'. Export session now to protect your work.');}
}
function validateStats(s){if(!s||s.schema!==E.SCHEMA||!Array.isArray(s.features)||s.features.length!==META.features.length||!s.features.every(Number.isFinite))throw Error('Missing or incompatible image features.');for(const k of ['rgbHist','midHist'])if(!Array.isArray(s[k])||s[k].length!==256||!s[k].every(x=>Number.isFinite(x)&&x>=0))throw Error('Invalid tone histogram.');return s;}
function validateCurve(a){if(!a||!Number.isFinite(a.gamma)||!Number.isFinite(a.gain)||a.gamma<1/3-1e-8||a.gamma>3+1e-8||a.gain<.8-1e-8||a.gain>1.25+1e-8)throw Error('Target is outside this model’s gamma/gain range.');return a;}
function validateSession(s){
 if(s.version!==4||s.schema!==E.SCHEMA||!s.records||typeof s.records!=='object')throw Error('Not a compatible v4 session.');
 if(Object.keys(s.records).length>5000)throw Error('Session has too many stored images.');
 const out={...fresh(),...s,settings:{...DEFAULT_SETTINGS,...s.settings}};
 for(const [key,r]of Object.entries(out.records)){validateStats(r.stats);if(typeof r.id!=='string'||typeof r.group!=='string'||r.id!==key)throw Error('Invalid image record.');if(r.anchor)validateCurve(r.anchor);r.events=(r.events||[]).filter(e=>['too_dark','too_light','good','target','original'].includes(e.rating)&&Number.isFinite(e.gamma)&&e.gamma>=1/3&&e.gamma<=3&&Number.isFinite(e.gain)&&e.gain>=.8&&e.gain<=1.25).slice(-100);}
 if(out.model){out.model=E.validatedModel(out.model);if(out.model.featureCount!==META.features.length||JSON.stringify(out.model.names)!==JSON.stringify(META.names))throw Error('Model feature layout does not match this analyzer.');}
 out.groups=Object.fromEntries(Object.entries(out.groups||{}).filter(([k,v])=>['train','validation','ignore'].includes(v)));
 out.votes=out.votes||{};out.reports=Array.isArray(out.reports)?out.reports:[];out.legacy=Array.isArray(out.legacy)?out.legacy:[];out.round=Math.max(1,Math.floor(out.round)||1);return out;
}
function groupFromName(name){return name.replace(/\.[^.]+$/,'').toLowerCase().replace(/(?:[ _-]+(?:gamma[ _-]*(?:corrected|adjusted)|corrected|edited|copy)(?:[ _-]*\d+)?)$/i,'').trim()||'untitled';}
function roleFor(group){return state.groups[group]||(E.hash('scene:'+group)%5===0?'validation':'train');}
function recordFor(item){return state.records[item.id];}
function trainingRecords(){return Object.values(state.records).map(r=>({...r,role:roleFor(r.group)}));}
function legacyTone(s,p){
 const mean=s.mean<=p.idealMean?1:1+((s.mean-p.idealMean)/Math.max(.006,p.brightRef-p.idealMean))**2,r=s.ratio;
 const ratio=r<=p.targetRatio||r<=0?1:r>=.999999?3:Math.max(1,Math.log(p.targetRatio)/Math.log(r));
 let g=p.mode==='mean'?mean:p.mode==='ratio'?ratio:p.mode==='hybrid'?mean*p.hybridWeight+ratio*(1-p.hybridWeight):3**(-p.fixedSlider/100);
 g=clamp(1+(g-1)*p.strength,1,p.maxGamma);const d=clamp(p.highlightTarget/Math.max(s.p95**g,1e-6),1,p.maxGain);
 const b=!p.highlightFix||p.strength===0||g<=1.00000001?1:1+(d-1)*Math.min(1,(g-1)/.08)*Math.min(1,p.strength);
 return {gamma:g,gain:b,noOpScore:null,legacy:true,noop:g===1&&b===1,coverage:0};
}
function modelTone(item){
 if(item.predToken===modelToken&&item.prediction)return item.prediction;
 const p=item.stats.sampleCount<16?{gamma:1,gain:1,noop:true,noOpScore:null,unusable:true}:state.model?E.predict(item.stats.features,state.model):state.legacyBaseline?legacyTone(item.stats,state.legacyBaseline):E.predict(item.stats.features,null);
 item.predToken=modelToken;item.prediction=p;return p;
}
function previewTone(item){return manualPreview&&lbOpen()&&lbItem()?.id===item.id?manualPreview:modelTone(item);}
function gammaFor(item){return modelTone(item).gamma;}function gainFor(item){return modelTone(item).gain;}
function modelLabel(){return state.model?'FEATURE MODEL r'+state.model.revision:state.legacyBaseline?'LEGACY BASELINE':'UNTRAINED · ORIGINAL';}
function updateOptimizerStatus(){
 const votes=images.filter(i=>state.votes[i.id]?.round===state.round),n=votes.length,goal=state.settings.roundSize?Math.min(images.length,state.settings.roundSize):images.length;
 const good=votes.filter(i=>state.votes[i.id].rating==='good').length,targets=votes.filter(i=>['target','original'].includes(state.votes[i.id].rating)).length;
 $('roundSummary').textContent=`Round ${state.round} · ${n}/${goal||0} reviewed · ${good} Good · ${targets} explicit targets`;
 $('lbRoundStatus').textContent=optimizerBusy?lastStatus:`Round ${state.round} · ${n}/${goal||0} reviewed · ${modelLabel()}`;
 const all=trainingRecords(),labeled=all.filter(r=>E.labelFor(r,state.settings.step)),tr=labeled.filter(r=>r.role==='train'),va=labeled.filter(r=>r.role==='validation');
 $('dataCount').innerHTML=`${images.length} <small>images loaded</small>`;
 $('recordCount').textContent=`${labeled.length} labeled examples saved across ${new Set(labeled.map(r=>r.folder)).size} folders · ${all.length} analyzed images`;
 $('splitCount').textContent=`${tr.length} training / ${va.length} validation examples · scene-group split`;
 const pending=state.legacy.filter(r=>!r.migrated).length;$('pendingLegacy').hidden=!pending;$('pendingLegacy').textContent=`${pending} legacy examples await source images for full feature extraction.`;
 $('modelName').textContent=state.model?`Feature model r${state.model.revision}`:state.legacyBaseline?'Legacy preview':'Untrained';
 $('modelDescription').textContent=`${META.features.length} image measurements → 16 → 8 → gamma, gain, no-change score`;
 $('modelStatus').textContent=state.model?`${state.model.training?.imageCount??'Imported'} training examples. ${state.importedModelUnverified?'Imported weights: validation independence is not verified.':'Validation is excluded from fitting.'}`:state.legacyBaseline?'The v3 rule is only a temporary preview. Train builds the new feature model from recovered examples.':'No correction until you train. “Good” approves this unchanged result.';
 $('onboard').hidden=!!state.model;
 for(const id of ['train','lbOptimize'])$(id).disabled=optimizerBusy||loading||tr.length<4;
 $('undoModel').disabled=optimizerBusy||!state.undo;$('viewReport').disabled=!state.reports.length;$('lbReportButton').disabled=!state.reports.length;
 $('exportModel').disabled=!state.model;$('trainProgress').hidden=!optimizerBusy;$('cancelTrain').hidden=!optimizerBusy;
 if(lbOpen())lbSyncRating();
}
function controlsFromState(){for(const [id,key]of [['epochs','epochs'],['regularization','regularization'],['toneStep','step'],['roundSize','roundSize']])$(id).value=state.settings[key];for(const k of ['autoTrain','autoAdvance'])$(k).checked=state.settings[k];}
function refreshAll(){modelToken++;renderGallery();updateOptimizerStatus();if(lbOpen()){manualPreview=null;updateSidebar();lbRequestCorrection();}}
function updateAll(){for(const i of visibleCards)renderCorrected(i);updateOptimizerStatus();}
function imageFromUrl(url){return new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(Error('Browser cannot decode this image.'));i.src=url;});}
function isImageFile(f){return /\.(png|jpe?g|webp|avif|bmp)$/i.test(f.name);}
async function fileIdentity(f,providedBytes=null){
 const bytes=providedBytes||await f.arrayBuffer();
 if(globalThis.crypto?.subtle){const digest=await crypto.subtle.digest('SHA-256',bytes);return 'sha256:'+Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');}
 // Fallback for restricted file:// environments. Used for bookkeeping, never prediction.
 const a=new Uint8Array(bytes);let h1=2166136261,h2=2246822519;for(const b of a){h1=Math.imul(h1^b,16777619);h2=Math.imul(h2^b,3266489917);}return `fallback:${a.length}:${h1>>>0}:${h2>>>0}`;
}
function isAnimated(buffer,name){
 const b=new Uint8Array(buffer),v=new DataView(buffer),text=(p,n)=>String.fromCharCode(...b.subarray(p,p+n));
 if(/\.png$/i.test(name)&&b.length>8){let p=8;while(p+12<=b.length){const len=v.getUint32(p),type=text(p+4,4);if(type==='acTL')return true;if(type==='IEND')break;p+=12+len;}}
 if(/\.webp$/i.test(name)&&b.length>12){let p=12;while(p+8<=b.length){const type=text(p,4),len=v.getUint32(p+4,true);if(type==='ANIM'||type==='ANMF')return true;p+=8+len+(len%2);}}
 if(/\.avif$/i.test(name)&&b.length>16&&text(4,4)==='ftyp'){const end=Math.min(v.getUint32(0),b.length);for(let p=8;p+4<=end;p+=4)if(text(p,4)==='avis')return true;}
 return false;
}
async function analyzeFile(file){
 const url=URL.createObjectURL(file);
 try{const bytes=await file.arrayBuffer();if(isAnimated(bytes,file.name))throw Error("Animated images are not supported; use a still frame.");const img=await imageFromUrl(url),width=img.naturalWidth,height=img.naturalHeight;if(!width||!height)throw Error('Empty image.');
 const scale=Math.min(1,256/Math.max(width,height)),w=Math.max(1,Math.round(width*scale)),h=Math.max(1,Math.round(height*scale));
 const c=document.createElement('canvas');c.width=w;c.height=h;const ctx=c.getContext('2d',{willReadFrequently:true,colorSpace:'srgb'});ctx.drawImage(img,0,0,w,h);
 const frame=ctx.getImageData(0,0,w,h),stats=E.analyze(frame.data,w,h),id=await fileIdentity(file,bytes),key=file.webkitRelativePath||file.name;
 let record=state.records[id];
 if(!record){record={id,name:file.name,key,size:file.size,width,height,folder:key.includes('/')?key.split('/')[0]:'Loose images',group:groupFromName(file.name),stats,anchor:null,events:[]};state.records[id]=record;migrateLegacy(record);}
 else record.stats=stats; // Versioned source reanalysis, never output pixels.
 const vote=state.votes[id],rating=vote?.round===state.round?vote.rating:'';
 return {file,url,img:c,width,height,stats,id,key,preview:frame,thumb:c.toDataURL('image/png'),rating};
 }catch(e){URL.revokeObjectURL(url);throw e;}
}
async function loadFiles(fileList){
 if(optimizerBusy){status('Cancel training before loading another folder.');return;}
 const files=[...fileList].filter(isImageFile).sort((a,b)=>(a.webkitRelativePath||a.name).localeCompare(b.webkitRelativePath||b.name,undefined,{numeric:true}));
 if(!files.length){status('Choose PNG, JPEG, WebP, AVIF or BMP images. Animated formats are excluded.');return;}
 if(files.length>2000){status('Choose at most 2,000 images per load. Previous work is unchanged.');return;}
 clearTimeout(autoTimer);loading=true;const generation=++loadGeneration;if(lbDialog.open)lbClose();galleryObserver?.disconnect();visibleCards.clear();images.forEach(x=>URL.revokeObjectURL(x.url));images=[];
 $('gallery').innerHTML='<div class="empty">Analyzing histograms, colour and spatial features…</div>';let failed=[],duplicates=0;const ids=new Set();
 for(let i=0;i<files.length;i++){
  if(generation!==loadGeneration)return;
  try{const item=await analyzeFile(files[i]);if(generation!==loadGeneration){URL.revokeObjectURL(item.url);return;}if(ids.has(item.id)){URL.revokeObjectURL(item.url);duplicates++;}else{ids.add(item.id);images.push(item);}}
  catch(e){failed.push(files[i].name+': '+e.message);}
  if(i%4===0){status(`Analyzing ${i+1}/${files.length} · full feature extraction stays local.`);await new Promise(r=>setTimeout(r,0));}
 }
 loading=false;$('fileErrors').textContent=failed.length?`${failed.length} decode failures. `+failed.slice(0,3).join(' · '):'';
 saveSoon();renderGallery();status(`${images.length} images ready. ${duplicates?duplicates+' byte-identical duplicates omitted. ':''}Global model and all prior examples were kept. Click a preview to inspect.`);
}
function migrateLegacy(r){
 const match=state.legacy.find(x=>!x.migrated&&x.name===r.name&&(!x.size||x.size===r.size)&&(!x.width||x.width===r.width)&&(!x.height||x.height===r.height)&&(!x.stats||Math.abs(x.stats.mean-r.stats.mean)<.025));
 if(!match)return;
 const usable=(match.events||[]).filter(e=>e.gamma>=1/3&&e.gamma<=3&&e.gain>=.8&&e.gain<=1.25);r.events=usable;
 const approved=usable.find(e=>e.rating==='good');if(approved)r.anchor={gamma:approved.gamma,gain:approved.gain,source:'legacy_good',time:approved.time};
 match.migrated=r.id;
}
function recordVisible(item){const role=roleFor(recordFor(item).group),filter=$('galleryFilter').value;return filter==='all'||filter==='unrated'&&!item.rating||filter===role||filter==='targets'&&!!recordFor(item).anchor;}

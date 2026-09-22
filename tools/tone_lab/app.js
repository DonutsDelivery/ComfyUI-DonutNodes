"use strict";
function renderGallery(){
 galleryObserver?.disconnect();visibleCards.clear();const g=$('gallery');g.textContent='';
 if(!images.length){g.innerHTML='<div class="empty">Choose a folder to view and teach its images. Stored examples remain available for training.</div>';return;}
 galleryObserver=new IntersectionObserver(entries=>{for(const e of entries){const i=Number(e.target.dataset.index);if(e.isIntersecting){visibleCards.add(i);renderCorrected(i);}else visibleCards.delete(i);}},{rootMargin:'250px'});
 let shown=0;
 images.forEach((item,i)=>{
   if(!recordVisible(item))return;shown++;const r=recordFor(item),role=roleFor(r.group),p=modelTone(item),card=document.createElement('article');card.className='card'+(role==='validation'?' heldout':'');card.dataset.index=i;
   const caption=state.model?'Feature model':state.legacyBaseline?'Legacy baseline':'Untrained · unchanged';
   card.innerHTML=`<div class="cardhead"><div class="filename" title="${esc(item.key)}">${esc(item.file.name)}</div><span class="role-badge role-${role}">${role}</span><button class="inspect" type="button">Inspect ↗</button></div><div class="pair" tabindex="0" role="button" aria-label="Inspect ${esc(item.file.name)}"><figure><img src="${item.thumb}" alt="Original" loading="lazy"><figcaption>Original</figcaption></figure><figure><canvas class="corrected"></canvas><figcaption>${caption}</figcaption></figure></div><div class="metrics"><span>Gamma <b>${fmt(E.slider(p.gamma),1)}</b> · exp ${fmt(p.gamma)}</span><span>Gain <b>${fmt((p.gain-1)*100,1)}%</b></span><span>${r.anchor?'Saved target: '+r.anchor.source:'No exact target'}</span><span>${p.noop?'No change':p.coverage>.2?'Outside training range':''}${item.stats.flat?' · flat tones':''}</span></div><div class="judge"><span class="hint">Rate the global model result:</span><button class="too-dark" data-rate="too_dark">Too dark</button><button class="good" data-rate="good">Good</button><button class="too-light" data-rate="too_light">Too bright</button><button class="skip" data-rate="skip">Skip</button></div>`;
   card.querySelector('.inspect').onclick=e=>openLightbox(i,e.currentTarget);const pair=card.querySelector('.pair');pair.onclick=()=>openLightbox(i,pair);pair.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openLightbox(i,pair);}};
   for(const b of card.querySelectorAll('[data-rate]')){b.classList.toggle('active',b.dataset.rate===item.rating);b.disabled=optimizerBusy;b.onclick=()=>setImageRating(i,b.dataset.rate);}
   g.appendChild(card);galleryObserver.observe(card);
 });if(!shown)g.innerHTML='<div class="empty">No images match this filter.</div>';
}
function renderCorrected(i){
 const item=images[i],card=$('gallery').querySelector(`.card[data-index="${i}"]`);if(!item||!card)return;
 const can=card.querySelector('canvas.corrected'),f=item.preview,p=modelTone(item);can.width=f.width;can.height=f.height;
 const ctx=can.getContext('2d',{colorSpace:'srgb'}),out=ctx.createImageData(f.width,f.height),lut=new Uint8ClampedArray(256);
 for(let k=0;k<256;k++)lut[k]=clamp((k/255)**p.gamma*p.gain,0,1)*255;
 for(let k=0;k<f.data.length;k+=4){out.data[k]=lut[f.data[k]];out.data[k+1]=lut[f.data[k+1]];out.data[k+2]=lut[f.data[k+2]];out.data[k+3]=f.data[k+3];}ctx.putImageData(out,0,0);
}
function nextUnratedIndex(from){for(let j=1;j<=images.length;j++){const i=(from+j)%images.length;if(!images[i].rating)return i;}return -1;}
function afterVote(index){
 saveSoon();renderGallery();updateOptimizerStatus();if(lbOpen())updateSidebar();
 const reviewed=images.filter(i=>i.rating).length,goal=state.settings.roundSize?Math.min(state.settings.roundSize,images.length):images.length;
 if(state.settings.autoTrain&&reviewed>=goal&&goal){clearTimeout(autoTimer);autoTimer=setTimeout(trainModel,450);}
 else if(state.settings.autoAdvance&&lbOpen()){const next=nextUnratedIndex(index);if(next>=0)setTimeout(()=>{if(lbOpen()&&!optimizerBusy)lbShowImage(next);},130);else status('Batch reviewed. Train + next round will fit the one shared model.');}
}
function setImageRating(index,rating){
 if(optimizerBusy||loading||manualPreview||!images[index]||lbOpen()&&lb.busy)return;
 if(Date.now()-lastVoteTime<140)return;lastVoteTime=Date.now();
 const item=images[index],r=recordFor(item),p=modelTone(item);
 if(r.stats.sampleCount<16&&rating!=='skip'){status('Too few opaque analysis pixels. Exclude or skip this image.');return;}
 const event={rating,gamma:p.gamma,gain:p.gain,time:new Date().toISOString(),modelRevision:state.model?.revision||0,round:state.round};
 if(rating!=='skip'){
   r.events.push(event);r.events=r.events.slice(-100);
   if(rating==='good'&&!r.anchor)r.anchor={gamma:p.gamma,gain:p.gain,source:'good',time:event.time};
 }
 state.votes[item.id]={...event,round:state.round};item.rating=rating;
 status(rating==='good'?'Approval recorded. Existing exact targets remain fixed.':rating==='skip'?'Skipped; this vote is not training data.':r.anchor?'Vote recorded. This example still fits its saved target; use Fine-tune to explicitly revise that target.':'Directional feedback saved for the global model.');afterVote(index);
}
function manualFrom(g,b){return {gamma:clamp(g,1/3,3),gain:clamp(b,.8,1.25)};}
function updateManualControls(){if(!manualPreview)return;$('manualGamma').value=E.slider(manualPreview.gamma);$('manualGammaN').value=+E.slider(manualPreview.gamma).toFixed(4);$('manualGain').value=(manualPreview.gain-1)*100;$('manualGainN').value=+((manualPreview.gain-1)*100).toFixed(4);}
function beginManual(tone){if(!lbOpen()||optimizerBusy)return;manualPreview=manualFrom(tone.gamma,tone.gain);updateManualControls();updateSidebar();lbRequestCorrection();}
function returnToModel(){manualPreview=null;updateSidebar();lbRequestCorrection();}
function saveTarget(kind='manual'){
 if(!lbOpen()||optimizerBusy||lb.busy)return;const item=lbItem(),r=recordFor(item);if(r.stats.sampleCount<16){status('Not enough opaque pixels to train.');return;}
 const p=kind==='original'?{gamma:1,gain:1}:manualPreview;if(!p)return;
 r.anchor={gamma:p.gamma,gain:p.gain,source:kind,time:new Date().toISOString()};
 const rating=kind==='original'?'original':'target';r.events.push({...r.anchor,rating,round:state.round});r.events=r.events.slice(-100);state.votes[item.id]={rating,round:state.round,gamma:p.gamma,gain:p.gain};item.rating=rating;
 manualPreview=null;updateSidebar();lbRequestCorrection();status('Target saved for learning. The preview has returned to the global model—not your target. Train to update every prediction.');afterVote(lb.index);
}
function updateSidebar(){
 if(!lbOpen())return;const item=lbItem(),r=recordFor(item),p=modelTone(item),m=!!manualPreview;
 $('previewBadge').textContent=m?'MANUAL TARGET PREVIEW':'MODEL OUTPUT';$('previewBadge').classList.toggle('manual',m);$('targetEditor').hidden=!m;$('useModel').hidden=!m;$('editTarget').hidden=m;
 $('predictionSource').textContent=m?'This explicit target preview is separate from model inference.':'Pixels and one model only. Rating history is not an input.';
 $('modelReadout').textContent=`${modelLabel()} · gamma ${fmt(E.slider(p.gamma),1)} · gain ${fmt((p.gain-1)*100,1)}%`+(p.noOpScore===null?'':` · no-change score ${fmt(p.noOpScore,2)} (uncalibrated)`)+(p.coverage>.2?' · many measurements outside training range':'');
 $('anchorInfo').textContent=r.anchor?`Saved ${r.anchor.source} target: gamma ${fmt(E.slider(r.anchor.gamma),1)}, gain ${fmt((r.anchor.gain-1)*100,1)}%. Good votes do not replace it.`:'No approved target. Fine-tune to teach both sliders precisely.';
 $('viewTarget').disabled=!r.anchor||optimizerBusy;$('removeTarget').disabled=!r.anchor||optimizerBusy;$('imageRole').value=roleFor(r.group);$('sceneGroup').value=r.group;
 $('featureSummary').innerHTML=[['Mean',item.stats.mean],['Contrast σ',item.stats.std],['p05',item.stats.p05],['Median',item.stats.p50],['p95',item.stats.p95],['Entropy',item.stats.entropy]].map(([k,v])=>`<span>${k} <b>${fmt(v)}</b></span>`).join('');
 const fv=$('featureValues');fv.textContent='';let group='';for(let i=0;i<META.names.length;i++){if(META.groups[i]!==group){group=META.groups[i];const h=document.createElement('div');h.className='feature-heading';h.textContent=group;fv.appendChild(h);}const line=document.createElement('div');line.className='feature-line';line.innerHTML=`<span>${esc(META.names[i])}</span><b>${fmt(item.stats.features[i],4)}</b>`;fv.appendChild(line);}
 drawHistogram();lbSyncRating();if($('sensitivityDetails').open)renderSensitivity();
}
function drawHistogram(){
 if(!lbOpen())return;const item=lbItem(),tone=previewTone(item),a=item.preview.data,after=Array(64).fill(0);let n=0;
 for(let i=0;i<a.length;i+=4)if(a[i+3]>=250){const r=clamp((a[i]/255)**tone.gamma*tone.gain,0,1),g=clamp((a[i+1]/255)**tone.gamma*tone.gain,0,1),b=clamp((a[i+2]/255)**tone.gamma*tone.gain,0,1),y=.299*r+.587*g+.114*b;after[Math.min(63,Math.floor(y*64))]++;n++;}
 for(let i=0;i<64;i++)after[i]/=n||1;const before=item.stats.hist,canvas=$('histogram'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,max=Math.max(...before,...after,.01);ctx.clearRect(0,0,w,h);ctx.strokeStyle='#344249';ctx.lineWidth=1;for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(i*w/4,0);ctx.lineTo(i*w/4,h);ctx.stroke();}
 for(const [hist,color,dash]of [[before,'#a1afba',[6,5]],[after,'#ffbd70',[]]]){ctx.strokeStyle=color;ctx.setLineDash(dash);ctx.lineWidth=2;ctx.beginPath();hist.forEach((v,i)=>{const x=4+i/63*(w-8),y=h-5-v/max*(h-10);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}ctx.setLineDash([]);
}
function renderSensitivity(){const out=$('sensitivityValues');out.textContent='';if(!lbOpen()||!state.model){out.textContent='Train a model first.';return;}for(const x of E.sensitivity(lbItem().stats.features,state.model)){const line=document.createElement('div');line.className='feature-line';line.innerHTML=`<span>${esc(x.name)}</span><b>${x.effect>=0?'+':''}${fmt(x.effect,2)} gamma</b>`;out.appendChild(line);}}
async function changeGroupRole(action){
 if(!lbOpen()||optimizerBusy)return;const r=recordFor(lbItem()),group=action==='group'?$('sceneGroup').value.trim().slice(0,200):r.group,role=$('imageRole').value;if(!group){$('sceneGroup').value=r.group;return;}
 if(state.model&&!confirm('Changing the train/validation grouping resets trained weights to avoid validation leakage. Targets are kept. Continue?')){updateSidebar();return;}
 if(state.model){state.model=null;state.undo=null;state.importedModelUnverified=false;state.legacyBaseline=null;modelToken++;}
 if(action==='group')r.group=group;else state.groups[group]=role;
 saveSoon();renderGallery();updateSidebar();lbRequestCorrection();status('Scene grouping updated. Related images in the same group share a single data role.');
}
function workerTrain(payload){return new Promise((resolve,reject)=>{
 const source=`const E=(${buildFeatureEngine.toString()})(); self.onmessage=async e=>{try{const result=await E.train(e.data,p=>self.postMessage({progress:p}));self.postMessage({result});}catch(err){self.postMessage({error:err.message});}};`;
 const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
 try{worker=new Worker(url);}catch(e){URL.revokeObjectURL(url);reject(Error('This browser blocked the local training worker. Try another browser; no data was changed.'));return;}
 URL.revokeObjectURL(url);worker.onmessage=e=>{if(e.data.progress){const p=e.data.progress;$('trainProgress').value=p.epoch/p.epochs;status(`Training shared weights · epoch ${p.epoch}/${p.epochs} · regularized training loss ${fmt(p.loss,4)}`);}else{worker.terminate();worker=null;e.data.error?reject(Error(e.data.error)):resolve(e.data.result);}};
 worker.onerror=e=>{worker?.terminate();worker=null;reject(Error(e.message||'Worker failed.'));};worker.postMessage(payload);
 });}
function setBusy(b){optimizerBusy=b;for(const el of document.querySelectorAll('header input,header select,#resetTools button,#imageRole,#saveGroup,#editTarget,#saveTarget,#acceptOriginal'))el.disabled=b;updateOptimizerStatus();if(lbOpen())lbSetBusy(lb.busy);}
function cancelTraining(){trainToken++;worker?.terminate();worker=null;setBusy(false);status('Training cancelled. Previous model and all feedback are unchanged.');}
async function trainModel(){
 if(optimizerBusy||loading)return;if(manualPreview){status('Save the manual target or return to the model before training.');return;}
 clearTimeout(autoTimer);const records=trainingRecords(),token=++trainToken,base=state.model?structuredClone(state.model):null,oldLegacy=state.legacyBaseline?{...state.legacyBaseline}:null,round=state.round;
 const labeled=records.filter(r=>r.role==='train'&&E.labelFor(r,state.settings.step));if(labeled.length<4){status('Collect at least 4 labeled training images first. Validation does not train the model.');return;}
 const votes=Object.values(state.votes).filter(v=>v.round===state.round),human={good:0,too_dark:0,too_light:0,target:0,original:0,skip:0};for(const v of votes)if(v.rating in human)human[v.rating]++;
 setBusy(true);$('trainProgress').value=0;status('Fitting one feature model to all saved training folders. Validation is excluded.');
 try{const baselineCurves=oldLegacy?Object.fromEntries(records.map(r=>[r.id,legacyTone(r.stats,oldLegacy)])):null;const result=await workerTrain({records,model:base,baselineCurves,epochs:state.settings.epochs,regularization:state.settings.regularization,step:state.settings.step,seed:49137+state.round*97});if(token!==trainToken)return;
   const improved=(!base&&!oldLegacy)||result.training.loss<result.previousTraining.loss-1e-6;
   const report={...result,time:new Date().toISOString(),round,human,applied:improved,legacyBefore:!!oldLegacy,importedModelUnverified:state.importedModelUnverified};
   if(improved){state.undo={model:base,legacyBaseline:oldLegacy,round,votes:structuredClone(state.votes),importedModelUnverified:state.importedModelUnverified};state.model=result.model;state.legacyBaseline=null;state.round++;state.votes={};images.forEach(i=>i.rating='');modelToken++;manualPreview=null;}
   state.reports.push(report);state.reports=state.reports.slice(-20);setBusy(false);saveSoon();renderGallery();if(lbOpen()){updateSidebar();lbRequestCorrection();}status(improved?'New shared model applied. Review its predictions; validation results are in the report.':'Candidate did not improve training feedback loss. Model and votes were kept.');showReport(report);
 }catch(e){if(token!==trainToken)return;setBusy(false);status('Training failed: '+e.message+' Your model and feedback were kept.');console.error(e);}
}
function showReport(r){
 if(!r)return;$('reportHeading').textContent=r.applied?`Round ${r.round} → ${r.round+1}: feature model updated`:`Round ${r.round}: model kept`;
 const h=r.human||{},n=(h.good||0)+(h.too_dark||0)+(h.too_light||0);
 $('reportSummary').textContent=`Frozen-model judgments: ${h.good||0}/${n} Good, ${h.too_light||0} too bright, ${h.too_dark||0} too dark. ${h.target||0} manual targets; ${h.original||0} original approvals. Fitted ${r.learnedWeights} shared weights/biases on ${r.training.n} training images in ${r.groups} groups. ${r.validation.n} validation images were never fitted.`;
 const drift=r.changes.filter(x=>x.targetDrift>.012),messages=[];
 if(!r.validation.n)messages.push('No labeled validation examples yet. Generalization has not been measured.');
 if(r.validation.n&&r.validation.loss>r.previousValidation.loss)messages.push('Validation feedback loss increased. The training improvement did not transfer to this validation set.');
 if(r.training.n<30)messages.push('Small training set for a 169-feature model. Add diverse examples; watch validation rather than training fit.');
 if(drift.length)messages.push(`${drift.length} stored targets differ from predictions by more than 3/255 RMS. They were not silently overridden.`);
 if(r.legacyBefore)messages.push('Before metrics use your imported v3 baseline. The new candidate uses the feature network, not the old formula.');
 if(r.importedModelUnverified)messages.push('The initial imported model has unknown training membership. Validation independence is not verified.');
 $('reportWarnings').textContent=messages.join(' ')||'Training fit is not a visual-quality score. Review the new global model before accepting it as a baseline.';
 const metrics=[['Feedback loss','loss',1,4],['Target RGB-curve RMS /255','curveRMS',255,2],['Gamma slider MAE','gammaMAE',1,2],['Gain MAE, percentage points','gainMAE',1,2]];
 const table=$('reportMetrics');table.textContent='';for(const [name,key,scale,d]of metrics){const tr=document.createElement('tr');for(const v of [name,`${fmt(r.previousTraining[key]===null?NaN:r.previousTraining[key]*scale,d)} → ${fmt(r.training[key]===null?NaN:r.training[key]*scale,d)}`,`${fmt(r.previousValidation[key]===null?NaN:r.previousValidation[key]*scale,d)} → ${fmt(r.validation[key]===null?NaN:r.validation[key]*scale,d)}`]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}table.appendChild(tr);}
 const tbody=$('imageChanges');tbody.textContent='';for(const x of r.changes){const tr=document.createElement('tr');for(const v of [x.name,x.role,`${fmt(E.slider(x.gammaBefore),1)} → ${fmt(E.slider(x.gammaAfter),1)}`,`${fmt((x.gainBefore-1)*100,1)} → ${fmt((x.gainAfter-1)*100,1)}`,`${x.midDelta>=0?'+':''}${fmt(x.midDelta*255,1)}`]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}tbody.appendChild(tr);}
 if(!$('roundReport').open)$('roundReport').showModal();
}
function download(name,text,type){const u=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),3000);}
function exportSession(){saveNow();download('donut-tone-lab-feature-session-v4.json',JSON.stringify({...state,exportedAt:new Date().toISOString(),undo:state.undo},null,2),'application/json');status('Session exported, including features, immutable targets, history and global model. No image pixels included.');}
function exportModel(){if(!state.model)return;download('donut-tone-model-v4.json',JSON.stringify({version:4,type:'donut-tone-model',schema:E.SCHEMA,algorithm:E.ALGORITHM,analysis:{longEdge:256,colorSpace:'srgb',opaqueAlphaMinimum:250},model:state.model},null,2),'application/json');status('Model-only export contains global weights and normalization—not image records, filenames or ratings.');}
function importLegacy(s){
 const history=s.optimizer?.history||[],rows=s.images||[],names=new Set([...rows.map(r=>r.file||r.key),...history.map(e=>e.key)]),list=[];
 for(const key of names){if(typeof key!=='string')continue;const row=rows.find(r=>(r.file||r.key)===key),es=history.filter(e=>e.key===key);const events=es.map(e=>({rating:e.rating,gamma:Number(e.judgedGamma),gain:Number(e.judgedGain),time:e.time,round:e.round,legacy:true}));
  if(row&&['good','too_dark','too_light'].includes(row.rating)){const snap=row.rating_snapshot;events.push({rating:row.rating,gamma:Number(snap?.gamma??row.gamma_exponent),gain:Number(snap?.gain??row.brightness_gain),time:snap?.time||s.exported_at,legacy:true});}
  if(!events.length)continue;list.push({key,name:key.split('/').pop(),size:row?.file_size||Number((es[0]?.imageId||'').split('|')[1])||0,width:row?.width,height:row?.height,stats:row?.stats||es[0]?.stats,events});}
 const existing=new Set(state.legacy.map(r=>r.key));state.legacy.push(...list.filter(r=>!existing.has(r.key)));
 if(!state.model&&s.parameters){const p=s.parameters;if(['mean','ratio','hybrid','fixed'].includes(p.mode)&&['strength','maxGamma','idealMean','brightRef','targetRatio','hybridWeight','highlightTarget','maxGain','fixedSlider'].every(k=>Number.isFinite(p[k])))state.legacyBaseline={...p};}
 for(const r of Object.values(state.records))migrateLegacy(r);
 modelToken++;saveSoon();refreshAll();status(`Imported ${list.length} legacy examples. Reload their source folders to compute new features. Earliest Good becomes a stable target; saved global v3 parameters are only a temporary preview.`);
}
async function importFile(file){
 if(optimizerBusy||loading){status("Wait for image analysis/training to finish before importing.");return;}try{if(file.size>60*1024*1024)throw Error('JSON is larger than 60 MB.');const s=JSON.parse(await file.text());
 if(s.version===3||s.version===2||s.version===1){importLegacy(s);return;}
 if(s.type==='donut-tone-model'){
  const model=E.validatedModel(s.model);if(model.featureCount!==META.features.length||JSON.stringify(model.names)!==JSON.stringify(META.names))throw Error('Model feature schema mismatch.');
  if(Object.keys(state.records).length&&!confirm('Load these global weights? Existing targets are kept, but their validation independence cannot be verified for imported weights.'))return;
  state.undo={model:state.model,legacyBaseline:state.legacyBaseline,round:state.round,votes:structuredClone(state.votes),importedModelUnverified:state.importedModelUnverified};state.model=model;state.legacyBaseline=null;state.importedModelUnverified=true;state.round++;state.votes={};images.forEach(i=>i.rating='');saveSoon();refreshAll();status('Global model imported. No image-specific history is needed for prediction.');return;
 }
 const next=validateSession(s);if(Object.keys(state.records).length&&!confirm('Replace this local session with the imported session? Export first to keep the current work.'))return;
 if(lbOpen())lbClose();images.forEach(i=>URL.revokeObjectURL(i.url));images=[];state=next;controlsFromState();saveSoon();refreshAll();status('Session restored. Choose a source folder to view it; all saved training folders remain available to the learner.');
 }catch(e){status('Import rejected: '+e.message+'. Existing state was not replaced.');console.error(e);}
}
function exportCSV(){
 const names=['file','scene_group','role',...META.names,'predicted_gamma_slider','predicted_gain_percent','target_gamma_slider','target_gain_percent'];
 const escape=v=>'"'+String(v??'').replace(/^[=+@-]/,"'$&").replaceAll('"','""')+'"';
 const rows=images.map(i=>{const r=recordFor(i),p=modelTone(i);return [i.key,r.group,roleFor(r.group),...i.stats.features,E.slider(p.gamma),(p.gain-1)*100,r.anchor?E.slider(r.anchor.gamma):'',r.anchor?(r.anchor.gain-1)*100:''];});download('donut-tone-image-measurements-v4.csv',[names,...rows].map(r=>r.map(escape).join(',')).join('\n'),'text/csv');
}
function setupApp(){
 $('folder').onchange=e=>{const fs=[...e.target.files];e.target.value='';loadFiles(fs);};$('files').onchange=e=>{const fs=[...e.target.files];e.target.value='';loadFiles(fs);};$('importData').onchange=e=>{const f=e.target.files[0];e.target.value='';if(f)importFile(f);};
 $('exportSession').onclick=exportSession;$('exportModel').onclick=exportModel;$('exportFeatures').onclick=exportCSV;
 $('train').onclick=trainModel;$('lbOptimize').onclick=trainModel;$('cancelTrain').onclick=cancelTraining;
 $('viewReport').onclick=()=>showReport(state.reports.at(-1));$('lbReportButton').onclick=()=>showReport(state.reports.at(-1));
 $('reportClose').onclick=()=>$('roundReport').close();$('reportNext').onclick=()=>{$('roundReport').close();const i=nextUnratedIndex(-1);if(i>=0)openLightbox(i);};
 $('galleryFilter').onchange=renderGallery;$('inspectNext').onclick=()=>{const i=nextUnratedIndex(-1);if(i>=0)openLightbox(i);else if(images.length)openLightbox(0);};
 for(const [id,key]of [['epochs','epochs'],['regularization','regularization'],['toneStep','step'],['roundSize','roundSize']])$(id).onchange=()=>{state.settings[key]=Number($(id).value);saveSoon();updateOptimizerStatus();};
 for(const key of ['autoAdvance','autoTrain'])$(key).onchange=()=>{state.settings[key]=$(key).checked;saveSoon();};
 $('editTarget').onclick=()=>beginManual(modelTone(lbItem()));$('useModel').onclick=returnToModel;$('viewTarget').onclick=()=>{if(recordFor(lbItem()).anchor)beginManual(recordFor(lbItem()).anchor);};
 $('saveTarget').onclick=()=>saveTarget('manual');$('acceptOriginal').onclick=()=>saveTarget('original');
 for(const [range,number,kind]of [['manualGamma','manualGammaN','gamma'],['manualGain','manualGainN','gain']]){
   function update(v){if(!manualPreview)return;v=Number(v);if(!Number.isFinite(v))return;if(kind==='gamma')manualPreview.gamma=E.exponent(clamp(v,-100,100));else manualPreview.gain=1+clamp(v,-20,25)/100;updateManualControls();lbRequestCorrection();drawHistogram();}
   $(range).oninput=()=>update($(range).value);$(number).oninput=()=>update($(number).value);
 }
 $('removeTarget').onclick=()=>{if(!lbOpen()||!confirm('Remove the saved target and directional constraints for this example? The global model is unchanged.'))return;const r=recordFor(lbItem());r.anchor=null;r.events=[];saveSoon();updateSidebar();updateOptimizerStatus();};
 $('imageRole').onchange=()=>changeGroupRole('role');$('saveGroup').onclick=()=>changeGroupRole('group');$('sensitivityDetails').ontoggle=()=>{if($('sensitivityDetails').open)renderSensitivity();};
 $('resetControls').onclick=()=>{state.settings={...DEFAULT_SETTINGS};controlsFromState();saveSoon();status('Training controls reset. Learned weights and targets were kept.');};
 $('resetModel').onclick=()=>{if(!confirm('Reset the global model to identity? All saved targets and measurements are kept.'))return;state.model=null;state.legacyBaseline=null;state.undo=null;state.importedModelUnverified=false;state.round++;state.votes={};images.forEach(i=>i.rating='');saveSoon();refreshAll();status('Model reset to untrained. Train again from the saved examples.');};
 $('clearRound').onclick=()=>{if(!confirm('Clear only the visible round votes? Saved training targets and history remain.'))return;state.votes={};images.forEach(i=>i.rating='');saveSoon();renderGallery();updateOptimizerStatus();};
 $('clearData').onclick=()=>{if(!confirm('Delete all saved examples, weights and reports from this v4 session? Export first to keep a backup.'))return;if(lbOpen())lbClose();state=fresh();images.forEach(i=>URL.revokeObjectURL(i.url));images=[];saveSoon();controlsFromState();refreshAll();status('New empty session.');};
 $('undoModel').onclick=()=>{if(!state.undo||optimizerBusy)return;const u=state.undo;state.model=u.model;state.legacyBaseline=u.legacyBaseline;state.round=u.round;state.votes=u.votes;state.importedModelUnverified=u.importedModelUnverified;state.undo=null;for(const i of images)i.rating=state.votes[i.id]?.round===state.round?state.votes[i.id].rating:'';saveSoon();refreshAll();status('Previous global model restored. Saved teaching examples were kept.');};
 const drop=$('drop');for(const t of ['dragenter','dragover'])drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add('over');});for(const t of ['dragleave','drop'])drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove('over');});drop.addEventListener('drop',e=>loadFiles(e.dataTransfer.files));
 window.addEventListener('beforeunload',()=>{saveNow();});
}


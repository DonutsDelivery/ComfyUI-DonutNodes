
"use strict";
/* Pure, versioned analyzer and global predictor. No files, names, IDs or ratings
   are inputs to predict(). This same function is used by gallery, lightbox and worker. */
function buildFeatureEngine(){
  'use strict';
  const SCHEMA='donut_srgb256_features_v4.0', ALGORITHM='donut_feature_mlp_v4.0';
  const LOGG=Math.log(3), LOGB=Math.log(1.25), H1=16, H2=8;
  const clip=(x,a,b)=>Math.max(a,Math.min(b,x));
  const avg=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:0;
  const variance=(a,m=avg(a))=>a.length?avg(a.map(x=>(x-m)**2)):0;
  const q=(a,p)=>{if(!a.length)return 0;const z=(a.length-1)*p,i=Math.floor(z);return a[i]+(a[Math.ceil(z)]-a[i])*(z-i);};
  const sigmoid=x=>1/(1+Math.exp(-clip(x,-40,40)));
  const slider=g=>-100*Math.log(g)/Math.log(3);
  const exponent=s=>3**(-s/100);
  function random(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
  function hash(s){let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return h>>>0;}
  const lin=x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4;
  function analyze(data,w,h){
    if(!Number.isInteger(w)||!Number.isInteger(h)||w<1||h<1||data.length!==w*h*4)throw Error('Invalid analysis pixels.');
    const ys=[],linear=[],ss=[],cs=[],channels=[[],[],[]],hist=Array(64).fill(0),satHist=Array(8).fill(0),rgbHist=Array(256).fill(0),midHist=Array(256).fill(0);
    const tiles=Array.from({length:16},()=>[]),valid=new Uint8Array(w*h),ymap=new Float64Array(w*h);
    let blacks=0,whites=0,anyBlack=0,anyWhite=0,neutral=0,centerSum=0,centerN=0,borderSum=0,borderN=0;
    for(let p=0;p<w*h;p++){
      if(data[4*p+3]<250)continue;
      const r=data[4*p]/255,g=data[4*p+1]/255,b=data[4*p+2]/255,y=.299*r+.587*g+.114*b;
      const mx=Math.max(r,g,b),mn=Math.min(r,g,b),ch=mx-mn,s=mx>1e-12?ch/mx:0;
      valid[p]=1;ymap[p]=y;ys.push(y);linear.push(.2126*lin(r)+.7152*lin(g)+.0722*lin(b));ss.push(s);cs.push(ch);
      channels[0].push(r);channels[1].push(g);channels[2].push(b);
      hist[Math.min(63,Math.floor(y*64))]++;satHist[Math.min(7,Math.floor(s*8))]++;
      rgbHist[data[4*p]]+=.299;rgbHist[data[4*p+1]]+=.587;rgbHist[data[4*p+2]]+=.114;
      blacks+=y<=1/255;whites+=y>=254/255;anyBlack+=mn<=1/255;anyWhite+=mx>=254/255;neutral+=ch<.05;
      const x=p%w,ypos=Math.floor(p/w),tx=Math.min(3,Math.floor(x/w*4)),ty=Math.min(3,Math.floor(ypos/h*4));tiles[ty*4+tx].push(y);
      if(tx>=1&&tx<=2&&ty>=1&&ty<=2){centerSum+=y;centerN++;}else{borderSum+=y;borderN++;}
    }
    const n=ys.length;ys.sort((a,b)=>a-b);ss.sort((a,b)=>a-b);cs.sort((a,b)=>a-b);channels.forEach(a=>a.sort((a,b)=>a-b));
    const mean=avg(ys),std=Math.sqrt(variance(ys,mean)),lo=q(ys,.25),hi=q(ys,.75);let midN=0;
    for(let p=0;p<w*h;p++)if(valid[p]&&ymap[p]>=lo&&ymap[p]<=hi){midHist[data[4*p]]+=.299;midHist[data[4*p+1]]+=.587;midHist[data[4*p+2]]+=.114;midN++;}
    for(let k=0;k<256;k++){rgbHist[k]/=n||1;midHist[k]/=midN||1;}
    for(let k=0;k<64;k++)hist[k]/=n||1;for(let k=0;k<8;k++)satHist[k]/=n||1;
    const names=[],groups=[],features=[];
    function add(name,v,group){names.push(name);features.push(Number.isFinite(v)?v:0);groups.push(group);}
    hist.forEach((v,i)=>add('Luma bin '+String(i).padStart(2,'0'),v,'Luma histogram'));
    const qs=[.01,.05,.10,.25,.50,.75,.90,.95,.99],quantiles=qs.map(p=>q(ys,p));
    qs.forEach((v,i)=>add('p'+Math.round(v*100).toString().padStart(2,'0'),quantiles[i],'Tonal shape'));
    add('Mean luma',mean,'Tonal shape');add('Luma standard deviation',std,'Tonal shape');
    add('Skewness / 4',std>1e-6?clip(avg(ys.map(y=>((y-mean)/std)**3)),-4,4)/4:0,'Tonal shape');
    add('Excess kurtosis / 20',std>1e-6?clip(avg(ys.map(y=>((y-mean)/std)**4))-3,-3,20)/20:0,'Tonal shape');
    const entropy=-hist.reduce((s,p)=>s+(p?p*Math.log2(p):0),0)/6;
    add('Histogram entropy / 6',entropy,'Tonal shape');add('Geometric mean luma',n?Math.exp(avg(ys.map(x=>Math.log(Math.max(x,1/255))))):0,'Tonal shape');
    add('Linear-light mean',avg(linear),'Tonal shape');add('Linear-light standard deviation',Math.sqrt(variance(linear)),'Tonal shape');
    for(const [name,a,b]of [['p99 − p01',.99,.01],['p95 − p05',.95,.05],['p90 − p10',.9,.1],['Interquartile range',.75,.25]])add(name,q(ys,a)-q(ys,b),'Exposure & range');
    const bounds=[0,.05,.2,.4,.6,.85,.97,1.000001];for(let i=0;i<7;i++)add('Zone '+bounds[i].toFixed(2)+'–'+Math.min(1,bounds[i+1]).toFixed(2),ys.filter(y=>y>=bounds[i]&&y<bounds[i+1]).length/(n||1),'Exposure & range');
    add('Near-black luma',blacks/(n||1),'Exposure & range');add('Near-white luma',whites/(n||1),'Exposure & range');
    add('Any RGB channel near black',anyBlack/(n||1),'Exposure & range');add('Any RGB channel near white',anyWhite/(n||1),'Exposure & range');
    add('Highlight headroom',1-q(ys,.99),'Exposure & range');add('Median / p95',q(ys,.5)/Math.max(q(ys,.95),1e-6),'Exposure & range');
    for(let c=0;c<3;c++){let a=channels[c],prefix=['R','G','B'][c];add(prefix+' mean',avg(a),'Colour & saturation');add(prefix+' standard deviation',Math.sqrt(variance(a)),'Colour & saturation');for(const p of [.05,.5,.95])add(prefix+' p'+Math.round(p*100),q(a,p),'Colour & saturation');}
    add('R mean − G mean',avg(channels[0])-avg(channels[1]),'Colour & saturation');add('B mean − G mean',avg(channels[2])-avg(channels[1]),'Colour & saturation');
    add('Neutral pixel fraction',neutral/(n||1),'Colour & saturation');
    for(const [name,a]of [['Chroma',cs],['Saturation',ss]]){add(name+' mean',avg(a),'Colour & saturation');add(name+' standard deviation',Math.sqrt(variance(a)),'Colour & saturation');add(name+' p90',q(a,.9),'Colour & saturation');}
    satHist.forEach((v,i)=>add('Saturation bin '+i,v,'Colour & saturation'));
    const tmean=tiles.map(a=>a.length?avg(a):mean),tstd=tiles.map(a=>Math.sqrt(variance(a)));
    tmean.forEach((v,i)=>add('Tile '+Math.floor(i/4)+','+(i%4)+' mean',v,'Spatial & texture'));
    tstd.forEach((v,i)=>add('Tile '+Math.floor(i/4)+','+(i%4)+' contrast',v,'Spatial & texture'));
    add('Between-tile variation',Math.sqrt(variance(tmean)),'Spatial & texture');add('Mean local contrast',avg(tstd),'Spatial & texture');add('Center − border luma',centerSum/(centerN||1)-borderSum/(borderN||1),'Spatial & texture');
    let dx=0,dy=0,nx=0,ny=0,edge=0,lap=0,nlap=0;
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){let p=y*w+x;if(!valid[p])continue;
      if(x+1<w&&valid[p+1]){const d=Math.abs(ymap[p+1]-ymap[p]);dx+=d;nx++;edge+=d>.08;}
      if(y+1<h&&valid[p+w]){const d=Math.abs(ymap[p+w]-ymap[p]);dy+=d;ny++;edge+=d>.08;}
      if(x>0&&x+1<w&&y>0&&y+1<h&&valid[p-1]&&valid[p+1]&&valid[p-w]&&valid[p+w]){lap+=((4*ymap[p]-ymap[p-1]-ymap[p+1]-ymap[p-w]-ymap[p+w])/4)**2;nlap++;}}
    add('Horizontal gradient',dx/(nx||1),'Spatial & texture');add('Vertical gradient',dy/(ny||1),'Spatial & texture');add('Edge fraction',edge/(nx+ny||1),'Spatial & texture');add('Laplacian RMS / 4',Math.sqrt(lap/(nlap||1)),'Spatial & texture');
    return {schema:SCHEMA,features,names,groups,hist,midHist,rgbHist,sampleCount:n,opaqueFraction:n/(w*h),mean,std,p05:q(ys,.05),p50:q(ys,.5),p95:q(ys,.95),ratio:q(ys,.5)/Math.max(q(ys,.95),1e-6),entropy,flat:q(ys,.95)-q(ys,.05)<.03,analysisWidth:w,analysisHeight:h};
  }
  function curve(hist,g,b){let y=0;for(let k=1;k<256;k++)if(hist[k])y+=hist[k]*Math.min(1,(k/255)**g*b);return y;}
  function curveGrad(hist,g,b){let value=0,dg=0,db=0;for(let k=1;k<256;k++)if(hist[k]){const x=k/255,v=x**g*b,w=hist[k];value+=w*Math.min(1,v);if(v<1){dg+=w*v*g*Math.log(x);db+=w*v;}}return {value,dg,db};}
  function curveRMS(hist,a,b){let z=0;for(let k=0;k<256;k++)if(hist[k]){const x=k/255,d=Math.min(1,x**a.gamma*a.gain)-Math.min(1,x**b.gamma*b.gain);z+=hist[k]*d*d;}return Math.sqrt(z);}
  function identityDistance(a){let z=0;for(let i=1;i<32;i++){const x=i/32;z+=(Math.min(1,x**a.gamma*a.gain)-x)**2;}return Math.sqrt(z/31);}
  function layer(ni,no,rng,scale=1){return {ni,no,w:Array.from({length:ni*no},()=> (rng()*2-1)*Math.sqrt(6/(ni+no))*scale),b:Array(no).fill(0)};}
  function initialize(count,stats,seed=49137){const rng=random(seed);return {format:ALGORITHM,schema:SCHEMA,featureCount:count,names:stats.names.slice(),mu:Array(count).fill(0),sd:Array(count).fill(1),layers:[layer(count,H1,rng),layer(H1,H2,rng),layer(H2,3,rng,.02)],trained:false,revision:0};}
  function standardize(features,m){return features.map((x,i)=>clip((x-m.mu[i])/m.sd[i],-6,6));}
  function forward(features,m){const acts=[standardize(features,m)];for(let i=0;i<m.layers.length;i++){const l=m.layers[i],input=acts[i],out=[];for(let j=0;j<l.no;j++){let sum=l.b[j];for(let k=0;k<l.ni;k++)sum+=l.w[j*l.ni+k]*input[k];out.push(i===m.layers.length-1?sum:Math.tanh(sum));}acts.push(out);}const raw=acts[3],lg=LOGG*Math.tanh(raw[0]),lb=LOGB*Math.tanh(raw[1]);return {gamma:Math.exp(lg),gain:Math.exp(lb),logGamma:lg,logGain:lb,noOpScore:sigmoid(raw[2]),acts};}
  function predict(features,m){
    if(!m||!m.trained)return {gamma:1,gain:1,noOpScore:null,untrained:true,noop:true,coverage:0};
    if(features.length!==m.featureCount)throw Error('Feature dimension mismatch.');
    const a=forward(features,m),distance=identityDistance(a),noop=distance<.0025||(a.noOpScore>=.9&&distance<.015);
    let outside=0;for(let i=0;i<features.length;i++)outside+=Math.abs((features[i]-m.mu[i])/m.sd[i])>3;
    return {gamma:noop?1:a.gamma,gain:noop?1:a.gain,noOpScore:a.noOpScore,noop,coverage:outside/features.length,rawGamma:a.gamma,rawGain:a.gain};
  }
  function validatedModel(m){
    if(!m||m.format!==ALGORITHM||m.schema!==SCHEMA||!m.trained)throw Error('Not a supported trained Tone Lab feature model.');
    const n=m.featureCount;if(!Number.isInteger(n)||n<1||n>1000||!Array.isArray(m.names)||m.names.length!==n||!Array.isArray(m.mu)||m.mu.length!==n||!Array.isArray(m.sd)||m.sd.length!==n)throw Error('Invalid model dimensions.');
    for(let i=0;i<n;i++)if(!Number.isFinite(m.mu[i])||!Number.isFinite(m.sd[i])||m.sd[i]<=0||typeof m.names[i]!=='string')throw Error('Invalid model preprocessing.');
    const shapes=[[n,H1],[H1,H2],[H2,3]];if(!Array.isArray(m.layers)||m.layers.length!==3)throw Error('Invalid layer count.');
    m.layers.forEach((l,i)=>{const [ni,no]=shapes[i];if(l.ni!==ni||l.no!==no||!Array.isArray(l.w)||!Array.isArray(l.b)||l.w.length!==ni*no||l.b.length!==no||![...l.w,...l.b].every(x=>Number.isFinite(x)&&Math.abs(x)<1e5))throw Error('Invalid network weights.');});
    return JSON.parse(JSON.stringify(m));
  }
  function labelFor(r,step=.025){
    if(r.anchor)return {kind:'target',target:r.anchor,weight:r.anchor.source==='manual'||r.anchor.source==='original'?1.5:1};
    const es=(r.events||[]).filter(e=>e.rating==='too_dark'||e.rating==='too_light');if(!es.length)return null;
    let lower=null,upper=null,streak=0,last=null;
    for(const e of es){const v=curve(r.stats.midHist,e.gamma,e.gain);streak=last?.rating===e.rating?streak+1:1;
      if(e.rating==='too_light'){upper=upper===null?v:Math.min(upper,v);if(lower!==null&&lower>=upper)lower=null;}
      else{lower=lower===null?v:Math.max(lower,v);if(upper!==null&&upper<=lower)upper=null;}last=e;}
    const margin=Math.min(.06,step*1.3**Math.min(4,streak-1));
    return {kind:lower!==null&&upper!==null?'bracket':last.rating,lower,upper,margin,observed:last,weight:.7};
  }
  function sampleLoss(r,a,label,grad=true){
    let loss=0,dg=0,db=0,dn=0;
    if(label.kind==='target'){
      const t=label.target,eg=a.logGamma-Math.log(t.gamma),eb=a.logGain-Math.log(t.gain);
      loss=.5*((eg/.35)**2+(eb/.12)**2);dg=eg/.35**2;db=eb/.12**2;
      const no=identityDistance(t)<.004?1:0,p=clip(a.noOpScore,1e-8,1-1e-8);
      loss+=.12*(-no*Math.log(p)-(1-no)*Math.log(1-p));dn=.12*(p-no);
    }else{
      const m=curveGrad(r.stats.midHist,a.gamma,a.gain);let err=0;
      if(label.kind==='bracket')err=m.value-(label.lower+label.upper)/2;
      else if(label.kind==='too_light')err=Math.max(0,m.value-Math.max(0,label.upper-label.margin));
      else err=Math.min(0,m.value-Math.min(1,label.lower+label.margin));
      loss=.5*(err/.05)**2;dg=err/.05**2*m.dg;db=err/.05**2*m.db;
      // Direction alone does not identify gamma AND gain. Weakly prefer retaining
      // the judged gain instead of achieving every request through brightness.
      const eb=a.logGain-Math.log(label.observed.gain);loss+=.025*(eb/.12)**2;db+=.05*eb/.12**2;
    }
    return {loss:loss*label.weight,dg:dg*label.weight,db:db*label.weight,dn:dn*label.weight};
  }
  function evaluate(records,m,step=.025,overrides=null){
    let loss=0,n=0,ex=0,rms=0,gs=0,bs=0,dirs=0,satisfied=0,noops=0;
    for(const r of records){const label=labelFor(r,step);if(!label)continue;const pred=overrides?.[r.id]||(m?.trained?predict(r.stats.features,m):{gamma:1,gain:1,noOpScore:.5});
      const a={...pred,noOpScore:pred.noOpScore??.5,logGamma:Math.log(pred.gamma),logGain:Math.log(pred.gain)};loss+=sampleLoss(r,a,label,false).loss;n++;
      if(label.kind==='target'){const d=curveRMS(r.stats.rgbHist,pred,label.target);rms+=d*d;gs+=Math.abs(slider(pred.gamma)-slider(label.target.gamma));bs+=100*Math.abs(pred.gain-label.target.gain);ex++;if(label.target.gamma===1&&label.target.gain===1)noops++;}
      else{const mid=curve(r.stats.midHist,pred.gamma,pred.gain);dirs++;satisfied+=label.kind==='bracket'?mid>=label.lower&&mid<=label.upper:label.kind==='too_light'?mid<label.upper-.005:mid>label.lower+.005;}}
    return {n,loss:n?loss/n:null,targets:ex,curveRMS:ex?Math.sqrt(rms/ex):null,gammaMAE:ex?gs/ex:null,gainMAE:ex?bs/ex:null,directional:dirs,directionSatisfied:satisfied,originalTargets:noops};
  }
  async function train(payload,progress=()=>{}){
    const all=payload.records.filter(r=>r.stats?.schema===SCHEMA&&r.stats.sampleCount>=16&&labelFor(r,payload.step));
    const tr=all.filter(r=>r.role==='train'),va=all.filter(r=>r.role==='validation');
    if(tr.length<4||new Set(tr.map(r=>r.group)).size<3)throw Error('Collect feedback on at least 4 training images in 3 scene groups. Validation images are not used for training.');
    const count=tr[0].stats.features.length;if(tr.some(r=>r.stats.features.length!==count))throw Error('Mixed feature versions. Reload source images to reanalyze.');
    let m=payload.model?.trained?validatedModel(payload.model):initialize(count,tr[0].stats,payload.seed||49137);
    const beforeModel=payload.model?.trained?validatedModel(payload.model):null;
    const mu=Array(count).fill(0),sd=Array(count).fill(0);
    for(const r of tr)for(let i=0;i<count;i++)mu[i]+=r.stats.features[i]/tr.length;
    for(const r of tr)for(let i=0;i<count;i++)sd[i]+=(r.stats.features[i]-mu[i])**2/tr.length;
    for(let i=0;i<count;i++)sd[i]=Math.max(.03,Math.sqrt(sd[i]));
    if(m.trained){const l=m.layers[0],oldW=l.w.slice();for(let j=0;j<l.no;j++)for(let i=0;i<count;i++){l.b[j]+=oldW[j*count+i]*(mu[i]-m.mu[i])/m.sd[i];l.w[j*count+i]=oldW[j*count+i]*sd[i]/m.sd[i];}}
    m.mu=mu;m.sd=sd;m.trained=true;
    const rng=random(payload.seed||49137),epochs=clip(Math.round(payload.epochs||400),50,2000),lr=.004,lambda=clip(payload.regularization??.01,.0001,.5);
    const labels=tr.map(r=>labelFor(r,payload.step));
    const packs=m.layers.map(l=>({mw:new Float64Array(l.w.length),vw:new Float64Array(l.w.length),mb:new Float64Array(l.b.length),vb:new Float64Array(l.b.length)}));
    let iter=0,best=JSON.parse(JSON.stringify(m)),bestLoss=Infinity,trace=[];
    // Weight each scene group equally: near-duplicates do not get extra voting power.
    const sizes={};tr.forEach(r=>sizes[r.group]=(sizes[r.group]||0)+1);const ng=Object.keys(sizes).length;
    for(let epoch=0;epoch<epochs;epoch++){
      const grads=m.layers.map(l=>({w:new Float64Array(l.w.length),b:new Float64Array(l.b.length)}));let total=0;
      for(let idx=0;idx<tr.length;idx++){
        const r=tr[idx],a=forward(r.stats.features,m),v=sampleLoss(r,a,labels[idx]),weight=1/(ng*sizes[r.group]);total+=weight*v.loss;
        let delta=[v.dg*LOGG*(1-Math.tanh(a.acts[3][0])**2),v.db*LOGB*(1-Math.tanh(a.acts[3][1])**2),v.dn].map(x=>x*weight);
        for(let li=2;li>=0;li--){const l=m.layers[li],inp=a.acts[li],next=Array(l.ni).fill(0),g=grads[li];
          for(let j=0;j<l.no;j++){g.b[j]+=delta[j];for(let i=0;i<l.ni;i++){g.w[j*l.ni+i]+=delta[j]*inp[i];next[i]+=l.w[j*l.ni+i]*delta[j];}}
          if(li>0)next.forEach((v,i)=>next[i]=v*(1-inp[i]**2));delta=next;}
      }
      let norm=0;for(let li=0;li<3;li++){const l=m.layers[li],g=grads[li];for(let j=0;j<l.w.length;j++){g.w[j]+=lambda*l.w[j];total+=.5*lambda*l.w[j]*l.w[j];norm+=g.w[j]**2;}for(const v of g.b)norm+=v*v;}
      if(!Number.isFinite(total))throw Error('Training became non-finite. Previous model was kept.');
      if(total<bestLoss){bestLoss=total;best=JSON.parse(JSON.stringify(m));}
      const factor=Math.min(1,5/Math.max(1e-12,Math.sqrt(norm)));iter++;
      const rate=lr*(.2+.8*(1-epoch/epochs));
      for(let li=0;li<3;li++)for(const key of ['w','b']){const l=m.layers[li],g=grads[li][key],mm=packs[li]['m'+key],vv=packs[li]['v'+key];for(let j=0;j<g.length;j++){const d=g[j]*factor;mm[j]=.9*mm[j]+.1*d;vv[j]=.999*vv[j]+.001*d*d;l[key][j]-=rate*(mm[j]/(1-.9**iter))/(Math.sqrt(vv[j]/(1-.999**iter))+1e-8);}}
      if(epoch%20===0||epoch===epochs-1){const v={epoch:epoch+1,epochs,loss:total};trace.push(v);progress(v);await new Promise(r=>setTimeout(r,0));}
    }
    const finalTrain=evaluate(tr,best,payload.step),previousTrain=evaluate(tr,beforeModel,payload.step,payload.baselineCurves),validation=evaluate(va,best,payload.step),previousValidation=evaluate(va,beforeModel,payload.step,payload.baselineCurves);
    best.revision=(payload.model?.revision||0)+1;best.training={imageCount:tr.length,groupCount:ng,validationCount:va.length,epochs,regularization:lambda,step:payload.step||.025,seed:payload.seed||49137};
    best.trainingKeys=undefined; // No image records are required by or exported with the model.
    const changes=all.map(r=>{const old=payload.baselineCurves?.[r.id]||predict(r.stats.features,beforeModel),next=predict(r.stats.features,best);return {id:r.id,name:r.name,role:r.role,gammaBefore:old.gamma,gammaAfter:next.gamma,gainBefore:old.gain,gainAfter:next.gain,midDelta:curve(r.stats.midHist,next.gamma,next.gain)-curve(r.stats.midHist,old.gamma,old.gain),targetDrift:r.anchor?curveRMS(r.stats.rgbHist,next,r.anchor):null};});
    return {model:best,training:finalTrain,validation,previousTraining:previousTrain,previousValidation,trace,changes,learnedWeights:best.layers.reduce((s,l)=>s+l.w.length+l.b.length,0),groups:ng};
  }
  function sensitivity(features,m){if(!m?.trained)return [];const base=predict(features,m);return features.map((v,i)=>{const f=features.slice(),step=.2*m.sd[i];f[i]+=step;const p=predict(f,m);return {name:m.names[i],effect:slider(p.gamma)-slider(base.gamma),gainEffect:100*(p.gain-base.gain)};}).sort((a,b)=>Math.abs(b.effect)-Math.abs(a.effect)).slice(0,8);}
  return {SCHEMA,ALGORITHM,H1,H2,clip,slider,exponent,hash,analyze,predict,forward,initialize,validatedModel,train,evaluate,labelFor,curve,curveGrad,curveRMS,identityDistance,sensitivity};
}


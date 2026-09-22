// Run from any directory: node --test tools/tone_lab/test_trainer.cjs
// Tests use synthetic pixels/targets only; no personal model or image is required.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {spawnSync} = require('node:child_process');
const os = require('node:os');
const root = __dirname;
const scripts = ['engine.js', 'state.js', 'app.js', 'lightbox.js'];
const source = fs.readFileSync(path.join(root, scripts[0]), 'utf8');
const E = vm.runInNewContext(source + '\nbuildFeatureEngine()', {setTimeout});
const plain = value => JSON.parse(JSON.stringify(value));
function example(i, role = 'train') {
    const w=16, h=16, data=new Uint8ClampedArray(w*h*4);
    for (let p=0; p<w*h; p++) {
        data[p*4]=(p*3+i*17)%256; data[p*4+1]=(p*5+i*23)%256;
        data[p*4+2]=(p*7+i*37)%256; data[p*4+3]=255;
    }
    return {id:'fixture-'+i, name:'fixture-'+i+'.png', group:'scene-'+i, role,
        stats:E.analyze(data,w,h), events:[],
        anchor:{gamma:1.2+i*.01, gain:1.01, source:'manual'}};
}

test('all assets parse and entry uses only local classic scripts in order', () => {
    const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
    assert.deepEqual([...html.matchAll(/<script src="([^"]+)"/g)].map(m=>m[1]), scripts);
    assert.ok(html.includes('<link rel="stylesheet" href="style.css">'));
    for (const file of scripts) new vm.Script(fs.readFileSync(path.join(root,file),'utf8'));
    assert.equal(/https?:\/\//.test(html), false);
});

test('feature schema and architecture remain v4 compatible', () => {
    const r=example(1), m=E.initialize(169,r.stats,21);
    assert.equal(E.SCHEMA,'donut_srgb256_features_v4.0');
    assert.equal(E.ALGORITHM,'donut_feature_mlp_v4.0');
    assert.equal(r.stats.features.length,169);
    assert.equal(new Set(r.stats.names).size,169);
    assert.equal(m.layers.reduce((n,l)=>n+l.w.length+l.b.length,0),2883);
    assert.equal(E.predict(r.stats.features,null).gamma,1);
});

test('a model learns exact synthetic targets without mutating its records', async () => {
    const records=Array.from({length:6},(_,i)=>example(i));
    const snapshot=JSON.stringify(records);
    const r=await E.train({records,epochs:60,regularization:.01,step:.025,seed:2026});
    assert.ok(r.training.loss < r.previousTraining.loss);
    assert.equal(JSON.stringify(records),snapshot);
    assert.equal(r.model.trained,true);
});

test('validation labels and measurements never change fitted weights or normalization', async () => {
    const records=Array.from({length:6},(_,i)=>example(i));
    const val=example(7,'validation');
    const payload={records:[...records,val],epochs:50,seed:99,step:.025};
    const a=await E.train(payload);
    const altered=plain(val); altered.anchor.gamma=.65;
    altered.stats.features=altered.stats.features.map(v=>v+10);
    const b=await E.train({...payload,records:[...records,altered]});
    assert.deepEqual(plain(a.model),plain(b.model));
    assert.equal(a.validation.n,1);
});

test('inference ignores filenames, target records and rating history', () => {
    const r=example(3), m=E.initialize(169,r.stats,82);m.trained=true;
    const before=plain(E.predict(r.stats.features,m));
    r.name='other.png';r.id='renamed';r.events=[{rating:'too_light'}];r.anchor.gamma=2.5;
    assert.deepEqual(plain(E.predict(r.stats.features,m)),before);
});

test('the standalone bundle has inline assets and refuses overwrite', () => {
    const temp=fs.mkdtempSync(path.join(os.tmpdir(),'tone-lab-'));
    const output=path.join(temp,'trainer.html');
    try {
        const python=process.env.PYTHON || 'python3';
        const args=[path.join(root,'build_standalone.py'),output];
        const built=spawnSync(python,args,{encoding:'utf8'});
        assert.equal(built.status,0,built.stderr);
        const html=fs.readFileSync(output,'utf8');
        assert.equal(/<script src=|<link rel="stylesheet"/.test(html),false);
        const js=html.match(/<script>([\s\S]*?)<\/script>/)[1];
        new vm.Script(js);
        assert.ok(js.includes(source));
        assert.notEqual(spawnSync(python,args,{encoding:'utf8'}).status,0);
        assert.equal(fs.readFileSync(output,'utf8'),html);
    } finally {fs.rmSync(temp,{recursive:true,force:true});}
});

#!/usr/bin/env node
// Bake the same narrow frontend migration into a NEW workflow JSON.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const [input,output] = process.argv.slice(2);
if (!input || !output || path.resolve(input) === path.resolve(output)) {
    console.error('Usage: node tools/prepare_tone_lab.cjs input.json NEW-output.json');process.exit(1);
}
try {
    const source = fs.readFileSync(path.join(__dirname,'../web/donut_tone_lab_model.js'),'utf8').replace(/^export /gm,'');
    const migrate = vm.runInNewContext(source+';addToneLabToV5');
    const workflow = JSON.parse(fs.readFileSync(input,'utf8'));
    const report = migrate(workflow);
    if (!report.changed) throw Error('Workflow was not changed: '+report.reason);
    fs.writeFileSync(output,JSON.stringify(workflow,null,2)+'\n',{flag:'wx'});
    console.log('Created '+output+' with Tone Lab disabled. Verify panel -> Run -> PNG metadata -> reload before distribution.');
} catch (error) {console.error(error.message);process.exit(1);}

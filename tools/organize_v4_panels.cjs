// Optional authoring helper: persist the same migrations used by the frontend.
// Writes a NEW JSON file; never overwrites the input or an existing destination.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function load(file, name) {
    const source = fs.readFileSync(path.join(__dirname,'..','web',file),'utf8')
        .replace(/export (function|const)/g,'$1');
    return vm.runInNewContext(`${source}\n${name}`,{});
}
function migrate(workflow) {
    if (!['V4 Beta','V5'].includes(workflow?.extra?.donut_workflow?.release)) throw new Error('Expected a tagged Donut V4 Beta or V5 workflow.');
    const repair = load('donut_seedvr2_workflow_repair.js','repairSeedVR2Workflow');
    const categorize = load('donut_panel_categories_model.js','organizeV4Panels');
    const report = repair(workflow);
    if (report.warnings.length) throw new Error(report.warnings.join('\n'));
    const panels = categorize(workflow).length;
    const split = load('donut_panel_categories_model.js','splitV4FinishingPanels');
    const finishingPanels = split(workflow).length;
    load('donut_panel_categories_model.js','arrangeV4ByFrequency')(workflow);
    return {...report,panels,finishingPanels};
}
if (require.main === module) {
    try {
        const [input,output,...extra] = process.argv.slice(2);
        if (!input || !output || extra.length) throw new Error('Usage: node tools/organize_v4_panels.cjs input.json NEW-output.json');
        const workflow = JSON.parse(fs.readFileSync(input,'utf8'));
        const report = migrate(workflow);
        fs.writeFileSync(output,JSON.stringify(workflow,null,2)+'\n',{flag:'wx'});
        console.log(JSON.stringify(report));
    } catch (error) {console.error(error.message); process.exitCode=1;}
}
module.exports = {migrate};

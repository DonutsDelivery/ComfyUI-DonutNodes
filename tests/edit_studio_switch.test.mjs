import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source = readFileSync(new URL('../web/donut_edit_studio.js', import.meta.url), 'utf8');
test('Editing and Use B switches toggle empty-reference state on and off directly', () => {
    const values = {enabled:true, use_reference_b:false, image_a:'', image_b:''};
    const controls = new Map();
    const element = () => Object.assign(new EventTarget(), {setAttribute(){}, append(){}});
    const context = vm.createContext({element, controls, get:name=>values[name],
        commit:(name,value)=>{values[name]=value;}, render(){}});
    vm.runInContext(source.slice(source.indexOf('    function toggle('), source.indexOf('    function field(')),context);
    for(const name of ['enabled','use_reference_b']) {
        const initial = values[name];
        const button = context.toggle(name,name);
        button.dispatchEvent(new Event('click',{cancelable:true}));
        assert.equal(values[name],!initial);
        button.dispatchEvent(new Event('click',{cancelable:true}));
        assert.equal(values[name],initial);
    }
});

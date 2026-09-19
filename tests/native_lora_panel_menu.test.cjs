const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname,'../web/donut_panel_categories_dom.js'),'utf8')
    .replace(/^import .*;\n/gm,'').replace(/export (function|const)/g,'$1');

test('panel opens native searchable combo with full current catalog, without prefiltering the selected file',()=>{
    let menu, button, changes=0;
    const selected='characters/selected.safetensors';
    const select={dataset:{},value:selected,options:['None',selected,'style/watercolour.safetensors','other/a.safetensors','other/b.safetensors'].map(value=>({value})),
        getAttribute:()=> 'Installed LoRA 1',setAttribute(){},dispatchEvent(){changes++;},
        insertAdjacentElement(_where,element){button=element;}};
    const root={querySelectorAll:()=>[select]};
    const context={document:{createElement:()=>({events:{},isConnected:true,setAttribute(){},addEventListener(name,fn){this.events[name]=fn;},getBoundingClientRect:()=>({left:5,bottom:10})})},
        LiteGraph:{ContextMenu:function(values,options){menu={values,options};}}, LGraphCanvas:{}, app:{canvas:{}}, Event:class {}, MouseEvent:class {}};
    const upgrade=vm.runInNewContext(source+'\nupgradePanelLoraPickers',context);
    assert.equal(upgrade(root),1); assert.equal(upgrade(root),0);
    const click=()=>button.events.click({detail:1,preventDefault(){},stopPropagation(){}});
    click(); assert.deepEqual(Array.from(menu.values),select.options.map(o=>o.value));
    assert.equal(menu.options.className,'dark'); assert.equal(changes,0);
    assert.equal(context.LGraphCanvas.active_canvas,context.app.canvas);
    menu.options.callback('watercolour');assert.equal(select.value,selected);
    menu.options.callback('style/watercolour.safetensors');assert.equal(changes,1);
    assert.equal(select.value,'style/watercolour.safetensors');
    select.options.push({value:'new/installed.safetensors'});click();
    assert.ok(menu.values.includes('new/installed.safetensors'));
    button.isConnected=false;menu.options.callback(selected);assert.equal(changes,1);
});

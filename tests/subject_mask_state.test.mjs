import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readSubjectMask, subjectMaskPrompt, acceptSubjectMaskResult} from '../web/donut_subject_mask_state.js';
const record = {version:1,image:'B',source:'a'.repeat(64),mask:'donutmask:'+'b'.repeat(64),width:100,height:80};
test('mask state survives JSON save/reload',()=>assert.deepEqual(readSubjectMask(JSON.stringify(record),'B'),record));
test('another B or malformed state is not reused',()=>{
    assert.equal(readSubjectMask(record,'new-B'),null);
    assert.equal(readSubjectMask({...record,mask:'../file'},'B'),null);
    assert.equal(readSubjectMask('not json','B'),null);
});
test('preview queues only the mask helper, not the generation workflow',()=>{
    const prompt=subjectMaskPrompt('B','birefnet.safetensors','request-1');
    assert.deepEqual(Object.keys(prompt.output),['request-1']);
    assert.equal(prompt.output['request-1'].class_type,'DonutSubjectMaskPreview');
    assert.equal(prompt.output['request-1'].inputs.request_id,'request-1');
});
test('stale preview from replaced B never overwrites current selection',()=>{
    const job={nodeId:'job',reference:'B'}, output={donut_subject_mask:[record]};
    assert.deepEqual(acceptSubjectMaskResult(job,'job','B',output),record);
    assert.equal(acceptSubjectMaskResult(job,'other-job','B',output),null);
    assert.equal(acceptSubjectMaskResult(job,'job','new-B',output),null);
    assert.equal(acceptSubjectMaskResult(null,'job','B',output),null);
});

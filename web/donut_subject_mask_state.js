export function readSubjectMask(value, reference) {
    try {
        const record = typeof value === 'string' ? JSON.parse(value) : value;
        if (record?.version !== 1 || record.image !== reference || !/^donutmask:[a-f0-9]{64}$/.test(record.mask)
            || !/^[a-f0-9]{64}$/.test(record.source) || !Number.isInteger(record.width) || record.width < 1
            || !Number.isInteger(record.height) || record.height < 1) return null;
        return record;
    } catch { return null; }
}

export function subjectMaskPrompt(reference, model, requestId, prompt = '', threshold = 0.5) {
    return {
        output:{[requestId]:{class_type:'DonutSubjectMaskPreview', inputs:{image_b:reference, model_name:model, request_id:requestId, ...(model === 'sam3.1_multiplex_fp16.safetensors' ? {mask_b_prompt:prompt,mask_b_threshold:threshold} : {})}}},
        workflow:{},
    };
}

export function acceptSubjectMaskResult(job, nodeId, reference, output) {
    if (!job || job.nodeId !== nodeId || job.reference !== reference) return null;
    return readSubjectMask(output?.donut_subject_mask?.[0], reference);
}

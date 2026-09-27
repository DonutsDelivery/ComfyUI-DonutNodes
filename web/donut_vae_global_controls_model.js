export const VAE_SHARED_WIDGETS = ['vae_damage_correction','vae_damage_strength'];

export function vaeCorrectionMirrorWidgets(config, widgetName, excluded, resolve) {
    const paths = config?.vae_correction_global?.targets?.[widgetName] || [];
    const seen = new Set(excluded ? [excluded] : []), result = [];
    for (const path of paths) {
        if (!Array.isArray(path)) continue;
        const node = resolve(path);
        const widget = node?.widgets?.find(item => item?.name === widgetName);
        if (!node || !widget || seen.has(node)) continue;
        seen.add(node);
        result.push({node,widget});
    }
    return result;
}

// The base decoder retains the old workflow's primary VAE choice. Use those
// serialized values to migrate legacy per-stage settings exactly once.
export function prepareVaeCorrectionMigration(panel, resolve) {
    const config = panel?.properties?.donut_app_controls;
    const shared = config?.vae_correction_global;
    if (!shared || config.vae_correction_initialized === 1) return null;
    const source = resolve(shared.source_path || []);
    const sourceWidgets = Object.fromEntries(VAE_SHARED_WIDGETS.map(name =>
        [name,source?.widgets?.find(widget => widget?.name === name)]));
    if (Object.values(sourceWidgets).some(widget => !widget)) return null;
    const mirrors = Object.fromEntries(VAE_SHARED_WIDGETS.map(name =>
        [name,vaeCorrectionMirrorWidgets(config,name,null,resolve)]));
    if (VAE_SHARED_WIDGETS.some(name => !mirrors[name].length
            || mirrors[name].length !== (shared.targets?.[name] || []).length)) return null;
    const updates = [];
    for (const name of VAE_SHARED_WIDGETS) {
        for (const {node,widget} of mirrors[name]) {
            if (node === source || Object.is(widget.value,sourceWidgets[name].value)) continue;
            updates.push({node,widget,value:sourceWidgets[name].value});
        }
    }
    return {config,updates};
}

// Presentation-only migration for the V4 panels. Node values, sockets and
// prompts remain owned by their existing modules. Works with live graphs and
// serialized workflows; never infer a widget's meaning from a numeric node ID.
const pathKey = path => JSON.stringify((path || []).map(String));
const cleanTitle = title => String(title || '').replace(/^Advanced\s*[·:]\s*/i, '');
export const SIZE_FIELDS = [
    ['resolution_mode', 'Output sizing mode'], ['aspect_ratio', 'Output aspect ratio'],
    ['megapixels', 'Output megapixels'], ['width', 'Custom output width'],
    ['height', 'Custom output height'], ['multiple', 'Pixel grid'],
];
export function widgetValue(node, name) {
    const live = node?.widgets?.find(widget => widget.name === name);
    if (live) return live.value;
    return node?.widgets_values_named?.[name];
}
function hasWidget(node, name) {
    return !!node?.widgets?.some(widget => widget.name === name)
        || Object.hasOwn(node?.widgets_values_named || {}, name);
}
export function graphEntries(root) {
    const definitions = new Map((root?.definitions?.subgraphs || []).map(graph => [String(graph.id), graph]));
    const result = [];
    function walk(graph, path = [], ancestors = new Set()) {
        if (!graph || ancestors.has(graph)) return;
        const next = new Set(ancestors).add(graph);
        for (const node of graph.nodes || graph._nodes || []) {
            const at = [...path, node.id];
            result.push({node, path:at});
            walk(node.subgraph || definitions.get(String(node.type)), at, next);
        }
    }
    walk(root);
    return result;
}
export function panelRole(panel) {
    const config = panel.properties?.donut_app_controls;
    if (!Array.isArray(config?.groups)) return null;
    const saved = panel.properties.donut_panel_role;
    if (['models','loras','prompts','guidance','generate','save'].includes(saved)) return saved;
    const title = String(panel.title || '').toLowerCase();
    if (/generate.*finish/.test(title)) return 'generate';
    if (/save images/.test(title)) return 'save';
    if (/seed.*guidance/.test(title)) return 'guidance';
    if (/prompts/.test(title)) return 'prompts';
    if (/loras/.test(title)) return 'loras';
    if (/models/.test(title)) return 'models';
    return null; // Leave custom/unknown panels and the wildcard library alone.
}
const bucket = (title, rank, advanced = false) => ({title, rank, advanced});
function classify(role, group, control) {
    const original = group.donut_source_title || group.title;
    const title = cleanTitle(original), w = control.widget || '';
    const advanced = group.advanced === true;
    if (role === 'models') {
        if (/^uncensorfix_/.test(w)) return bucket('UncensorFix', 70.5);
        if (/Fusion/.test(title)) {
            if (!advanced) return bucket('Fusion', 70);
            if (/^tap_|^per_layer_weights$/.test(w)) return bucket('Fusion · image taps', 71, true);
            if (/^projector_/.test(w)) return bucket('Fusion · projector', 72, true);
            if (/^fusion_/.test(w)) return bucket('Fusion · transformer', 73, true);
            return bucket('Fusion · execution and interface', 74, true);
        }
        if (/Model merge/i.test(title)) {
            if (/^blocks\./.test(w)) return bucket('Model merge · diffusion blocks', 82, true);
            if (['execution_mode','ratio_mode','body_ratio','fusion_ratio'].includes(w)) return bucket('Model merge · ratios and execution', 80, true);
            return bucket('Model merge · input and conditioning weights', 81, true);
        }
        const names = ['Model setup','Primary model','Secondary model','Text encoder','VAE','Upscale model','Face detector','SAM'];
        const at = names.indexOf(title);
        if (at >= 0) return bucket(title === 'Upscale model' ? 'Donut upscale model' : title === 'SAM' ? 'Face-detail segmentation · SAM' : title, at * 5, advanced);
    }
    if (role === 'loras' && title === 'LoRA settings') {
        if (w === 'global_block_vector') return bucket('Global block weights', 20);
        if (['safe_stack','safe_limit','fusion_aware','max_fusion_boost'].includes(w)) return bucket('LoRA safety and fusion', 30);
        return bucket('LoRA setup', 10);
    }
    if (role === 'guidance') {
        if (/^variance_/.test(w)) return bucket(advanced ? 'Seed variance · advanced' : 'Seed variance', advanced ? 21 : 20, advanced);
        if (w === 'alpha' || /^nag_/.test(w)) return bucket('Negative attention guidance · NAG', 10, advanced);
        if (/Shared seed/.test(title)) return bucket('Shared seed', 0, advanced);
    }
    if (role === 'generate') {
        if (group.donut_image_size) return bucket('Image size', 0);
        if (w === 'batch_size') return bucket('Batch', 5);
        if (/AuraFlow/i.test(title)) return bucket('AuraFlow sampling', 12);
        if (/SeedVR2.*post/i.test(title)) {
            const basic = ['enabled','seedvr2_upscale_factor','seedvr2_model_name','seedvr2_vae_name','seedvr2_color_correction'];
            return bucket(basic.includes(w) ? 'SeedVR2 · post upscale' : 'SeedVR2 · post upscale · advanced', basic.includes(w) ? 60 : 61, !basic.includes(w));
        }
        if (title === 'Generation and finish') {
            if (['denoise','tiled_diffusion','rescale_factor','upscale_1_enabled'].includes(w)) return bucket('Donut hires · first upscale', 30);
            if (['denoise_2','upscale_2_enabled'].includes(w)) return bucket('Donut hires · second upscale', 50);
            if (['denoise_1','max_faces'].includes(w) || control.mode === 'bypass') return bucket('Face detail', 40);
            return bucket('Base sampling', 10);
        }
        const stage = /First upscale/i.test(title) ? ['first',30] : /Second upscale/i.test(title) ? ['second',50] : null;
        if (stage) {
            const [label, rank] = stage;
            // Legacy replacement-engine controls stay available, but must not
            // masquerade as the new post-processing pass.
            if (w === 'upscale_engine') return bucket(`Donut hires · ${label} upscale · alternative engine`, rank + 3, true);
            if (/^seedvr2_/.test(w)) return bucket(`Donut hires · ${label} upscale · SeedVR2 replacement settings`, rank + 4, true);
            if (['rescale_factor','denoise','enabled','tiled_diffusion'].includes(w)) return bucket(`Donut hires · ${label} upscale`, rank);
            return bucket(`Donut hires · ${label} upscale · ${/^nag_/.test(w) ? 'NAG' : 'advanced'}`, rank + (/^nag_/.test(w) ? 2 : 1), true);
        }
        if (/Face detail|Differential diffusion/i.test(title)) {
            const category = /^nag_/.test(w) ? 'NAG' : /^(bbox_|sam_|drop_size)/.test(w) ? 'detection and masks' : 'sampling';
            return bucket(`Face detail · ${category}`, category === 'sampling' ? 41 : category === 'detection and masks' ? 42 : 43, true);
        }
        if (/Base pass/i.test(title)) return bucket(/^nag_/.test(w) ? 'Base sampling · NAG' : 'Base sampling · advanced', 13, true);
        if (/Sampler|ER SDE/i.test(title)) return bucket(`Base sampling · ${title.toLowerCase()}`, 14, true);
    }
    if (role === 'save' && title === 'Final save') {
        if (['extension','quality','optimize_image','lossless_webp','dpi'].includes(w)) return bucket('Format and compression', 20);
        if (['embed_workflow','show_previews'].includes(w)) return bucket('Metadata and previews', 30);
        return bucket('Destination and filenames', 10);
    }
    return bucket(title, group.donut_category_rank ?? (role === 'models' ? 60 : role === 'generate' ? 20 : 50), advanced);
}
export function categorizeGroups(role, groups) {
    const result = [];
    for (const group of groups) {
        if (group.donut_category_fixed) { result.push(group); continue; }
        // These groups have renderers with state (prompt variants, row pickers,
        // shared wildcards). Keep the entire renderer configuration together.
        if (group.loras || group.prompt_sets || group.shared_prompt_tools || group.wildcard_library || !group.controls?.length) {
            result.push({...group, donut_category_rank:group.loras || group.shared_prompt_tools ? 0 : group.prompt_sets ? 10 : group.donut_category_rank ?? 50});
            continue;
        }
        const parts = new Map();
        for (const control of group.controls) {
            const category = classify(role, group, control);
            const key = `${category.rank}/${category.title}/${category.advanced}`;
            if (!parts.has(key)) parts.set(key, {
                ...group, title:category.title, advanced:category.advanced,
                donut_source_title:group.donut_source_title || group.title,
                donut_category_rank:category.rank, donut_category_fixed:true, controls:[],
            });
            parts.get(key).controls.push(control);
        }
        result.push(...parts.values());
    }
    // Standard controls for one stage can originate from several old groups.
    // Merge only metadata-compatible categories; keep custom/conditional groups
    // separate. The fixed marker prevents reclassifying a promoted second-stage
    // scale as the first-stage scale on the next import.
    const merged = [], byCategory = new Map();
    for (const group of result) {
        const {controls, donut_source_title, ...metadata} = group;
        const key = JSON.stringify(Object.fromEntries(Object.entries(metadata).sort(([a],[b]) => a.localeCompare(b))));
        if (group.donut_category_fixed && byCategory.has(key)) byCategory.get(key).controls.push(...controls);
        else {
            const value = {...group, ...(controls ? {controls:[...controls]} : {})};
            merged.push(value);
            if (group.donut_category_fixed) byCategory.set(key,value);
        }
    }
    for (const group of merged) if (/^Donut hires · (first|second) upscale$/.test(group.title)) {
        const priority = control => /enabled$/.test(control.widget || '') ? 0 : control.widget === 'rescale_factor' ? 1 : 2;
        group.controls.sort((a,b) => priority(a) - priority(b));
    }
    return merged.sort((a,b) => a.donut_category_rank - b.donut_category_rank);
}
function sameFamily(a, b) {
    return pathKey(a.properties?.donut_app_controls?.seed_path) === pathKey(b.properties?.donut_app_controls?.seed_path);
}
function moveGroups(source, target, predicate) {
    const from = source.properties.donut_app_controls;
    const moving = from.groups.filter(predicate);
    if (!moving.length) return;
    from.groups = from.groups.filter(group => !predicate(group));
    target.properties.donut_app_controls.groups.push(...moving.map(group => ({...group,donut_category_fixed:false})));
}
export function organizeV4Panels(root) {
    const entries = graphEntries(root);
    const panels = entries.map(entry => entry.node).filter(node => panelRole(node));
    const originals = new Map(panels.map(panel => [panel, JSON.stringify(panel.properties)]));
    for (const panel of panels) {
        panel.properties.donut_panel_role = panelRole(panel);
        panel.properties.donut_columns = 'sections';
    }
    for (const generate of panels.filter(panel => panelRole(panel) === 'generate')) {
        // Cross-panel ownership requires V4's explicit shared seed path.
        // A custom panel with no family metadata is categorized in place only.
        if (!generate.properties.donut_app_controls.seed_path?.length) continue;
        const family = panels.filter(panel => sameFamily(panel, generate));
        if (family.filter(panel => panelRole(panel) === 'generate').length !== 1) continue;
        const models = family.filter(panel => panelRole(panel) === 'models');
        if (models.length === 1) moveGroups(models[0], generate, group => /AuraFlow/.test(group.donut_source_title || group.title));
        const guidance = family.filter(panel => panelRole(panel) === 'guidance');
        const prompts = family.filter(panel => panelRole(panel) === 'prompts');
        if (guidance.length === 1 && prompts.length === 1) {
            for (const group of guidance[0].properties.donut_app_controls.groups) {
                const controls = (group.controls || []).filter(control => ['edit_negative','separator'].includes(control.widget));
                if (!controls.length) continue;
                group.controls = group.controls.filter(control => !controls.includes(control));
                prompts[0].properties.donut_app_controls.groups.push({...group, title:'Prompt composition · editing override and separator', donut_source_title:'Prompt composition · editing override and separator', donut_category_fixed:false, advanced:true, controls});
            }
        }
        const seed = generate.properties.donut_app_controls.seed_path;
        const studios = entries.filter(({node}) => (node.comfyClass || node.type) === 'DonutEditStudio'
            && pathKey(node.properties?.donut_seed_path) === pathKey(seed)
            && SIZE_FIELDS.every(([name]) => hasWidget(node, name)));
        if (studios.length !== 1) continue;
        const {node:studio, path} = studios[0];
        const groups = generate.properties.donut_app_controls.groups;
        if (!groups.some(group => group.donut_image_size && pathKey(group.donut_image_size) === pathKey(path))) {
            groups.unshift({title:'Image size', advanced:false, donut_image_size:path,
                description:'Global output size for generation and editing. Reference-A sizing modes take effect while Editing is enabled; otherwise the preset size is used. The readout in Edit Studio shows the effective size.',
                controls:SIZE_FIELDS.map(([widget,title]) => ({path:[...path],widget,title})),
            });
        }
    }
    for (const panel of panels) {
        const config = panel.properties.donut_app_controls;
        config.groups = categorizeGroups(panelRole(panel), config.groups);
    }
    return panels.filter(panel => originals.get(panel) !== JSON.stringify(panel.properties));
}

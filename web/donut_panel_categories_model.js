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
    if (['models','loras','prompts','guidance','generate','save','hires','face','post'].includes(saved)) return saved;
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
        // Per-stage NAG groups expose only the enable toggle; the shared NAG
        // settings live in the guidance panel's global group instead.
        if (/^nag_enabled$/.test(w) && /upscale/i.test(title)) {
            const stageRank = /First upscale/i.test(title) ? 30 : 50;
            return bucket(`Donut · ${/First/i.test(title) ? 'first' : 'second'} upscale · NAG on/off`, stageRank, true);
        }
        if (/^nag_/.test(w) && /upscale|Face detail/i.test(title)) return null; // moved to the shared NAG panel
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
            if (!category) continue; // null = reclassified elsewhere; the control is dropped from this panel
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
// Per-stage NAG groups expose only each stage's enable toggle; the shared
// NAG settings (phi/auto/tau/sigmas/ref) live once on the base sampler's
// widgets shown inside the guidance panel. Values are bound to the base
// node explicitly, and a save that still carries stage-level duplicates
// heals by moving those controls out of the stage panels on load.
export const NAG_SHARED_WIDGETS = ['nag_phi','nag_auto_phi','nag_phi_scale','nag_tau','nag_sigma_start','nag_sigma_end','nag_ref_boost','nag_ref_boost_a','nag_fit_mode'];
function consolidateNagGlobals(entries, generate) {
    const panels = entries.map(entry => entry.node);
    const config = generate.properties.donut_app_controls;
    if (!Array.isArray(config?.groups)) return;
    const base = config.groups.find(group => /Base sampling · NAG$/.test(group.title));
    if (!base?.controls?.length) return;
    const baseWidgets = new Map(base.controls.map(control => [control.widget, control]));
    const shared = new Set(NAG_SHARED_WIDGETS);
    const stages = panels.filter(n => ['generate','hires','face'].includes(panelRole(n)));
    for (const stage of stages) {
        if (stage === generate) continue;
        const groups = stage.properties.donut_app_controls?.groups;
        if (!Array.isArray(groups)) continue;
        let touched = false;
        for (const group of groups) {
            if (!/NAG/.test(group.title) || !Array.isArray(group.controls)) continue;
            const kept = group.controls.filter(control => {
                if (!shared.has(control.widget) || !baseWidgets.has(control.widget)) return true;
                // Preserve a stage that deliberately differs? No — lockstep by
                // user preference: stage widgets lose their own copies and the
                // shared group silently drives every stage.
                touched = true;
                return false;
            });
            if (kept.length !== group.controls.length) { group.controls = kept; }
        }
        if (touched && !groups.some(group => group.controls?.some(control => /^nag_enabled$/.test(control.widget)))) continue;
        if (touched && !groups.length) stage.properties.donut_app_controls.groups = groups.filter(group => group.controls?.length);
        void touched;
    }
    // Ensure the base NAG group carries every shared widget even if the
    // saved base group predates one of them (auto phi, for instance).
    for (const widget of NAG_SHARED_WIDGETS) {
        if (baseWidgets.has(widget)) continue;
        const stageWithWidget = stages.flatMap(stage => (stage.properties.donut_app_controls?.groups || []).flatMap(group => group.controls || []))
            .find(control => control.widget === widget);
        if (stageWithWidget) base.controls.push({...stageWithWidget, path:[...base.controls[0].path]});
    }
}
export function organizeV4Panels(root) {
    const entries = graphEntries(root);
    const panels = entries.map(entry => entry.node).filter(node => panelRole(node));
    const originals = new Map(panels.map(panel => [panel, JSON.stringify(panel.properties)]));
    for (const panel of panels) {
        panel.properties.donut_panel_role = panelRole(panel);
        panel.properties.donut_columns = 'sections';
    }
    for (const panel of panels.filter(panel => panelRole(panel) === 'loras')) {
        const groups = panel.properties.donut_app_controls.groups;
        const execution = groups.flatMap(group => group.controls || []).find(control => control.widget === 'execution_mode');
        const owner = execution && entries.find(entry => pathKey(entry.path) === pathKey(execution.path))?.node;
        if (!execution || !hasWidget(owner, 'chunk_lora') || groups.some(group => group.controls?.some(control => control.widget === 'chunk_lora'))) continue;
        groups.push({title:'Experimental · low VRAM', advanced:true, donut_category_fixed:true,
            controls:[['chunk_lora','LoRA chunking'],['chunk_edit_mlp','Edit MLP chunking'],['chunk_edit_norm','Edit normalization chunking']]
                .map(([widget,title]) => ({path:[...execution.path],widget,title}))});
    }
    for (const generate of panels.filter(panel => panelRole(panel) === 'generate')) {
        // Cross-panel ownership requires V4's explicit shared seed path.
        // A custom panel with no family metadata is categorized in place only.
        consolidateNagGlobals(entries, generate);
        if (!generate.properties.donut_app_controls.seed_path?.length) continue;
        const family = panels.filter(panel => sameFamily(panel, generate));
        if (family.filter(panel => panelRole(panel) === 'generate').length !== 1) continue;
        const models = family.filter(panel => panelRole(panel) === 'models');
        if (models.length === 1) moveGroups(models[0], generate, group => /AuraFlow/.test(group.donut_source_title || group.title));
        const guidance = family.filter(panel => panelRole(panel) === 'guidance');
        const prompts = family.filter(panel => panelRole(panel) === 'prompts');
        if (models.length === 1 && guidance.length === 1) {
            const modelGroups = models[0].properties.donut_app_controls.groups;
            const fusionControls = modelGroups.flatMap(group => group.controls || []);
            // compatibility_preset can be promoted onto the outer subgraph in V5.
            // The experiment widgets live on the actual inner Fusion Control, so
            // prefer an existing inner-only widget as the path anchor.
            const fusionAnchor = fusionControls.find(control => control.widget === 'uncensorfix_controls')
                || fusionControls.find(control => control.widget === 'tap_method')
                || fusionControls.find(control => control.widget === 'compatibility_preset');
            const guidanceGroups = guidance[0].properties.donut_app_controls.groups;
            const experimentWidgets = [
                ['nag_text_energy_compensation', 'Text-energy compensation'],
                ['nag_batch_txtfusion', 'Batch equal-length text fusion'],
            ];
            const experimentTitles = new Map(experimentWidgets);
            // Workflows saved while an earlier binding was active carry the
            // injected controls with stale paths; repoint them to the current
            // anchor instead of skipping, so the workflow heals on load.
            for (const group of guidanceGroups) {
                for (const control of group.controls || []) {
                    const title = experimentTitles.get(control.widget);
                    if (title === undefined) continue;
                    if (fusionAnchor && JSON.stringify(control.path) !== JSON.stringify(fusionAnchor.path)) {
                        control.path = [...fusionAnchor.path];
                        control.title = title;
                    }
                }
            }
            const existing = new Set(guidanceGroups.flatMap(group => group.controls || []).map(control => control.widget));
            if (fusionAnchor && experimentWidgets.some(([widget]) => !existing.has(widget))) {
                guidanceGroups.push({
                    title:'Advanced · NAG experiments', advanced:true, donut_category_fixed:false,
                    description:'Experimental text-fusion compatibility controls. Both default Off.',
                    controls:experimentWidgets
                        .filter(([widget]) => !existing.has(widget))
                        .map(([widget,title]) => ({path:[...fusionAnchor.path],widget,title})),
                });
            }
        }
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
        const role = panelRole(panel);
        config.groups = categorizeGroups(['hires','face','post'].includes(role) ? 'generate' : role, config.groups);
    }
    return panels.filter(panel => originals.get(panel) !== JSON.stringify(panel.properties));
}

// Split only the standard, tagged saved layout. Existing control paths and
// renderer configurations move intact; no generation node or setting changes.
export function splitV4FinishingPanels(root) {
    if (!['V4 Beta','V5'].includes(root?.extra?.donut_workflow?.release) || !Array.isArray(root.nodes)
            || !Array.isArray(root.extra?.donut_layout?.columns)) return [];
    organizeV4Panels(root);
    const sources = root.nodes.filter(node => panelRole(node) === 'generate');
    if (sources.length !== 1) return [];
    const source = sources[0], config = source.properties.donut_app_controls;
    const columns = root.extra.donut_layout.columns;
    const at = columns.findIndex(column => column.some(id => String(id) === String(source.id)));
    if (at < 0) return [];
    const definitions = [
        ['first', '07 · First upscale', 'hires', '#83bbd9', /^Donut hires · first upscale/],
        ['face', '08 · Face detail', 'face', '#e4a4bf', /^Face detail/],
        ['second', '09 · Second upscale', 'hires', '#83bbd9', /^Donut hires · second upscale/],
        ['post', '10 · SeedVR2 upscale', 'post', '#90c9ae', /^SeedVR2 · post upscale/],
    ];
    const created = [];
    let next = Math.max(root.last_node_id || 0, ...graphEntries(root).map(({node}) => Number(node.id) || 0));
    for (const [stage, title, role, color, pattern] of definitions) {
        const groups = config.groups.filter(group => pattern.test(group.title));
        if (!groups.length) continue;
        // A previously split/customized family must not acquire duplicate panels.
        if (root.nodes.some(node => node.properties?.donut_finish_stage === stage && sameFamily(node,source))) continue;
        config.groups = config.groups.filter(group => !groups.includes(group));
        const panel = {
            id:++next, type:'DonutWorkflowPanel', title, pos:[...(source.pos || [0,0])],
            size:[480,400], flags:{}, order:root.nodes.length, mode:0, inputs:[], outputs:[],
            properties:{donut_panel_role:role, donut_finish_stage:stage, donut_columns:'sections',
                panel_width:480, panel_min_width:480,
                donut_app_controls:{...config, title, groups}},
            color, bgcolor:'#1b242c', widgets_values:[],
        };
        root.nodes.push(panel); created.push(panel);
    }
    if (!created.length) return [];
    root.last_node_id = next;
    source.title = '06 · Generate';
    const stageId = stage => created.find(node => node.properties.donut_finish_stage === stage)?.id;
    const additions = [['first','face'],['second','post']].map(stages => stages.map(stageId).filter(id => id != null)).filter(ids => ids.length);
    columns.splice(at + 1, 0, ...additions);
    const appInputs = root.extra.linearData?.inputs;
    if (Array.isArray(appInputs)) {
        const index = appInputs.findIndex(([id]) => String(id) === String(source.id));
        appInputs.splice(index < 0 ? appInputs.length : index + 1, 0,
            ...created.map(panel => [panel.id,'workflow_controls']));
    }
    for (const node of root.nodes) {
        if (node.type === 'DonutLatestPreview') node.title = '11 · Latest result';
        if (panelRole(node) === 'save') node.title = '12 · Save images';
    }
    return created;
}

// Apply once so later user rearrangements survive saving/reopening. Settings,
// execution nodes and controls stay untouched; this only orders presentation.
export function arrangeV4ByFrequency(root) {
    const layout = root?.extra?.donut_layout;
    if (!['V4 Beta','V5'].includes(root?.extra?.donut_workflow?.release) || !Array.isArray(layout?.columns)
            || layout.panel_order === 'setup-finish-iterate-v2') return false;
    const nodes = root.nodes || [];
    const unique = predicate => {
        const matches = nodes.filter(predicate);
        return matches.length === 1 ? matches[0] : null;
    };
    const roles = Object.fromEntries(['models','loras','generate','save','prompts','guidance']
        .map(role => [role,unique(node => panelRole(node) === role)]));
    const stages = Object.fromEntries(['first','face','second','post']
        .map(stage => [stage,unique(node => node.properties?.donut_finish_stage === stage)]));
    const edit = unique(node => node.type === 'DonutEditStudio');
    const result = unique(node => node.type === 'DonutLatestPreview');
    if (![...Object.values(roles),...Object.values(stages),edit,result].every(Boolean)) return false;
    const reference = unique(node => node.type === 'DonutReferenceStudio');
    const wildcards = unique(node => node.properties?.donut_app_controls?.groups?.some(group => group.wildcard_library));
    const order = [roles.models,roles.loras,roles.generate,stages.first,stages.second,
        stages.face,stages.post,roles.save,edit,roles.prompts,roles.guidance,result];
    const titles = ['Models','LoRAs & block weights','Generation setup','First upscale','Second upscale',
        'Face detail','SeedVR2 upscale','Save images','Image setup & editing','Prompts','Seed & guidance','Latest result'];
    order.forEach((node,index) => {node.title = `${String(index+1).padStart(2,'0')} · ${titles[index]}`;});
    const columns = [[roles.models,roles.generate],[roles.loras],[stages.first,stages.second],
        [stages.face,stages.post,roles.save],[edit,reference],[roles.prompts,wildcards],[roles.guidance,result]]
        .map(column => column.filter(Boolean).map(node => node.id));
    const assigned = new Set(columns.flat().map(String));
    // Keep existing auxiliary nodes reachable, beneath setup rather than
    // between the frequently adjusted controls and the preview.
    for (const id of layout.columns.flat()) {
        if (!assigned.has(String(id))) {columns[0].push(id); assigned.add(String(id));}
    }
    layout.columns = columns;
    layout.panel_order = 'setup-finish-iterate-v2';
    const inputs = root.extra.linearData?.inputs;
    if (Array.isArray(inputs)) {
        const rank = new Map([...order.slice(0,9),reference,roles.prompts,wildcards,roles.guidance,result]
            .filter(Boolean).map((node,index) => [String(node.id),index]));
        inputs.sort((a,b) => (rank.get(String(a[0])) ?? Infinity) - (rank.get(String(b[0])) ?? Infinity));
    }
    return true;
}

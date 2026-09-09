export function createProgress(api, getGraph) {
    const root = document.createElement('section');
    root.className = 'donut-execution-progress';
    const label = document.createElement('p'), bar = document.createElement('progress');
    bar.max = 1; bar.value = 0;
    bar.setAttribute('aria-label', 'Current generation stage progress');
    label.textContent = 'Ready'; root.append(label, bar);
    let run, active;
    const title = id => {
        let graph = getGraph(), node;
        for (const part of String(id ?? '').split(':')) {
            node = graph?.getNodeById?.(Number(part)) ?? graph?.getNodeById?.(part);
            if (!node) break;
            graph = node.subgraph;
        }
        return node?.title || node?.type || `Node ${id}`;
    };
    const accepts = detail => detail.prompt_id && detail.prompt_id === run;
    const indeterminate = text => { label.textContent = text; bar.removeAttribute('value'); };
    const handlers = {
        execution_start: ({detail}) => { run = detail.prompt_id; active = null; indeterminate('Starting generation…'); },
        executing: ({detail}) => {
            if (!accepts(detail) || detail.node == null) return;
            active = detail.node;
            indeterminate(title(detail.display_node ?? active));
        },
        progress_state: ({detail}) => {
            if (!accepts(detail)) return;
            const states = Object.values(detail.nodes || {});
            const state = states.find(s => s.state === 'running' && s.node_id === active)
                || states.filter(s => s.state === 'running').at(-1);
            if (!state) return;
            active = state.node_id;
            const name = title(state.display_node_id ?? state.real_node_id ?? active);
            if (state.max > 0 && state.value > 0) {
                bar.value = Math.min(1, Math.max(0, state.value / state.max));
                label.textContent = `${name} · ${state.value}/${state.max} · ${Math.round(bar.value * 100)}%`;
            } else indeterminate(name);
        },
        progress: ({detail}) => {
            if (!accepts(detail) || !(detail.max > 0)) return;
            active = detail.node ?? active;
            bar.value = Math.min(1, Math.max(0, detail.value / detail.max));
            label.textContent = `${title(active)} · ${detail.value}/${detail.max} · ${Math.round(bar.value * 100)}%`;
        },
        execution_success: ({detail}) => { if (accepts(detail)) { bar.value = 1; label.textContent = 'Generation complete'; run = null; } },
        execution_error: ({detail}) => { if (accepts(detail)) { bar.value = 0; label.textContent = `Failed · ${title(detail.node_id ?? active)}`; run = null; } },
        execution_interrupted: ({detail}) => { if (accepts(detail)) { bar.value = 0; label.textContent = 'Generation cancelled'; run = null; } },
    };
    return {element: root,
        attach() { for (const [name, fn] of Object.entries(handlers)) api.addEventListener(name, fn); },
        detach() { for (const [name, fn] of Object.entries(handlers)) api.removeEventListener(name, fn); },
    };
}

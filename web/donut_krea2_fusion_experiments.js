import { app } from "../../scripts/app.js";

const NODE_NAME = "DonutKrea2FusionControl";
const CLASSIC = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0";
const NEUTRAL = Array(12).fill("1.0").join(",");

const EXPERIMENTS = {
  "Experiment · NAG-friendly mean": {
    tap_method: "Donut 12-tap gains", tap_profile: "classic", tap_strength: 1.0,
    tap_formula: "scale_around_1", tap_normalization: "mean_gain",
  },
  "Experiment · NAG-friendly static RMS": {
    tap_method: "Donut 12-tap gains", tap_profile: "classic", tap_strength: 1.0,
    tap_formula: "scale_around_1", tap_normalization: "rms_gain",
  },
  "Experiment · NAG-friendly power 0.60": {
    tap_method: "Donut 12-tap gains", tap_profile: "classic", tap_strength: 0.60,
    tap_formula: "geometric_power", tap_normalization: "none",
  },
  "Experiment · soft tensor RMS 0.75": {
    tap_method: "Donut 12-tap gains", tap_profile: "classic", tap_strength: 0.75,
    tap_formula: "scale_around_1", tap_normalization: "tensor_rms",
  },
};

const COMMON = {
  projector_method: "Donut projector-input gains",
  projector_profile: "off",
  projector_strength: 1.0,
  projector_formula: "scale_around_1",
  projector_normalization: "none",
  fusion_method: "Standard Krea2 fusion",
  fusion_strength: 1.0,
};

function widget(node, name) {
  return node?.widgets?.find(item => item.name === name);
}

function fusionDescendant(graph) {
  for (const node of graph?.nodes || []) {
    const type = node.comfyClass || node.type || node.properties?.["Node name for S&R"];
    if (type === NODE_NAME && widget(node, "tap_method")) return node;
    const nested = fusionDescendant(node.subgraph);
    if (nested) return nested;
  }
}

function appendChoices(preset) {
  let values = preset?.options?.values;
  if (typeof values === "function") values = values();
  if (!Array.isArray(values)) return;
  const next = [...values];
  for (const name of Object.keys(EXPERIMENTS)) if (!next.includes(name)) next.push(name);
  preset.options.values = next;
}

function applyExperiment(node, name) {
  const recipe = EXPERIMENTS[name];
  if (!recipe || !widget(node, "tap_method")) return false;
  const preset = widget(node, "compatibility_preset");
  node._donutApplyingKrea2Preset = true;
  try {
    for (const [key, value] of Object.entries({...COMMON, ...recipe})) {
      const item = widget(node, key);
      if (!item) continue;
      item.value = value;
      item.callback?.(value, app.canvas, node);
    }
    const tapWeights = widget(node, "per_layer_weights");
    if (tapWeights) tapWeights.value = CLASSIC;
    const projectorWeights = widget(node, "projector_layer_weights");
    if (projectorWeights) projectorWeights.value = NEUTRAL;
    if (preset) preset.value = name;
  } finally {
    node._donutApplyingKrea2Preset = false;
  }
  node.setDirtyCanvas?.(true, true);
  return true;
}

function applyOuterExperimentInputs(node, name) {
  const recipe = EXPERIMENTS[name];
  if (!recipe || widget(node, "tap_method")) return;

  // V4 exposes tap_strength on the outer Generate/Fusion subgraph. That value
  // is an actual subgraph input and therefore wins over the nested widget at
  // execution time. Keep it in lockstep with the experiment recipe; otherwise
  // power 0.60 is executed as power 1.0 and collapses back to the full classic
  // profile (which can be identical to Rebalance for float32 conditioning).
  const strength = widget(node, "tap_strength");
  if (!strength) return;
  strength.value = recipe.tap_strength;
  strength.callback?.(recipe.tap_strength, app.canvas, node);
  node.setDirtyCanvas?.(true, true);
}

function decorate(node) {
  const preset = widget(node, "compatibility_preset");
  if (!preset) return;
  appendChoices(preset);
  if (preset._donutExperimentPresets) return;
  preset._donutExperimentPresets = true;
  const previous = preset.callback;
  preset.callback = function(value) {
    const result = previous?.apply(this, arguments);
    if (!EXPERIMENTS[value]) return result;
    applyOuterExperimentInputs(node, value);
    const target = widget(node, "tap_method") ? node : fusionDescendant(node.subgraph);
    if (target) applyExperiment(target, value);
    preset.value = value;
    return result;
  };
  if (EXPERIMENTS[preset.value]) {
    applyOuterExperimentInputs(node, preset.value);
    const target = widget(node, "tap_method") ? node : fusionDescendant(node.subgraph);
    if (target) applyExperiment(target, preset.value);
  }
}

function visit(graph) {
  for (const node of graph?.nodes || []) {
    decorate(node);
    visit(node.subgraph);
  }
}

app.registerExtension({
  name: "Donut.Krea2FusionExperiments",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
      const result = created?.apply(this, arguments);
      queueMicrotask(() => decorate(this));
      return result;
    };
  },
  afterConfigureGraph() {
    // Run after the stable preset/simple-mode extensions have installed their
    // callbacks so this experimental wrapper remains the outermost handler.
    queueMicrotask(() => visit(app.rootGraph));
  },
});

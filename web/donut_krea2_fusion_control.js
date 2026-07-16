import { app } from "../../scripts/app.js";

const NODE_NAME = "DonutKrea2FusionControl";
const CUSTOM = "Custom settings";

const TAP_DONUT = "Donut 12-tap gains";
const TAP_REBALANCE = "nova452 Rebalance operation";
const PROJECTOR_DONUT = "Donut projector-input gains";
const PROJECTOR_BYPASS_2 = "Krea2FilterBypass 2vector diff";
const PROJECTOR_BYPASS_3 = "Krea2FilterBypass 3vector diff";
const FUSION_STANDARD = "Standard Krea2 fusion";
const FUSION_ENHANCER = "capitan01R Krea2T-Enhancer operation";

const PROFILE_VECTORS = {
  off: "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
  classic: "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0",
  deep_2: "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,1.0,1.0",
  deep_3: "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,1.0",
};

const PRESETS = {
  "COPY settings: Krea2FilterBypass 2vector": {
    tap_method: TAP_DONUT,
    tap_profile: "off",
    tap_strength: 1.0,
    tap_formula: "scale_around_1",
    tap_normalization: "tensor_rms",
    projector_method: PROJECTOR_BYPASS_2,
    projector_profile: "off",
    projector_strength: 1.0,
    projector_formula: "scale_around_1",
    projector_normalization: "none",
    fusion_method: FUSION_STANDARD,
    fusion_strength: 1.0,
  },
  "COPY settings: Krea2FilterBypass 3vector": {
    tap_method: TAP_DONUT,
    tap_profile: "off",
    tap_strength: 1.0,
    tap_formula: "scale_around_1",
    tap_normalization: "tensor_rms",
    projector_method: PROJECTOR_BYPASS_3,
    projector_profile: "off",
    projector_strength: 1.0,
    projector_formula: "scale_around_1",
    projector_normalization: "none",
    fusion_method: FUSION_STANDARD,
    fusion_strength: 1.0,
  },
  "COPY settings: nova452 ConditioningKrea2Rebalance profile @ tap strength 1": {
    tap_method: TAP_REBALANCE,
    tap_profile: "classic",
    tap_strength: 1.0,
    tap_formula: "scale_around_1",
    tap_normalization: "none",
    projector_method: PROJECTOR_DONUT,
    projector_profile: "off",
    projector_strength: 1.0,
    projector_formula: "scale_around_1",
    projector_normalization: "none",
    fusion_method: FUSION_STANDARD,
    fusion_strength: 1.0,
  },
  "COPY settings: capitan01R Krea2T-Enhancer defaults": {
    tap_method: TAP_DONUT,
    tap_profile: "off",
    tap_strength: 1.0,
    tap_formula: "scale_around_1",
    tap_normalization: "tensor_rms",
    projector_method: PROJECTOR_DONUT,
    projector_profile: "off",
    projector_strength: 1.0,
    projector_formula: "scale_around_1",
    projector_normalization: "none",
    fusion_method: FUSION_ENHANCER,
    fusion_strength: 1.0,
  },
  "HYBRID settings: Rebalance + Krea2T-Enhancer": {
    tap_method: TAP_REBALANCE,
    tap_profile: "classic",
    tap_strength: 1.0,
    tap_formula: "scale_around_1",
    tap_normalization: "none",
    projector_method: PROJECTOR_DONUT,
    projector_profile: "off",
    projector_strength: 1.0,
    projector_formula: "scale_around_1",
    projector_normalization: "none",
    fusion_method: FUSION_ENHANCER,
    fusion_strength: 1.0,
  },
  "HYBRID settings: Rebalance + Krea2FilterBypass 2vector": {
    tap_method: TAP_REBALANCE,
    tap_profile: "classic",
    tap_strength: 1.0,
    tap_formula: "scale_around_1",
    tap_normalization: "none",
    projector_method: PROJECTOR_BYPASS_2,
    projector_profile: "off",
    projector_strength: 1.0,
    projector_formula: "scale_around_1",
    projector_normalization: "none",
    fusion_method: FUSION_STANDARD,
    fusion_strength: 1.0,
  },
  "HYBRID settings: Rebalance + Krea2FilterBypass 3vector": {
    tap_method: TAP_REBALANCE,
    tap_profile: "classic",
    tap_strength: 1.0,
    tap_formula: "scale_around_1",
    tap_normalization: "none",
    projector_method: PROJECTOR_BYPASS_3,
    projector_profile: "off",
    projector_strength: 1.0,
    projector_formula: "scale_around_1",
    projector_normalization: "none",
    fusion_method: FUSION_STANDARD,
    fusion_strength: 1.0,
  },

  "DONUT settings: RMS-balanced classic": {
    tap_method: TAP_DONUT,
    tap_profile: "classic",
    tap_strength: 1.0,
    tap_formula: "scale_around_1",
    tap_normalization: "tensor_rms",
    projector_method: PROJECTOR_DONUT,
    projector_profile: "off",
    projector_strength: 1.0,
    projector_formula: "scale_around_1",
    projector_normalization: "none",
    fusion_method: FUSION_STANDARD,
    fusion_strength: 1.0,
  },
  "DONUT settings: RMS-balanced classic + Krea2T-Enhancer": {
    tap_method: TAP_DONUT,
    tap_profile: "classic",
    tap_strength: 1.0,
    tap_formula: "scale_around_1",
    tap_normalization: "tensor_rms",
    projector_method: PROJECTOR_DONUT,
    projector_profile: "off",
    projector_strength: 1.0,
    projector_formula: "scale_around_1",
    projector_normalization: "none",
    fusion_method: FUSION_ENHANCER,
    fusion_strength: 1.0,
  },
};

const MANAGED_WIDGETS = [
  "tap_profile",
  "tap_strength",
  "tap_formula",
  "tap_normalization",
  "projector_profile",
  "projector_strength",
  "projector_formula",
  "projector_normalization",
  "fusion_strength",
  "per_layer_weights",
  "projector_layer_weights",
];

const SETTINGS_WIDGETS = [
  "tap_method",
  ...MANAGED_WIDGETS,
  "projector_method",
  "fusion_method",
];

const HIDDEN_PREFIX = "donuthidden-";

function widget(node, name) {
  return node.widgets?.find((item) => item.name === name);
}

function hideWidget(item) {
  if (!item || (typeof item.type === "string" && item.type.startsWith(HIDDEN_PREFIX))) return;
  item._donutKrea2Type = item.type;
  item._donutKrea2ComputeSize = item.computeSize;
  item.type = HIDDEN_PREFIX + item.type;
  item.computeSize = () => [0, -4];
  item.hidden = true;
}

function showWidget(item) {
  if (!item || typeof item.type !== "string" || !item.type.startsWith(HIDDEN_PREFIX)) return;
  item.type = item._donutKrea2Type;
  item.computeSize = item._donutKrea2ComputeSize;
  item.hidden = false;
  delete item._donutKrea2Type;
  delete item._donutKrea2ComputeSize;
}

function updateVisibility(node) {
  const tapMethod = widget(node, "tap_method")?.value;
  const tapProfile = widget(node, "tap_profile")?.value;
  const projectorMethod = widget(node, "projector_method")?.value;
  const projectorProfile = widget(node, "projector_profile")?.value;
  const fusionMethod = widget(node, "fusion_method")?.value;

  const visible = new Set();
  visible.add("tap_profile");
  if (tapProfile !== "off") {
    visible.add("tap_strength");
    if (tapMethod === TAP_DONUT) {
      visible.add("tap_formula");
      visible.add("tap_normalization");
    }
    visible.add("per_layer_weights");
  }

  if (projectorMethod === PROJECTOR_DONUT) {
    visible.add("projector_profile");
    if (projectorProfile !== "off") {
      visible.add("projector_strength");
      visible.add("projector_formula");
      visible.add("projector_normalization");
      visible.add("projector_layer_weights");
    }
  } else {
    visible.add("projector_strength");
  }

  if (fusionMethod === FUSION_ENHANCER) visible.add("fusion_strength");

  for (const name of MANAGED_WIDGETS) {
    const item = widget(node, name);
    if (visible.has(name)) showWidget(item);
    else hideWidget(item);
  }

  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], size[0]), size[1]]);
  node.setDirtyCanvas?.(true, true);
}

function applyPreset(node, name) {
  const values = PRESETS[name];
  if (!values) {
    updateVisibility(node);
    return;
  }

  node._donutApplyingKrea2Preset = true;
  try {
    for (const [widgetName, value] of Object.entries(values)) {
      const item = widget(node, widgetName);
      if (!item) continue;
      item.value = value;
      item.callback?.(value);
    }
  } finally {
    node._donutApplyingKrea2Preset = false;
  }
  updateVisibility(node);
}

app.registerExtension({
  name: "donut.krea2FusionControl.presets",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onCreated?.apply(this, arguments);
      const node = this;
      const presetWidget = widget(node, "compatibility_preset");

      for (const name of SETTINGS_WIDGETS) {
        const item = widget(node, name);
        if (!item) continue;
        const previous = item.callback;
        item.callback = function (value) {
          const callbackResult = previous?.apply(this, arguments);
          if (name === "tap_profile" && PROFILE_VECTORS[value]) {
            const weights = widget(node, "per_layer_weights");
            if (weights) weights.value = PROFILE_VECTORS[value];
          } else if (name === "projector_profile" && PROFILE_VECTORS[value]) {
            const weights = widget(node, "projector_layer_weights");
            if (weights) weights.value = PROFILE_VECTORS[value];
          } else if (name === "per_layer_weights") {
            const profile = widget(node, "tap_profile");
            if (profile) profile.value = "custom";
          } else if (name === "projector_layer_weights") {
            const profile = widget(node, "projector_profile");
            if (profile) profile.value = "custom";
          }
          if (!node._donutApplyingKrea2Preset && presetWidget) {
            presetWidget.value = CUSTOM;
          }
          updateVisibility(node);
          return callbackResult;
        };
      }

      if (presetWidget) {
        const previous = presetWidget.callback;
        presetWidget.callback = function (value) {
          const callbackResult = previous?.apply(this, arguments);
          applyPreset(node, value);
          return callbackResult;
        };
      }

      updateVisibility(node);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      updateVisibility(this);
      return result;
    };
  },
});

import { app } from "../../scripts/app.js";

const NODE_NAME = "DonutKrea2FusionControl";
const SIMPLE = "Simple";
const ADVANCED = "Advanced";
const CUSTOM = "Custom";
const UNCENSORFIX = "UncensorFix";

const TAP_DONUT = "Donut 12-tap gains";
const TAP_REBALANCE = "nova452 Rebalance operation";
const PROJECTOR_DONUT = "Donut projector-input gains";
const PROJECTOR_BYPASS_2 = "Krea2FilterBypass 2vector diff";
const PROJECTOR_BYPASS_3 = "Krea2FilterBypass 3vector diff";
const FUSION_STANDARD = "Standard Krea2 fusion";
const FUSION_ENHANCER = "capitan01R Krea2T-Enhancer operation";

const LEGACY_TO_SIMPLE = {
  "Custom settings": CUSTOM,
  "COPY settings: Krea2FilterBypass 2vector": "Bypass 2",
  "COPY settings: Krea2FilterBypass 3vector": "Bypass 3",
  "COPY settings: nova452 ConditioningKrea2Rebalance profile @ tap strength 1": "Rebalance",
  "COPY settings: capitan01R Krea2T-Enhancer defaults": "Enhancer",
  "HYBRID settings: Rebalance + Krea2T-Enhancer": "Rebalance + Enhancer",
  "HYBRID settings: Rebalance + Krea2FilterBypass 2vector": "Rebalance + Bypass 2",
  "HYBRID settings: Rebalance + Krea2FilterBypass 3vector": "Rebalance + Bypass 3",
  "DONUT settings: RMS-balanced classic": "Balanced",
  "DONUT settings: RMS-balanced classic + Krea2T-Enhancer": "Balanced + Enhancer",
  "DONUT settings: Krea2 C33 TeacherFix EMA5000": UNCENSORFIX,
  "TeacherFix": UNCENSORFIX,
};

const SIMPLE_PRESETS = {
  "Bypass 2": {
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
  "Bypass 3": {
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
  Rebalance: {
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
  Enhancer: {
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
  "Rebalance + Enhancer": {
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
  "Rebalance + Bypass 2": {
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
  "Rebalance + Bypass 3": {
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
  Balanced: {
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
  "Balanced + Enhancer": {
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
  [UNCENSORFIX]: {
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
    fusion_method: FUSION_STANDARD,
    fusion_strength: 1.0,
  },
};

const SIMPLE_PROJECTOR_STRENGTH = new Set([
  "Bypass 2",
  "Bypass 3",
  "Rebalance + Bypass 2",
  "Rebalance + Bypass 3",
]);

const SIMPLE_FUSION_STRENGTH = new Set([
  "Enhancer",
  "Rebalance + Enhancer",
  "Balanced + Enhancer",
]);

const ADVANCED_WIDGETS = [
  "tap_method",
  "tap_profile",
  "per_layer_weights",
  "tap_strength",
  "tap_formula",
  "tap_normalization",
  "projector_method",
  "projector_profile",
  "projector_layer_weights",
  "projector_strength",
  "projector_formula",
  "projector_normalization",
  "fusion_method",
  "fusion_strength",
];

const HIDDEN_PREFIX = "donuthidden-";

function widget(node, name) {
  return node.widgets?.find((item) => item.name === name);
}

function simplifyPresetName(name) {
  return LEGACY_TO_SIMPLE[name] ?? name;
}

function normalizePresetWidget(node) {
  const preset = widget(node, "compatibility_preset");
  if (!preset) return;
  const simplified = simplifyPresetName(preset.value);
  if (simplified !== preset.value) preset.value = simplified;
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

function resize(node) {
  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], size[0]), size[1]]);
  node.setDirtyCanvas?.(true, true);
}

function syncSimpleStrength(node, presetName = widget(node, "compatibility_preset")?.value) {
  presetName = simplifyPresetName(presetName);
  const strength = Number(widget(node, "tap_strength")?.value ?? 1.0);
  if (!Number.isFinite(strength)) return;

  if (SIMPLE_PROJECTOR_STRENGTH.has(presetName)) {
    const projectorStrength = widget(node, "projector_strength");
    if (projectorStrength) projectorStrength.value = strength;
  }
  if (SIMPLE_FUSION_STRENGTH.has(presetName)) {
    const fusionStrength = widget(node, "fusion_strength");
    if (fusionStrength) fusionStrength.value = strength;
  }
}

function applySimplePreset(node, name) {
  name = simplifyPresetName(name);
  const values = name === "Off" && widget(node, "uncensorfix_controls") ? SIMPLE_PRESETS[UNCENSORFIX] : SIMPLE_PRESETS[name];
  if (!values) return;

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
  const composition = widget(node, "uncensorfix_controls");
  if (composition) composition.value = name === UNCENSORFIX ? "Fusion + LoRA" : "Fusion only";
}

function updateModeVisibility(node) {
  normalizePresetWidget(node);
  const mode = widget(node, "ui_mode")?.value ?? ADVANCED;
  if (mode === SIMPLE) {
    for (const name of ADVANCED_WIDGETS) {
      const item = widget(node, name);
      if (name === "tap_strength") showWidget(item);
      else hideWidget(item);
    }
    resize(node);
    return;
  }

  const visible = new Set(["tap_method", "tap_profile", "projector_method", "fusion_method"]);
  const tapMethod = widget(node, "tap_method")?.value;
  const tapProfile = widget(node, "tap_profile")?.value;
  const projectorMethod = widget(node, "projector_method")?.value;
  const projectorProfile = widget(node, "projector_profile")?.value;
  const fusionMethod = widget(node, "fusion_method")?.value;
  const preset = simplifyPresetName(widget(node, "compatibility_preset")?.value);

  if (tapProfile !== "off") {
    visible.add("tap_strength");
    visible.add("per_layer_weights");
    if (tapMethod === TAP_DONUT) {
      visible.add("tap_formula");
      visible.add("tap_normalization");
    }
  }
  if (preset === UNCENSORFIX) visible.add("tap_strength");

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

  for (const name of ADVANCED_WIDGETS) {
    const item = widget(node, name);
    if (visible.has(name)) showWidget(item);
    else hideWidget(item);
  }
  resize(node);
}

function scheduleVisibility(node) {
  queueMicrotask(() => updateModeVisibility(node));
}

app.registerExtension({
  name: "donut.krea2FusionControl.simpleModeTeacherFix",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onCreated?.apply(this, arguments);
      const node = this;
      const modeWidget = widget(node, "ui_mode");
      const presetWidget = widget(node, "compatibility_preset");
      const tapStrengthWidget = widget(node, "tap_strength");

      if (modeWidget) {
        const previous = modeWidget.callback;
        modeWidget.callback = function (value) {
          const callbackResult = previous?.apply(this, arguments);
          if (value === SIMPLE) syncSimpleStrength(node);
          scheduleVisibility(node);
          return callbackResult;
        };
      }

      if (presetWidget) {
        const previous = presetWidget.callback;
        presetWidget.callback = function (value) {
          // A queued strength callback from an earlier selection must not
          // restore that preset after the user explicitly selects Off/Custom.
          const selectionVersion = (node._donutKrea2PresetSelectionVersion ?? 0) + 1;
          node._donutKrea2PresetSelectionVersion = selectionVersion;
          const callbackResult = previous?.apply(this, arguments);
          const simplified = simplifyPresetName(value);
          applySimplePreset(node, simplified);
          presetWidget.value = simplified;
          if (widget(node, "ui_mode")?.value === SIMPLE) {
            queueMicrotask(() => {
              if (widget(node, "ui_mode")?.value !== SIMPLE) return;
              if (node._donutKrea2PresetSelectionVersion !== selectionVersion) return;
              syncSimpleStrength(node, simplified);
              updateModeVisibility(node);
            });
          }
          scheduleVisibility(node);
          return callbackResult;
        };
      }

      if (tapStrengthWidget) {
        const previous = tapStrengthWidget.callback;
        tapStrengthWidget.callback = function (value) {
          const presetBefore = simplifyPresetName(presetWidget?.value);
          const selectionVersion = node._donutKrea2PresetSelectionVersion ?? 0;
          const callbackResult = previous?.apply(this, arguments);
          if (widget(node, "ui_mode")?.value === SIMPLE && presetBefore && presetBefore !== CUSTOM) {
            queueMicrotask(() => {
              if (widget(node, "ui_mode")?.value !== SIMPLE) return;
              if ((node._donutKrea2PresetSelectionVersion ?? 0) !== selectionVersion) return;
              if (presetWidget) presetWidget.value = presetBefore;
              syncSimpleStrength(node, presetBefore);
              updateModeVisibility(node);
            });
          }
          scheduleVisibility(node);
          return callbackResult;
        };
      }

      scheduleVisibility(node);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      normalizePresetWidget(this);
      if (widget(this, "ui_mode")?.value === SIMPLE) syncSimpleStrength(this);
      scheduleVisibility(this);
      return result;
    };
  },
});

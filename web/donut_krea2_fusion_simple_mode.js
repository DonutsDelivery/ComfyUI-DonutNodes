import { app } from "../../scripts/app.js";

const NODE_NAME = "DonutKrea2FusionControl";
const SIMPLE = "Simple";
const ADVANCED = "Advanced";
const CUSTOM = "Custom";
const NEUTRAL = "Neutral";
const UNCENSORFIX = "UncensorFix";
const REFUSAL_V1 = "Refusal Reduction v1";
const REFUSAL_V2 = "Refusal Reduction v2";
const WEIGHTS_NONE = "None";
const WEIGHTS_ON = "Fusion + UncensorFix weights";
const WEIGHTS_OFF = "Fusion only";
// Embedded weights used to be presets. They are a separate selector now; these
// old preset values only exist so saved workflows still load and migrate.
const LEGACY_WEIGHT_PRESETS = new Set([UNCENSORFIX, REFUSAL_V1, REFUSAL_V2]);
// Compositions that apply weights via the selector. The old LoRA-only modes
// belong to the legacy preset path and are left alone.
const COMPOSITIONS = new Set([WEIGHTS_OFF, WEIGHTS_ON, "Fusion + LoRA"]);

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
  "DONUT settings: RMS-balanced classic, raw NAG negative": "Balanced + raw NAG",
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
  "Balanced + raw NAG": {
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
    nag_match_taps: false,
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
  [NEUTRAL]: {
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

for (const preset of Object.values(SIMPLE_PRESETS)) {
  if (preset.nag_match_taps === undefined) preset.nag_match_taps = true;
}

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
  "nag_text_energy_compensation",
  "nag_batch_txtfusion",
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
  // An old weights preset (still loadable) means the neutral fusion settings.
  const neutral = name === "Off" && widget(node, "uncensorfix_controls") || LEGACY_WEIGHT_PRESETS.has(name);
  const values = neutral ? SIMPLE_PRESETS[NEUTRAL] : SIMPLE_PRESETS[name];
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
}

// Weights are chosen by one selector. The old composition widget is kept in sync
// (and hidden) because saved workflows and the backend still read it.
function syncEmbeddedWeights(node) {
  const weights = widget(node, "embedded_weights");
  const composition = widget(node, "uncensorfix_controls");
  const preset = widget(node, "compatibility_preset");
  if (!weights || !composition) return;
  const legacyPreset = simplifyPresetName(preset?.value);
  if (COMPOSITIONS.has(composition.value)) {
    if (LEGACY_WEIGHT_PRESETS.has(legacyPreset)) {
      weights.value = composition.value === WEIGHTS_OFF ? WEIGHTS_NONE : legacyPreset;
      preset.value = NEUTRAL;
    } else if (composition.value === WEIGHTS_OFF) {
      weights.value = WEIGHTS_NONE;
    }
    composition.value = weights.value === WEIGHTS_NONE ? WEIGHTS_OFF : WEIGHTS_ON;
    hideWidget(composition);
  }
  const strength = widget(node, "uncensorfix_strength");
  if (strength) {
    strength.label = "weights_strength";
    if (weights.value === WEIGHTS_NONE && COMPOSITIONS.has(composition.value)) hideWidget(strength);
    else showWidget(strength);
  }
  // Weights are not presets any more; keep an old value listed only while it still drives a legacy workflow.
  if (preset?.options?.values) {
    preset._donutAllPresets ||= [...preset.options.values];
    preset.options.values = preset._donutAllPresets.filter(
      name => !LEGACY_WEIGHT_PRESETS.has(name) || name === preset.value);
  }
}

function updateModeVisibility(node) {
  normalizePresetWidget(node);
  syncEmbeddedWeights(node);
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
  if (LEGACY_WEIGHT_PRESETS.has(preset)) visible.add("tap_strength");

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

  // Experiment toggles are always visible in Advanced mode; hidden in Simple.
  visible.add("nag_text_energy_compensation");
  visible.add("nag_batch_txtfusion");

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

      const weightsWidget = widget(node, "embedded_weights");
      if (weightsWidget) {
        const previous = weightsWidget.callback;
        weightsWidget.callback = function () {
          const callbackResult = previous?.apply(this, arguments);
          const composition = widget(node, "uncensorfix_controls");
          if (composition && COMPOSITIONS.has(composition.value)) {
            composition.value = weightsWidget.value === WEIGHTS_NONE ? WEIGHTS_OFF : WEIGHTS_ON;
          }
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

      syncEmbeddedWeights(node);
      scheduleVisibility(node);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      normalizePresetWidget(this);
      syncEmbeddedWeights(this);
      if (widget(this, "ui_mode")?.value === SIMPLE) syncSimpleStrength(this);
      scheduleVisibility(this);
      return result;
    };
  },
});

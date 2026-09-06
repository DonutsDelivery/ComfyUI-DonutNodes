import { app } from "../../scripts/app.js";

const NODE_NAME = "DonutKrea2FusionControl";
const SIMPLE = "Simple";
const ADVANCED = "Advanced";
const CUSTOM = "Custom settings";
const TEACHERFIX = "DONUT settings: Krea2 C33 TeacherFix EMA5000";

const TAP_DONUT = "Donut 12-tap gains";
const PROJECTOR_DONUT = "Donut projector-input gains";
const FUSION_ENHANCER = "capitan01R Krea2T-Enhancer operation";

const TEACHERFIX_VALUES = {
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
  fusion_method: "Standard Krea2 fusion",
  fusion_strength: 1.0,
};

const SIMPLE_PROJECTOR_STRENGTH = new Set([
  "COPY settings: Krea2FilterBypass 2vector",
  "COPY settings: Krea2FilterBypass 3vector",
  "HYBRID settings: Rebalance + Krea2FilterBypass 2vector",
  "HYBRID settings: Rebalance + Krea2FilterBypass 3vector",
]);

const SIMPLE_FUSION_STRENGTH = new Set([
  "COPY settings: capitan01R Krea2T-Enhancer defaults",
  "HYBRID settings: Rebalance + Krea2T-Enhancer",
  "DONUT settings: RMS-balanced classic + Krea2T-Enhancer",
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

function applyTeacherFixPreset(node) {
  node._donutApplyingKrea2Preset = true;
  try {
    for (const [name, value] of Object.entries(TEACHERFIX_VALUES)) {
      const item = widget(node, name);
      if (item) item.value = value;
    }
  } finally {
    node._donutApplyingKrea2Preset = false;
  }
}

function updateModeVisibility(node) {
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
  const preset = widget(node, "compatibility_preset")?.value;

  if (tapProfile !== "off") {
    visible.add("tap_strength");
    visible.add("per_layer_weights");
    if (tapMethod === TAP_DONUT) {
      visible.add("tap_formula");
      visible.add("tap_normalization");
    }
  }
  // TeacherFix uses tap_strength as its LoRA strength even though its
  // normal tap profile is intentionally off.
  if (preset === TEACHERFIX) visible.add("tap_strength");

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
          const callbackResult = previous?.apply(this, arguments);
          if (value === TEACHERFIX) applyTeacherFixPreset(node);
          if (widget(node, "ui_mode")?.value === SIMPLE) {
            queueMicrotask(() => {
              if (widget(node, "ui_mode")?.value !== SIMPLE) return;
              syncSimpleStrength(node, value);
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
          const presetBefore = presetWidget?.value;
          const callbackResult = previous?.apply(this, arguments);
          if (widget(node, "ui_mode")?.value === SIMPLE && presetBefore && presetBefore !== CUSTOM) {
            // The legacy Advanced UI intentionally changes the preset label to
            // Custom when any setting is edited. Restore it in a microtask so
            // this remains correct regardless of extension registration order:
            // an outer legacy callback may still run after this callback returns.
            queueMicrotask(() => {
              if (widget(node, "ui_mode")?.value !== SIMPLE) return;
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
      if (widget(this, "ui_mode")?.value === SIMPLE) syncSimpleStrength(this);
      scheduleVisibility(this);
      return result;
    };
  },
});

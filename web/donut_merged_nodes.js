import { app } from "../../scripts/app.js";

/**
 * Donut merged-node widget visibility.
 *
 * Several Donut nodes were consolidated into multipurpose nodes with a mode
 * selector. ComfyUI has no native conditional widgets, so the Python side
 * declares the union of every mode's widgets as optional and ignores the
 * irrelevant ones at runtime. This extension hides the widgets that don't
 * belong to the currently selected mode so each mode shows a clean UI.
 *
 * Purely cosmetic: if this script fails to load the nodes still work, they
 * just show all widgets at once. Hidden widgets keep their values and still
 * serialize, so toggling modes is lossless.
 */

// node id (NODE_CLASS_MAPPINGS key) -> { selector widget, value -> active widget names }
const CONFIG = {
  "ModelMergeZIT": {
    selector: "granularity",
    modes: {
      grouped: ["early", "lowmid", "upmid", "late", "x_embedder", "t_embedder", "cap_embedder", "refiners", "final", "other"],
      blocks: ["layer_0","layer_1","layer_2","layer_3","layer_4","layer_5","layer_6","layer_7","layer_8","layer_9","layer_10","layer_11","layer_12","layer_13","layer_14","layer_15","layer_16","layer_17","layer_18","layer_19","layer_20","layer_21","layer_22","layer_23","layer_24","layer_25","layer_26","layer_27","layer_28","layer_29","x_embedder","t_embedder","cap_embedder","context_refiner","noise_refiner","final_layer","norm_final","other"],
    },
  },
  "Donut Block Calibration": {
    selector: "preset",
    modes: {
      "Manual": ["input_blocks_strength", "middle_blocks_strength", "output_1_blocks_strength", "output_2_blocks_strength"],
      "Auto (full strength)": [],
    },
  },
  "Donut Spectral Sharpener": {
    selector: "reference_source",
    modes: {
      external_reference: ["enhancement_strength", "frequency_bands", "spectral_mode", "blend_factor"],
      generated_noise: ["enhancement_strength", "frequency_bands", "noise_type", "noise_strength", "noise_scale", "noise_saturation", "spectral_mode", "blend_factor"],
      self_amplify: ["enhancement_strength", "frequency_bands", "spectral_mode", "blend_factor"],
    },
  },
  "DonutLoRACivitAILookup": {
    selector: "source",
    modes: {
      "LoRA File": ["lora_name", "api_key", "force_refresh"],
      "Hash": ["hash", "api_key"],
    },
  },
  "DonutSampler": {
    selector: "mode",
    modes: {
      simple: [],
      advanced: ["cfg_curve", "add_noise", "start_at_step", "end_at_step", "return_with_leftover_noise"],
      multi_model: ["cfg_curve", "add_noise", "start_at_step", "end_at_step", "return_with_leftover_noise", "randomize_seed_per_model", "switch_at_step_1", "switch_at_step_2"],
    },
  },
  "DonutDetailerUnified": {
    selector: "formula",
    modes: {
      scale_weight_bias: ["swb_Scale_in", "swb_Weight_in", "swb_Bias_in", "swb_Scale_out0", "swb_Weight_out0", "swb_Bias_out0", "swb_Scale_out2", "swb_Weight_out2", "swb_Bias_out2"],
      k_s1_s2: ["ks_Multiplier_in", "ks_S1_in", "ks_S2_in", "ks_Multiplier_out0", "ks_S1_out0", "ks_S2_out0", "ks_Multiplier_out2", "ks_S1_out2", "ks_S2_out2"],
      direct: ["dir_Weight_in", "dir_Bias_in", "dir_Weight_out0", "dir_Bias_out0", "dir_Weight_out2", "dir_Bias_out2"],
    },
  },
  "DonutImageAdjust": {
    selector: "operation",
    modes: {
      auto_gamma: ["ag_method", "ag_strength", "ag_target", "ag_percentile", "ag_direction"],
      gamma: ["gm_gamma", "gm_dither"],
      white_balance: ["wb_method", "wb_strength", "wb_clip_percent"],
      histogram_stretch: ["hs_black_point", "hs_white_point", "hs_mode"],
      local_contrast: ["lc_radius", "lc_amount", "lc_protect_tones", "lc_fast_blur"],
      cas: ["cas_sharpness", "cas_contrast"],
    },
  },
};

const HIDDEN = "donuthidden-";

function hideWidget(w) {
  if (!w || (typeof w.type === "string" && w.type.startsWith(HIDDEN))) return;
  w._donutType = w.type;
  w._donutCompute = w.computeSize;
  w.type = HIDDEN + w.type;
  w.computeSize = () => [0, -4];
  w.hidden = true;            // modern ComfyUI frontend honors this; legacy ignores it
}

function showWidget(w) {
  if (!w || typeof w.type !== "string" || !w.type.startsWith(HIDDEN)) return;
  w.type = w._donutType;
  w.computeSize = w._donutCompute;
  w.hidden = false;
  delete w._donutType;
  delete w._donutCompute;
}

function applyVisibility(node, cfg) {
  if (!node.widgets) return;
  const sel = node.widgets.find((w) => w.name === cfg.selector);
  if (!sel) return;
  const active = new Set(cfg.modes[sel.value] || []);
  for (const name of cfg.managed) {
    if (name === cfg.selector) continue;
    const w = node.widgets.find((x) => x.name === name);
    if (!w) continue; // name may be an input slot, not a widget — skip
    if (active.has(name)) showWidget(w);
    else hideWidget(w);
  }
  const sz = node.computeSize();
  node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "donut.mergedNodes.widgetVisibility",
  beforeRegisterNodeDef(nodeType, nodeData) {
    const cfg = CONFIG[nodeData?.name];
    if (!cfg) return;
    // union of every mode's widgets = the set this node manages
    cfg.managed = [...new Set(Object.values(cfg.modes).flat())];

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const self = this;
      const sel = this.widgets?.find((w) => w.name === cfg.selector);
      if (sel) {
        const prev = sel.callback;
        sel.callback = function () {
          const rr = prev?.apply(this, arguments);
          applyVisibility(self, cfg);
          return rr;
        };
      }
      applyVisibility(this, cfg);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      applyVisibility(this, cfg);
      return r;
    };
  },
});

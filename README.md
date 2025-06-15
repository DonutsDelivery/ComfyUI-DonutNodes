# ComfyUI-DonutNodes

A comprehensive collection of advanced model manipulation nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), featuring fine-grained control over model parameters, advanced model merging capabilities, CLIP encoding control, and LoRA stack management.

## 🚀 Features

- **🍩 Donut Detailer Series** - Fine-grained control over model weight/bias parameters
- **🔀 DonutWidenMerge** - Advanced TIES+WIDEN model merging with memory optimization
- **📝 Donut Clip Encode** - Advanced CLIP encoding with multiple mixing modes
- **📚 LoRA Stack Management** - Apply multiple LoRAs with individual block adjustments

---

## 📦 Installation

1. Clone into your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/DonutsDelivery/ComfyUI-DonutDetailer.git
   ```
2. Restart ComfyUI
3. Look for nodes under **DonutNodes** category

---

# 🍩 Donut Detailer Series

Fine-grained control over model and LoRA weight/bias parameters for precise model behavior adjustment.

## Model Detailing Nodes

### 🍩 Donut Detailer (Model)

**Purpose:** Basic weight/bias scaling with separate input/output group controls.

#### Parameters
| Name          | Type   | Description                                                      |
| ------------- | ------ | ---------------------------------------------------------------- |
| `Scale_in`    | FLOAT  | Scale factor for input-group weight adjustments                 |
| `Weight_in`   | FLOAT  | Input-group weight multiplier                                   |
| `Bias_in`     | FLOAT  | Input-group bias multiplier                                     |
| `Scale_out0`  | FLOAT  | Scale factor for first output-group weight adjustments          |
| `Weight_out0` | FLOAT  | First output-group weight multiplier                            |
| `Bias_out0`   | FLOAT  | First output-group bias multiplier                              |
| `Scale_out2`  | FLOAT  | Scale factor for second output-group weight adjustments         |
| `Weight_out2` | FLOAT  | Second output-group weight multiplier                           |
| `Bias_out2`   | FLOAT  | Second output-group bias multiplier                             |

**Formula:**
- **Weights:** `1 – Scale × Weight`
- **Biases:** `1 + Scale × Bias`

### 🍩 Donut Detailer 2 (Model)

**Purpose:** Advanced coefficient-based control mimicking Supermerger's "Adjust" functionality.

#### Parameters
| Name         | Type   | Description                                               |
| ------------ | ------ | --------------------------------------------------------- |
| `K_in`       | FLOAT  | Base coefficient for input-group weight/bias             |
| `S1_in`      | FLOAT  | Weight-scale factor for input-group                      |
| `S2_in`      | FLOAT  | Bias-scale factor for input-group                        |
| `K_out0`     | FLOAT  | Base coefficient for first output-group weight/bias      |
| `S1_out0`    | FLOAT  | Weight-scale factor for first output-group               |
| `S2_out0`    | FLOAT  | Bias-scale factor for first output-group                 |
| `K_out2`     | FLOAT  | Base coefficient for second output-group weight/bias     |
| `S1_out2`    | FLOAT  | Weight-scale factor for second output-group              |
| `S2_out2`    | FLOAT  | Bias-scale factor for second output-group                |

**Formula:**
```
Weight multiplier = 1 – (K × S1 × 0.01)
Bias multiplier   = 1 + (K × S2 × 0.02)
```

### 🍩 Donut Detailer 4 (Model)

**Purpose:** Direct weight/bias multipliers with no complex formulas.

#### Parameters
| Name          | Type   | Description                                    |
| ------------- | ------ | ---------------------------------------------- |
| `Weight_in`   | FLOAT  | Input-group weight multiplier (direct)        |
| `Bias_in`     | FLOAT  | Input-group bias multiplier (direct)          |
| `Weight_out0` | FLOAT  | First output-group weight multiplier (direct) |
| `Bias_out0`   | FLOAT  | First output-group bias multiplier (direct)   |
| `Weight_out2` | FLOAT  | Second output-group weight multiplier (direct)|
| `Bias_out2`   | FLOAT  | Second output-group bias multiplier (direct)  |

**Effect:** Direct multiplication (default 1.0 = bypass).

### 🍩 Donut Detailer XL Blocks (Model)

**Purpose:** Granular per-block control for SDXL UNet architecture.

#### Parameters
| Name        | Type   | Description                                             |
| ----------- | ------ | ------------------------------------------------------- |
| `W_block_*` | FLOAT  | Weight multiplier for each SDXL UNet block             |
| `B_block_*` | FLOAT  | Bias multiplier for each SDXL UNet block               |

**Effect:** Direct control over every major SDXL UNet block for highly granular tweaking.

## LoRA Detailing Nodes

### 🍩 Donut Detailer LoRA 6 (LoRA)

**Purpose:** Scale LoRA layers and save modified LoRA patches.

#### Parameters
| Name          | Type   | Description                                      |
| ------------- | ------ | ------------------------------------------------ |
| `Weight_down` | FLOAT  | Down-layer weight multiplier                    |
| `Bias_down`   | FLOAT  | Down-layer bias multiplier                      |
| `Weight_mid`  | FLOAT  | Mid-layer weight multiplier                     |
| `Bias_mid`    | FLOAT  | Mid-layer bias multiplier                       |
| `Weight_up`   | FLOAT  | Up-layer weight multiplier                      |
| `Bias_up`     | FLOAT  | Up-layer bias multiplier                        |

**Effect:** Scales LoRA down/mid/up layers' weights & biases with save capability.

---

# 🔀 DonutWidenMerge

Advanced model merging using the WIDEN algorithm, optimized for SDXL diffusion models with memory-efficient processing.

## ✨ Overview

DonutWidenMerge combines multiple AI models into a single model by intelligently selecting and blending the most important parameter changes. Unlike simple averaging, it uses the WIDEN algorithm to identify which parameters matter most and how they should be combined.

The node analyzes each parameter in your models, classifies them by their role in the neural network (attention layers, convolutions, etc.), and applies specialized merging logic for each type. This results in merged models that preserve the strengths of each input model while maintaining stability.

## 🧮 How WIDEN Works

1. **Analysis** - For each parameter, the algorithm computes how much it changed from the base model and in what direction
2. **Classification** - Parameters are classified by their role (attention, convolution, normalization, etc.)
3. **Ranking** - Changes are ranked by significance using layer-specific criteria
4. **Selection** - Only the most significant changes that exceed certain thresholds are merged
5. **Weighting** - Selected changes are weighted based on their importance and layer type

This selective approach prevents the "muddy" results often seen with simple model averaging.

## 🛠️ Nodes

### DonutWidenMergeUNet
Merges UNet models using WIDEN algorithm with SDXL-specific optimizations. Takes a base model plus up to 11 additional models.

### DonutWidenMergeCLIP
Same WIDEN functionality optimized for CLIP text encoders.

### 🎭 DonutFillerModel / DonutFillerClip
Placeholder nodes for unused model slots when merging fewer than 12 models.

## ⚙️ Parameters

### 💪 merge_strength (0.1 - 3.0, default: 1.0)
Controls the overall intensity of the merge. Higher values make the other models influence the base model more strongly.

- **0.5** - Subtle influence, preserves base model character
- **1.0** - Balanced merge (recommended starting point)  
- **1.5** - Strong influence from other models
- **2.0+** - Very aggressive, may cause instability

### 🎯 widen_threshold (0.1 - 3.0, default: 1.0)
Determines how picky the algorithm is about which parameters to merge. Higher values mean fewer parameters get merged, but they're more significant.

- **0.5** - Less selective, merges more parameters
- **1.0** - Balanced selection (recommended)
- **1.5** - More selective, only obvious improvements
- **2.0+** - Very selective, minimal changes

### 🔧 widen_calibration (0.1 - 3.0, default: 1.0)
Adjusts how the algorithm weighs parameter importance. This affects which parameters are considered most valuable to merge.

- **0.5** - Conservative weighting
- **1.0** - Standard weighting (recommended)
- **1.5** - Aggressive weighting, emphasizes standout changes
- **2.0+** - Very aggressive, may over-emphasize some changes

### 🔄 enable_renorm (True/False, default: True)
Renormalization helps keep the merged model stable by adjusting parameter magnitudes to reasonable ranges. Usually should stay enabled.

### 📐 renorm_mode ("magnitude", "calibrate", "none")
How to stabilize the merged parameters:

- **"magnitude"** - Simple and fast stabilization (recommended)
- **"calibrate"** - More sophisticated but slower stabilization  
- **"none"** - No stabilization (may cause instability)

### 📦 batch_size (10 - 500, default: 50/75)
Not used in current zero-accumulation implementation, kept for compatibility.

## 📋 Usage Examples

### 🕊️ Basic Usage
Start with all defaults and only adjust `merge_strength`:
- Use **0.8-1.2** for subtle merges
- Use **1.0-1.5** for balanced merges
- Use **1.5-2.0** for strong merges

### 🎨 Fine-Tuning
- If the default merge is too aggressive, increase `widen_threshold` to **1.2-1.5**
- If it's too conservative, decrease `widen_threshold` to **0.7-0.9**

### 🌈 Many Models (8-12)
When merging many models, use lower `merge_strength` (**0.8-1.0**) to prevent overwhelming the base model.

## 💾 Memory Management

The node uses "zero-accumulation" processing - it works on one parameter at a time instead of loading everything into memory:

- 📊 Memory usage stays constant regardless of model size
- 🚀 Can merge large models on systems with limited RAM
- 🛑 Automatically monitors memory and stops safely if needed
- 💨 No memory leaks or accumulation during processing

## 🔧 Technical Notes

- 🧠 Implements full zero-accumulation processing for minimal memory footprint
- 🎯 SDXL-optimized with layer-specific thresholds and importance weighting
- 🔢 Supports up to 12 model merging simultaneously
- 📊 Real-time memory monitoring with automatic safety cutoffs
- 🔄 Smart caching prevents redundant processing of identical merges

---

# 📝 Donut Clip Encode (Mix Only)

Advanced CLIP encoding node for SDXL with multiple mixing modes and precise control over the two CLIP branches.

## Features

- **Mix Mode** - Multiple presets for blending CLIP branches
- **Strength Mode** - Direct strength control per branch
- **8 Different Presets** - Various blending strategies
- **Resolution Scaling** - Internal CLIP resolution control

## Parameters

| Name                      | Type    | Description                                                                                              |
|---------------------------|---------|----------------------------------------------------------------------------------------------------------|
| `clip`                    | CLIP    | The CLIP encoder from your SDXL pipeline                                                                |
| `width` / `height`        | INT     | Base image resolution (scaled internally)                                                               |
| `text_g` / `text_l`       | STRING  | "Guidance" and "Language" prompts for the two CLIP branches                                             |
| `mode`                    | ENUM    | `Mix Mode` or `Strength Mode`                                                                           |
| `clip_gl_mix`             | FLOAT   | (0.0–1.0) Mix slider for all mix-based presets                                                          |
| `vs_mix`                  | FLOAT   | (0.0–1.0) Ratio slider for **Split vs Pooled** preset                                                   |
| `clip_g_strength`         | FLOAT   | Strength for branch **g** (Strength Mode only)                                                          |
| `clip_l_strength`         | FLOAT   | Strength for branch **l** (Strength Mode only)                                                          |
| `preset`                  | ENUM    | Choose from 8 different blending presets                                                                |

## Modes & Presets

### 🔀 Mix Mode Presets

| Preset                   | Behavior                                                                                      |
|--------------------------|-----------------------------------------------------------------------------------------------|
| **Default**              | Single-pass joint encode                                                                      |
| **Split Only**           | Two-pass encode, linear blend by `clip_gl_mix`                                                |
| **Continuous**           | Gamma-blended joint↔split via `clip_gl_mix^(1/3)`                                            |
| **Split vs Pooled**      | Split sequence full, gamma blend pooled summary by `vs_mix^0.3`                              |
| **Split vs Continuous**  | Linear blend between split-only and continuous                                                |
| **Default vs Split**     | Linear blend joint vs split-only                                                              |
| **Default vs Continuous**| Linear blend joint vs continuous                                                              |
| **Strength Blend**       | Blend three embeddings with custom weights                                                    |

### 🏋️ Strength Mode
Direct control over branch strengths:
```python
cond = cond_g * clip_g_strength + cond_l * clip_l_strength
pooled = pooled_g * clip_g_strength + pooled_l * clip_l_strength
```

---

# 📚 LoRA Stack Management

## DonutLoRAStack

Apply multiple LoRAs with individual block adjustments and precise control.

### Features

- **Multiple LoRA Slots** - Apply up to multiple LoRAs simultaneously
- **Individual Controls** - Each LoRA has its own weight settings
- **Block Vector Support** - Fine-grained per-block adjustments
- **On/Off Switches** - Enable/disable individual LoRAs
- **Model & CLIP Weights** - Separate control for model and CLIP components

### Parameters (Per LoRA Slot)

| Name                | Type    | Description                                          |
|---------------------|---------|------------------------------------------------------|
| `switch_N`          | BOOLEAN | Enable/disable this LoRA slot                        |
| `lora_name_N`       | STRING  | LoRA file selection                                  |
| `model_weight_N`    | FLOAT   | Model weight for this LoRA                          |
| `clip_weight_N`     | FLOAT   | CLIP weight for this LoRA                           |
| `block_vector_N`    | STRING  | Block-wise weight adjustments (comma-separated)     |

### Block Vector Format

Block vectors allow per-block weight control using comma-separated values:
```
1,1,1,1,1,1,1,1,1,1,1,1  # Equal weight for all blocks
0.5,0.7,1.0,1.2,0.8,...  # Custom weights per block
```

## DonutApplyLoRAStack

Applies the configured LoRA stack to models and CLIP encoders.

### Inputs/Outputs

| Type   | Input/Output | Description                                          |
|--------|--------------|------------------------------------------------------|
| MODEL  | Input/Output | Model to apply LoRA stack to                        |
| CLIP   | Input/Output | CLIP encoder to apply LoRA stack to                 |
| LORA_STACK | Input    | LoRA stack configuration from DonutLoRAStack        |

---

## 💡 Usage Tips

### Donut Detailer Series
- Use **Donut Detailer 2** for the closest mimic of Supermerger's "Adjust"
- Use **Donut Detailer 4** for straightforward weight/bias multipliers
- Use **LoRA 6** to save custom-scaled LoRA patches
- Use **XL Blocks** when you need per-block control in SDXL

### DonutWidenMerge
- Start with recommended presets for your use case
- Gradually increase `above_average_value_ratio` for stronger effects
- Lower `temperature` for more aggressive merging
- Use `score_calibration_value` for fine-tuning final strength

### Memory Management
- **Low VRAM?** Reduce `batch_size` to 10-20
- **High VRAM?** Increase `batch_size` to 50-100 for faster processing
- **Out of memory errors?** Lower batch size and ensure no other models are loaded

### Donut Clip Encode
- Try **Split Only** at `clip_gl_mix=0.5` for an even split
- In **Continuous**, raise `clip_gl_mix > 0.7` for strong split bias
- **Strength Mode** is handy for direct CFG control per branch
- **Strength Blend** lets you combine all three core embeddings with custom weights

### LoRA Stack Management
- Use block vectors for fine-grained control over LoRA application
- Disable unused slots to improve performance
- Experiment with different model vs CLIP weight ratios

---

## 🔬 Technical Details

### TIES Algorithm
The DonutWidenMerge nodes implement TIES (Task-Informed Expert Selection):

1. **Parameter Importance Analysis** - Identifies the most significant differences between models
2. **TIES Trimming** - Removes small changes that might be noise
3. **Sign Election** - Resolves conflicts when models disagree on parameter directions
4. **Batch Processing** - Memory-efficient processing of large models
5. **Forced Merging** - Ensures important parameters always get merged

### Performance Optimizations
- **Memory efficient:** Processes models in batches to avoid VRAM overflow
- **Scalable:** Handles models from SD1.5 to SDXL and beyond
- **Fast:** Optimized tensor operations with automatic cleanup
- **Reliable:** Extensive error handling and fallback mechanisms

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Based on the TIES algorithm from ["Resolving Interference When Merging Models"](https://arxiv.org/abs/2306.01708)
- Inspired by WIDEN techniques for model expansion
- Built for the ComfyUI ecosystem

---

**Happy model manipulation!** 🍩✨

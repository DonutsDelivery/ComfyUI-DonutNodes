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
Advanced model merging nodes implementing the TIES+WIDEN algorithm with memory-efficient batch processing.

## Features
- **Memory-optimized TIES+WIDEN merging** - Handles large models without VRAM overflow
- **Batch processing** - Configurable batch sizes for optimal performance
- **Intelligent parameter selection** - Automatically identifies most important parameters to merge
- **Simplified controls** - Streamlined parameters for easier use while maintaining flexibility
- **Support for both UNet and CLIP models** - Complete workflow integration

## Available Nodes
### DonutWidenMergeUNet
Merges two UNet models using the TIES+WIDEN algorithm.

### DonutWidenMergeCLIP
Merges two CLIP models using the TIES+WIDEN algorithm.

## Parameters

### Core Merge Strength
#### `merge_strength` (0.1 - 3.0, default: 1.0)
Primary merge strength controlling how much influence the "other" model has on the base model.
- **0.5** - Subtle merge, preserves base model characteristics
- **1.0** - Balanced merge, equal influence from both models
- **1.5** - Strong merge, other model dominates
- **2.0** - Very aggressive merge, can completely override base features
- **3.0** - Maximum strength, dramatic transformations

#### `temperature` (0.1 - 10.0, default: 0.8) 
Controls merge "confidence" through inverse scaling. Lower = more aggressive.
- **0.5** - Cold merge: 2x strength, very aggressive changes
- **0.8** - Cool merge: 1.25x strength, moderate confidence  
- **1.0** - Neutral: No temperature scaling
- **2.0** - Hot merge: 0.5x strength, gentle blending

### Advanced Controls
#### `threshold` (0.0 - 0.1, default: 0.00005)
Threshold for removing small parameter changes (noise filtering and merge cutoff).

#### `enable_ties` (Boolean, default: True)
Enables TIES algorithm for intelligent conflict resolution when models disagree.

#### `forced_merge_ratio` (0.0 - 1.0, default: 0.0)
Ratio of parameters to force merge regardless of threshold. Set to 0 to disable forced merging.

#### `batch_size` (10 - 100, default: 30)
Memory management - how many parameters to process simultaneously.

## Mathematical Formula
```
effective_strength = merge_strength × (1.0 / temperature)
merged_parameter = base_parameter + (other_parameter - base_parameter) × effective_strength
```

## Recommended Settings

### Style Transfer (Dramatic Changes)
```
merge_strength: 1.5-2.5
temperature: 0.4-0.6  
threshold: 0.00005
enable_ties: True
forced_merge_ratio: 0.1-0.2
```

### Fine-tuning Blend (Subtle Improvements)
```
merge_strength: 0.8-1.2
temperature: 0.8-1.0
threshold: 0.0001
enable_ties: True
forced_merge_ratio: 0.0
```

### Memory-Constrained Environments
```
batch_size: 10-20
(Reduce batch size if experiencing VRAM issues)
```

## What's New
- **Simplified parameters**: Reduced complexity while maintaining full functionality
- **Unified strength control**: Single `merge_strength` parameter replaces multiple strength controls
- **Streamlined thresholding**: Combined noise filtering and merge thresholds into one parameter
- **Cleaner forced merge control**: `forced_merge_ratio` replaces boolean toggle for better granular control
- **Removed unused features**: Eliminated parameters that weren't being utilized

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

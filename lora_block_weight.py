import folder_paths
import comfy.utils
import comfy.lora
import os
import torch
import numpy as np
import nodes
import re
import json
from comfy.cli_args import args
from safetensors.torch import safe_open
import ast
import logging

from server import PromptServer
from .libs import utils


model_path = folder_paths.models_dir
utils.add_folder_path_and_extensions("lbw_models", [os.path.join(model_path, "lbw_models")], {'.safetensors'})


def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def load_lbw_preset(filename):
    path = os.path.join(os.path.dirname(__file__), "..", "resources", filename)
    path = os.path.abspath(path)
    preset_list = []

    if os.path.exists(path):
        with open(path, 'r') as file:
            for line in file:
                preset_list.append(line.strip())

        return preset_list
    else:
        return []


def parse_unet_num(s):
    if s[1] == '.':
        return int(s[0])
    else:
        return int(s)


class MakeLBW:
    @classmethod
    def INPUT_TYPES(s):
        preset = ["Preset"]  # 20
        preset += load_lbw_preset("lbw-preset.txt")
        preset += load_lbw_preset("lbw-preset.custom.txt")
        preset = [name for name in preset if not name.startswith('@')]

        lora_names = folder_paths.get_filename_list("loras")
        lora_dirs = [os.path.dirname(name) for name in lora_names]
        lora_dirs = ["All"] + list(set(lora_dirs))

        return {"required": {"model": ("MODEL",),
                             "clip": ("CLIP", ),
                             "category_filter": (lora_dirs,),
                             "lora_name": (lora_names, ),
                             "inverse": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False", "tooltip": "Apply the following weights for each block:\nTrue: 1 - weight\nFalse: weight"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": ""}),
                             "A": ("FLOAT", {"default": 4.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "B": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "preset": (preset,),
                             "block_vector": ("STRING", {"multiline": True, "placeholder": "block weight vectors", "default": "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1", "pysssss.autocomplete": False}),
                             "bypass": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                             }
                }

    RETURN_TYPES = ("LBW_MODEL", "STRING")
    RETURN_NAMES = ("lbw_model", "populated_vector")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/LoraBlockWeight"

    DESCRIPTION = "Instead of directly applying the LoRA Block Weight to the MODEL, it is generated in a separate LBW_MODEL form."

    def __init__(self):
        self.loaded_lora = None

    def doit(self, model, clip, lora_name, inverse, seed, A, B, preset, block_vector, bypass=False, category_filter=None):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        block_weights, muted_weights, populated_vector = LoraLoaderBlockWeight.load_lbw(model, clip, lora, inverse, seed, A, B, block_vector)
        lbw_model = {
                        'blocks': block_weights,
                        'muted': muted_weights
                    }
        return lbw_model, populated_vector


class LoraLoaderBlockWeight:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        preset = ["Preset"]  # 20
        preset += load_lbw_preset("lbw-preset.txt")
        preset += load_lbw_preset("lbw-preset.custom.txt")
        preset = [name for name in preset if not name.startswith('@')]

        lora_names = folder_paths.get_filename_list("loras")
        lora_dirs = [os.path.dirname(name) for name in lora_names]
        lora_dirs = ["All"] + list(set(lora_dirs))

        return {"required": {"model": ("MODEL",),
                             "clip": ("CLIP", ),
                             "category_filter": (lora_dirs,),
                             "lora_name": (lora_names, ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "inverse": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False", "tooltip": "Apply the following weights for each block:\nTrue: 1 - weight\nFalse: weight"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": ""}),
                             "A": ("FLOAT", {"default": 4.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "B": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "preset": (preset,),
                             "block_vector": ("STRING", {"multiline": True, "placeholder": "block weight vectors", "default": "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1", "pysssss.autocomplete": False}),
                             "bypass": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                             }
                }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "populated_vector")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/LoraBlockWeight"

    @staticmethod
    def validate(vectors):
        if len(vectors) < 12:
            return False

        for x in vectors:
            if x in ['R', 'r', 'U', 'u', 'A', 'a', 'B', 'b'] or is_numeric_string(x):
                continue
            else:
                subvectors = x.strip().split(' ')
                for y in subvectors:
                    y = y.strip()
                    if y not in ['R', 'r', 'U', 'u', 'A', 'a', 'B', 'b'] and not is_numeric_string(y):
                        return False

        return True

    @staticmethod
    def convert_vector_value(A, B, vector_value):
        def simple_vector(x):
            if x in ['U', 'u']:
                ratio = np.random.uniform(-1.5, 1.5)
                ratio = round(ratio, 2)
            elif x in ['R', 'r']:
                ratio = np.random.uniform(0, 3.0)
                ratio = round(ratio, 2)
            elif x == 'A':
                ratio = A
            elif x == 'a':
                ratio = A/2
            elif x == 'B':
                ratio = B
            elif x == 'b':
                ratio = B/2
            elif is_numeric_string(x):
                ratio = float(x)
            else:
                ratio = None

            return ratio

        v = simple_vector(vector_value)
        if v is not None:
            ratios = [v]
        else:
            ratios = [simple_vector(x) for x in vector_value.split(" ")]

        return ratios

    @staticmethod
    def norm_value(value):  # make to int if 1.0 or 0.0
        if value == 1:
            return 1
        elif value == 0:
            return 0
        else:
            return value

    @staticmethod
    def block_spec_parser(loaded, spec):
        if not spec.startswith("%"):
            return spec
        else:
            items = [x.strip() for x in spec[1:].split(',')]

            input_blocks_set = set()
            middle_blocks_set= set()
            output_blocks_set = set()
            double_blocks_set = set()
            single_blocks_set = set()
            layers_blocks_set = set()  # Z-Image/Lumina2 DiT layers
            krea_blocks_set = set()  # Krea 2 single-stream DiT blocks

            for key, v in loaded.items():
                if isinstance(key, tuple):
                    k = key[0]
                else:
                    k = key

                k_unet = k[len("diffusion_model."):]

                if k_unet.startswith("input_blocks."):
                    k_unet_num = k_unet[len("input_blocks."):len("input_blocks.")+2]
                    k_unet_int = parse_unet_num(k_unet_num)
                    input_blocks_set.add(k_unet_int)
                elif k_unet.startswith("middle_block."):
                    k_unet_num = k_unet[len("middle_block."):len("middle_block.")+2]
                    k_unet_int = parse_unet_num(k_unet_num)
                    middle_blocks_set.add(k_unet_int)
                elif k_unet.startswith("output_blocks."):
                    k_unet_num = k_unet[len("output_blocks."):len("output_blocks.")+2]
                    k_unet_int = parse_unet_num(k_unet_num)
                    output_blocks_set.add(k_unet_int)
                elif k_unet.startswith("double_blocks."):
                    k_unet_num = k_unet[len("double_blocks."):len("double_blocks.") + 2]
                    k_unet_int = parse_unet_num(k_unet_num)
                    double_blocks_set.add(k_unet_int)
                elif k_unet.startswith("single_blocks."):
                    k_unet_num = k_unet[len("single_blocks."):len("single_blocks.") + 2]
                    k_unet_int = parse_unet_num(k_unet_num)
                    single_blocks_set.add(k_unet_int)
                elif k_unet.startswith("layers."):
                    k_unet_num = k_unet[len("layers."):len("layers.") + 2]
                    k_unet_int = parse_unet_num(k_unet_num)
                    layers_blocks_set.add(k_unet_int)
                elif k_unet.startswith("blocks."):
                    k_unet_num = k_unet[len("blocks."):len("blocks.") + 2]
                    k_unet_int = parse_unet_num(k_unet_num)
                    krea_blocks_set.add(k_unet_int)

            pat1 = re.compile(r"(default|base)=([0-9.]+)")
            pat2 = re.compile(r"(in|out|mid|double|single|layer|block)([0-9]+)-([0-9]+)=([0-9.]+)")
            pat3 = re.compile(r"(in|out|mid|double|single|layer|block)([0-9]+)=([0-9.]+)")
            pat4 = re.compile(r"(in|out|mid|double|single|layer|block)=([0-9.]+)")

            base_spec = None
            default_spec = 1.0

            for item in items:
                match = pat1.match(item)
                if match:
                    if match[1] == 'base':
                        base_spec = match[2]
                        continue

                    if match[1] == 'default':
                        default_spec = match[2]
                        continue

            if base_spec is None:
                base_spec = default_spec

            input_blocks = [default_spec] * len(input_blocks_set)
            middle_blocks = [default_spec] * len(middle_blocks_set)
            output_blocks = [default_spec] * len(output_blocks_set)
            double_blocks = [default_spec] * len(double_blocks_set)
            single_blocks = [default_spec] * len(single_blocks_set)
            layers_blocks = [default_spec] * len(layers_blocks_set)  # Z-Image/Lumina2
            krea_blocks = [default_spec] * len(krea_blocks_set)  # Krea 2

            for item in items:
                match = pat2.match(item)
                if match:
                    for x in range(int(match[2])-1, int(match[3])):
                        value = float(match[4])

                        if x < 0:
                            continue

                        if match[1] == 'in' and len(input_blocks) > x:
                            input_blocks[x] = value
                        elif match[1] == 'out' and len(output_blocks) > x:
                            output_blocks[x] = value
                        elif match[1] == 'mid' and len(middle_blocks) > x:
                            middle_blocks[x] = value
                        elif match[1] == 'double' and len(double_blocks) > x:
                            double_blocks[x] = value
                        elif match[1] == 'single' and len(single_blocks) > x:
                            single_blocks[x] = value
                        elif match[1] == 'layer' and len(layers_blocks) > x:
                            layers_blocks[x] = value
                        elif match[1] == 'block' and len(krea_blocks) > x:
                            krea_blocks[x] = value

                    continue

                match = pat3.match(item)
                if match:
                    value = float(match[3])
                    x = int(match[2]) - 1

                    if x < 0:
                        continue

                    if match[1] == 'in' and len(input_blocks) > x:
                        input_blocks[x] = value
                    elif match[1] == 'out' and len(output_blocks) > x:
                        output_blocks[x] = value
                    elif match[1] == 'mid' and len(middle_blocks) > x:
                        middle_blocks[x] = value
                    elif match[1] == 'double' and len(double_blocks) > x:
                        double_blocks[x] = value
                    elif match[1] == 'single' and len(single_blocks) > x:
                        single_blocks[x] = value
                    elif match[1] == 'layer' and len(layers_blocks) > x:
                        layers_blocks[x] = value
                    elif match[1] == 'block' and len(krea_blocks) > x:
                        krea_blocks[x] = value

                    continue

                match = pat4.match(item)
                if match:
                    value = float(match[2])

                    if match[1] == 'in':
                        input_blocks = [value] * len(input_blocks)
                    elif match[1] == 'out':
                        output_blocks = [value] * len(output_blocks)
                    elif match[1] == 'mid':
                        middle_blocks = [value] * len(middle_blocks)
                    elif match[1] == 'double':
                        double_blocks = [value] * len(double_blocks)
                    elif match[1] == 'single':
                        single_blocks = [value] * len(single_blocks)
                    elif match[1] == 'layer':
                        layers_blocks = [value] * len(layers_blocks)
                    elif match[1] == 'block':
                        krea_blocks = [value] * len(krea_blocks)

                    continue

            # concat specs
            res = [str(base_spec)]
            for x in (input_blocks + middle_blocks + output_blocks + double_blocks + single_blocks + layers_blocks + krea_blocks):
                res.append(str(x))

            return ",".join(res)

    @staticmethod
    def transform_separate_to_fused_qkv(lora, key_map, model):
        """
        Transform a LoRA with separate Q/K/V attention keys to work with fused QKV models.
        This handles Lumina2/Z-Image LoRAs trained on diffusers implementations.

        The approach: Create a fused LoRA by concatenating B matrices (Q, K, V along output dim)
        and creating a block-diagonal A matrix. This produces a single LoRA that patches the
        entire fused QKV weight.
        """
        new_lora = dict(lora)  # Copy original

        # Get the model's state dict to understand the QKV dimensions
        sd = model.model.state_dict()

        # Find all layers that have separate Q/K/V in the LoRA
        # Pattern: diffusion_model.layers.X.attention.to_q.lora_A.weight
        layer_pattern = re.compile(r'diffusion_model\.layers\.(\d+)\.attention\.to_([qkv])\.lora_([AB])\.weight')

        # Group by layer number
        layers_data = {}
        keys_to_remove = set()
        for key in list(lora.keys()):
            match = layer_pattern.match(key)
            if match:
                layer_num = int(match.group(1))
                qkv_type = match.group(2)  # q, k, or v
                ab_type = match.group(3)   # A or B

                if layer_num not in layers_data:
                    layers_data[layer_num] = {}
                if qkv_type not in layers_data[layer_num]:
                    layers_data[layer_num][qkv_type] = {}
                layers_data[layer_num][qkv_type][ab_type] = key
                keys_to_remove.add(key)

        if not layers_data:
            return lora

        # For each layer, create fused QKV LoRA entries
        for layer_num, qkv_data in layers_data.items():
            # Check if we have all Q, K, V with both A and B
            if 'q' not in qkv_data or 'k' not in qkv_data or 'v' not in qkv_data:
                continue
            if 'A' not in qkv_data['q'] or 'B' not in qkv_data['q']:
                continue
            if 'A' not in qkv_data['k'] or 'B' not in qkv_data['k']:
                continue
            if 'A' not in qkv_data['v'] or 'B' not in qkv_data['v']:
                continue

            # Get the model's fused QKV key to understand dimensions
            model_qkv_key = f"diffusion_model.layers.{layer_num}.attention.qkv.weight"
            if model_qkv_key not in sd:
                continue

            # Get LoRA weights
            q_A = lora[qkv_data['q']['A']]  # [rank, in_dim]
            q_B = lora[qkv_data['q']['B']]  # [q_out_dim, rank]
            k_A = lora[qkv_data['k']['A']]  # [rank, in_dim]
            k_B = lora[qkv_data['k']['B']]  # [k_out_dim, rank]
            v_A = lora[qkv_data['v']['A']]  # [rank, in_dim]
            v_B = lora[qkv_data['v']['B']]  # [v_out_dim, rank]

            # Get dimensions
            rank_q = q_A.shape[0]
            rank_k = k_A.shape[0]
            rank_v = v_A.shape[0]
            in_dim = q_A.shape[1]
            q_out_dim = q_B.shape[0]
            k_out_dim = k_B.shape[0]
            v_out_dim = v_B.shape[0]

            # Total fused rank is sum of individual ranks
            total_rank = rank_q + rank_k + rank_v
            total_out_dim = q_out_dim + k_out_dim + v_out_dim

            # Create block-diagonal A matrix: [total_rank, in_dim]
            # Each block is independent, so we stack them
            fused_A = torch.zeros(total_rank, in_dim, dtype=q_A.dtype, device=q_A.device)
            fused_A[0:rank_q, :] = q_A
            fused_A[rank_q:rank_q+rank_k, :] = k_A
            fused_A[rank_q+rank_k:, :] = v_A

            # Create fused B matrix: [total_out_dim, total_rank]
            # B matrices are block-diagonal too
            fused_B = torch.zeros(total_out_dim, total_rank, dtype=q_B.dtype, device=q_B.device)
            fused_B[0:q_out_dim, 0:rank_q] = q_B
            fused_B[q_out_dim:q_out_dim+k_out_dim, rank_q:rank_q+rank_k] = k_B
            fused_B[q_out_dim+k_out_dim:, rank_q+rank_k:] = v_B

            # Create the fused LoRA entry
            fused_key_base = f"diffusion_model.layers.{layer_num}.attention.qkv"
            new_lora[f"{fused_key_base}.lora_A.weight"] = fused_A
            new_lora[f"{fused_key_base}.lora_B.weight"] = fused_B

            # Handle alpha - use Q's alpha if present (they should all be the same)
            alpha_q_key = qkv_data['q']['A'].replace('.lora_A.weight', '.alpha')
            if alpha_q_key in lora:
                # Scale alpha by rank ratio since we changed the rank
                # Original: alpha / rank_q, now we want same scaling
                # But since B @ A gives same result, we keep alpha as-is per component
                # Actually with block-diagonal, each component still has its own effective rank
                # Use the original alpha (they should be consistent)
                new_lora[f"{fused_key_base}.alpha"] = lora[alpha_q_key]

            # Add the fused key to key_map
            key_map[fused_key_base] = model_qkv_key

        # Remove the original separate Q/K/V keys
        for key in keys_to_remove:
            if key in new_lora:
                del new_lora[key]
            # Also remove associated alpha keys
            alpha_key = key.replace('.lora_A.weight', '.alpha').replace('.lora_B.weight', '.alpha')
            if alpha_key in new_lora:
                del new_lora[alpha_key]

        return new_lora

    @staticmethod
    def load_lbw(model, clip, lora, inverse, seed, A, B, block_vector):
        key_map = comfy.lora.model_lora_keys_unet(model.model)
        # Only process CLIP keys if clip model is provided (some LoRAs are UNet-only)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        # Check if we have Lumina2/Z-Image LoRA with separate attention keys
        # that need to be mapped to fused QKV format
        lora_has_separate_attn = any("attention.to_q" in k or "attention.to_k" in k or "attention.to_v" in k for k in lora.keys())

        # Check model state dict directly for fused QKV
        model_sd = model.model.state_dict()
        model_has_fused_qkv = any("attention.qkv" in k for k in model_sd.keys())

        if lora_has_separate_attn and model_has_fused_qkv:
            # Transform separate Q/K/V LoRA keys to work with fused QKV model
            lora = LoraLoaderBlockWeight.transform_separate_to_fused_qkv(lora, key_map, model)

        # Handle to_out.0 -> out mapping for attention output projections
        lora_has_to_out = any(".attention.to_out.0." in k for k in lora.keys())
        model_sd = model.model.state_dict()
        if lora_has_to_out:
            # Find unique layer numbers that need to_out mapping
            layer_pattern = re.compile(r'diffusion_model\.layers\.(\d+)\.attention\.to_out\.0\.lora_[AB]\.weight')
            layers_needing_out_mapping = set()
            for lora_key in lora.keys():
                match = layer_pattern.match(lora_key)
                if match:
                    layers_needing_out_mapping.add(int(match.group(1)))

            for layer_num in layers_needing_out_mapping:
                base_key = f"diffusion_model.layers.{layer_num}.attention.to_out.0"
                model_key = f"diffusion_model.layers.{layer_num}.attention.out.weight"
                if model_key in model_sd and base_key not in key_map:
                    key_map[base_key] = model_key

        loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)

        block_vector = LoraLoaderBlockWeight.block_spec_parser(loaded, block_vector)

        block_vector = block_vector.split(":")
        if len(block_vector) > 1:
            block_vector = block_vector[1]
        else:
            block_vector = block_vector[0]

        vector = block_vector.split(",")

        if not LoraLoaderBlockWeight.validate(vector):
            preset_dict = load_preset_dict()
            if len(vector) > 0 and vector[0].strip() in preset_dict:
                vector = preset_dict[vector[0].strip()].split(",")
            else:
                raise ValueError(f"[LoraLoaderBlockWeight] invalid block_vector '{block_vector}'")

        # sort: input, middle, output, others
        input_blocks = []
        middle_blocks = []
        output_blocks = []
        double_blocks = []
        single_blocks = []
        layers_blocks = []  # Z-Image/Lumina2 DiT layers
        krea_blocks = []  # Krea 2 single-stream DiT blocks
        others = []
        for key, v in loaded.items():
            if isinstance(key, tuple):
                k = key[0]
            else:
                k = key

            k_unet = k[len("diffusion_model."):]

            if k_unet.startswith("input_blocks."):
                k_unet_num = k_unet[len("input_blocks."):len("input_blocks.")+2]
                input_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("middle_block."):
                k_unet_num = k_unet[len("middle_block."):len("middle_block.")+2]
                middle_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("output_blocks."):
                k_unet_num = k_unet[len("output_blocks."):len("output_blocks.")+2]
                output_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("double_blocks."):
                k_unet_num = k_unet[len("double_blocks."):len("double_blocks.")+2]
                double_blocks.append((key, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("single_blocks."):
                k_unet_num = k_unet[len("single_blocks."):len("single_blocks.")+2]
                single_blocks.append((key, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("layers."):
                k_unet_num = k_unet[len("layers."):len("layers.")+2]
                layers_blocks.append((key, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("blocks."):
                k_unet_num = k_unet[len("blocks."):len("blocks.")+2]
                krea_blocks.append((key, v, parse_unet_num(k_unet_num), k_unet))
            else:
                others.append((k, v, k_unet))

        input_blocks = sorted(input_blocks, key=lambda x: x[2])
        middle_blocks = sorted(middle_blocks, key=lambda x: x[2])
        output_blocks = sorted(output_blocks, key=lambda x: x[2])
        double_blocks = sorted(double_blocks, key=lambda x: x[2])
        single_blocks = sorted(single_blocks, key=lambda x: x[2])
        layers_blocks = sorted(layers_blocks, key=lambda x: x[2])
        krea_blocks = sorted(krea_blocks, key=lambda x: x[2])

        # prepare patch
        np.random.seed(seed % (2**31))
        populated_vector_list = []
        ratios = []
        ratio = 1.0
        vector_i = 1

        last_k_unet_num = None

        block_weights = {}
        muted_weights = []

        for k, v, k_unet_num, k_unet in (input_blocks + middle_blocks + output_blocks + double_blocks + single_blocks + layers_blocks + krea_blocks):
            if last_k_unet_num != k_unet_num and len(vector) > vector_i:
                ratios = LoraLoaderBlockWeight.convert_vector_value(A, B, vector[vector_i].strip())
                ratio = ratios.pop(0)

                if inverse:
                    populated_ratio = 1 - ratio
                else:
                    populated_ratio = ratio

                populated_vector_list.append(LoraLoaderBlockWeight.norm_value(populated_ratio))

                vector_i += 1
            else:
                if len(ratios) > 0:
                    ratio = ratios.pop(0)
                else:
                    pass # use last used ratio if no more user specified ratio is given

                if inverse:
                    populated_ratio = 1 - ratio
                else:
                    populated_ratio = ratio

            last_k_unet_num = k_unet_num

            if populated_ratio != 0:
                block_weights[k] = v, populated_ratio
            else:
                muted_weights.append(k)

        # prepare base patch
        ratios = LoraLoaderBlockWeight.convert_vector_value(A, B, vector[0].strip())
        ratio = ratios.pop(0)

        if inverse:
            populated_ratio = 1 - ratio
        else:
            populated_ratio = ratio

        populated_vector_list.insert(0, LoraLoaderBlockWeight.norm_value(populated_ratio))

        for k, v, k_unet in others:
            if populated_ratio != 0:
                block_weights[k] = v, populated_ratio
            else:
                muted_weights.append(k)

        populated_vector = ','.join(map(str, populated_vector_list))
        return block_weights, muted_weights, populated_vector

    @staticmethod
    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, inverse, seed, A, B, block_vector):
        block_weights, muted_weights, populated_vector = LoraLoaderBlockWeight.load_lbw(model, clip, lora, inverse, seed, A, B, block_vector)

        new_modelpatcher = model.clone()
        new_clip = clip.clone() if clip is not None else None

        muted_weights = set(muted_weights)

        for k, v in block_weights.items():
            weights, ratio = v

            # Extract string key for checking (handle tuple keys with offset info)
            key_str = k[0] if isinstance(k, tuple) else k

            if k in muted_weights:
                pass
            elif 'text' in key_str or 'encoder' in key_str:
                if new_clip is not None:
                    final_strength = strength_clip * ratio
                    new_clip.add_patches({k: weights}, final_strength)
            else:
                final_strength = strength_model * ratio
                new_modelpatcher.add_patches({k: weights}, final_strength)

        return new_modelpatcher, new_clip, populated_vector

    def doit(self, model, clip, lora_name, strength_model, strength_clip, inverse, seed, A, B, preset, block_vector, bypass=False, category_filter=None):
        if strength_model == 0 and strength_clip == 0 or bypass:
            return model, clip, ""

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora, populated_vector = LoraLoaderBlockWeight.load_lora_for_models(model, clip, lora, strength_model, strength_clip, inverse, seed, A, B, block_vector)
        return model_lora, clip_lora, populated_vector


class ApplyLBW:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "clip": ("CLIP", ),
                    "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "lbw_model": ("LBW_MODEL",),
                }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/LoraBlockWeight"

    DESCRIPTION = "Apply LBW_MODEL to MODEL and CLIP"

    @staticmethod
    def doit(model, clip, strength_model, strength_clip, lbw_model):
        block_weights = lbw_model['blocks']
        muted_weights = lbw_model['muted']

        new_modelpatcher = model.clone()
        new_clip = clip.clone()

        muted_weights = set(muted_weights)

        for k, v in block_weights.items():
            weights, ratio = v

            if k in muted_weights:
                pass
            elif 'text' in k or 'encoder' in k:
                new_clip.add_patches({k: weights}, strength_clip * ratio)
            else:
                new_modelpatcher.add_patches({k: weights}, strength_model * ratio)

        return new_modelpatcher, new_clip


class XY_Capsule_LoraBlockWeight:
    def __init__(self, x, y, target_vector, label, storage, params):
        self.x = x
        self.y = y
        self.target_vector = target_vector
        self.reference_vector = None
        self.label = label
        self.storage = storage
        self.another_capsule = None
        self.params = params

    def set_reference_vector(self, vector):
        self.reference_vector = vector

    def set_x_capsule(self, capsule):
        self.another_capsule = capsule

    def set_result(self, image, latent):
        if self.another_capsule is not None:
            logging.info(f"XY_Capsule_LoraBlockWeight: ({self.another_capsule.x, self.y}) is processed.")
            self.storage[(self.another_capsule.x, self.y)] = image
        else:
            logging.info(f"XY_Capsule_LoraBlockWeight: ({self.x, self.y}) is processed.")

    def patch_model(self, model, clip):
        lora_name, strength_model, strength_clip, inverse, block_vectors, seed, A, B, heatmap_palette, heatmap_alpha, heatmap_strength, xyplot_mode = self.params

        try:
            if self.y == 0:
                target_vector = self.another_capsule.target_vector if self.another_capsule else self.target_vector
                model, clip, _ = LoraLoaderBlockWeight().doit(model, clip, lora_name, strength_model, strength_clip, inverse,
                                                              seed, A, B, "", target_vector)
            elif self.y == 1:
                reference_vector = self.another_capsule.reference_vector if self.another_capsule else self.reference_vector
                model, clip, _ = LoraLoaderBlockWeight().doit(model, clip, lora_name, strength_model, strength_clip, inverse,
                                                              seed, A, B, "", reference_vector)
        except:
            self.storage[(self.another_capsule.x, self.y)] = "fail"
            pass

        return model, clip

    def pre_define_model(self, model, clip, vae):
        if self.y < 2:
            model, clip = self.patch_model(model, clip)

        return model, clip, vae

    def get_result(self, model, clip, vae):
        _, _, _, _, _, _, _, _, heatmap_palette, heatmap_alpha, heatmap_strength, xyplot_mode = self.params

        if self.y < 2:
            return None

        if self.y == 2:
            # diff
            weighted_image = self.storage[(self.another_capsule.x, 0)]
            reference_image = self.storage[(self.another_capsule.x, 1)]

            if weighted_image == "fail" or reference_image == "fail":
                image = "fail"
            else:
                image = torch.abs(weighted_image - reference_image)
                self.storage[(self.another_capsule.x, self.y)] = image

        elif self.y == 3:
            import matplotlib.cm as cm
            # heatmap
            image = self.storage[(self.another_capsule.x, 0)]

            if image == "fail":
                image = utils.empty_pil_tensor(8,8)
                latent = utils.empty_latent()
                return image, latent
            else:
                image = image.clone()

                diff_image = torch.abs(self.storage[(self.another_capsule.x, 2)])

                heatmap = torch.sum(diff_image, dim=3, keepdim=True)

                min_val = torch.min(heatmap)
                max_val = torch.max(heatmap)
                heatmap = (heatmap - min_val) / (max_val - min_val)
                heatmap *= heatmap_strength

                # viridis / magma / plasma / inferno / cividis
                if heatmap_palette == "magma":
                    colormap = cm.magma
                elif heatmap_palette == "plasma":
                    colormap = cm.plasma
                elif heatmap_palette == "inferno":
                    colormap = cm.inferno
                elif heatmap_palette == "cividis":
                    colormap = cm.cividis
                else:
                    # default: viridis
                    colormap = cm.viridis

                heatmap = torch.from_numpy(colormap(heatmap.squeeze())).unsqueeze(0)
                heatmap = heatmap[..., :3]

                image = heatmap_alpha * heatmap + (1 - heatmap_alpha) * image

        latent = nodes.VAEEncode().encode(vae, image)[0]
        return image, latent

    def getLabel(self):
        return self.label


def load_preset_dict():
    preset = ["Preset"]  # 20
    preset += load_lbw_preset("lbw-preset.txt")
    preset += load_lbw_preset("lbw-preset.custom.txt")

    dict = {}
    for x in preset:
        if not x.startswith('@'):
            item = x.split(':')
            if len(item) > 1:
                dict[item[0]] = item[1]

    return dict


class XYInput_LoraBlockWeight:
    @staticmethod
    def resolve_vector_string(vector_string, preset_dict):
        vector_string = vector_string.strip()

        if vector_string in preset_dict:
            return vector_string, preset_dict[vector_string]

        vector_infos = vector_string.split(':')

        if len(vector_infos) > 1:
            return vector_infos[0], vector_infos[1]
        elif len(vector_infos) > 0:
            return vector_infos[0], vector_infos[0]
        else:
            return None, None

    @classmethod
    def INPUT_TYPES(cls):
        preset = ["Preset"]  # 20
        preset += load_lbw_preset("lbw-preset.txt")
        preset += load_lbw_preset("lbw-preset.custom.txt")

        default_vectors = "SD-NONE/SD-ALL\nSD-ALL/SD-ALL\nSD-INS/SD-ALL\nSD-IND/SD-ALL\nSD-INALL/SD-ALL\nSD-MIDD/SD-ALL\nSD-MIDD0.2/SD-ALL\nSD-MIDD0.8/SD-ALL\nSD-MOUT/SD-ALL\nSD-OUTD/SD-ALL\nSD-OUTS/SD-ALL\nSD-OUTALL/SD-ALL"

        lora_names = folder_paths.get_filename_list("loras")
        lora_dirs = [os.path.dirname(name) for name in lora_names]
        lora_dirs = ["All"] + list(set(lora_dirs))

        return {"required": {
                             "category_filter": (lora_dirs, ),
                             "lora_name": (lora_names, ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "inverse": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "A": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "B": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "preset": (preset,),
                             "block_vectors": ("STRING", {"multiline": True, "default": default_vectors, "placeholder": "{target vector}/{reference vector}", "pysssss.autocomplete": False}),
                             "heatmap_palette": (["viridis", "magma", "plasma", "inferno", "cividis"], ),
                             "heatmap_alpha":  ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "heatmap_strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "xyplot_mode": (["Simple", "Diff", "Diff+Heatmap"],),
                             }}

    RETURN_TYPES = ("XY", "XY")
    RETURN_NAMES = ("X (vectors)", "Y (effect_compares)")

    FUNCTION = "doit"
    CATEGORY = "InspirePack/LoraBlockWeight"

    def doit(self, lora_name, strength_model, strength_clip, inverse, seed, A, B, preset, block_vectors, heatmap_palette, heatmap_alpha, heatmap_strength, xyplot_mode, category_filter=None):
        xy_type = "XY_Capsule"

        preset_dict = load_preset_dict()
        common_params = lora_name, strength_model, strength_clip, inverse, block_vectors, seed, A, B, heatmap_palette, heatmap_alpha, heatmap_strength, xyplot_mode

        storage = {}
        x_values = []
        x_idx = 0
        for block_vector in block_vectors.split("\n"):
            if block_vector == "":
                continue

            item = block_vector.split('/')

            if len(item) > 0:
                target_vector = item[0].strip()
                ref_vector = item[1].strip() if len(item) > 1 else ''

                x_item = None
                label, block_vector = XYInput_LoraBlockWeight.resolve_vector_string(target_vector, preset_dict)
                _, ref_block_vector = XYInput_LoraBlockWeight.resolve_vector_string(ref_vector, preset_dict)
                if label is not None:
                    x_item = XY_Capsule_LoraBlockWeight(x_idx, 0, block_vector, label, storage, common_params)
                    x_idx += 1

                if x_item is not None and ref_block_vector is not None:
                    x_item.set_reference_vector(ref_block_vector)

                if x_item is not None:
                    x_values.append(x_item)

        if xyplot_mode == "Simple":
            y_values = [XY_Capsule_LoraBlockWeight(0, 0, '', 'target', storage, common_params)]
        elif xyplot_mode == "Diff":
            y_values = [XY_Capsule_LoraBlockWeight(0, 0, '', 'target', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 1, '', 'reference', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 2, '', 'diff', storage, common_params)]
        else:
            y_values = [XY_Capsule_LoraBlockWeight(0, 0, '', 'target', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 1, '', 'reference', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 2, '', 'diff', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 3, '', 'heatmap', storage, common_params)]

        return (xy_type, x_values), (xy_type, y_values),


class LoraBlockInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "model": ("MODEL", ),
                        "clip": ("CLIP", ),
                        "lora_name": (folder_paths.get_filename_list("loras"), ),
                        "block_info": ("STRING", {"multiline": True}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    CATEGORY = "InspirePack/LoraBlockWeight"

    OUTPUT_NODE = True

    RETURN_TYPES = ()
    FUNCTION = "doit"

    @staticmethod
    def extract_info(model, clip, lora):
        key_map = comfy.lora.model_lora_keys_unet(model.model)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)

        def parse_unet_num(s):
            if s[1] == '.':
                return int(s[0])
            else:
                return int(s)

        input_block_count = set()
        input_blocks = []
        input_blocks_map = {}

        middle_block_count = set()
        middle_blocks = []
        middle_blocks_map = {}

        output_block_count = set()
        output_blocks = []
        output_blocks_map = {}

        text_block_count1 = set()
        text_blocks1 = []
        text_blocks_map1 = {}

        text_block_count2 = set()
        text_blocks2 = []
        text_blocks_map2 = {}

        double_block_count = set()
        double_blocks = []
        double_blocks_map = {}

        single_block_count = set()
        single_blocks = []
        single_blocks_map = {}

        others = []
        for key, v in loaded.items():
            if isinstance(key, tuple):
                k = key[0]
            else:
                k = key

            k_unet = k[len("diffusion_model."):]

            if k_unet.startswith("input_blocks."):
                k_unet_num = k_unet[len("input_blocks."):len("input_blocks.")+2]
                k_unet_int = parse_unet_num(k_unet_num)

                input_block_count.add(k_unet_int)
                input_blocks.append(k_unet)
                if k_unet_int in input_blocks_map:
                    input_blocks_map[k_unet_int].append(k_unet)
                else:
                    input_blocks_map[k_unet_int] = [k_unet]

            elif k_unet.startswith("middle_block."):
                k_unet_num = k_unet[len("middle_block."):len("middle_block.")+2]
                k_unet_int = parse_unet_num(k_unet_num)

                middle_block_count.add(k_unet_int)
                middle_blocks.append(k_unet)
                if k_unet_int in middle_blocks_map:
                    middle_blocks_map[k_unet_int].append(k_unet)
                else:
                    middle_blocks_map[k_unet_int] = [k_unet]

            elif k_unet.startswith("output_blocks."):
                k_unet_num = k_unet[len("output_blocks."):len("output_blocks.")+2]
                k_unet_int = parse_unet_num(k_unet_num)

                output_block_count.add(k_unet_int)
                output_blocks.append(k_unet)
                if k_unet_int in output_blocks_map:
                    output_blocks_map[k_unet_int].append(k_unet)
                else:
                    output_blocks_map[k_unet_int] = [k_unet]

            elif k_unet.startswith("double_blocks."):
                k_unet_num = k_unet[len("double_blocks."):len("double_blocks.") + 2]
                k_unet_int = parse_unet_num(k_unet_num)

                double_block_count.add(k_unet_int)
                double_blocks.append(k_unet)
                if k_unet_int in double_blocks_map:
                    double_blocks_map[k_unet_int].append(k_unet)
                else:
                    double_blocks_map[k_unet_int] = [k_unet]

            elif k_unet.startswith("single_blocks."):
                k_unet_num = k_unet[len("single_blocks."):len("single_blocks.") + 2]
                k_unet_int = parse_unet_num(k_unet_num)

                single_block_count.add(k_unet_int)
                single_blocks.append(k_unet)
                if k_unet_int in single_blocks_map:
                    single_blocks_map[k_unet_int].append(k_unet)
                else:
                    single_blocks_map[k_unet_int] = [k_unet]

            elif k_unet.startswith("er.text_model.encoder.layers."):
                k_unet_num = k_unet[len("er.text_model.encoder.layers."):len("er.text_model.encoder.layers.")+2]
                k_unet_int = parse_unet_num(k_unet_num)

                text_block_count1.add(k_unet_int)
                text_blocks1.append(k_unet)
                if k_unet_int in text_blocks_map1:
                    text_blocks_map1[k_unet_int].append(k_unet)
                else:
                    text_blocks_map1[k_unet_int] = [k_unet]

            elif k_unet.startswith("r.encoder.block."):
                k_unet_num = k_unet[len("r.encoder.block."):len("r.encoder.block.")+2]
                k_unet_int = parse_unet_num(k_unet_num)

                text_block_count2.add(k_unet_int)
                text_blocks2.append(k_unet)
                if k_unet_int in text_blocks_map2:
                    text_blocks_map2[k_unet_int].append(k_unet)
                else:
                    text_blocks_map2[k_unet_int] = [k_unet]

            else:
                others.append(k_unet)

        text = ""

        input_blocks = sorted(input_blocks)
        middle_blocks = sorted(middle_blocks)
        output_blocks = sorted(output_blocks)
        double_blocks = sorted(double_blocks)
        single_blocks = sorted(single_blocks)
        others = sorted(others)

        if len(input_block_count) > 0:
            text += f"\n-------[Input blocks] ({len(input_block_count)}, Subs={len(input_blocks)})-------\n"
            input_keys = sorted(input_blocks_map.keys())
            for x in input_keys:
                text += f" IN{x}: {len(input_blocks_map[x])}\n"

        if len(middle_block_count) > 0:
            text += f"\n-------[Middle blocks] ({len(middle_block_count)}, Subs={len(middle_blocks)})-------\n"
            middle_keys = sorted(middle_blocks_map.keys())
            for x in middle_keys:
                text += f" MID{x}: {len(middle_blocks_map[x])}\n"

        if len(output_block_count) > 0:
            text += f"\n-------[Output blocks] ({len(output_block_count)}, Subs={len(output_blocks)})-------\n"
            output_keys = sorted(output_blocks_map.keys())
            for x in output_keys:
                text += f" OUT{x}: {len(output_blocks_map[x])}\n"

        if len(double_block_count) > 0:
            text += f"\n-------[Double blocks(MMDiT)] ({len(double_block_count)}, Subs={len(double_blocks)})-------\n"
            double_keys = sorted(double_blocks_map.keys())
            for x in double_keys:
                text += f" DOUBLE{x}: {len(double_blocks_map[x])}\n"

        if len(single_block_count) > 0:
            text += f"\n-------[Single blocks(DiT)] ({len(single_block_count)}, Subs={len(single_blocks)})-------\n"
            single_keys = sorted(single_blocks_map.keys())
            for x in single_keys:
                text += f" SINGLE{x}: {len(single_blocks_map[x])}\n"

        text += f"\n-------[Base blocks] ({len(text_block_count1) + len(text_block_count2) + len(others)}, Subs={len(text_blocks1) + len(text_blocks2) + len(others)})-------\n"
        text_keys1 = sorted(text_blocks_map1.keys())
        for x in text_keys1:
            text += f" TXT_ENC{x}: {len(text_blocks_map1[x])}\n"

        text_keys2 = sorted(text_blocks_map2.keys())
        for x in text_keys2:
            text += f" TXT_ENC{x} [B]: {len(text_blocks_map2[x])}\n"

        for x in others:
            text += f" {x}\n"

        return text

    def doit(self, model, clip, lora_name, block_info, unique_id):
        lora_path = folder_paths.get_full_path("loras", lora_name)

        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        text = LoraBlockInfo.extract_info(model, clip, lora)

        PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": unique_id, "widget_name": "block_info", "type": "text", "data": text})
        return {}


class LoadLBW:
    @classmethod
    def INPUT_TYPES(s):
        files = folder_paths.get_filename_list('lbw_models')
        return {"required": {
            "lbw_model": [sorted(files), ]},
        }

    RETURN_TYPES = ("LBW_MODEL",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/LoraBlockWeight"

    DESCRIPTION = "Load LBW_MODEL from .lbw.safetensors file"

    @staticmethod
    def decode_dict(encoded_dict, tensor_dict):
        original_dict = {}

        def decode_value(value):
            if isinstance(value, str) and value.startswith('t') and value[1:].isdigit():
                return tensor_dict[value]
            return value

        for k, tuple_value in encoded_dict.items():
            decoded_tuple = tuple(decode_value(v) for v in tuple_value[0][1])
            key = ast.literal_eval(k) if isinstance(k, str) and (k.startswith('(') or k.startswith('[')) else k
            original_dict[key] = ((tuple_value[0][0], decoded_tuple), tuple_value[1])

        return original_dict

    @staticmethod
    def load(file):
        tensor_dict = comfy.utils.load_torch_file(file)

        with safe_open(file, framework="pt") as f:
            metadata = f.metadata()

        encoded_dict = json.loads(metadata.get('blocks', '{}'))
        muted_blocks = ast.literal_eval(metadata.get('muted_blocks', '[]'))

        decoded_dict = LoadLBW.decode_dict(encoded_dict, tensor_dict)

        lbw_model = {
            'blocks': decoded_dict,
            'muted': muted_blocks
        }

        return lbw_model, metadata

    def doit(self, lbw_model):
        lbw_path = folder_paths.get_full_path("lbw_models", lbw_model)
        lbw_model, _ = LoadLBW.load(lbw_path)
        return (lbw_model,)


class SaveLBW:
    def __init__(self):
        self.output_dir = folder_paths.get_folder_paths('lbw_models')[-1]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "lbw_model": ("LBW_MODEL", ),
                              "filename_prefix": ("STRING", {"default": "ComfyUI"}) },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ()
    FUNCTION = "doit"

    OUTPUT_NODE = True

    CATEGORY = "InspirePack/LoraBlockWeight"

    DESCRIPTION = "Save LBW_MODEL as a .lbw.safetensors file"

    @staticmethod
    def encode_dict(original_dict):
        tensor_dict = {}
        encoded_dict = {}
        counter = 0

        def generate_unique_id():
            nonlocal counter
            counter += 1
            return f"t{counter}"

        def encode_value(value):
            if isinstance(value, torch.Tensor):
                unique_id = generate_unique_id()
                tensor_dict[unique_id] = value
                return unique_id
            return value

        for k, tuple_value in original_dict.items():
            encoded_tuple = tuple(encode_value(v) for v in tuple_value[0][1])
            encoded_dict[str(k)] = (tuple_value[0][0], encoded_tuple), tuple_value[1]

        return encoded_dict, tensor_dict

    @staticmethod
    def save(lbw_model, file, metadata):
        metadata['format'] = 'Inspire LBW 1.0'
        weighted_blocks = lbw_model['blocks']
        metadata['muted_blocks'] = str(lbw_model['muted'])
        encoded_dict, tensor_dict = SaveLBW.encode_dict(weighted_blocks)
        metadata['blocks'] = json.dumps(encoded_dict)

        comfy.utils.save_torch_file(tensor_dict, file, metadata=metadata)

    def doit(self, lbw_model, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        # support save metadata for lbw sharing
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = {}
        if not args.disable_metadata:
            metadata = {"prompt": prompt_info}
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        file = f"{filename}_{counter:05}_.lbw.safetensors"

        results = list()
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": "output"
        })

        file = os.path.join(full_output_folder, file)

        SaveLBW.save(lbw_model, file, metadata)

        return {}


NODE_CLASS_MAPPINGS = {
    "XY Input: Lora Block Weight //Inspire": XYInput_LoraBlockWeight,
    "LoraLoaderBlockWeight //Inspire": LoraLoaderBlockWeight,
    "LoraBlockInfo //Inspire": LoraBlockInfo,
    "MakeLBW //Inspire": MakeLBW,
    "ApplyLBW //Inspire": ApplyLBW,
    "SaveLBW //Inspire": SaveLBW,
    "LoadLBW //Inspire": LoadLBW,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XY Input: Lora Block Weight //Inspire": "XY Input: LoRA Block Weight",
    "LoraLoaderBlockWeight //Inspire": "LoRA Loader (Block Weight)",
    "LoraBlockInfo //Inspire": "LoRA Block Info",
    "MakeLBW //Inspire": "Make LoRA Block Weight",
    "ApplyLBW //Inspire": "Apply LoRA Block Weight",
    "SaveLBW //Inspire": "Save LoRA Block Weight",
    "LoadLBW //Inspire": "Load LoRA Block Weight",
}

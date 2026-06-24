import torch
import copy

class DonutBlockCalibration:
    """
    Calibrates merged model block magnitudes to match a reference model.
    Works like automatic Donut Detailer XL Blocks adjustment.
    """
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_model": ("MODEL",),      # Model after merge
                "reference_model": ("MODEL",),   # Model to copy block magnitudes from
                "input_blocks_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),     # Input blocks calibration strength
                "middle_blocks_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),    # Middle blocks calibration strength
                "output_1_blocks_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),  # Output blocks 0-4 calibration strength
                "output_2_blocks_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),  # Output blocks 5-8 calibration strength
            },
            "optional": {
                "preset": (["Manual", "Auto (full strength)"], {"default": "Manual"}),  # Auto forces all strengths to 1.0
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "calibration_info")
    FUNCTION = "calibrate"
    CATEGORY = "donut/calibration"

    def extract_block_magnitudes(self, model):
        """Extract average magnitude for each SDXL block from a model"""
        block_magnitudes = {}
        
        # Get model parameters
        if hasattr(model, 'model'):
            state_dict = model.model.state_dict()
        elif hasattr(model, 'unet'):
            state_dict = model.unet.state_dict()
        else:
            state_dict = model.state_dict()
        
        # Group parameters by blocks (same logic as Donut Detailer XL Blocks)
        block_groups = {
            'input_blocks': {},
            'middle_block': {},
            'output_blocks': {},
            'out': {}
        }
        
        # Process each parameter
        for param_name, param in state_dict.items():
            # Skip non-diffusion model parameters
            if not param_name.startswith('diffusion_model.'):
                continue
                
            # Remove diffusion_model prefix
            clean_name = param_name[len('diffusion_model.'):]
            
            # Classify into blocks
            if clean_name.startswith('input_blocks.'):
                block_idx = clean_name.split('.')[1]
                if block_idx not in block_groups['input_blocks']:
                    block_groups['input_blocks'][block_idx] = []
                block_groups['input_blocks'][block_idx].append(param)
                
            elif clean_name.startswith('middle_block.'):
                block_idx = clean_name.split('.')[1]
                if block_idx not in block_groups['middle_block']:
                    block_groups['middle_block'][block_idx] = []
                block_groups['middle_block'][block_idx].append(param)
                
            elif clean_name.startswith('output_blocks.'):
                block_idx = clean_name.split('.')[1]
                if block_idx not in block_groups['output_blocks']:
                    block_groups['output_blocks'][block_idx] = []
                block_groups['output_blocks'][block_idx].append(param)
                
            elif clean_name.startswith('out.'):
                if 'main' not in block_groups['out']:
                    block_groups['out']['main'] = []
                block_groups['out']['main'].append(param)
        
        # Calculate average magnitude for each block
        for block_type, blocks in block_groups.items():
            block_magnitudes[block_type] = {}
            for block_idx, params in blocks.items():
                if params:
                    # Calculate average magnitude across all parameters in block
                    total_magnitude = 0.0
                    param_count = 0
                    for param in params:
                        total_magnitude += torch.norm(param).item()
                        param_count += 1
                    block_magnitudes[block_type][block_idx] = total_magnitude / param_count if param_count > 0 else 0.0
        
        return block_magnitudes

    def apply_block_calibration(self, merged_model, reference_model, input_strength, middle_strength, output_1_strength, output_2_strength):
        """Apply block-level magnitude calibration to match reference model"""
        # Extract magnitudes from both models
        ref_magnitudes = self.extract_block_magnitudes(reference_model)
        merged_magnitudes = self.extract_block_magnitudes(merged_model)
        
        # Get merged model parameters
        if hasattr(merged_model, 'model'):
            target_state_dict = merged_model.model.state_dict()
        elif hasattr(merged_model, 'unet'):
            target_state_dict = merged_model.unet.state_dict()
        else:
            target_state_dict = merged_model.state_dict()
        
        calibration_applied = 0
        calibration_details = []
        skipped_groups = []
        
        # Determine which block groups to process based on strength
        process_input = input_strength > 0.0
        process_middle = middle_strength > 0.0
        process_output_1 = output_1_strength > 0.0
        process_output_2 = output_2_strength > 0.0
        
        # Track skipped groups for reporting
        if not process_input:
            skipped_groups.append("Input Blocks")
        if not process_middle:
            skipped_groups.append("Middle Blocks")
        if not process_output_1:
            skipped_groups.append("Output Blocks 0-4")
        if not process_output_2:
            skipped_groups.append("Output Blocks 5-8")
        
        with torch.no_grad():
            for param_name, param in target_state_dict.items():
                # Skip non-diffusion model parameters
                if not param_name.startswith('diffusion_model.'):
                    continue
                    
                # Remove diffusion_model prefix
                clean_name = param_name[len('diffusion_model.'):]
                
                # Determine which block this parameter belongs to and get appropriate strength
                block_type = None
                block_idx = None
                calibration_strength = 0.0
                
                if clean_name.startswith('input_blocks.'):
                    if not process_input:
                        continue  # Skip processing if strength is 0
                    block_type = 'input_blocks'
                    block_idx = clean_name.split('.')[1]
                    calibration_strength = input_strength
                elif clean_name.startswith('middle_block.'):
                    if not process_middle:
                        continue  # Skip processing if strength is 0
                    block_type = 'middle_block'
                    block_idx = clean_name.split('.')[1]
                    calibration_strength = middle_strength
                elif clean_name.startswith('output_blocks.'):
                    output_block_num = int(clean_name.split('.')[1])
                    if output_block_num <= 4:
                        if not process_output_1:
                            continue  # Skip processing if strength is 0
                        calibration_strength = output_1_strength
                    else:
                        if not process_output_2:
                            continue  # Skip processing if strength is 0
                        calibration_strength = output_2_strength
                    block_type = 'output_blocks'
                    block_idx = clean_name.split('.')[1]
                elif clean_name.startswith('out.'):
                    if not process_output_2:  # Out block grouped with output_2
                        continue  # Skip processing if strength is 0
                    block_type = 'out'
                    block_idx = 'main'
                    calibration_strength = output_2_strength
                
                # Apply calibration if we have both reference and merged magnitudes
                if (block_type and block_idx and calibration_strength > 0.0 and
                    block_type in ref_magnitudes and block_idx in ref_magnitudes[block_type] and
                    block_type in merged_magnitudes and block_idx in merged_magnitudes[block_type]):
                    
                    ref_mag = ref_magnitudes[block_type][block_idx]
                    merged_mag = merged_magnitudes[block_type][block_idx]
                    
                    # Calculate scaling factor
                    if merged_mag > 1e-8:  # Avoid division by zero
                        raw_scale_factor = ref_mag / merged_mag
                        # Apply calibration strength (1.0 = full calibration, 0.0 = no change)
                        scale_factor = 1.0 + (raw_scale_factor - 1.0) * calibration_strength
                        param.data.mul_(scale_factor)
                        calibration_applied += 1
                        
                        # Record calibration details for first parameter in each block
                        block_key = f"{block_type}_{block_idx}"
                        if not any(detail['block'] == block_key for detail in calibration_details):
                            calibration_details.append({
                                'block': block_key,
                                'reference_mag': ref_mag,
                                'merged_mag': merged_mag,
                                'scale_factor': scale_factor,
                                'group_strength': calibration_strength
                            })
        
        return calibration_applied, calibration_details, skipped_groups

    def calibrate(self, merged_model, reference_model, input_blocks_strength, middle_blocks_strength, output_1_blocks_strength, output_2_blocks_strength, preset="Manual"):
        # Auto preset forces full strength on all block groups; Manual honors the sliders
        if preset == "Auto (full strength)":
            input_blocks_strength = 1.0
            middle_blocks_strength = 1.0
            output_1_blocks_strength = 1.0
            output_2_blocks_strength = 1.0

        # Clone the merged model to avoid modifying the original
        calibrated_model = copy.deepcopy(merged_model)
        
        print(f"[BLOCK CALIBRATION] Starting calibration:")
        print(f"  Input Blocks: {input_blocks_strength:.2f}")
        print(f"  Middle Blocks: {middle_blocks_strength:.2f}")  
        print(f"  Output 1 Blocks (0-4): {output_1_blocks_strength:.2f}")
        print(f"  Output 2 Blocks (5-8): {output_2_blocks_strength:.2f}")
        
        try:
            calibration_count, calibration_details, skipped_groups = self.apply_block_calibration(
                calibrated_model, reference_model, input_blocks_strength, middle_blocks_strength, output_1_blocks_strength, output_2_blocks_strength
            )
            
            print(f"[BLOCK CALIBRATION] Successfully calibrated {calibration_count} parameters")
            if skipped_groups:
                print(f"[BLOCK CALIBRATION] Skipped groups: {', '.join(skipped_groups)}")
            
            # Create detailed calibration report
            calibration_info = f"""╔═ BLOCK CALIBRATION RESULTS ═╗
║ Input Blocks Strength: {input_blocks_strength:.2f}
║ Middle Blocks Strength: {middle_blocks_strength:.2f}
║ Output 1 Blocks Strength: {output_1_blocks_strength:.2f}
║ Output 2 Blocks Strength: {output_2_blocks_strength:.2f}
║ Parameters Calibrated: {calibration_count}
║ Status: ✓ Successfully applied block magnitude matching
╠═══════════════════════════════════════════════════╣"""
            
            if skipped_groups:
                calibration_info += f"\n║ SKIPPED GROUPS (strength = 0):"
                for group in skipped_groups:
                    calibration_info += f"\n║ • {group}"
                calibration_info += "\n╠═══════════════════════════════════════════════════╣"
            
            calibration_info += "\n║ BLOCK MAGNITUDE ADJUSTMENTS:"
            
            for detail in calibration_details:
                block_name = detail['block'].replace('_', ' ').title()
                ref_mag = detail['reference_mag']
                merged_mag = detail['merged_mag']
                scale = detail['scale_factor']
                strength = detail['group_strength']
                calibration_info += f"\n║ {block_name}: {merged_mag:.4f} → {ref_mag:.4f} (×{scale:.3f}) [str:{strength:.2f}]"
            
            calibration_info += f"""
╠═══════════════════════════════════════════════════╣
║ Granular control: Each block group can be calibrated
║ independently. Set strength to 0 to skip processing
║ entire block groups for computational efficiency.
╚═══════════════════════════════════════════════════╝"""
            
            return (calibrated_model, calibration_info)
            
        except Exception as e:
            error_info = f"""╔═ BLOCK CALIBRATION ERROR ═╗
║ Error: {str(e)}
║ Status: ✗ Calibration failed
╚═══════════════════════════════════════════════════╝"""
            print(f"[BLOCK CALIBRATION] Error: {e}")
            return (calibrated_model, error_info)


class DonutSimpleCalibration(DonutBlockCalibration):
    """
    Thin alias of DonutBlockCalibration. Kept registered so already-saved
    workflows deserialize byte-identically. Only requires merged and reference
    models as inputs and always runs the engine with the Auto (full strength) preset.
    """
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_model": ("MODEL",),      # Model after merge
                "reference_model": ("MODEL",),   # Model to copy block magnitudes from
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "calibration_info")
    FUNCTION = "calibrate"
    CATEGORY = "donut/calibration"

    def calibrate(self, merged_model, reference_model):
        # Delegate to the shared engine with full calibration (Auto preset)
        return super().calibrate(
            merged_model=merged_model,
            reference_model=reference_model,
            input_blocks_strength=1.0,
            middle_blocks_strength=1.0,
            output_1_blocks_strength=1.0,
            output_2_blocks_strength=1.0,
            preset="Auto (full strength)",
        )


NODE_CLASS_MAPPINGS = {
    "Donut Block Calibration": DonutBlockCalibration,
    "Donut Simple Calibration": DonutSimpleCalibration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Donut Block Calibration": "Donut Block Calibration",
    # "Donut Simple Calibration" intentionally omitted from the menu (deprecated/merged
    # into Donut Block Calibration via the "preset" widget). The class stays registered
    # in NODE_CLASS_MAPPINGS above so saved workflows still deserialize.
}
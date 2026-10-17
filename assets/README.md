# Bundled numerical assets

`uncensorfix.f32` contains 3,457,232 bytes of raw little-endian float32 factors.
Target names, tensor shapes, order, and alpha values are defined in
`uncensorfix_weights.py`. For each target, the up tensor precedes the down tensor.

SHA-256: `f3c817bd957e6d47883346237b5e067697f0b9e1c9909bd06353da455949aacf`

These bytes are identical to the decoded payload previously embedded in Python
source. The file contains data only: no executable code or original checkpoint
header. The loader verifies size and hash before interpreting it as tensors.
It ships with the node package and requires no download or user configuration.

`refusal_reduction_v2.bf16` contains 30,344,832 bytes of raw little-endian bfloat16
factors for the Krea2 TextFusion Refusal-Reduction LoRA (Capitan01R, Civitai model
2775340). It holds the 35 v2 targets in the order listed in
`refusal_reduction_weights.py`; for each target the up tensor (`lora_B`) precedes the
down tensor (`lora_A`), rank 64, alpha 64. v1 is its first 32 block targets, bit-identical
to the original v1 file, so one asset serves both presets.

SHA-256: `8d2798e11b79f90c28744e6615a40ff1d8d2ee30fa86f47e2dfddb71467480ba`

Source files (Civitai SHA-256 prefixes): `refusal_reduction_v2.safetensors` `2b5ce7ce…`,
`Krea2_TextFusion_Refusal_Reduction.safetensors` (v1) `84ec722d…`. The file contains data
only, with no safetensors header and no executable code. The loader verifies size and
hash before interpreting it as tensors. It ships with the node package and requires no
download or user configuration.

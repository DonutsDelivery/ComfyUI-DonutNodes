# Bundled numerical assets

`uncensorfix.f32` contains 3,457,232 bytes of raw little-endian float32 factors.
Target names, tensor shapes, order, and alpha values are defined in
`uncensorfix_weights.py`. For each target, the up tensor precedes the down tensor.

SHA-256: `f3c817bd957e6d47883346237b5e067697f0b9e1c9909bd06353da455949aacf`

These bytes are identical to the decoded payload previously embedded in Python
source. The file contains data only: no executable code or original checkpoint
header. The loader verifies size and hash before interpreting it as tensors.
It ships with the node package and requires no download or user configuration.

# TeacherFix bundled weights

The distribution must include `krea2_c33_teacherfix_ema5000.safetensors` in this
folder as a normal binary file, not a Git LFS pointer or a download placeholder.
The original file was supplied for this preset by the repository maintainer.
It is preserved byte-for-byte, including its original metadata.

- File size: 3,470,548 bytes (3.31 MiB).
- SHA-256: `db3c2b7612828120e7ef9cc8fe77124c6fd8de2e38f150599e62abd9695f6beb`.
- Contents: 33 text-fusion targets; 99 tensors; rank 4 and alpha 4.

The node reads only this package-relative file and validates its size, checksum,
factor pairs, scope and finite values before applying it. No `models/loras`
search, external file selection, automatic download or separate LoRA node is
involved. Other presets and strength zero never read the asset.

Do not re-export or change its dtype: the preset intentionally identifies the
original bytes. No separate asset license was included in its metadata; this
README records provenance and does not assign a new license to the weights.

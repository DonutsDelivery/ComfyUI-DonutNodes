# Edit Studio outpainting — 2026-09-19

Local implementation; not published. Restart ComfyUI and refresh its frontend to
load both halves before generating an outpaint.

The existing version-1 mask document now optionally contains `outpaint` placement
(scale, normalized x/y anchor, overlap in output pixels). No node inputs, widget
orders, links, or output dimensions were added or changed. Both the original
Edit Studio and the independent-crop adapter construct A on the existing output
canvas before encoding. The full source A is fitted into the placement; the
normal source crop is inactive while outpainting is enabled. B remains separate.

Pixels outside placed A are always selected, including after erase/invert and
feathering. This prevents an extended base edge from being composited back into
newly generated space. Inside A, the automatic overlap and user strokes are
feathered inward; unselected pixels remain exact relative to the resized, placed
base. Enabled finishing upscale stages still change output resolution normally.

Validation:
- 75 Python tests passed across inpainting, Edit Studio, and independent crops.
- 29 JavaScript tests passed across mask editor and reference geometry.
- Browser verified the actual editor module using a local synthetic image:
  enabling outpaint displays 1024 × 1024 / 1.00 MP, alignment updates placement,
  drag updates the horizontal anchor, mode switching retains placement, and
  applying/reopening preserves the mask document and placement controls.
- Browser verified the directional canvas presets with a 512 × 768 source and
  a 1 MP starting budget. Side-by-side selected 1184 × 864 (0.98 MP); stacked
  selected 576 × 1760 (0.97 MP). The preview placed A on the requested side,
  selected the new half, and returned the new dimensions on apply.
- Syntax and whitespace checks passed.
- Updated the existing pipeline assertion to account for the already-wired
  SeedVR2 stage between Face Detailer and the final preserving composite.
- No GPU generation or human assessment of seam/content quality was performed.

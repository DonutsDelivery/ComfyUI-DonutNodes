# Third-Party Notices

`Donut Krea2 Fusion Control` includes explicitly named compatibility presets derived from the following projects and community artifacts. The presets are labelled `COPY settings:` in the node UI. Selecting one writes its values into the node's visible method/strength/profile settings; the preset label is not a hidden runtime override. No external files are loaded at runtime.

## ComfyUI-Krea2T-Enhancer

Source: https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer

The `COPY settings: capitan01R Krea2T-Enhancer defaults` preset selects the visible enhancer fusion method at strength 1. The implementation adapts the upstream two-pass text-fusion algorithm and preserves its default constants and tensor operation order for compatibility.

MIT License

Copyright (c) 2026 capitan01R

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## ComfyUI-ConditioningKrea2Rebalance

Source: https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance

The `COPY settings: nova452 ConditioningKrea2Rebalance profile @ tap strength 1` preset selects the visible Rebalance tap method and classic profile with a reduced multiplier of 1. The implementation reproduces the upstream dtype conversion and operation order while intentionally changing the upstream default multiplier. The upstream project is licensed under Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Donut's compatibility implementation is modified to support four independently routed conditioning inputs and visible method/profile/strength controls.

## Krea 2 projector-bypass community files

The following presets embed the exact single `[1,12]` `projector.diff` tensor values from the user-supplied community safetensors files:

- `COPY settings: Krea2FilterBypass 2vector`
- `COPY settings: Krea2FilterBypass 3vector`

They reproduce standard ComfyUI `.diff` patch semantics at `projector_strength=1.0` without reading either external file at runtime.

## Hybrid and Donut presets

Presets labelled `HYBRID settings:` combine the separately attributed operations above at a reduced Rebalance multiplier of 1; they do not claim to originate from either upstream project. Presets labelled `DONUT settings:` are Donut-authored RMS-preserved tap configurations, optionally combined with the attributed Enhancer operation.

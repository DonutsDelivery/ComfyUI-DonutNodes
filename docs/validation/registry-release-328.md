# DonutNodes 3.0.28

Applies Fusion Rebalance/taps to NAG's negative stream so it matches the
positive. That mismatch was leftover grain with Rebalance+NAG. DonutSampler
and the standalone krea2-nag node both receive the same marked transform.
Balanced + raw NAG keeps RMS-balanced positive with an unmatched NAG
negative. V5 wires `negative_raw` into sampler/upscale/detailer `negative`
so Turbo-off CFG can use a live negative; Turbo still zeros at sample time.

Focused tests: NAG fusion taps, NAG integration, grounding/NAG, Fusion
preset, inpaint/workflow wiring. Syntax compilation and diff checks pass.
No full GPU generation was packaged into this release; live Rebalance+NAG
A/B was run before publish.

Release commit `4251e25` pushed to main. Registry upload succeeded.

Exact version checked 2026-09-20 19:21 UTC: `NodeVersionStatusPending`.
Download `https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.28/node.zip`.
Staging ZIP SHA-256:
`7305e29a20e2c9edd1549468cbcbc1c8354995af7ea79ec83db9b9ee455b1756`.

Reload `workflows/v5/DonutWF_v5.json` in ComfyUI. If this is the Civitai
copy, upload that JSON by hand.

# 3.0.9 release validation

## Scope

Makes NAG's separate negative-conditioning stream use the active Krea2 Fusion
tap transform. This keeps RMS-balanced and Rebalance-based Fusion presets
consistent between the sampler's positive conditioning and NAG's negative text
stream. The transform is marked in conditioning metadata so it is never applied
twice.

## Validation

Focused tensor tests cover RMS-balanced, Rebalance, neutral pass-through, and
double-application protection. The Fusion/Workflow Panel UI suites also pass.

Published successfully at 2026-09-09T22:10:24Z using Comfy CLI 1.15.0 from a
fresh Registry-specific staging directory.

Exact-version check at 2026-09-09T22:10:50Z: NodeVersionStatusPending with no
status reason. The published 159-entry ZIP has SHA-256
`6d797911392e15b10a8ffc3f7a0911faea1fcba06d9a9e265c3cf61f9ea06eca`, matching
the validated staging archive byte-for-byte. Upload is verified; Registry
approval remains unverified.

No recurring follow-up is scheduled: this release changes Krea2 conditioning
math only and does not add scan-sensitive operations or package content.

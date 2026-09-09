> Superseded diagnosis: see [actual registry scan findings](registry-scan-findings.md). The API exposes reasons with `include_status_reason=true`.

# Registry release audit — 2026-09-09

Public source: https://api.comfy.org/nodes/donutnodes/versions

## Scope and limits

The public API returned 95 releases: 91 Active, 2 Flagged (2.0.4 and 3.0.0), and 2 Pending (3.0.1 and 3.0.2). This is current status, not a history of moderation transitions. Active releases could have been flagged previously; the response cannot establish that. Missing version numbers are not evidence of flags. No scan reason or moderation report is included. Therefore no historical flagged-to-approved repair can be confirmed from this data.

## Published archive findings

| Release | Current status | Civitai downloader | Supporting-model downloader | Dependency subprocess probes |
|---|---|---|---|---|
| 2.0.3 | Active | Present | Absent | Absent |
| 2.0.4 | Flagged | Present | Absent | Present |
| 3.0.0 | Flagged | Present | Present | Present |
| 3.0.1 | Pending | Present | Removed | Present |
| 3.0.2 | Pending | Present | Absent | Removed |

The Civitai downloader, server routes, and browser UI are byte-identical between 2.0.3 and 2.0.4. That rules out a change to these files as the difference between those archives; it does not establish that unchanged code cannot be flagged under a different scanner or review.

2.0.4 adds `donut_dependencies.py`, including `subprocess.run` at line 176. It runs fixed diagnostic Python snippets with `shell=False` and a timeout. Other changes include LoRA extraction, model merging, fusion presets, workflow repair, save-path handling and a substantial FaceDetailer rewrite. The subprocess addition is a plausible scanner trigger, not a confirmed cause.

3.0.0 → 3.0.1 removes `donut_model_downloads.py` and `model_sources.json`, changes initialization and the downloader panel to manual installation, and updates packaging/version metadata. The subprocess probes remain unchanged.

3.0.1 → 3.0.2 removes the probes and their UI input/documentation, while retaining metadata-only diagnostics. It also fixes HTML widget width during title-bar selection/dragging and updates the shared layout imports.

A textual scan for Python process launch and dynamic evaluation patterns across runtime Python/JavaScript files found the new subprocess call in 2.0.4, 3.0.0 and 3.0.1, absent in 2.0.3 and 3.0.2. The `model.eval()` matches are normal PyTorch evaluation-mode calls, not Python code evaluation. This focused scan is not a comprehensive security audit.

## Conclusion

There is no confirmed successful remediation after the currently flagged releases yet. 3.0.2 eliminates the identified subprocess addition. Its eventual approval would support, but not prove, the hypothesis because multiple files changed and registry policies/review timing may differ. A definitive cause requires the registry scan report or reviewer explanation.

## Archive SHA-256

- 2.0.3: `568850d49a06a574aa0d6cac8817a4eb508ee65518a9901d653618075bdb5de4`
- 2.0.4: `2d7301be20b608c6cab7db789f77cc2063ff97ca8a838d52124ae46b07c8c154`
- 3.0.0: `de7246b5992ce353a92dff7e4a316fe94ff3a429bb5f2dc9009497d722f14ed4`
- 3.0.1: `34b85eb09b2dd9e1a3754d3dc6be6ffd120d59dda45042f625700ae6f93d05ac`
- 3.0.2: `ebb904afcc6fd5677e1aed1813dbe6d65e0e42b1018c15345aaddf16942f24a8`

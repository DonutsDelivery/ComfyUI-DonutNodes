# 3.0.11 release validation

## Scope

The V4 Workflow Panel's Face detail control now falls back to the stable
`DonutFaceDetailer` node type if a ComfyUI subgraph reload remaps the internal
node ID. If the workflow genuinely lacks that node, the panel says how to
recover instead of only reporting an unavailable control.

## Validation

The V4 workflow's direct Face Detailer target and semantic fallback were
validated. The workflow reload suite, the focused bypass-projector suite, and
the Fusion-preset suite passed (31 tests total).

The release was packed and published directly with Comfy CLI 1.15.0 from a
fresh Registry-specific staging directory. The packaged archive excluded tests,
development baselines, and the automatic downloader backend; the staging step
installed the registry manual-download panel. The published ZIP downloaded and
passed an archive integrity check. It contains the version `3.0.11` workflow
fallback and has SHA-256
`8015ffaea69d5b065946063f6f88739e80580f0e6240e60deb4c900866f7ff2d`.

At 2026-09-10T15:51:37Z, the exact-version Registry check returned
`NodeVersionStatusPending` with no status reason. Upload is verified; Registry
approval is unverified. Because this release changes package contents, a
five-minute follow-up is scheduled until the exact version becomes Active or
Flagged.

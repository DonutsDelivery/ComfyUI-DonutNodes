# Release verification

Before preparing a release, read [the publishing guide](docs/publishing.md) for
known scan triggers, package validation, and distribution requirements.

After every Comfy Registry publication, verify that the exact published version
passes registry review. A successful upload or working ZIP download does not
mean the release is approved or available through normal installation/update
discovery.

- Immediately check the exact version at
  `https://api.comfy.org/nodes/donutnodes/versions?include_status_reason=true` and verify its published ZIP.
- If its status is Pending, ask the user a simple yes-or-no question about
  whether they want recurring review checks. Do not create a recurring monitor
  unless the user answers yes. When requested, check hourly until the version
  becomes Active or Flagged. Stay quiet while it remains Pending; notify the
  user when it passes, is flagged, or checking fails and needs action.
- If the user declines recurring checks, record that approval is unverified in
  the release notes and do not schedule a monitor.
- If Flagged, promptly notify the user and investigate the registry report and
  relevant package changes. Do not guess the cause or silently abandon the
  release. A replacement release requires the same verification.
- Only call the release approved after confirming Active for that exact version.
  Report upload success separately from approval, and record the version,
  checked status, and check time in the release validation notes.

The purpose is to prevent a shipped fix from remaining unavailable unnoticed
for weeks because publication succeeded but registry review did not.

# User-facing panels and workflow verification

Treat panel layout, grouping, labels, dropdowns, presets, promoted inputs, and
workflow migrations as potential behavior changes. A control rendering correctly
does not prove that generation uses its displayed value.

- Trace affected controls through their target widgets, nested subgraph inputs,
  and backend consumers. Check for stale paths, remapped IDs, duplicate controls,
  and outer inputs overriding inner widget values. Displayed labels must describe
  the behavior actually queued, including legacy composition modes.
- Preserve saved user choices when rendering or reloading. Do not silently apply
  new defaults or reapply preset recipes over manual adjustments. Keep any needed
  legacy migration narrowly scoped and test both migrated and current workflows.
- Exercise relevant transitions, not just initial defaults: enabled to disabled,
  one preset to another, active presets to Off/None, and manual adjustments
  followed by save/reload. Verify that obsolete hidden settings cannot remain
  effective after a feature is disabled. Preserve independent controls according
  to their documented behavior.
- For changes affecting panel bindings, serialization, presets, or workflow
  structure, set distinctive valid values through the actual user-facing panels
  and queue using ComfyUI's Run button. Enable the affected stages so they execute.
  Direct API submission bypasses the panel path and is not end-to-end proof.
- Save a PNG with workflow metadata enabled. Compare the observed panel values
  at queue time against both its embedded workflow and execution prompt. Resolve
  linked inputs to their sources, including promoted inputs and seed domains.
  Metadata proves serialized inputs; inspect backend logic or runtime evidence
  separately when the concern is whether those inputs are actually honored.
- Reload the saved workflow and verify that the same choices survive. Confirm
  that the browser loaded the changed frontend code; restart the backend when
  needed to verify Python changes. Do not count an old running version as proof
  of a new fix.
- Add focused regression tests for discovered failures. Static checks and unit
  tests supplement the UI generation check; they do not replace it. For purely
  cosmetic changes with no binding or workflow changes, visual and interaction
  checks are sufficient.
- Use a separate audit workflow/session and preserve the user's working settings.
  Record the tested transitions, expected/actual values, output PNG paths, and
  verification limits in docs/validation. If generation is blocked, report that
  explicitly rather than claiming end-to-end success.
- When a fix changes the distributed workflow JSON, explicitly tell the user
  which updated JSON must be uploaded manually to Civitai.

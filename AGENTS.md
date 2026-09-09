# Release verification

Before preparing a release, read [the publishing guide](docs/publishing.md) for
known scan triggers, package validation, and distribution requirements.

After every Comfy Registry publication, verify that the exact published version
passes registry review. A successful upload or working ZIP download does not
mean the release is approved or available through normal installation/update
discovery.

- Immediately check the exact version at
  `https://api.comfy.org/nodes/donutnodes/versions?include_status_reason=true` and verify its published ZIP.
- If its status is Pending, arrange a recurring follow-up in the current task
  (every five minutes when automation tools are available). Keep checking until
  the version becomes Active or Flagged. Stay quiet while it remains Pending;
  notify the user when it passes, is flagged, or checking fails and needs action.
- If automated follow-up is unavailable, explicitly report that approval is
  still unverified and that follow-up has not been scheduled. Never imply that
  someone will check later without actually arranging it.
- If Flagged, promptly notify the user and investigate the registry report and
  relevant package changes. Do not guess the cause or silently abandon the
  release. A replacement release requires the same verification.
- Only call the release approved after confirming Active for that exact version.
  Report upload success separately from approval, and record the version,
  checked status, and check time in the release validation notes.

The purpose is to prevent a shipped fix from remaining unavailable unnoticed
for weeks because publication succeeded but registry review did not.

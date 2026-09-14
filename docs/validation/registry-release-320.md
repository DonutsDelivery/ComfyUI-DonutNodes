# DonutNodes 3.0.20 release validation

Fixes the crash class behind user reports of
`AttributeError: 'NoneType' object has no attribute 'lower'` in
`DonutEditStudio._load_edit_lora`: `folder_paths.get_full_path` returns
`None` for a LoRA that is not installed, and `comfy.utils.load_torch_file`
crashed on it. Edit Studio and the sibling LoRA stack call sites
(`DonutSafeApplyLoRAStack`, `donut_lora_nodes`, `krea2_edit_integration`)
now raise a `FileNotFoundError` naming the missing file. The regular edit
path no longer silently no-ops on a missing LoRA (it previously produced an
unedited image with no error), `VALIDATE_INPUTS` reports a missing
identity-edit LoRA at queue time, and Edit Studio recovers renamed or
relocated LoRAs through the shared basename resolver without
auto-downloading.

Verified separately that the reported `DonutInpaintComposite`/9th-output
mismatch was already fixed by 3.0.17: the 3.0.16 CDN archive lacks
`donut_inpaint.py` and the `DONUT_INPAINT` output; 3.0.17+ contain both.
Registry state at release time: 3.0.19 Active ("Passed automated checks"),
only 3.0.0–3.0.4 and 2.0.4 banned from the Sept 7–9 scan round.

Validation: 17/17 Edit Studio tests (including new missing-LoRA,
basename-recovery, and validation-time-message regressions), 34/34
stack+inpaint tests, 45/45 streamlining tests, `py_compile` and
`git diff --check` clean. The 24 pre-existing full-suite import errors
(`No module named 'comfy.sd'`) reproduce on the clean tree via
`git stash` baseline — environment lacks a ComfyUI runtime, not a
regression.

Git commit: `a7233fe` (`main`). Clean source archive SHA-256:
`e09965a23161997b76c6b53be93c7a7fa55b6647a27705a8975d764f0b7d478a`.

Upload succeeded via local CLI (comfy-cli 1.20.0; no publish workflow in
the repository). The exact published ZIP was downloaded and reports
version 3.0.20 with the new guards present (6 `_require_lora_path` /
`_resolve_lora` references in `DonutEditStudio.py`). Published ZIP
SHA-256: `60833a4e8c6225300d10e14cc25547436c2a4e467d72815abd88dc7c2e814ccc`.
ZIP: https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.20/node.zip
162 files; no tests, credentials, `config.yaml`, `.env`, or nested
`node.zip` inside the archive.

Exact version checked through
`/nodes/donutnodes/versions?include_status_reason=true`:
`NodeVersionStatusPending`, with an empty status reason. Approval is
unverified; no recurring monitor was created. Hourly follow-ups require
explicit user opt-in.

# Publishing DonutNodes

## Upload is not approval

A successful upload and a working ZIP download do not mean a version has passed
registry review. Follow the release verification rules in [AGENTS.md](../AGENTS.md).
Check the exact version, including its scan reasons:

https://api.comfy.org/nodes/donutnodes/versions?include_status_reason=true

The `status_reason` field may contain a JSON-encoded list of findings with file,
line, matched pattern, scanner, and severity. If a cached listing omits a new
release, check `/nodes/donutnodes/versions/<version>` as well. Record the status
and UTC check time. While Pending, schedule five-minute follow-ups until Active
or Flagged. Never describe Pending as approved or assume it means manual review.

## Scan triggers and false positives

The following were actually reported for DonutNodes 2.0.4–3.0.2. They are examples,
not a complete scanner specification or a guarantee about future decisions.

| Source pattern | Reported finding | Guidance |
| --- | --- | --- |
| Literal `importlib.import_module("cv2")` | Bytecode manipulation/import evasion | Use a normal function-local import for a fixed optional dependency. Preserve lazy loading and error handling. |
| `subprocess.run(...)` diagnostic probes | Command injection risk and environment manipulation | Avoid shipping optional process-launch diagnostics in the registry package. Fixed commands and `shell=False` did not prevent findings. |
| Reading `HF_TOKEN` and using `requests.get(...)` | Environment manipulation and network operations | Legitimate network features can be flagged. Document their behavior and seek review or explicitly omit them from the restricted distribution. |
| Base85 weight data embedded in Python strings | Minified code and privilege escalation | Keep numerical data in a documented non-executable asset; verify size and checksum. Random encoded strings can match command-name and semicolon rules. |
| A local graph helper's `.connect(...)` | Network operations | A match may be unrelated to networking. Inspect the actual code and exclude development-only authoring tools from runtime packages. |
| Tests reading environment variables or embedding CSS | Environment manipulation and minified code | Keep tests in GitHub source, but exclude development tests from the runtime ZIP. |

All thirteen findings on 3.0.2 were labelled `info`, yet the version was Flagged.
Do not treat low severity as approval. An earlier Active version containing a
feature also does not guarantee a later version will pass.

The [official registry standards](https://docs.comfy.org/registry/standards)
prohibit Python `eval`/`exec`, runtime package installation through subprocess,
and code obfuscation. The publishing CLI's local warnings are not equivalent to
the registry's server-side scan. Ordinary PyTorch `model.eval()` is not Python
dynamic code execution; review context instead of blindly removing matches.

Do not rename, obscure, or re-encode suspicious operations merely to evade a
pattern. Fix actual risks, make harmless behavior explicit, or request registry
review with the exact finding and an explanation. Never claim a guessed cause
when the report is available. Removing one trigger may leave several others.

## Package validation

- Inspect the actual ZIP, not just the working tree or `.comfyignore`.
- Exclude credentials, caches, tests, and development-only tools.
- Verify required assets are present. Git-based packaging may omit new untracked
  assets; explicitly declare necessary assets in packaging configuration.
- Verify weight bytes/checksums and run relevant behavioral tests when changing
  data representation, imports, or distribution contents.
- Download the published ZIP and verify it matches the validated package.
- Record distribution differences clearly. GitHub source may retain optional
  features omitted from the registry ZIP, but registration and UI must handle
  their absence. Do not promise feature parity without testing both builds.

See [actual scan findings](validation/registry-scan-findings.md) for the incident
that established these examples. Approval remains a registry decision.

## GitHub and registry distributions

GitHub source includes the optional automatic supporting-model downloader and
its full UI. Registry archives exclude `donut_model_downloads.py` and use the
manual model-files panel in `distribution/registry/donut_model_downloads.js`.
The initializer registers the optional backend only when its file is present.
Neither distribution needs the retired subprocess probes.

To prepare a registry release, pack with `comfy node pack`, then run:

```sh
python tools/prepare_registry.py node.zip /tmp/donut-registry-release
cd /tmp/donut-registry-release
comfy node publish
```

Use a new staging directory for each release. Never publish directly from the
full GitHub checkout: packing exclusions remove the backend, but staging also
selects the matching manual UI. Keep the same version number in both sources
and verify that the runtime asset and model catalog are included in the ZIP.

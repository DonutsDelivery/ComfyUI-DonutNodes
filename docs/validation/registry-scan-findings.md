# Actual registry scan findings

Source: https://api.comfy.org/nodes/donutnodes/versions?include_status_reason=true

This supersedes the hypotheses in registry-release-audit.md. The public API DOES supply scan reasons when include_status_reason=true is requested.


## 3.0.3 — NodeVersionStatusPending

No findings yet.

## 3.0.2 — NodeVersionStatusFlagged

- 1 × python_bytecode_manipulation: donut_dependencies.py (info)
- 1 × python_minified_code: uncensorfix_weights.py (info)
- 10 × python_privilege_escalation: uncensorfix_weights.py (info)
- 1 × python_network_operations: tools/streamline_workflow.py (info)

## 3.0.1 — NodeVersionStatusFlagged

- 1 × python_bytecode_manipulation: donut_dependencies.py (info)
- 1 × python_command_injection_risk: donut_dependencies.py (info)
- 1 × python_environment_manipulation: donut_dependencies.py (info)
- 1 × python_minified_code: uncensorfix_weights.py (info)
- 10 × python_privilege_escalation: uncensorfix_weights.py (info)
- 1 × python_network_operations: tools/streamline_workflow.py (info)

## 3.0.0 — NodeVersionStatusFlagged

- 1 × python_bytecode_manipulation: donut_dependencies.py (info)
- 1 × python_command_injection_risk: donut_dependencies.py (info)
- 1 × python_environment_manipulation: donut_dependencies.py (info)
- 1 × python_environment_manipulation: donut_model_downloads.py (info)
- 1 × python_network_operations: donut_model_downloads.py (info)
- 1 × python_network_operations: tools/streamline_workflow.py (info)
- 1 × python_minified_code: uncensorfix_weights.py (info)
- 10 × python_privilege_escalation: uncensorfix_weights.py (info)

## 2.0.4 — NodeVersionStatusFlagged

- 1 × python_bytecode_manipulation: donut_dependencies.py (info)
- 1 × python_command_injection_risk: donut_dependencies.py (info)
- 1 × python_environment_manipulation: donut_dependencies.py (info)
- 1 × python_environment_manipulation: tests/test_lora_cleanup_browser.py (info)
- 1 × python_environment_manipulation: tests/test_streamlining.py (info)
- 1 × python_minified_code: uncensorfix_weights.py (info)
- 1 × python_minified_code: tests/test_lora_cleanup_browser.py (info)
- 10 × python_privilege_escalation: uncensorfix_weights.py (info)
- 1 × python_network_operations: tools/streamline_workflow.py (info)

## 2.0.3 — NodeVersionStatusActive

Passed automated checks

## Source review and remediation direction

- OpenCV: the flagged fixed `importlib.import_module("cv2")` imports the optional image library; it does not manipulate bytecode. A normal function-local `import cv2` preserves lazy import and error isolation while making intent explicit.
- UncensorFix: eleven findings in 3.0.2 point to base85-encoded float32 weight data, not executable statements. The loader decompresses with a size bound, verifies SHA-256, and interprets bytes as float32 tensors. Store numerical data as a documented non-executable asset instead of opaque Python string literals, retaining exact bytes and checksum. Do not alter model behavior or encode strings differently merely to avoid matching.
- Workflow authoring tool: `Graph.connect` modifies local JSON graph edges; it does not open a socket. The development-only `tools/streamline_workflow.py` need not ship in the runtime package. Exclude authoring tools in packaging; retain readable source in the repository.
- The subprocess probes really did appear in earlier reports: removing them reduced findings by two. Removing the supporting-model downloader also removed two findings. Neither removed the remaining findings, so neither was a sufficient remediation.
- All thirteen 3.0.2 findings are marked info, but the version is Flagged. Do not interpret info severity as approval.
- 3.0.3 still contains the three affected sources, so its reload fix does not address those existing scan findings. It was Pending when checked.

No new release or runtime changes were made in this investigation. Final approval remains a registry decision; report these findings if source/asset cleanup still receives a flag.

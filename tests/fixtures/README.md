# Workflow regression input

The eight real-workflow migration checks use the supplied `ComfyUI_00016_.json`.
Set `DONUT_WORKFLOW_SOURCE` to that file when running `test_streamlining.py`.
Alternatively, place a gzip-compressed copy at `tests/fixtures/ComfyUI_00016_.json.gz`.
Without the source, those eight checks are explicitly skipped; the 37 remaining
Python unit/contract tests still run. The full user workflow and the migrated
workflow are delivered separately rather than committed as test data.

Original upload SHA-256:
`d8be7eda6700ee4e9d35d36a53411c12b2a070363c178f2dd17748dc147fa9bc`

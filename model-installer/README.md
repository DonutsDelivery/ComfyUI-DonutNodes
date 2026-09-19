# DonutNodes model installer

Extract this folder into your ComfyUI directory (beside `main.py`), or into the
Windows portable directory. Keep the files together.

- Windows: double-click `install-models.bat`.
- Linux/macOS: open a terminal here and run `sh install-models.sh`.

Python 3.9 or newer is required. The Windows launcher checks for portable
ComfyUI's bundled Python first. Otherwise Python must be available on PATH.
The installer finds ComfyUI or asks for its directory. You can also pass
`--comfyui /path/to/ComfyUI`. Use `--list` to inspect downloads without installing.

This installs **all entries in the shared DonutNodes model catalog**, including
optional model alternatives and both SeedVR2 sizes. It shows the total size at
startup. Existing files with matching SHA-256 checksums are skipped; conflicting
files are preserved and reported. Completed downloads survive reruns; interrupted
partial files are removed and that model starts again on retry.

Files come directly from the catalog's upstream HTTPS links and go into
`ComfyUI/models/<folder>/<filename>`. Size and SHA-256 are verified before each
download becomes available. This installer uses the standard models directory;
it does not read custom `extra_model_paths.yaml` mappings.

For restricted upstream models, it asks for an API token only when access fails.
Input is hidden and is not saved. You can alternatively set `HF_TOKEN` or
`CIVITAI_API_KEY`. Account access or license acceptance must be handled with the
upstream provider. The installer does not request an administrator password;
choose a ComfyUI directory you can write to.

Restart ComfyUI afterward. Model files do not install or update custom nodes or
ComfyUI itself; use Manager for those dependencies.

This is a separate manual/Civitai auxiliary package. It is excluded from the
Comfy Registry release.

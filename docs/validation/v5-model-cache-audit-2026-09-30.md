# V5 model and adapter cache audit

Audited DonutNodes HEAD `2806ebe12caada993e6305e3f441ce61ab22a11c` on
2026-09-30. Pre-existing wheel-navigation changes were preserved. Scope was
narrowed to V5 after the user confirmed it does not use Widen Merge. No
production code was changed for this audit; no release or push was performed.

## Result

No general stale-adapter defect was established for ordinary V5 LoRA row
On/Off, strength changes, Krea2 merge changes, or SDA disabling. This is a
source/CPU-contract result, not proof of a clean run on the friend's RTX 5090.
The reported reset benefit was intermittent, and the user has not experienced
it. No before/after images or exact software versions from that machine were
available.

## Paths examined

- V5 uses `DonutLoRALoader` / `DonutApplyLoRAStackSafe`, grouped Krea2 merge,
  Fusion Control and its embedded UncensorFix, Edit Studio, prompt/variance,
  SDA/base sampling, the two hires stages, Face Detailer, and optional SeedVR2.
  Companion nodes in the Required node packs subgraph are installation markers;
  listing them in JSON does not prove they execute.
- Dynamic LoRA enable/strength edits are committed into `slots_json`. The loader
  builds the stack from that state each execution, applies it to the current
  upstream model, and omits disabled rows. Disabled/zero-weight rows do not
  recycle a previous output model. File checks supplement ordinary Comfy input
  invalidation; `IS_CHANGED` does not replace input-value comparisons.
- Regular patches are appended to model clones. Runtime bypass makes fresh
  adapter copies and binds them to the sampling root. Its ejection helper
  avoids restoring an earlier merge forward that another cleanup has removed.
  Rebinding the same plan retires an older owner on the same physical root.
- Grouped Single model mode returns the current primary input. Merge builds a
  new plan; exact source swaps have a unique attachment identity, and LoRAs
  routed to retained model2 receive source identities. Off/zero at a downstream
  node legitimately preserves adapters supplied by other upstream nodes.
- UncensorFix uses fresh factors/adapters. Fusion runtime projector/enhancer
  changes restore forwards in `finally`. Its modern composition inputs are
  authoritative; legacy execution-mode selectors can be overridden by the
  documented upstream mode. That is not evidence of cache leakage.
- SDA off/zero calls the base sampler with the current input. Enabled SDA
  clones the patcher. Scoped bypass owns/ejects each forward's adapters; native
  keyframes reset. Its cached verified CPU adapter data is keyed by path,
  device, inode, size, mtime and ctime. Retaining file data while disabled does
  not by itself apply it.
- Prompt expansion tracks text seed, resolved text and wildcard file stamps.
  CLIP's local encoding dictionary lives inside one `encode` call. Variance
  settings are normal inputs, and its optional upstream implementation copies
  metadata and constructs new noisy embeddings rather than mutating cached
  input embeddings.
- NAG prepares per-call state and model-local wrappers. Dynamic stage wrappers
  retain preparation for their owning model; their sigma-list cache does not
  select a previous unrelated model or negative prompt. Stage contexts reset
  in `finally`.
- SeedVR2 retains unmodified model/VAE resources with filename/size/mtime
  identity. A disabled stage returns the input image without sampling. A
  Donut-engine switch clears the stage's SeedVR2 resources.

## Confirmed narrower gap: changing files on disk

`DonutEditStudio.IS_CHANGED` fingerprints reference image contents and wildcard
files but omits the edit LoRA file. Replacing that file under the same name,
with otherwise identical node inputs, produces the same signature. Comfy can
therefore retain the old Edit Studio output. This is not a failed ordinary
LoRA-name/strength/enable transition: those change normal cache inputs.

`tests/test_v5_cache_invalidation_audit.py` contains an expected-failure
regression asserting that replacement should invalidate the signature. Its
expected failure records an UNFIXED issue. In the integrated sampler branch,
recorded edit metadata can cause a fresh LoRA read when the sampler executes;
the missing Edit Studio invalidation alone does not establish that every
subsequent V5 sample uses the old file.

SDA and SeedVR2 refresh their internal file caches when they execute, but do not
publish their file identities through `IS_CHANGED`. A fully unchanged cached
output can therefore bypass those refresh checks. Standard upstream loaders
also generally require re-execution/reloading after replacing a file under the
same name. No model-file replacement was reported in this incident.

Dynamic LoRA stamps use path/mtime/size. A replacement that deliberately retains
both timestamp and size is another limitation; no such replacement was
reproduced or attributed to this report.

## Validation

Isolated existing suites passed:

| Suite | Passed |
| --- | ---: |
| Krea2 merge | 13 |
| Merge/LoRA injection composition | 3 |
| Regular LoRA patch routing | 12 |
| SDA native scheduling/cache/cleanup | 22 |
| Model-level txtfusion guard | 40 |
| Total | 90 |

Additional extracted-source CPU checks passed four lifecycle contracts:
strength changes followed by Off restore the base forward; same-plan clone
rebinding plus late old-owner ejection leaves the new owner active; abandoned
owner finalization restores the base; runtime moves do not replace canonical
adapter weights. They used real tiny PyTorch Linear layers, Donut's actual
rebinding/ejection helpers and the actual upstream BypassInjectionManager /
BypassForwardHook, with explicit CPU device, adapter and patcher-owner doubles.
A fifth check demonstrated the unchanged Edit Studio file signature.

Reproducer: `docs/validation/v5-cache-audit-transitions.py`. It currently uses
the audit checkout and upstream snapshot paths under `/tmp`; those sources
are not bundled. The separate expected-failure regression uses only the repo.

The existing UncensorFix merge suite could not complete: it ran 22 tests with
17 dependency errors (`comfy.sd` unavailable). An initial combined invocation
also hit the tests package import error. These are not counted as passed and
were not classified as production bugs. See `/tmp/donut-cache-audit-tests/`.
No full ComfyUI backend, GPU checkpoint generation, UI Run, or metadata PNG
comparison was performed. Production panel bindings/serialization were not
changed by this audit.

## External source snapshots

- ComfyUI `051ddedaae52d4042abcde4e110446ea835211cd`, especially
  `comfy/model_patcher.py`, `comfy/model_management.py`,
  `comfy/weight_adapter/bypass.py`, and execution caching. Current core detaches
  older same-root patchers (ejecting their injections) before loading the new
  patcher. Its `clone_has_same_weights` helper compares injection keys rather
  than contents, but this snapshot's model loader does not call that helper;
  this observation was NOT promoted into a confirmed stale-weights defect.
- KreaSeedVarianceEnhancer `1515d23a5b399a44ccd97482e1105a43937268ae`.
- ComfyUI-Krea2-NAG `0afb38dfc4ae4040d621ac5a46e1761b83fa2a43`.

These are source snapshots, not verified versions installed on the friend's
machine. The inspected variance dependency saves CPU/CUDA RNG states before
seeding its noise and restores them in `finally`; a persistent RNG change was
not found in that source.

## RTX 5090 interpretation

CUDA/PyTorch caches memory allocations and compiled kernels; those mechanisms
are not designed to select or retain facial identities. Different VRAM,
precision, quantized operations, attention backends, dynamic loading and
software versions can exercise different code paths and reveal a state bug.
No 5090-specific caching or kernel defect was established.

Sources: https://docs.pytorch.org/docs/main/notes/cuda.html and
https://docs.nvidia.com/cuda/archive/12.9.0/blackwell-compatibility-guide/index.html.
A useful next comparison would hold all seeds and resolved prompts fixed,
compare cold/warm executions, and distinguish model unloading alone from
unloading plus execution-cache reset. That comparison was not run here.

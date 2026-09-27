# FaceDetailer VRAM investigation — 2026-09-27

## Observed failure

The local ComfyUI log records a FaceDetailer failure at 00:22:11
(Europe/Copenhagen). A 408 × 560 crop was resized to 896 × 1216, with a
1024² pixel target. VAE encoding completed in 0.2 seconds. Sampling then failed
inside the NAG block's MLP, in Donut's `_LinearLoKrBypassAdapter.h` at
`out.transpose(-1, -2).flatten(-2)`. The stack went through a composite bypass
adapter. Error cleanup also ran out of memory while reinjecting adapter weights.

Execution history identifies `krea2RealVae_v10.safetensors`, NAG enabled, and
experimental bypass for the model merge and adapter path. The optional LoRA,
MLP and Q/K-normalization chunking settings were all false. This failure
occurred before face decoding or VAE damage correction; it did not use the
2× upscale VAE. The log reports 9,011 MiB peak tensor allocation and 9,248 MiB
peak reservation for the failed execution on the 12 GB RTX 4070.

A later, already-existing user run succeeded with an 832 × 1216 face canvas.
This is evidence of a tight memory margin, not proof of a precise safe size.
No prompts, images or raw execution history are included in this report.

## Changes

- Bound inference intermediates for linear LoKr to 1024 tokens per batch.
  The existing Kronecker-factor operations and alpha/rank/strength scaling
  are retained, including decomposed factors. The result is copied into a
  single output buffer.
- Apply the same bound to stacks containing linear LoKr and plain LoRA, so
  component summation also happens on these smaller batches. The previous
  optional path only covered stacks consisting entirely of plain LoRA.
- Make FaceDetailer apply the existing Krea2 MLP/Q/K memory helper to the
  actual model passed to each sampling cycle. Previously only the sampler and
  hires paths called it. Existing option values are respected.

Autograd uses the original full-sequence path. No face resolution, attention
geometry, model strength, NAG setting, VAE setting or workflow JSON is changed.
No compatible LoKr-specific operation was found in the installed ComfyUI
operations or Comfy Kitchen sources; the existing native tensor math is reused.

## Validation limits

Source inspection only. No implementation tests, parity checks, GPU benchmark
or generation were run. The existing successful run predates this patch and
does not validate it. The expected reduction in temporary allocations has not
been measured, and OOM avoidance is not yet confirmed.

The queue was empty before a backend restart through ComfyUI Manager. Execution
history was backed up locally outside the repository. The startup log at
`2026-09-26T22:28:21Z` and successful DonutNodes import at `22:28:27Z` confirm
the restart. The local queue endpoint responded at `22:28:49Z`, with no running
or pending jobs. No generation was submitted. These are startup observations,
not implementation or image-quality tests. See the
[3.0.40 release record](release-3.0.40.md) for publication status.

## Failure after the first patch

The user's run at 00:55:40 (Europe/Copenhagen) still failed. The traceback
includes the updated FaceDetailer line numbers, and LoKr execution coverage was
logged before the failure. Its 832 × 1216 crop encoded successfully. The first
OOM occurred in the NAG block's main MLP down projection, while Comfy Kitchen's
eager FP8 path dequantized the base weight. A second OOM occurred while error
cleanup reinjected adapter weights. MLP and normalization chunking were still
disabled in that execution. An already-running user job subsequently completed
at 00:59:29 with an 896 × 1216 crop. Size alone does not explain these failures.

Source review also found a reference cycle in `_trace_lokr_calls`: the adapter
stored a callback that strongly captured the adapter's bound `h` method. Once
a runtime adapter was ejected, this cycle could retain its GPU tensors until
cyclic garbage collection. ComfyUI's `cleanup_models_gc` only requests a full
collection when it finds a dead loaded model; a retired adapter cycle need
not trigger that condition.

The callback now holds `weakref.WeakMethod` and resolves it only for the active
call. This preserves the existing coverage reporting without retaining the
retired adapter. Its contribution to peak VRAM has not yet been measured.

The user declined verification generations and requested installation only.
No tests or generations were run by the agent. With the queue empty, execution
history was backed up locally and the backend restarted to load this follow-up.
The startup log records DonutNodes importing at `2026-09-26T23:02:57Z`; the
queue endpoint was ready at `23:02:59Z` with no running or pending work. Existing
MLP/normalization option values were preserved. Startup succeeded, but the
reported OOM is not yet confirmed resolved. Both patches are included in the
3.0.40 release preparation.

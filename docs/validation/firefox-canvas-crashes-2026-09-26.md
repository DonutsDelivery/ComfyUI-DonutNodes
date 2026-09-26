# Firefox canvas crash investigation — 2026-09-26

## Evidence from existing reports

Read the local Firefox crash store and pending minidumps. No reproduction or
generation was started. Times below are local Europe/Copenhagen (UTC+02:00).

| Time | Process / crashing thread | Failure |
| --- | --- | --- |
| 10:35:22 | Main / `CanvasRenderer` | `SIGSEGV / SEGV_ACCERR` |
| 10:39:24 | Main; crash-store stack only | Same failure and first scanned libxul offset |
| 10:42:48 | Main / `CanvasRenderer` | `SIGSEGV / SEGV_ACCERR` |
| 10:46:59 | Main / `CanvasRenderer` | `SIGSEGV / SEGV_ACCERR` |

The thread names were decoded from each available minidump's thread-name stream
and matched to its exception thread ID. Local report identifiers and raw crash
data are omitted from this public summary.

All four stacks start with an instruction address outside executable libxul
memory, followed by a scanned frame at `libxul.so + 0x3bb529d`. In the 10:35 and
10:42 dumps, the fault address maps to non-executable jemalloc memory. A scanned
frame is less reliable than an unwound frame; it does not identify a C++ function
or a specific JavaScript drawing call.

Firefox is Arch Linux 154.0, build `20260818182641`, running on GNOME Wayland with
NVIDIA driver `610.57.04`. The recorded libxul build ID matches the installed
binary: `9a090c3e4988a92ce6ef9a98466f13ee16a84b84`. `pacman -Qkk firefox`
reported 100 files and zero altered files. Matching debug information was not
available locally or from the queried Arch debuginfod endpoint, so no exact C++
function was resolved.

The 10:46:59 report has 28,237,438,976 bytes (26.3 GiB) of available physical
memory, `LinuxUnderMemoryPressure: 0`, and no `OOMAllocationSize`. The 10:39:24
report also has 12.6 GiB available. These crashes are not established as memory
exhaustion. No kernel OOM-kill, NVIDIA Xid, or hardware-error event was found for
the inspected interval. NVIDIA `invalid mmap context` messages occur at crash
times, but their timing alone does not establish driver causation.

## Diagnosis and targeted workaround

The confirmed failure site is Firefox's native canvas renderer, inside its main
process. The reports do not establish which workflow draw operation triggers it
or whether the underlying defect belongs to Firefox, the driver, or another
component. The existing VAE sampling OOM is a separate recorded exception in the
Python process.

The report records forced WebRender (`gfx.webrender.all: true`), with the GPU
process unused. The installed binary includes `gfx.canvas.accelerated`; no user
override for that preference was present in the inspected profile.

An initial diagnostic workaround was to set **`gfx.canvas.accelerated = false`**
in Firefox's `about:config`, then restart Firefox. The user declined this because
canvas performance matters. Mozilla's
[graphics platform implementation](https://github.com/mozilla-firefox/firefox/blob/main/gfx/thebes/gfxPlatform.cpp)
uses that preference for accelerated Canvas2D, and `UseRemoteCanvas()` follows
the accelerated Canvas2D state. The preference is also documented in Mozilla's
[default preferences](https://github.com/mozilla-firefox/firefox/blob/main/modules/libpref/init/StaticPrefList.yaml).
This changes browser canvas rendering; it does not change ComfyUI's Python/CUDA
generation or saved image dimensions. Browser drawing may be slower.

## Limits and state

Crash logs and relevant source were inspected. No Firefox preference, running
browser, ComfyUI process or workflow setting was changed. No crash report was
uploaded. The browser-preference workaround was not applied, and the exact
triggering draw operation remains open.

## Follow-up: reduce redundant Donut drawing

Source inspection found a recurring path in the existing frontend:

1. The visible independent-crop panel polls every 500 ms and calls the edit
   studio's `render()`.
2. `draw()` reset both reference canvases' width and height on every call,
   including when the image, crop, mask and dimensions were unchanged.
3. Rewritten text labels triggered the panel subtree mutation observer, which
   ran category synchronization and scheduled a graph layout pass.
4. Every layout pass requested a full graph canvas redraw even when no node
   size or position changed.

The local frontend now:

- Skips reference drawing when the source image, geometry, mask and display
  dimensions are unchanged. Hidden/unmounted reference canvases skip drawing.
- Resizes canvas buffers only when their dimensions change. Transform, alpha
  and stroke width are explicitly reset for actual redraws. A `contextrestored`
  event invalidates the preview state so restored buffers are repainted.
- Runs category adapters for added/removed elements, ignoring text-only
  replacement. Resize observers still handle actual panel size changes.
- Coalesces control input/change refreshes into one pending task, preserving
  the existing task boundary that protects checkbox commits.
- Requests a graph redraw from automatic layout only when a node moves or
  resizes, and avoids repeatedly assigning an identical cached layout size.

All layout imports use `?v=16` so the panels share the same module instance.
The canvas changes use the existing GPU-capable context; no software-rendering
hint or browser preference override was added. Polling remains available for
widget synchronization.

This removes identifiable redundant work. It is not proof of the native crash's
root cause or of crash avoidance. Source paths and diffs were reviewed; no tests,
browser reproduction, generation or performance benchmark were run. Reload the
ComfyUI tab to load the changed JavaScript. Runtime behavior and crash avoidance
remain unverified.

## Continued crashes after the redraw changes

The user reported continued crashes. Subsequent local inspection found dumps
at 11:13:35 and 11:17:53 (Europe/Copenhagen) with the same main-process
`CanvasRenderer` exception, `SIGSEGV / SEGV_ACCERR`, and first scanned
`libxul.so + 0x3bb529d` frame. Available physical memory was approximately
9.7 GiB and 10.6 GiB respectively, with no recorded memory-pressure condition
or OOM allocation annotation. These are scanned frames, not a resolved native
call stack. The dumps do not prove which frontend revision was loaded.

The inspected kernel interval contains NVIDIA `invalid mmap context` messages,
including at 11:17:53 and 11:22:13, but no matching OOM-kill, Xid or hardware-error
entry. Correlation does not establish the driver as the cause.

Installed Firefox is 154.0. The configured Arch repository advertises 155.0.1-1;
no browser or system package upgrade was performed. Mozilla's
[155.0.1 release notes](https://www.firefox.com/en-US/firefox/155.0.1/releasenotes/)
include a CSS blur/backdrop-filter hang fix, which is not an established match
for these native crashes.

Prepared a separate local application-menu entry, **ComfyUI in Firefox
(XWayland)**. It launches Firefox with `GDK_BACKEND=x11`,
`MOZ_ENABLE_WAYLAND=0`, and `--new-instance`, opening the local ComfyUI URL.
Existing Firefox windows must be closed normally first so the normal profile
is available. It leaves hardware-acceleration preferences and the ordinary
Firefox launcher unchanged. This provides an optional way to isolate the
native Wayland path; it is not a confirmed crash fix. No browser was launched
or terminated, and no reproduction test was run. This local launcher is not
part of the DonutNodes distribution.

## Focus on the V4 DOM-panel regression

The user reports that rapid crashes began with the V4 DOM-panel integration,
after previously stable use of the same windowing setup. That is a useful
regression boundary. A crash on `CanvasRenderer` identifies the failure site;
it does not establish Wayland as the triggering change. Further work focuses
on the panel implementation with the existing GPU/windowing setup retained.

Source inspection of the 3.0.38 frontend found additional problems:

- Every visible app-control panel polls refresh callbacks every 500 ms. Its
  unchanged textareas called `fitTextarea`, which set height to `auto`, read
  `scrollHeight` and set the height again on every poll. The layout helper also
  read DOM dimensions and wrote node properties from widget-size callbacks
  used by canvas rendering.
- Every prompt-variant rebuild added its new prompt-tool callbacks to the
  panel-wide refresher array. That array was cleared only by a whole-panel
  rebuild, so variant add/remove/reorder operations retained old callbacks and
  detached DOM. Those callbacks continued running on each poll.
- Standalone reference-guidance crop overlays still assigned both canvas
  dimensions and redrew on every 500 ms poll. The 3.0.38 cache covered Edit
  Studio's preview canvases, but did not cover this separate overlay path.
- Edit/crop status text and slider bounds/values were written repeatedly even
  when unchanged, creating avoidable DOM work.

The 3.0.39 follow-up changes cache textarea measurements by value, width and font
metrics; update panel-height caches from resize observation; retain only the
current prompt variants' refreshers; cache standalone crop overlays with
context-restoration invalidation; and skip unchanged text and control writes.
Layout imports use `?v=17` consistently, including the Registry manual panel.

This is a source-level repair of identified panel defects. No implementation
tests, browser reproduction, performance recording, generation or crash-free
run was performed, and no crash trigger has been isolated yet. These follow-up
changes are not part of the published 3.0.38 ZIP; see
[the 3.0.39 release record](release-3.0.39.md) for publication status. A hard
refresh of the locally served ComfyUI tab loads them without restarting the
backend.

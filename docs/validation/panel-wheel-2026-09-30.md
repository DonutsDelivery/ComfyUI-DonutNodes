# Wheel navigation over Donut panels

## Change

`fitModule` installs a capture-phase, non-passive wheel listener on each Donut
panel. In the currently displayed graph container it passes the original event
to `app.canvas.processMouseWheel`, preserving coordinates, deltas, delta mode,
modifiers and the existing canvas zoom/pan preference. Events consumed by the
canvas are stopped before descendant controls or bubbling forwarders see them.
Events already prevented by the frontend are not forwarded again.

App Mode, other graph instances, dialogs, and absent handlers are excluded.
The canvas itself can decline events when navigation is disabled or the pointer
is outside its viewport. Removal detaches the listener; re-addition restores it.
All shared-layout imports, including the registry downloader replacement, now
use v18 so they load the same updated module.

No widget bindings, serialization, presets, backend consumers, or workflow JSON
changed. No replacement workflow or manual Civitai JSON upload is required.
No release or Git push was performed.

## Regression checks

`node --test tests/panel_wheel.test.mjs tests/layout_width.test.mjs`

Result: **14 passed**. Coverage includes original event identity and properties,
one dispatch, descendant suppression, already-handled events, App Mode, foreign
and removed graphs, dialogs, missing/declining handlers, cleanup and re-addition,
width preservation and shared layout imports.

The existing layout-width test double did not deliver ResizeObserver callbacks,
so its height assertion initially failed (520 instead of 800). The test now
delivers the observer callback explicitly, matching the existing cached-height
implementation; no production sizing logic changed.

## Browser interaction check

Used T3 collaborative browser with the actual new helper and a deliberately
minimal canvas double. Reusable fixture: `tests/panel_wheel_browser.html` (serve
the repository over local HTTP and open that path). This is not a ComfyUI session.

- Synthetic cancelable WheelEvents over panel text, focused number input, range,
  textarea and select: both directions forwarded exactly once, prevented, and
  field values retained (10 events).
- Already-prevented event: no second canvas call.
- Navigation disabled: no canvas change and event left unconsumed.
- Actual browser click toggled the button Off to On.
- Actual browser text entry changed Steps from 8 to 12; a further wheel event
  retained 12 and moved the fixture scale to 1.1.
- Strength remained 0.37; prompt remained `Saved prompt`.

The preview scroll tool did not emit wheel events, so wheel checks used
`dispatchEvent(new WheelEvent(...))`. This does not prove trusted hardware-wheel
behavior, native select-popup handling, or integration with ComfyUI's renderer.
The fixture uses a canvas double, not the real LiteGraph zoom arithmetic.

Screenshot:
`/home/user/.t3-hermes/userdata/browser-artifacts/browser-screenshot-localhost-munahyi6-f317c815.png`

No ComfyUI backend/frontend was available here, so no live workflow Run, generated
PNG, or save/reload test was performed. This change only routes navigation events;
live ComfyUI wheel interaction remains unverified. No user workflow was modified.

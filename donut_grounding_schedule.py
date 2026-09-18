"""Step-scheduled Krea2 semantic grounding in one uninterrupted DonutSampler run.

The base sampler still owns Edit Studio preparation, appearance tokens, masks,
Turbo resolution and CFG. Only the prepared semantic conditioning is varied.
Every distinct grounding resolution is encoded before denoising, not interpolated
between differently shaped text embeddings or encoded inside a model forward.
"""
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from inspect import signature
import math
import operator
import re

try:
    from .donut_krea2_sda import DonutSampler as _BaseDonutSampler
except ImportError:
    from donut_krea2_sda import DonutSampler as _BaseDonutSampler

CURVES = ("constant", "linear", "ease_in", "ease_out", "ease_in_out")
SUPPORTED_SAMPLERS = ("euler", "er_sde", "dpmpp_2m")
_TAG = "donut_grounding_px"
_WRAPPER_KEY = "donut_grounding_schedule"
_REQUEST = ContextVar("donut_grounding_request", default=None)


def _pixel_value(value):
    if isinstance(value, bool):
        raise ValueError("Grounding px must be an integer between 0 and 4096.")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("Grounding px must be an integer between 0 and 4096.") from exc
    if not 0 <= value <= 4096:
        raise ValueError("Grounding px must be an integer between 0 and 4096.")
    return value


def grounding_values(start, end, count, curve):
    """Return exact endpoints with intermediate resolutions rounded to 64 px.

    A one-step run uses end (its only step is also its last step). A zero-step
    advanced range has no schedule. Decreasing and equal endpoints are valid.
    """
    start, end = _pixel_value(start), _pixel_value(end)
    count = operator.index(count)
    if count < 0:
        raise ValueError("Grounding step count cannot be negative.")
    if curve not in CURVES[1:]:
        raise ValueError(f"Unknown dynamic grounding curve: {curve!r}")
    if count == 0:
        return ()
    if count == 1:
        return (end,)
    values = []
    for index in range(count):
        t = index / (count - 1)
        if curve == "ease_in":
            t *= t
        elif curve == "ease_out":
            t = 1 - (1 - t) ** 2
        elif curve == "ease_in_out":
            t = 2 * t * t if t < 0.5 else 1 - 2 * (1 - t) ** 2
        px = math.floor((start + (end - start) * t) / 64 + 0.5) * 64
        values.append(min(max(start, end), max(min(start, end), px)))
    values[0], values[-1] = start, end
    return tuple(values)


@dataclass(frozen=True)
class _Request:
    start: int
    end: int
    curve: str
    clip: object
    image: object
    image_b: object
    prompt: str
    negative_prompt: str
    original_positive: object
    turbo: bool


def _tag(conditioning, px):
    if conditioning is None:
        return None
    return [[tensor, dict(metadata, **{_TAG: px})] for tensor, metadata in conditioning]


class _SelectGrounding:
    """Select fully preprocessed conditions at the same step as dynamic CFG.

    This is a model-local PREDICT_NOISE wrapper, not a global sampler patch.
    The supported single-evaluation solvers share the base guider's completed-
    step semantics. Original conditions are restored even when sampling raises.
    """
    def __init__(self, values):
        self.values = tuple(values)

    def __call__(self, executor, x, timestep, model_options=None, seed=None):
        guider = executor.class_obj
        index = getattr(guider, "_step_index", None)
        cfg_values = getattr(guider, "cfg_values", ())
        if (not isinstance(index, int) or not 0 <= index < len(self.values)
                or len(cfg_values) != len(self.values)):
            raise RuntimeError("Scheduled grounding requires DonutSampler's step-aware CFG guider.")
        px = self.values[index]
        original = guider.conds
        selected = {}
        for key, conditions in original.items():
            if conditions is None:
                selected[key] = None
                continue
            selected[key] = [c for c in conditions if _TAG not in c or c[_TAG] == px]
            if conditions and not selected[key]:
                raise RuntimeError(f"Scheduled grounding has no {key} conditioning for {px} px.")
        guider.conds = selected
        try:
            return executor(x, timestep, model_options, seed)
        finally:
            guider.conds = original


class _SamplingGuard:
    def __init__(self, count):
        self.count = count

    def __call__(self, executor, model_wrap, sigmas, extra_args, callback, noise,
                 latent_image=None, denoise_mask=None, disable_pbar=False):
        schedule = [float(value) for value in sigmas]
        if (len(schedule) != self.count + 1
                or any(not math.isfinite(value) or value < 0 for value in schedule)
                or any(a <= b for a, b in zip(schedule, schedule[1:]))):
            raise ValueError("Scheduled grounding received a different or invalid sigma schedule.")
        # Reuse the read-only solver/preset inspector, NOT SDA's LoRA or 8-step
        # gate. This preserves the live Bleh preset and all of its solver options.
        try:
            from .donut_sda_sampler import inspect_sda_sampler
        except ImportError:
            from donut_sda_sampler import inspect_sda_sampler
        try:
            inspect_sda_sampler(executor.class_obj, SUPPORTED_SAMPLERS)
        except ValueError as exc:
            raise ValueError(str(exc).replace("SDA", "Scheduled grounding")) from exc
        return executor(model_wrap, sigmas, extra_args, callback, noise,
                        latent_image, denoise_mask, disable_pbar)


def _prepare_conditions(request, model, positive, negative, values):
    import nodes
    import comfy.patcher_extension
    try:
        from .krea2_edit_integration import scale_image_to_megapixels
        from .krea2_variance_integration import reapply_edit_variance
        from .krea2_nag_integration import sampler_negative
    except ImportError:
        from krea2_edit_integration import scale_image_to_megapixels
        from krea2_variance_integration import reapply_edit_variance
        from krea2_nag_integration import sampler_negative

    wrappers = comfy.patcher_extension.WrappersMP
    if not all(hasattr(wrappers, name) for name in ("PREDICT_NOISE", "SAMPLER_SAMPLE")):
        raise RuntimeError("Update ComfyUI to use scheduled grounding wrappers.")
    encoder = nodes.NODE_CLASS_MAPPINGS["Krea2EditGroundedEncode"]()
    image = scale_image_to_megapixels(request.image)
    options = {}
    if request.image_b is not None:
        options["image_b"] = scale_image_to_megapixels(request.image_b)

    # The base Edit Mode path already encoded start and applied variance/Turbo
    # metadata. Reuse it, and encode each other resolution once per polarity.
    cache = {request.start: (positive, negative)}
    all_positive, all_negative = [], []
    for px in dict.fromkeys(values):
        if px not in cache:
            pos = encoder.encode(request.clip, request.prompt, image=image,
                                 grounding_px=px, **options)[0]
            neg = encoder.encode(request.clip, request.negative_prompt, image=image,
                                 grounding_px=px, **options)[0]
            pos = reapply_edit_variance(pos, request.original_positive)
            neg = sampler_negative(neg, request.turbo)
            cache[px] = pos, neg
        pos, neg = cache[px]
        all_positive.extend(_tag(pos, px))
        if neg is not None:
            all_negative.extend(_tag(neg, px))

    patched = model.clone()
    patched.add_wrapper_with_key(wrappers.PREDICT_NOISE, _WRAPPER_KEY, _SelectGrounding(values))
    patched.add_wrapper_with_key(wrappers.SAMPLER_SAMPLE, _WRAPPER_KEY, _SamplingGuard(len(values)))
    return patched, all_positive, all_negative if negative is not None else None


class DonutSampler(_BaseDonutSampler):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(super().INPUT_TYPES())
        optional = inputs.setdefault("optional", {})
        # Append only, after SDA/NAG/inpaint widgets, for saved V4 compatibility.
        optional["grounding_schedule"] = (list(CURVES), {
            "default": "constant",
            "tooltip": "Edit Mode only. Constant uses Edit Studio's existing grounding_px. "
                       "Dynamic curves use start/end over the executed steps in one run. "
                       "Supports Euler, ER-SDE and DPM++ 2M, including verified Bleh presets; "
                       "NAG and multi-model sampling are not yet supported.",
        })
        optional["grounding_start_px"] = ("INT", {
            "default": 512, "min": 0, "max": 4096, "step": 64,
            "tooltip": "First-step semantic grounding resolution cap. Zero means native/unlimited, "
                       "not disabled grounding; use positive endpoints for a changing schedule.",
        })
        optional["grounding_end_px"] = ("INT", {
            "default": 1088, "min": 0, "max": 4096, "step": 64,
            "tooltip": "Last-step semantic grounding resolution. Higher values provide more "
                       "reference information, not a guaranteed identity-strength multiplier. "
                       "Distinct resolutions add encoding time and conditioning memory.",
        })
        return inputs

    def sample(self, *args, grounding_schedule="constant", grounding_start_px=512,
               grounding_end_px=1088, **kwargs):
        parent = super().sample
        # Even a nested, unrelated run must not inherit another run's request.
        token = _REQUEST.set(None)
        try:
            if grounding_schedule == "constant":
                return parent(*args, **kwargs)
            bound = signature(parent).bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = bound.arguments
            if not inputs.get("edit_mode", False):
                return parent(*args, **kwargs)
            start, end = _pixel_value(grounding_start_px), _pixel_value(grounding_end_px)
            grounding_values(start, end, 2, grounding_schedule)  # validate curve
            inputs["grounding_px"] = start
            if start == end:
                return parent(*bound.args, **bound.kwargs)
            if start == 0 or end == 0:
                raise ValueError("Zero grounding px means native/unlimited resolution, not no grounding. "
                                 "Use positive start/end values for a changing schedule.")
            if inputs.get("mode", "simple") not in ("simple", "advanced"):
                raise ValueError("Scheduled grounding currently supports simple/advanced Edit Mode, not multi_model.")
            if inputs.get("nag_options", {}).get("nag_enabled", False):
                raise ValueError("Disable NAG or select constant grounding; scheduled NAG conditioning is not yet supported.")
            name = inputs["sampler_name"]
            if name not in SUPPORTED_SAMPLERS and not re.fullmatch(r"bleh_preset_[0-9]+", name):
                raise ValueError("Scheduled grounding supports Euler, ER-SDE and DPM++ 2M (including verified Bleh presets).")
            inpaint = inputs.get("edit_inpaint")
            image = inpaint["image"] if inpaint is not None else inputs.get("source_image")
            request = _Request(start, end, grounding_schedule, inputs.get("clip"), image,
                               inputs.get("source_image_b"), inputs.get("edit_prompt", ""),
                               inputs.get("edit_negative_prompt", ""), inputs["positive"],
                               inputs.get("turbo_mode", False))
            _REQUEST.set(request)
            return parent(*bound.args, **bound.kwargs)
        finally:
            _REQUEST.reset(token)

    def _scheduled_run(self, parent, args, kwargs, advanced=False):
        request = _REQUEST.get()
        if request is None:
            return parent(*args, **kwargs)
        bound = signature(parent).bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        count = inputs["steps"]
        if advanced:
            count = max(0, min(count, inputs["end_at_step"]) - inputs["start_at_step"])
        values = grounding_values(request.start, request.end, count, request.curve)
        if not values:
            return parent(*args, **kwargs)
        inputs["model"], inputs["positive"], inputs["negative"] = _prepare_conditions(
            request, inputs["model"], inputs["positive"], inputs["negative"], values,
        )
        latent, info = parent(*bound.args, **bound.kwargs)
        shown = [str(value) for value in values]
        if len(shown) > 16:
            shown = shown[:8] + ["..."] + shown[-4:]
        return latent, (f"Grounding {request.curve} ({len(values)} steps): "
                        f"{' -> '.join(shown)} px; {len(set(values))} resolutions\n{info}")

    def run_simple(self, *args, **kwargs):
        return self._scheduled_run(super().run_simple, args, kwargs)

    def run_advanced(self, *args, **kwargs):
        return self._scheduled_run(super().run_advanced, args, kwargs, advanced=True)


NODE_CLASS_MAPPINGS = {"DonutSampler": DonutSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutSampler": "DonutSampler"}

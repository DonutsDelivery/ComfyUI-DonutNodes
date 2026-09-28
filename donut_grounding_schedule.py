"""Step-scheduled Krea2 grounding and NAG alpha in one DonutSampler run.

The base sampler still owns Edit Studio preparation, appearance tokens, masks,
Turbo resolution and CFG. Prepared semantic conditioning and NAG wrappers are
selected together by executed-step index. No text embeddings of different
lengths are interpolated or encoded inside a model forward.
"""
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import math
import operator
import re

try:
    from .donut_krea2_sda import DonutSampler as _BaseDonutSampler
except ImportError:
    from donut_krea2_sda import DonutSampler as _BaseDonutSampler

try:
    from .donut_grounding_nag import (
        capture_nag_preparations, get_nag_preparation, select_nag_options,
    )
except ImportError:
    from donut_grounding_nag import (
        capture_nag_preparations, get_nag_preparation, select_nag_options,
    )

CURVES = ("constant", "linear", "ease_in", "ease_out", "ease_in_out")
SUPPORTED_SAMPLERS = ("euler", "er_sde", "dpmpp_2m")
_TAG = "donut_grounding_px"
_WRAPPER_KEY = "donut_grounding_schedule"
_REQUEST = ContextVar("donut_grounding_request", default=None)
_NAG_REQUEST = ContextVar("donut_nag_schedule_request", default=None)
_ACTIVE_STAGE_NAG = ContextVar("donut_stage_nag_schedule", default=None)
_STAGE_NAG_WRAPPER_KEY = "donut_stage_nag_alpha_schedule"
NAG_ALPHA_WIDGETS = ("nag_alpha_schedule", "nag_alpha_start", "nag_alpha_end")


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
        t = _curve_position(t, curve)
        px = math.floor((start + (end - start) * t) / 64 + 0.5) * 64
        values.append(min(max(start, end), max(min(start, end), px)))
    values[0], values[-1] = start, end
    return tuple(values)


def _curve_position(t, curve):
    if curve == "ease_in":
        return t * t
    if curve == "ease_out":
        return 1 - (1 - t) ** 2
    if curve == "ease_in_out":
        return 2 * t * t if t < 0.5 else 1 - 2 * (1 - t) ** 2
    return t


def nag_alpha_values(start, end, count, curve):
    """Return one finite NAG blend alpha per executed step."""
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError("NAG alpha schedule values must be between 0 and 1.")
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NAG alpha schedule values must be finite numbers between 0 and 1.") from exc
    if (not math.isfinite(start) or not math.isfinite(end)
            or not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0):
        raise ValueError("NAG alpha schedule values must be finite numbers between 0 and 1.")
    count = operator.index(count)
    if count < 0:
        raise ValueError("NAG alpha schedule step count cannot be negative.")
    if curve not in CURVES[1:]:
        raise ValueError(f"Unknown dynamic NAG alpha curve: {curve!r}")
    if count == 0:
        return ()
    if count == 1:
        return (end,)
    values = tuple(start + (end - start) * _curve_position(index / (count - 1), curve)
                   for index in range(count))
    return (start, *values[1:-1], end)


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


@dataclass(frozen=True)
class _NagRequest:
    curve: str
    start: float
    end: float
    auto_phi: bool
    manual_phi: float
    phi_scale: float


def nag_alpha_schedule_input_types():
    """Append-only settings shared by auxiliary NAG sampling stages."""
    return {
        "nag_alpha_schedule": (list(CURVES), {
            "default": "constant",
            "tooltip": "NAG alpha over this stage's executed denoising steps. The global Settings / Configuration panel mirrors the same curve to every NAG stage; each stage spans its own effective step range.",
        }),
        "nag_alpha_start": ("FLOAT", {
            "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "First executed step's NAG alpha for the shared dynamic schedule.",
        }),
        "nag_alpha_end": ("FLOAT", {
            "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Last executed step's NAG alpha for the shared dynamic schedule.",
        }),
    }


def nag_request_from_options(options):
    """Read the shared alpha schedule from one stage's NAG keyword inputs."""
    curve = options.get("nag_alpha_schedule", "constant")
    if not options.get("nag_enabled", False) or curve == "constant":
        return None
    start = options.get("nag_alpha_start", 0.25)
    end = options.get("nag_alpha_end", 0.25)
    nag_alpha_values(start, end, 2, curve)  # validate even before a sampler is built
    return _NagRequest(
        curve, float(start), float(end), bool(options.get("nag_auto_phi", False)),
        float(options.get("nag_phi", 4.0)), float(options.get("nag_phi_scale", 1.0)),
    )


def without_nag_schedule_options(options):
    """Drop schedule-only widgets before forwarding NAG options upstream."""
    return {key: value for key, value in options.items() if key not in NAG_ALPHA_WIDGETS}


def _tag(conditioning, px):
    if conditioning is None:
        return None
    return [[tensor, dict(metadata, **{_TAG: px})] for tensor, metadata in conditioning]


class _SelectGrounding:
    """Select grounding and NAG variants at the same step as dynamic CFG.

    This is a model-local PREDICT_NOISE wrapper, not a global sampler patch.
    The supported single-evaluation solvers share the base guider's completed-
    step semantics. Original conditions are restored even when sampling raises.
    """
    def __init__(self, values=None, nag_wrappers=None, nag_alphas=None):
        self.values = None if values is None else tuple(values)
        self.nag_wrappers = nag_wrappers
        self.nag_alphas = None if nag_alphas is None else tuple(nag_alphas)

    def __call__(self, executor, x, timestep, model_options=None, seed=None):
        guider = executor.class_obj
        index = getattr(guider, "_step_index", None)
        cfg_values = getattr(guider, "cfg_values", ())
        count = len(self.values) if self.values is not None else len(self.nag_alphas or ())
        if (not isinstance(index, int) or not 0 <= index < count
                or len(cfg_values) != count):
            raise RuntimeError("Scheduled grounding/NAG requires DonutSampler's step-aware CFG guider.")
        px = self.values[index] if self.values is not None else None
        alpha = self.nag_alphas[index] if self.nag_alphas is not None else None
        original = guider.conds
        selected = original
        if self.values is not None:
            selected = {}
            for key, conditions in original.items():
                if conditions is None:
                    selected[key] = None
                    continue
                selected[key] = [c for c in conditions if _TAG not in c or c[_TAG] == px]
                if conditions and not selected[key]:
                    raise RuntimeError(f"Scheduled grounding has no {key} conditioning for {px} px.")
        if self.nag_wrappers is not None:
            key = (px, alpha)
            if key not in self.nag_wrappers:
                raise RuntimeError(f"Scheduled NAG has no prepared wrapper for step {index + 1}.")
            model_options = select_nag_options(model_options, self.nag_wrappers[key])
        guider.conds = selected
        try:
            return executor(x, timestep, model_options, seed)
        finally:
            guider.conds = original


class _SamplingGuard:
    def __init__(self, count):
        self.count = count

    @staticmethod
    def validate_sigmas(sigmas, count=None):
        schedule = [float(value) for value in sigmas]
        if ((count is not None and len(schedule) != count + 1)
                or any(not math.isfinite(value) or value < 0 for value in schedule)
                or any(a <= b for a, b in zip(schedule, schedule[1:]))):
            raise ValueError("Scheduled grounding/NAG received a different or invalid sigma schedule.")
        return tuple(schedule)

    @staticmethod
    def validate(sampler, sigmas, count=None):
        schedule = _SamplingGuard.validate_sigmas(sigmas, count)
        # Reuse the read-only solver/preset inspector, NOT SDA's LoRA or 8-step
        # gate. This preserves the live Bleh preset and all of its solver options.
        try:
            from .donut_sda_sampler import inspect_sda_sampler
        except ImportError:
            from donut_sda_sampler import inspect_sda_sampler
        try:
            inspect_sda_sampler(sampler, SUPPORTED_SAMPLERS)
        except ValueError as exc:
            raise ValueError(str(exc).replace("SDA", "Scheduled grounding/NAG")) from exc
        return tuple(schedule)

    def __call__(self, executor, model_wrap, sigmas, extra_args, callback, noise,
                 latent_image=None, denoise_mask=None, disable_pbar=False):
        schedule = self.validate_sigmas(sigmas, self.count)
        self.validate(executor.class_obj, schedule, self.count)
        return executor(model_wrap, sigmas, extra_args, callback, noise,
                        latent_image, denoise_mask, disable_pbar)


def _sigma_values(sigmas):
    if hasattr(sigmas, "detach"):
        sigmas = sigmas.detach().to("cpu").reshape(-1).tolist()
    else:
        sigmas = list(sigmas)
    return tuple(float(value) for value in sigmas)


def _prediction_sigma_index(timestep, sigmas):
    """Map a PREDICT_NOISE sigma to its executed denoising step."""
    try:
        values = _sigma_values(timestep)
    except (TypeError, ValueError):
        values = (float(timestep),)
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError("Dynamic stage NAG received an invalid denoising sigma.")
    sigma = values[0]
    if any(not math.isclose(value, sigma, rel_tol=1e-6, abs_tol=1e-7) for value in values[1:]):
        raise RuntimeError("Dynamic stage NAG received different sigmas in one prediction batch.")
    candidates = sigmas[:-1]
    if not candidates:
        raise RuntimeError("Dynamic stage NAG received no executed sigma steps.")
    index = min(range(len(candidates)), key=lambda item: abs(candidates[item] - sigma))
    if not math.isclose(candidates[index], sigma, rel_tol=1e-5, abs_tol=1e-7):
        raise RuntimeError("Dynamic stage NAG could not match the prediction sigma to an executed step.")
    return index


class _SelectStageNAGAlpha:
    def __call__(self, executor, x, timestep, model_options=None, seed=None):
        active = _ACTIVE_STAGE_NAG.get()
        if active is None:
            raise RuntimeError("Dynamic stage NAG ran outside its guarded sampler invocation.")
        index = _prediction_sigma_index(timestep, active.sigmas)
        alpha = active.alphas[index]
        replacements = active.wrappers.get(alpha)
        if replacements is None:
            raise RuntimeError(f"Dynamic stage NAG has no prepared wrapper for step {index + 1}.")
        return executor(x, timestep, select_nag_options(model_options, replacements), seed)


@dataclass(frozen=True)
class _ActiveStageNAG:
    sigmas: tuple
    alphas: tuple
    wrappers: dict


class _ScheduleStageNAG:
    """Schedule NAG from the actual sigma list used by an auxiliary stage."""
    def __init__(self, preparation, request):
        self.preparation = preparation
        self.request = request
        self.wrapper_cache = {}

    def __call__(self, executor, model_wrap, sigmas, extra_args, callback, noise,
                 latent_image=None, denoise_mask=None, disable_pbar=False):
        schedule = _SamplingGuard.validate_sigmas(sigmas)
        _SamplingGuard.validate(executor.class_obj, schedule)
        count = len(schedule) - 1
        alphas = nag_alpha_values(self.request.start, self.request.end, count, self.request.curve)
        cache_key = schedule
        wrappers = self.wrapper_cache.get(cache_key)
        if wrappers is None:
            try:
                from .krea2_nag_integration import resolve_nag_phi
            except ImportError:
                from krea2_nag_integration import resolve_nag_phi
            wrappers = {}
            for alpha in dict.fromkeys(alphas):
                phi = (resolve_nag_phi(self.request.manual_phi, alpha, True, self.request.phi_scale)
                       if self.request.auto_phi else self.request.manual_phi)
                wrappers[alpha] = self.preparation.wrappers_for(alpha=alpha, phi=phi)
            self.wrapper_cache[cache_key] = wrappers
        token = _ACTIVE_STAGE_NAG.set(_ActiveStageNAG(schedule, alphas, wrappers))
        try:
            return executor(model_wrap, sigmas, extra_args, callback, noise,
                            latent_image, denoise_mask, disable_pbar)
        finally:
            _ACTIVE_STAGE_NAG.reset(token)


def install_stage_nag_schedule(model, request):
    """Attach per-step alpha selection to an already prepared stage NAG model."""
    if request is None:
        return model
    preparation = get_nag_preparation(model)
    if preparation is None:
        raise RuntimeError("Dynamic NAG alpha could not capture the installed NAG preparation; alpha was not silently left static.")
    import comfy.patcher_extension
    wrappers = comfy.patcher_extension.WrappersMP
    if not all(hasattr(wrappers, name) for name in ("PREDICT_NOISE", "SAMPLER_SAMPLE")):
        raise RuntimeError("Update ComfyUI to use scheduled stage NAG wrappers.")
    patched = model.clone()
    patched.add_wrapper_with_key(wrappers.PREDICT_NOISE, _STAGE_NAG_WRAPPER_KEY,
                                 _SelectStageNAGAlpha())
    patched.add_wrapper_with_key(wrappers.SAMPLER_SAMPLE, _STAGE_NAG_WRAPPER_KEY,
                                 _ScheduleStageNAG(preparation, request))
    return patched


def _prepare_conditions(request, nag_request, model, positive, negative, values, nag_alphas):
    import comfy.patcher_extension

    if request is not None:
        import nodes
        try:
            from .krea2_edit_integration import scale_image_to_megapixels
            from .krea2_variance_integration import reapply_edit_variance
            from .krea2_nag_integration import sampler_negative
            from .DonutKrea2FusionControl import prepare_positive_conditioning_taps
        except ImportError:
            from krea2_edit_integration import scale_image_to_megapixels
            from krea2_variance_integration import reapply_edit_variance
            from krea2_nag_integration import sampler_negative
            from DonutKrea2FusionControl import prepare_positive_conditioning_taps

    wrappers = comfy.patcher_extension.WrappersMP
    if not all(hasattr(wrappers, name) for name in ("PREDICT_NOISE", "SAMPLER_SAMPLE")):
        raise RuntimeError("Update ComfyUI to use scheduled grounding/NAG wrappers.")

    cache = {}
    raw_negatives = {}
    if request is not None:
        encoder = nodes.NODE_CLASS_MAPPINGS["Krea2EditGroundedEncode"]()
        image = scale_image_to_megapixels(request.image)
        options = {}
        if request.image_b is not None:
            options["image_b"] = scale_image_to_megapixels(request.image_b)

        # The base Edit Mode path already encoded start and applied variance/Turbo
        # metadata. Reuse it, and encode each other resolution once per polarity.
        cache[request.start] = (positive, negative)
        for px in dict.fromkeys(values):
            if px not in cache:
                pos = encoder.encode(request.clip, request.prompt, image=image,
                                     grounding_px=px, **options)[0]
                raw_negative = encoder.encode(request.clip, request.negative_prompt, image=image,
                                              grounding_px=px, **options)[0]
                pos = reapply_edit_variance(pos, request.original_positive)
                pos = prepare_positive_conditioning_taps(model, pos)
                raw_negatives[px] = raw_negative
                cache[px] = (pos, sampler_negative(raw_negative, request.turbo))

    nag = get_nag_preparation(model)
    vary_alpha = nag_alphas is not None
    if vary_alpha and nag is None:
        raise RuntimeError("Dynamic NAG alpha could not capture the installed NAG preparation; alpha was not silently left static.")
    vary_negative = (request is not None and nag is not None and not nag.explicit_negative
                     and (nag.changes_negative or vary_alpha))
    select_nag = nag is not None and (vary_alpha or vary_negative)
    nag_wrappers = {} if select_nag else None

    if select_nag:
        pairs = (tuple(dict.fromkeys(zip(values, nag_alphas or (None,) * len(values))))
                 if values is not None else tuple((None, alpha) for alpha in dict.fromkeys(nag_alphas)))
        try:
            from .krea2_nag_integration import resolve_nag_phi
        except ImportError:
            from krea2_nag_integration import resolve_nag_phi
        for px, alpha in pairs:
            raw_negative = raw_negatives.get(px) if vary_negative and px is not None else None
            if vary_alpha:
                phi = (resolve_nag_phi(nag_request.manual_phi, alpha, True, nag_request.phi_scale)
                       if nag_request.auto_phi else nag_request.manual_phi)
            else:
                alpha = None
                phi = None
            nag_wrappers[(px, alpha)] = nag.wrappers_for(raw_negative, alpha=alpha, phi=phi)

    all_positive, all_negative = [], []
    if request is not None:
        for px in dict.fromkeys(values):
            pos, neg = cache[px]
            all_positive.extend(_tag(pos, px))
            if neg is not None:
                all_negative.extend(_tag(neg, px))

    count = len(values) if values is not None else len(nag_alphas)
    patched = model.clone()
    patched.add_wrapper_with_key(
        wrappers.PREDICT_NOISE, _WRAPPER_KEY,
        _SelectGrounding(values, nag_wrappers, nag_alphas),
    )
    patched.add_wrapper_with_key(wrappers.SAMPLER_SAMPLE, _WRAPPER_KEY, _SamplingGuard(count))
    return patched, (all_positive if request is not None else positive), (
        all_negative if negative is not None else None
    ) if request is not None else negative


class DonutSampler(_BaseDonutSampler):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(super().INPUT_TYPES())
        optional = inputs.setdefault("optional", {})
        # Append only, after SDA/NAG/inpaint widgets, for saved V4 compatibility.
        optional["grounding_schedule"] = (list(CURVES), {
            "default": "constant",
            "tooltip": "Preferred wiring: connect Edit Studio's grounding_schedule "
                       "output so editing settings live in one place. Constant "
                       "uses Edit Studio's grounding_px directly. Dynamic curves "
                       "reach start/end over the executed steps in one run, "
                       "Euler/ER-SDE/DPM++ 2M including Bleh presets. Supports NAG, "
                       "reference guidance and inpainting; multi-model runs are not yet supported.",
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

    @capture_nag_preparations(enabled=False)
    def sample(
        self,
        model,
        seed,
        steps,
        cfg_start,
        cfg_halfway,
        cfg_end,
        halfway_step,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        mode='simple',
        cfg_curve='linear',
        add_noise='enable',
        start_at_step=0,
        end_at_step=10000,
        return_with_leftover_noise='disable',
        randomize_seed_per_model='enable',
        switch_at_step_1=10,
        switch_at_step_2=15,
        model_2=None,
        model_3=None,
        edit_mode=False,
        source_image=None,
        vae=None,
        clip=None,
        edit_prompt='',
        edit_negative_prompt='',
        grounding_px=768,
        edit_model=None,
        turbo_mode=False,
        source_image_b=None,
        edit_inpaint=None,
        sda_enabled=False,
        sda_strength=1.0,
        *,
        grounding_schedule="constant",
        grounding_start_px=512,
        grounding_end_px=1088,
        nag_alpha_schedule="constant",
        nag_alpha_start=0.25,
        nag_alpha_end=0.25,
        **nag_options,
    ):
        inputs = dict(
            model=model,
            seed=seed,
            steps=steps,
            cfg_start=cfg_start,
            cfg_halfway=cfg_halfway,
            cfg_end=cfg_end,
            halfway_step=halfway_step,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            mode=mode,
            cfg_curve=cfg_curve,
            add_noise=add_noise,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            return_with_leftover_noise=return_with_leftover_noise,
            randomize_seed_per_model=randomize_seed_per_model,
            switch_at_step_1=switch_at_step_1,
            switch_at_step_2=switch_at_step_2,
            model_2=model_2,
            model_3=model_3,
            edit_mode=edit_mode,
            source_image=source_image,
            vae=vae,
            clip=clip,
            edit_prompt=edit_prompt,
            edit_negative_prompt=edit_negative_prompt,
            grounding_px=grounding_px,
            edit_model=edit_model,
            turbo_mode=turbo_mode,
            source_image_b=source_image_b,
            edit_inpaint=edit_inpaint,
            sda_enabled=sda_enabled,
            sda_strength=sda_strength,
        )
        inputs.update(nag_options)
        parent = super().sample
        # Even a nested, unrelated run must not inherit another run's request.
        token = _REQUEST.set(None)
        nag_token = _NAG_REQUEST.set(None)
        try:
            grounding_request = None
            if grounding_schedule != "constant" and inputs.get("edit_mode", False):
                start, end = _pixel_value(grounding_start_px), _pixel_value(grounding_end_px)
                grounding_values(start, end, 2, grounding_schedule)  # validate curve
                inputs["grounding_px"] = start
                if start != end:
                    if start == 0 or end == 0:
                        raise ValueError("Zero grounding px means native/unlimited resolution, not no grounding. "
                                         "Use positive start/end values for a changing schedule.")
                    inpaint = inputs.get("edit_inpaint")
                    image = inpaint["image"] if inpaint is not None else inputs.get("source_image")
                    grounding_request = _Request(
                        start, end, grounding_schedule, inputs.get("clip"), image,
                        inputs.get("source_image_b"), inputs.get("edit_prompt", ""),
                        inputs.get("edit_negative_prompt", ""), inputs["positive"],
                        inputs.get("turbo_mode", False),
                    )
                    _REQUEST.set(grounding_request)

            nag_request = None
            if inputs.get("nag_enabled", False) and nag_alpha_schedule != "constant":
                nag_alpha_values(nag_alpha_start, nag_alpha_end, 2, nag_alpha_schedule)
                nag_request = _NagRequest(
                    nag_alpha_schedule, float(nag_alpha_start), float(nag_alpha_end),
                    bool(inputs.get("nag_auto_phi", False)),
                    float(inputs.get("nag_phi", 4.0)),
                    float(inputs.get("nag_phi_scale", 1.0)),
                )
                _NAG_REQUEST.set(nag_request)

            if grounding_request is None and nag_request is None:
                return parent(**inputs)
            if inputs.get("mode", "simple") not in ("simple", "advanced"):
                raise ValueError("Dynamic grounding/NAG schedules support simple or advanced sampling, not multi_model.")
            name = inputs["sampler_name"]
            if name not in SUPPORTED_SAMPLERS and not re.fullmatch(r"bleh_preset_[0-9]+", name):
                raise ValueError("Dynamic grounding/NAG schedules support Euler, ER-SDE and DPM++ 2M, including verified Bleh presets.")
            with capture_nag_preparations():
                return parent(**inputs)
        finally:
            _REQUEST.reset(token)
            _NAG_REQUEST.reset(nag_token)

    def _scheduled_run(self, parent, inputs, advanced=False):
        request = _REQUEST.get()
        nag_request = _NAG_REQUEST.get()
        if request is None and nag_request is None:
            return parent(**inputs)
        count = inputs["steps"]
        if advanced:
            count = max(0, min(count, inputs["end_at_step"]) - inputs["start_at_step"])
        values = (grounding_values(request.start, request.end, count, request.curve)
                  if request is not None else None)
        alpha_values = (nag_alpha_values(nag_request.start, nag_request.end, count, nag_request.curve)
                        if nag_request is not None else None)
        if count == 0 or (values is not None and not values) or (alpha_values is not None and not alpha_values):
            return parent(**inputs)
        inputs["model"], inputs["positive"], inputs["negative"] = _prepare_conditions(
            request, nag_request, inputs["model"], inputs["positive"], inputs["negative"],
            values, alpha_values,
        )
        latent, info = parent(**inputs)
        summaries = []
        if values is not None:
            shown = [str(value) for value in values]
            if len(shown) > 16:
                shown = shown[:8] + ["..."] + shown[-4:]
            summaries.append(f"Grounding {request.curve} ({len(values)} steps): "
                             f"{' -> '.join(shown)} px; {len(set(values))} resolutions")
        if alpha_values is not None:
            shown = [f"{value:.3g}" for value in alpha_values]
            if len(shown) > 16:
                shown = shown[:8] + ["..."] + shown[-4:]
            phi_mode = "auto phi" if nag_request.auto_phi else "manual phi"
            summaries.append(f"NAG alpha {nag_request.curve} ({len(alpha_values)} steps): "
                             f"{' -> '.join(shown)}; {phi_mode}")
        return latent, ("\n".join(summaries) + f"\n{info}")

    def run_simple(
        self,
        model,
        seed,
        steps,
        cfg_start,
        cfg_halfway,
        cfg_end,
        halfway_step,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
    ):
        inputs = dict(
            model=model,
            seed=seed,
            steps=steps,
            cfg_start=cfg_start,
            cfg_halfway=cfg_halfway,
            cfg_end=cfg_end,
            halfway_step=halfway_step,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
        )
        return self._scheduled_run(super().run_simple, inputs)

    def run_advanced(
        self,
        model,
        add_noise,
        noise_seed,
        steps,
        cfg_start,
        cfg_halfway,
        cfg_end,
        halfway_step,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        cfg_curve='linear',
        denoise=1.0,
    ):
        inputs = dict(
            model=model,
            add_noise=add_noise,
            noise_seed=noise_seed,
            steps=steps,
            cfg_start=cfg_start,
            cfg_halfway=cfg_halfway,
            cfg_end=cfg_end,
            halfway_step=halfway_step,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            return_with_leftover_noise=return_with_leftover_noise,
            cfg_curve=cfg_curve,
            denoise=denoise,
        )
        return self._scheduled_run(super().run_advanced, inputs, advanced=True)



NODE_CLASS_MAPPINGS = {"DonutSampler": DonutSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutSampler": "DonutSampler"}

"""Scheduler-aware quantization for distilled/Turbo sampling."""

from bisect import bisect_left


# User-verified bong_tangent denoise points for an 8-step Turbo schedule.
# Interpolating this normalized curve keeps the scheduler's shape for other
# supported step counts while preserving the exact 8-step values.
_BONG_TANGENT_PROFILE = (
    (0.0, 0.0),
    (1.0 / 8.0, 0.225),
    (2.0 / 8.0, 0.308),
    (3.0 / 8.0, 0.359),
    (4.0 / 8.0, 0.400),
    (5.0 / 8.0, 0.444),
    (6.0 / 8.0, 0.502),
    (7.0 / 8.0, 0.606),
    (1.0, 1.0),
)


def _interpolate_profile(position, profile):
    for index in range(1, len(profile)):
        right_x, right_y = profile[index]
        if position <= right_x:
            left_x, left_y = profile[index - 1]
            amount = (position - left_x) / (right_x - left_x)
            return left_y + amount * (right_y - left_y)
    return profile[-1][1]


def turbo_denoise_points(supported_steps, scheduler):
    """Return valid ``(effective_steps, denoise)`` points for Turbo mode."""
    supported_steps = int(supported_steps)
    if supported_steps < 1:
        raise ValueError("Turbo mode requires at least one supported step.")

    if scheduler == "bong_tangent":
        return [
            (step, _interpolate_profile(step / supported_steps, _BONG_TANGENT_PROFILE))
            for step in range(1, supported_steps + 1)
        ]

    return [
        (step, step / supported_steps)
        for step in range(1, supported_steps + 1)
    ]


def resolve_turbo_sampling(supported_steps, requested_denoise, scheduler):
    """Resolve Turbo controls to ComfyUI execution arguments.

    Returns ``(effective_steps, execution_denoise, matched_denoise)``.
    ``matched_denoise`` is the scheduler's user-facing noise position. ComfyUI's
    denoise argument must instead be ``effective_steps / supported_steps`` so
    it reconstructs the model's complete supported schedule before slicing it.
    Ties select the higher matched denoise/effective-step point.
    """
    supported_steps = int(supported_steps)
    requested_denoise = min(1.0, max(0.0, float(requested_denoise)))
    points = turbo_denoise_points(supported_steps, scheduler)
    denoises = [point[1] for point in points]
    right = bisect_left(denoises, requested_denoise)

    if right == 0:
        selected = points[0]
    elif right == len(points):
        selected = points[-1]
    else:
        lower = points[right - 1]
        upper = points[right]
        if requested_denoise - lower[1] < upper[1] - requested_denoise:
            selected = lower
        else:
            selected = upper

    effective_steps, matched_denoise = selected
    execution_denoise = effective_steps / supported_steps
    return effective_steps, execution_denoise, matched_denoise

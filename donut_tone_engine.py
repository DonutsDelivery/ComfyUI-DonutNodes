"""Inference-only port of Donut Tone Lab v4.0's 169-feature analyzer.

The feature order, normalization, output activations and identity gate are a
versioned contract with `donut_tone_lab_feature_learner_v4.html`. No filenames,
ratings, histories, or per-image learned overrides enter this module.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

SCHEMA = "donut_srgb256_features_v4.0"
ALGORITHM = "donut_feature_mlp_v4.0"
FEATURE_COUNT = 169
SHAPES = ((FEATURE_COUNT, 16), (16, 8), (8, 3))


def analyze_rgba(pixels: np.ndarray) -> dict[str, Any]:
    """Analyze an already prepared uint8 sRGB proxy, exactly in v4 feature order.

    Image resizing is intentionally separate. This function has no browser,
    Pillow, ComfyUI, Torch or model dependency. Alpha < 250 is excluded, as in v4.
    """
    a = np.asarray(pixels)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 4 or min(a.shape[:2]) < 1:
        raise ValueError("Tone Lab analysis expects nonempty HWC uint8 RGBA pixels")
    h, w = a.shape[:2]
    valid = a[..., 3] >= 250
    rgb = a[..., :3].astype(np.float64) / 255.0
    ymap = .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]
    colours = rgb[valid]
    ys = np.sort(ymap[valid])
    n = len(ys)

    def avg(x):
        return float(np.mean(x)) if np.size(x) else 0.0

    def var(x):
        return float(np.var(x)) if np.size(x) else 0.0

    def q(x, p):
        # NumPy's linear quantile uses the same (n-1)*p interpolation as v4.
        return float(np.quantile(x, p, method="linear")) if np.size(x) else 0.0

    maximum = colours.max(axis=1) if n else np.empty(0)
    minimum = colours.min(axis=1) if n else np.empty(0)
    chroma = maximum - minimum
    saturation = np.divide(chroma, maximum, out=np.zeros_like(chroma), where=maximum > 1e-12)
    hist = np.bincount(np.minimum(63, np.floor(ys * 64)).astype(int), minlength=64) / max(1, n)
    sat_hist = np.bincount(np.minimum(7, np.floor(saturation * 8)).astype(int), minlength=8) / max(1, n)
    linear = np.where(colours <= .04045, colours / 12.92, ((colours + .055) / 1.055) ** 2.4)
    linear_y = linear @ np.array([.2126, .7152, .0722])
    mean, std = avg(ys), math.sqrt(var(ys))
    names, features = [], []

    def add(name, v):
        names.append(name)
        features.append(float(v) if np.isfinite(v) else 0.0)

    for i, v in enumerate(hist):
        add(f"Luma bin {i:02d}", v)
    for p in (.01, .05, .10, .25, .50, .75, .90, .95, .99):
        add(f"p{round(p * 100):02d}", q(ys, p))
    add("Mean luma", mean)
    add("Luma standard deviation", std)
    z = (ys - mean) / std if std > 1e-6 else np.empty(0)
    add("Skewness / 4", np.clip(avg(z ** 3), -4, 4) / 4 if z.size else 0)
    add("Excess kurtosis / 20", np.clip(avg(z ** 4) - 3, -3, 20) / 20 if z.size else 0)
    nonzero = hist[hist > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero))) / 6
    add("Histogram entropy / 6", entropy)
    add("Geometric mean luma", math.exp(avg(np.log(np.maximum(ys, 1 / 255)))) if n else 0)
    add("Linear-light mean", avg(linear_y))
    add("Linear-light standard deviation", math.sqrt(var(linear_y)))
    for name, high, low in (("p99 − p01", .99, .01), ("p95 − p05", .95, .05),
                            ("p90 − p10", .90, .10), ("Interquartile range", .75, .25)):
        add(name, q(ys, high) - q(ys, low))
    bounds = (0, .05, .2, .4, .6, .85, .97, 1.000001)
    for low, high in zip(bounds, bounds[1:]):
        add(f"Zone {low:.2f}–{min(1, high):.2f}", np.count_nonzero((ys >= low) & (ys < high)) / max(1, n))
    for name, mask in (("Near-black luma", ys <= 1 / 255), ("Near-white luma", ys >= 254 / 255),
                       ("Any RGB channel near black", minimum <= 1 / 255),
                       ("Any RGB channel near white", maximum >= 254 / 255)):
        add(name, np.count_nonzero(mask) / max(1, n))
    add("Highlight headroom", 1 - q(ys, .99))
    add("Median / p95", q(ys, .5) / max(q(ys, .95), 1e-6))
    for c, prefix in enumerate(("R", "G", "B")):
        channel = colours[:, c]
        add(prefix + " mean", avg(channel))
        add(prefix + " standard deviation", math.sqrt(var(channel)))
        for p in (.05, .5, .95):
            add(prefix + f" p{round(p * 100)}", q(channel, p))
    add("R mean − G mean", avg(colours[:, 0]) - avg(colours[:, 1]))
    add("B mean − G mean", avg(colours[:, 2]) - avg(colours[:, 1]))
    add("Neutral pixel fraction", np.count_nonzero(chroma < .05) / max(1, n))
    for name, values in (("Chroma", chroma), ("Saturation", saturation)):
        add(name + " mean", avg(values))
        add(name + " standard deviation", math.sqrt(var(values)))
        add(name + " p90", q(values, .9))
    for i, v in enumerate(sat_hist):
        add(f"Saturation bin {i}", v)

    tx = np.minimum(3, np.floor(np.arange(w) / w * 4)).astype(int)
    ty = np.minimum(3, np.floor(np.arange(h) / h * 4)).astype(int)
    tile_ids = ty[:, None] * 4 + tx[None, :]
    tiles = [ymap[valid & (tile_ids == i)] for i in range(16)]
    tile_means = [avg(t) if len(t) else mean for t in tiles]
    tile_std = [math.sqrt(var(t)) for t in tiles]
    for i, v in enumerate(tile_means):
        add(f"Tile {i // 4},{i % 4} mean", v)
    for i, v in enumerate(tile_std):
        add(f"Tile {i // 4},{i % 4} contrast", v)
    center = ((tx[None, :] >= 1) & (tx[None, :] <= 2) &
              (ty[:, None] >= 1) & (ty[:, None] <= 2))
    add("Between-tile variation", math.sqrt(var(tile_means)))
    add("Mean local contrast", avg(tile_std))
    add("Center − border luma", avg(ymap[valid & center]) - avg(ymap[valid & ~center]))
    dx = np.abs(np.diff(ymap, axis=1))[valid[:, :-1] & valid[:, 1:]]
    dy = np.abs(np.diff(ymap, axis=0))[valid[:-1, :] & valid[1:, :]]
    lap_valid = (valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1] &
                 valid[1:-1, :-2] & valid[1:-1, 2:])
    lap = ((4 * ymap[1:-1, 1:-1] - ymap[:-2, 1:-1] - ymap[2:, 1:-1] -
            ymap[1:-1, :-2] - ymap[1:-1, 2:]) / 4)[lap_valid]
    add("Horizontal gradient", avg(dx))
    add("Vertical gradient", avg(dy))
    add("Edge fraction", (np.count_nonzero(dx > .08) + np.count_nonzero(dy > .08)) / max(1, len(dx) + len(dy)))
    add("Laplacian RMS / 4", math.sqrt(avg(lap ** 2)))
    if len(features) != FEATURE_COUNT:
        raise RuntimeError("Internal Tone Lab feature schema mismatch")
    return {"features": features, "names": names, "sampleCount": n,
            "mean": mean, "p50": q(ys, .5), "p95": q(ys, .95), "entropy": entropy}


FEATURE_NAMES = tuple(analyze_rgba(np.zeros((1, 1, 4), dtype=np.uint8))["names"])


def validate_export(document: dict) -> dict:
    """Accept only v4 Export model only, never a session or executable weights."""
    if not isinstance(document, dict) or document.get("type") != "donut-tone-model" or document.get("version") != 4:
        raise ValueError("Select a v4 'Export model only' JSON, not a training session")
    if document.get("schema") != SCHEMA or document.get("algorithm") != ALGORITHM:
        raise ValueError("Incompatible Tone Lab schema/algorithm; expected v4.0")
    if document.get("analysis") != {"longEdge": 256, "colorSpace": "srgb", "opaqueAlphaMinimum": 250}:
        raise ValueError("Unsupported Tone Lab analysis configuration")
    m = document.get("model")
    if not isinstance(m, dict) or m.get("trained") is not True or m.get("schema") != SCHEMA or m.get("format") != ALGORITHM:
        raise ValueError("Missing or untrained Tone Lab v4.0 model")
    if m.get("featureCount") != FEATURE_COUNT or m.get("names") != list(FEATURE_NAMES):
        raise ValueError("Tone Lab feature names/order must exactly match all 169 v4.0 features")

    def vector(x, length, name, positive=False, bound=None):
        if not isinstance(x, list) or len(x) != length or any(type(v) not in (int, float) for v in x):
            raise ValueError(f"Invalid {name} length or numeric values")
        try:
            a = np.array(x, dtype=np.float64)
        except (OverflowError, ValueError) as e:
            raise ValueError(f"Invalid {name}") from e
        if not np.isfinite(a).all() or (positive and np.any(a <= 0)) or (bound and np.any(np.abs(a) >= bound)):
            raise ValueError(f"Invalid {name} values")
        return a

    mu = vector(m.get("mu"), FEATURE_COUNT, "mu")
    sd = vector(m.get("sd"), FEATURE_COUNT, "sd", positive=True)
    if not isinstance(m.get("layers"), list) or len(m["layers"]) != 3:
        raise ValueError("Tone Lab expects three dense layers")
    layers = []
    for i, (layer, (ni, no)) in enumerate(zip(m["layers"], SHAPES)):
        if not isinstance(layer, dict) or layer.get("ni") != ni or layer.get("no") != no:
            raise ValueError(f"Invalid layer {i} shape; expected {ni} -> {no}")
        layers.append((vector(layer.get("w"), ni * no, f"layer {i} weights", bound=1e5).reshape(no, ni),
                       vector(layer.get("b"), no, f"layer {i} biases", bound=1e5)))
    revision = m.get("revision", 0)
    if type(revision) is not int or revision < 0:
        raise ValueError("Invalid Tone Lab model revision")
    return {"mu": mu, "sd": sd, "layers": layers, "revision": revision}


def predict(features, model: dict) -> dict[str, Any]:
    """Use global weights only; match v4 tanh heads and its exact no-op gate."""
    f = np.asarray(features, dtype=np.float64)
    if f.shape != (FEATURE_COUNT,) or not np.isfinite(f).all():
        raise ValueError("Expected 169 finite Tone Lab features")
    with np.errstate(over="ignore", divide="ignore"):
        standardized = (f - model["mu"]) / model["sd"]
    x = np.clip(standardized, -6, 6)
    for i, (weights, bias) in enumerate(model["layers"]):
        x = weights @ x + bias
        if i < 2:
            x = np.tanh(x)
    gamma = math.exp(math.log(3) * math.tanh(x[0]))
    gain = math.exp(math.log(1.25) * math.tanh(x[1]))
    no_op = 1 / (1 + math.exp(-float(np.clip(x[2], -40, 40))))
    grid = np.arange(1, 32, dtype=np.float64) / 32
    distance = math.sqrt(float(np.mean((np.minimum(1, grid ** gamma * gain) - grid) ** 2)))
    noop = distance < .0025 or (no_op >= .9 and distance < .015)
    return {"gamma": 1.0 if noop else gamma, "gain": 1.0 if noop else gain,
            "rawGamma": gamma, "rawGain": gain, "noOpScore": no_op, "noop": noop,
            "coverage": float(np.mean(np.abs(standardized) > 3))}

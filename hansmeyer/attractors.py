"""Attractors: spatially varying (non-uniform) weights.

Hansmeyer 2010, "Uniform input mesh and specification of extrinsic parameters":
sets of weights are placed in the mesh's environment; for each face the
distances to the sets give their level of influence (eq. 8) and the face's
weights are the influence-weighted blend of the sets (eq. 9):

    c_a = f(d_a) h_a / sum_i f(d_i) h_i          w_x = sum_i w_{x,i} c_i

with f(d) = (1 - d)^t in the paper. Here
  * an attractor is a point or a space curve (distance = nearest point on it),
  * f is a choice of falloff curves over d / radius (the paper's power law included),
  * the main iteration schedule takes part in the blend as a "background" set
    with constant influence `background` (0 = the paper's pure eq. 8-9),
  * "modifier" attractors additionally scale or offset chosen weights near them.

Influence is measured at each face centroid, either at its current position or
at its "rest" position (where it sat on the input mesh, carried through subdivision).
"""

from __future__ import annotations

import copy
import json

import numpy as np

from .schedule import MAX_ITERATIONS, WEIGHT_NAMES

FALLOFFS = ("power", "linear", "smoothstep", "gaussian", "spline")
CURVE_TYPES = ("line", "circle", "helix", "sine", "lissajous", "polyline")
AXES = ("x", "y", "z")
SPLINE_KNOTS = np.linspace(0.0, 1.0, 5)
CURVE_SAMPLES = 256

DEFAULT_CURVE = {
    "type": "helix",
    "center": [0.0, 0.0, 0.0],
    "axis": "y",
    "radius": 1.0,  # circle, helix
    "height": 4.0,  # helix
    "turns": 2.0,  # helix
    "length": 4.0,  # sine
    "amplitude": 0.6,  # sine
    "waves": 2.0,  # sine
    "size": 1.5,  # lissajous
    "a": 3,
    "b": 2,
    "c": 1,
    "phase": 0.5,
    "start": [-2.0, 0.0, 0.0],  # line
    "end": [2.0, 0.0, 0.0],
    "points": [],  # polyline control points
    "closed": False,
    "smooth": True,
}

DEFAULT_ATTRACTOR = {
    "name": "A",
    "enabled": True,
    "kind": "point",  # "point" | "curve"
    "position": [0.0, 0.0, 0.0],
    "curve": DEFAULT_CURVE,
    "payload": "modifier",  # "set" (eq. 8-9 weight set) | "modifier"
    "strength": 1.0,  # h in eq. 8
    "radius": 2.0,  # distance at which the falloff reaches its end
    "falloff": {"type": "power", "tightness": 2.0, "spline": [1.0, 0.8, 0.5, 0.2, 0.0]},
    "weights": [],  # set payload: one weight dict per iteration
    "mods": [],  # modifier payload: [{"weight", "op": "scale"|"offset", "value", "from", "to"}]
}


# ------------------------------------------------------------------ defaults
def normalize(att: dict) -> dict:
    """Fill in defaults so every attractor has every field (presets stay short)."""
    a = copy.deepcopy(DEFAULT_ATTRACTOR)
    a.update({k: copy.deepcopy(v) for k, v in att.items() if k in DEFAULT_ATTRACTOR})
    curve = copy.deepcopy(DEFAULT_CURVE)
    curve.update(att.get("curve", {}))
    a["curve"] = curve
    fall = copy.deepcopy(DEFAULT_ATTRACTOR["falloff"])
    fall.update(att.get("falloff", {}))
    a["falloff"] = fall
    weights = [dict(w) for w in att.get("weights", [])][:MAX_ITERATIONS]
    weights += [{} for _ in range(MAX_ITERATIONS - len(weights))]
    a["weights"] = [{n: float(w.get(n, 0.0)) for n in WEIGHT_NAMES} for w in weights]
    a["mods"] = [
        {"weight": m.get("weight", "w_f"), "op": m.get("op", "scale"), "value": float(m.get("value", 1.0)),
         "from": int(m.get("from", 1)), "to": int(m.get("to", MAX_ITERATIONS))}
        for m in att.get("mods", [])
    ]
    for key, allowed in (("kind", ("point", "curve")), ("payload", ("set", "modifier"))):
        if a[key] not in allowed:
            raise ValueError(f"attractor {key} must be one of {allowed}")
    if a["falloff"]["type"] not in FALLOFFS or a["curve"]["type"] not in CURVE_TYPES:
        raise ValueError("unknown falloff or curve type")
    return a


def compact(att: dict) -> dict:
    """Inverse of normalize for saving: drop zero weights and unused curve data."""
    a = copy.deepcopy(att)
    if a["payload"] == "set":
        a["weights"] = [{k: v for k, v in w.items() if v != 0.0} for w in a["weights"]]
        while a["weights"] and not a["weights"][-1]:
            a["weights"].pop()
    else:
        a.pop("weights", None)
    if a["payload"] != "modifier":
        a.pop("mods", None)
    if a["kind"] == "point":
        a.pop("curve", None)
    else:
        a.pop("position", None)
        used = {"line": ["start", "end"], "circle": ["center", "axis", "radius"],
                "helix": ["center", "axis", "radius", "height", "turns"],
                "sine": ["center", "axis", "length", "amplitude", "waves"],
                "lissajous": ["center", "size", "a", "b", "c", "phase"],
                "polyline": ["points", "closed", "smooth"]}[a["curve"]["type"]]
        a["curve"] = {"type": a["curve"]["type"], **{k: a["curve"][k] for k in used}}
    return a


# ------------------------------------------------------------------ falloff
def falloff(u: np.ndarray, spec: dict) -> np.ndarray:
    """Influence in [0, 1] as a function of normalised distance u = d / radius."""
    u = np.maximum(np.asarray(u, float), 0.0)
    kind = spec.get("type", "power")
    if kind == "power":  # the paper's (1 - d)^t
        return np.clip(1.0 - u, 0.0, 1.0) ** max(float(spec.get("tightness", 2.0)), 1e-6)
    if kind == "linear":
        return np.clip(1.0 - u, 0.0, 1.0)
    if kind == "smoothstep":
        t = np.clip(u, 0.0, 1.0)
        return 1.0 - t * t * (3 - 2 * t)
    if kind == "gaussian":  # sigma = radius / 3
        return np.exp(-4.5 * u * u)
    if kind == "spline":
        vals = np.asarray(spec.get("spline", [1, 0.8, 0.5, 0.2, 0]), float)
        return np.interp(u, SPLINE_KNOTS, vals, right=vals[-1])
    raise ValueError(f"unknown falloff {kind!r}")


# ------------------------------------------------------------------- curves
def _frame(axis: str) -> np.ndarray:
    """Rotation taking local y (the curve's main axis) to the chosen world axis."""
    if axis == "y":
        return np.eye(3)
    if axis == "x":
        return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float).T
    return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float).T


def _catmull_rom(P: np.ndarray, closed: bool, samples: int) -> np.ndarray:
    n = len(P)
    if n < 3:
        t = np.linspace(0, 1, samples)[:, None]
        return P[0] * (1 - t) + P[-1] * t if n == 2 else np.repeat(P, samples, 0)
    if closed:
        ext = np.concatenate([P[-1:], P, P[:2]])
        segs = n
    else:
        ext = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
        segs = n - 1
    out = []
    per = max(samples // segs, 2)
    for i in range(segs):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t = np.linspace(0, 1, per, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    out.append(ext[segs + 1][None] if not closed else ext[1][None])
    return np.concatenate(out)


def curve_points(curve: dict, samples: int = CURVE_SAMPLES) -> np.ndarray:
    """Sample a curve as an (S, 3) polyline (closed curves repeat their first point)."""
    c = curve
    kind = c["type"]
    t = np.linspace(0.0, 1.0, samples)
    if kind == "line":
        a, b = np.asarray(c["start"], float), np.asarray(c["end"], float)
        return a + (b - a) * t[:, None]
    if kind == "polyline":
        P = np.asarray(c["points"], float).reshape(-1, 3)
        if len(P) == 0:
            return np.zeros((1, 3))
        if c.get("smooth", True) and len(P) >= 3:
            return _catmull_rom(P, bool(c.get("closed")), samples)
        return np.concatenate([P, P[:1]]) if c.get("closed") and len(P) > 2 else P
    if kind == "circle":
        th = 2 * np.pi * t
        local = np.stack([c["radius"] * np.cos(th), np.zeros_like(th), c["radius"] * np.sin(th)], 1)
    elif kind == "helix":
        th = 2 * np.pi * c["turns"] * t
        local = np.stack([c["radius"] * np.cos(th), (t - 0.5) * c["height"], c["radius"] * np.sin(th)], 1)
    elif kind == "sine":
        local = np.stack([c["amplitude"] * np.sin(2 * np.pi * c["waves"] * t), (t - 0.5) * c["length"],
                          np.zeros_like(t)], 1)
    elif kind == "lissajous":
        th = 2 * np.pi * t
        s = c["size"]
        local = np.stack([s * np.sin(c["a"] * th + c["phase"] * np.pi), s * np.sin(c["b"] * th),
                          s * np.sin(c["c"] * th + 0.5 * c["phase"] * np.pi)], 1)
    else:
        raise ValueError(f"unknown curve type {kind!r}")
    R = _frame(c.get("axis", "y")) if kind != "lissajous" else np.eye(3)
    return local @ R.T + np.asarray(c["center"], float)


def to_polyline(curve: dict, n: int = 8) -> dict:
    """Convert any curve into an editable polyline with n control points."""
    closed = curve["type"] in ("circle", "lissajous")
    P = curve_points(curve, 512)
    idx = np.linspace(0, len(P) - 1, n + (1 if closed else 0)).round().astype(int)
    pts = P[idx[:-1] if closed else idx]
    out = copy.deepcopy(curve)
    out.update({"type": "polyline", "points": pts.round(4).tolist(), "closed": closed, "smooth": True})
    return out


def load_polyline(path: str) -> np.ndarray:
    """Points from .csv/.txt (x y z per line, any separator) or .obj (v, ordered by `l` if present)."""
    if path.lower().endswith(".obj"):
        V, order = [], []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                p = line.split()
                if p and p[0] == "v":
                    V.append([float(x) for x in p[1:4]])
                elif p and p[0] == "l":
                    for tok in p[1:]:
                        i = int(tok.split("/")[0])
                        i = i - 1 if i > 0 else len(V) + i
                        if not order or order[-1] != i:
                            order.append(i)
        V = np.array(V, float)
        P = V[order] if order else V
    else:
        rows = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                vals = line.replace(",", " ").replace(";", " ").replace("\t", " ").split()
                try:
                    nums = [float(v) for v in vals]
                except ValueError:
                    continue  # header lines
                if len(nums) >= 3:
                    rows.append(nums[:3])
        P = np.array(rows, float)
    if len(P) < 2:
        raise ValueError(f"{path}: need at least two points")
    return P


# ----------------------------------------------------------------- distance
def _segment_distance(p: np.ndarray, A: np.ndarray, AB: np.ndarray, L2: np.ndarray) -> np.ndarray:
    """Distances from points p (c,3) to segments given per point (c,k,3) or shared (k,3); min over k."""
    p = p[:, None, :]
    t = np.clip(((p - A) * AB).sum(-1) / L2, 0.0, 1.0)
    return np.sqrt(((A + t[..., None] * AB - p) ** 2).sum(-1).min(1))


def distance_to_polyline(P: np.ndarray, Q: np.ndarray, chunk: int = 4096, window: int = 2) -> np.ndarray:
    """Distance from points P (n,3) to the polyline Q (m,3).

    Short polylines are solved exactly against every segment. Long (densely
    sampled) curves first find the nearest segment midpoint with one matrix
    product, then solve exactly on the `window` segments either side of it.
    That is ~20x faster and never underestimates; it can overestimate by at
    most half a segment length (curve length / 512 for sampled curves), and
    only for points whose nearest curve point lies away from that midpoint."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if len(Q) == 1:
        return np.linalg.norm(P - Q[0], axis=1)
    A, AB = Q[:-1], Q[1:] - Q[:-1]
    L2 = np.maximum((AB * AB).sum(1), 1e-30)
    n_seg = len(A)
    out = np.empty(len(P))
    if n_seg <= 4 * window + 2:
        for s in range(0, len(P), chunk):
            out[s:s + chunk] = _segment_distance(P[s:s + chunk], A, AB, L2)
        return out
    M = A + 0.5 * AB
    M2 = (M * M).sum(1)
    offsets = np.arange(-window, window + 1)
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk]
        nearest = np.argmin(M2[None, :] - 2.0 * (p @ M.T), axis=1)  # |p - m|^2 minus the constant |p|^2
        idx = np.clip(nearest[:, None] + offsets, 0, n_seg - 1)
        out[s:s + chunk] = _segment_distance(p, A[idx], AB[idx], L2[idx])
    return out


def geometry(att: dict) -> np.ndarray:
    """The attractor as a polyline (a single point for point attractors)."""
    if att["kind"] == "point":
        return np.asarray(att["position"], float)[None]
    return curve_points(att["curve"])


def influence(att: dict, P: np.ndarray, geom: np.ndarray | None = None) -> np.ndarray:
    """h * f(d / radius) at positions P."""
    geom = geometry(att) if geom is None else geom
    d = distance_to_polyline(P, geom)
    return float(att["strength"]) * falloff(d / max(float(att["radius"]), 1e-9), att["falloff"])


# ------------------------------------------------------------------ weights
def active(design) -> list[dict]:
    return [a for a in design.attractors if a.get("enabled", True)]


def face_positions(mesh, space: str) -> np.ndarray:
    if space == "rest" and "rest" in mesh.vattr:
        R = mesh.vattr["rest"]
        return mesh.face_reduce(R[mesh.face_idx]) / mesh.face_size[:, None]
    return mesh.face_centroid


def blend(design, mesh, level: int, base_weights: dict) -> dict:
    """Per-face weights for one iteration (eq. 8-9 plus modifiers)."""
    atts = active(design)
    if not atts:
        return base_weights
    P = face_positions(mesh, design.attractor_space)
    F = len(P)
    sets = [a for a in atts if a["payload"] == "set"]
    mods = [a for a in atts if a["payload"] == "modifier"
            and any(m["from"] <= level + 1 <= m["to"] for m in a["mods"])]
    W = {n: np.full(F, float(v)) for n, v in base_weights.items()}

    if sets:
        raw = [influence(a, P) for a in sets]
        bg = max(float(design.background), 0.0)
        total = bg + np.sum(raw, axis=0)
        ok = total > 1e-12
        for n in W:
            acc = bg * W[n]
            for a, r in zip(sets, raw):
                acc = acc + r * a["weights"][level].get(n, 0.0)
            W[n] = np.where(ok, acc / np.where(ok, total, 1.0), W[n])

    for a in mods:
        m = influence(a, P)
        for mod in a["mods"]:
            if not (mod["from"] <= level + 1 <= mod["to"]) or mod["weight"] not in W:
                continue
            if mod["op"] == "scale":
                W[mod["weight"]] = W[mod["weight"]] * (1.0 + (mod["value"] - 1.0) * m)
            else:
                W[mod["weight"]] = W[mod["weight"]] + mod["value"] * m
    return W


def make_provider(design):
    """Weight provider for the pipeline: (mesh, level, spec) -> weights (scalars or per-face arrays)."""
    if not active(design):
        return lambda mesh, level, spec: spec.weights
    return lambda mesh, level, spec: blend(design, mesh, level, spec.weights)


def level_signature(design, level: int):
    """What the attractors contribute to iteration `level` — used for per-level cache keys,
    so editing a weight set at iteration k only recomputes from k on."""
    atts = active(design)
    if not atts:
        return None
    common = []
    for a in atts:
        g = {k: a[k] for k in ("kind", "payload", "strength", "radius", "falloff")}
        g["geom"] = a["position"] if a["kind"] == "point" else a["curve"]
        if a["payload"] == "set":
            g["w"] = a["weights"][level]
        else:
            g["mods"] = [m for m in a["mods"] if m["from"] <= level + 1 <= m["to"]]
            if not g["mods"]:
                continue  # inactive at this iteration
        common.append(g)
    if not common:
        return None
    return json.dumps({"space": design.attractor_space, "bg": design.background, "a": common}, sort_keys=True)

"""Function layer stack (our extension, not Hansmeyer's method).

Each layer evaluates a function from functions.py and applies it in one of three ways:

  weight        per face, before an iteration:  v = amplitude * f(p) + offset, blended into one weight
                  add:      w + m v          multiply: w (1 + m v)      replace: lerp(w, v, m)
                  min/max:  lerp(w, min/max(w, v), m)
  displacement  per vertex, after an iteration: move along the vertex normal by m (amplitude f + offset),
                in local edge lengths (relative extrusion) or model units (absolute)
  fold          per vertex, after an iteration: V <- V + m * amplitude * (fold(V) - V)

m is the layer's mask: 1, or the normalised influence of a named attractor. Layers run
in list order, only in their iteration range [from, to] (1-based).
"""

from __future__ import annotations

import copy

import numpy as np

from . import functions
from .schedule import MAX_ITERATIONS

TARGETS = ("weight", "displacement", "fold")
BLENDS = ("add", "multiply", "replace", "min", "max")

DEFAULT_LAYER = {
    "name": "L",
    "enabled": True,
    "target": "weight",
    "function": "gyroid",
    "params": {},
    "weight": "w_f",
    "blend": "add",
    "amplitude": 0.3,
    "offset": 0.0,
    "from": 1,
    "to": MAX_ITERATIONS,
    "space": "rest",  # evaluate at "rest" (input-mesh) or "current" positions
    "mask": "",  # attractor name ("" = everywhere)
    "domain": {"fold": "none", "repeat": 1, "scale": 1.0, "c": 0.0, "params": {}},
}


def normalize(layer: dict) -> dict:
    out = copy.deepcopy(DEFAULT_LAYER)
    out.update({k: copy.deepcopy(v) for k, v in layer.items() if k in DEFAULT_LAYER})
    dom = copy.deepcopy(DEFAULT_LAYER["domain"])
    dom.update(layer.get("domain", {}))
    out["domain"] = dom
    if out["target"] not in TARGETS or out["blend"] not in BLENDS:
        raise ValueError("unknown layer target or blend mode")
    out["from"], out["to"] = int(out["from"]), int(out["to"])
    return out


def compact(layer: dict) -> dict:
    out = copy.deepcopy(layer)
    if out["target"] != "weight":
        for k in ("weight", "blend"):
            out.pop(k, None)
    if out["domain"].get("fold", "none") == "none":
        out.pop("domain")
    if not out["mask"]:
        out.pop("mask")
    return out


def active(design, level: int, targets) -> list[dict]:
    return [ly for ly in design.layers
            if ly["enabled"] and ly["target"] in targets and ly["from"] <= level + 1 <= ly["to"]]


def _mask(design, layer, P) -> np.ndarray | float:
    if not layer.get("mask"):
        return 1.0
    from .attractors import influence

    for a in design.attractors:
        if a["name"] == layer["mask"]:
            if not a["enabled"]:
                return 0.0
            return np.clip(influence(a, P) / max(float(a["strength"]), 1e-9), 0.0, 1.0)
    return 1.0  # mask attractor not found: no masking


def _field(layer, P) -> np.ndarray:
    return functions.evaluate(layer["function"], P, layer["params"], layer["domain"])


# ------------------------------------------------------------------ weights
def apply_weight_layers(design, mesh, level: int, W: dict) -> dict:
    layers = active(design, level, ("weight",))
    if not layers:
        return W
    from .attractors import face_positions

    F = mesh.n_faces
    W = {n: np.broadcast_to(np.asarray(v, float), (F,)).copy() for n, v in W.items()}
    cache = {}
    for ly in layers:
        if ly["weight"] not in W:
            continue
        if ly["space"] not in cache:
            cache[ly["space"]] = face_positions(mesh, ly["space"])
        P = cache[ly["space"]]
        v = ly["amplitude"] * _field(ly, P) + ly["offset"]
        m = _mask(design, ly, P)
        w = W[ly["weight"]]
        b = ly["blend"]
        if b == "add":
            w = w + m * v
        elif b == "multiply":
            w = w * (1.0 + m * v)
        elif b == "replace":
            w = w + m * (v - w)
        elif b == "min":
            w = w + m * (np.minimum(w, v) - w)
        else:
            w = w + m * (np.maximum(w, v) - w)
        W[ly["weight"]] = w
    return W


# ------------------------------------------------------------- post steps
def with_positions(mesh, V):
    """Same topology and attributes, new positions."""
    from .mesh import PolyMesh

    return PolyMesh(V, mesh.face_ptr, mesh.face_idx, vtype=mesh.vtype, fclass=mesh.fclass,
                    vattr=mesh.vattr, fattr=mesh.fattr, info=dict(mesh.info))


def post_process(design, mesh, level: int, relative: bool = True, vmask=None):
    """Displacement and fold layers, applied after iteration `level` (0-based)."""
    layers = active(design, level, ("displacement", "fold"))
    if not layers:
        return mesh
    V = mesh.V.copy()
    keep = np.ones(len(V))
    if "lock" in mesh.vattr:
        keep = keep * (mesh.vattr["lock"] <= level)  # locked for L iterations = held during levels 0..L-1
    if vmask is not None and len(vmask) == len(V):
        keep = keep * vmask
    for ly in layers:
        cur = with_positions(mesh, V)
        if ly["target"] == "displacement":
            P = cur.vattr["rest"] if ly["space"] == "rest" and "rest" in cur.vattr else V
            m = np.broadcast_to(np.asarray(_mask(design, ly, P), float), (len(V),))
            amount = (ly["amplitude"] * _field(ly, P) + ly["offset"]) * m * keep
            if relative:
                amount = amount * cur.vert_mean(cur.face_scale)
            V = V + cur.vert_normal * amount[:, None]
        else:
            target = functions.evaluate(ly["function"], V, ly["params"])
            m = np.broadcast_to(np.asarray(_mask(design, ly, V), float), (len(V),))
            V = V + (ly["amplitude"] * m * keep)[:, None] * (target - V)
    return with_positions(mesh, V)


def signature(design, level: int):
    out = []
    for ly in design.layers:
        if not ly["enabled"] or not (ly["from"] <= level + 1 <= ly["to"]):
            continue
        sig = {k: ly[k] for k in ("target", "function", "params", "weight", "blend", "amplitude", "offset",
                                  "space", "mask", "domain")}
        sig["fn"] = functions.signature(ly["function"])
        dom = ly["domain"].get("fold", "none")
        sig["dfn"] = functions.signature(dom) if dom != "none" else None
        out.append(sig)
    return out or None

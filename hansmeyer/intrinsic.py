"""Intrinsic weight specification (Hansmeyer 2010, "Differentiated input mesh using intrinsic
parameters" and "Further specification options").

  * Groups (tagging & locking, Fig. 9): faces / vertices of the input mesh are tagged;
    tags pass to every child face, each group has its own weight rules, and group
    vertices can be locked (held in place) for L iterations, giving creases and spikes.
  * Motifs (Fig. 6/7, eq. 10-11): vertices are classified by incident faces / edges
    ("4F4E"); each motif gets an attractor/deflector value U, which pulls face and edge
    points toward (+) or away from (-) such vertices with weights w6 / w7.
  * Measures (Fig. 8 + curvature): topological distance to the original vertices or
    edges, face planarity, or dihedral bend, normalised to [0, 1] per iteration, drive
    weight rules that interpolate between a value at 0 and a value at 1.
"""

from __future__ import annotations

import copy

import numpy as np

from .schedule import MAX_ITERATIONS

MEASURES = {
    "dist_vertex": "distance to original vertices",
    "dist_edge": "distance to original edges",
    "planarity": "non-planarity of the face",
    "bend": "bend (dihedral angle to neighbours)",
}
RULE_OPS = ("set", "scale", "add")
MAX_GROUPS = 62


# ------------------------------------------------------------------- motifs
def motif_label(code: int) -> str:
    return f"{int(code) // 100}F{int(code) % 100}E"


def parse_label(label: str) -> int:
    f, e = label.upper().rstrip("E").split("F")
    return int(f) * 100 + int(e)


def motif_counts(mesh) -> dict[str, int]:
    codes, counts = np.unique(mesh.motif_codes(), return_counts=True)
    return {motif_label(c): int(n) for c, n in zip(codes, counts)}


def motif_U(design, mesh) -> np.ndarray | None:
    """Per-vertex attractor/deflector value U from the design's motif table (eq. 10-11)."""
    table = {k: float(v) for k, v in design.motifs.items() if float(v) != 0.0}
    if not table:
        return None
    codes = mesh.motif_codes()
    U = np.zeros(mesh.n_verts)
    for label, u in table.items():
        try:
            U[codes == parse_label(label)] = u
        except ValueError:
            continue
    return U


# ----------------------------------------------------------------- measures
def _normalise(x: np.ndarray, pct: float = 95.0) -> np.ndarray:
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.ones_like(x)
    hi = np.percentile(x[finite], pct) if pct < 100 else x[finite].max()
    hi = hi if hi > 1e-12 else 1.0
    return np.clip(np.where(finite, x / hi, 1.0), 0.0, 1.0)


def vertex_distance(mesh, which: str) -> np.ndarray:
    """Edge hops from each vertex to the original vertices ("tv") or original edges ("te")."""
    key = "tv" if which == "dist_vertex" else "te"
    seeds = mesh.vattr.get(key, np.zeros(mesh.n_verts)) > 0.5
    return mesh.hop_distance(seeds)


def measure(mesh, name: str) -> np.ndarray:
    """Per-face measure in [0, 1]."""
    if name in ("dist_vertex", "dist_edge"):
        d = _normalise(vertex_distance(mesh, name), 100.0)
        return mesh.face_reduce(d[mesh.face_idx]) / mesh.face_size
    if name == "planarity":
        return _normalise(mesh.face_planarity)
    if name == "bend":
        return _normalise(mesh.face_bend)
    raise ValueError(f"unknown measure {name!r}")


# -------------------------------------------------------------------- rules
DEFAULT_RULE = {"enabled": True, "measure": "dist_edge", "weight": "w_f", "op": "set", "a": 0.0, "b": 0.3,
                "gamma": 1.0, "from": 1, "to": MAX_ITERATIONS}


def normalize_rule(r: dict) -> dict:
    out = copy.deepcopy(DEFAULT_RULE)
    out.update({k: v for k, v in r.items() if k in DEFAULT_RULE})
    if out["measure"] not in MEASURES or out["op"] not in RULE_OPS:
        raise ValueError("unknown intrinsic measure or op")
    return out


def _active_rules(design, level):
    return [r for r in design.intrinsic if r["enabled"] and r["from"] <= level + 1 <= r["to"]]


def apply_rules(design, mesh, level: int, W: dict) -> dict:
    rules = _active_rules(design, level)
    if not rules:
        return W
    F = mesh.n_faces
    W = {n: np.broadcast_to(np.asarray(v, float), (F,)).copy() for n, v in W.items()}
    cache = {}
    for r in rules:
        if r["weight"] not in W:
            continue
        if r["measure"] not in cache:
            cache[r["measure"]] = measure(mesh, r["measure"])
        t = cache[r["measure"]] ** max(float(r["gamma"]), 1e-6)
        v = r["a"] + (r["b"] - r["a"]) * t
        if r["op"] == "set":
            W[r["weight"]] = v
        elif r["op"] == "scale":
            W[r["weight"]] = W[r["weight"]] * v
        else:
            W[r["weight"]] = W[r["weight"]] + v
    return W


# ------------------------------------------------------------------- groups
DEFAULT_GROUP = {"name": "G", "enabled": True, "faces": [], "verts": [], "lock": 0, "lock_faces": False, "rules": []}


def normalize_group(g: dict) -> dict:
    out = copy.deepcopy(DEFAULT_GROUP)
    out.update({k: copy.deepcopy(v) for k, v in g.items() if k in DEFAULT_GROUP})
    out["faces"] = sorted({int(i) for i in out["faces"]})
    out["verts"] = sorted({int(i) for i in out["verts"]})
    out["rules"] = [{"weight": m.get("weight", "w_f"), "op": m.get("op", "scale"), "value": float(m.get("value", 1.0)),
                     "from": int(m.get("from", 1)), "to": int(m.get("to", MAX_ITERATIONS))} for m in out["rules"]]
    return out


def decorate_base(design, mesh) -> None:
    """Attach the attributes that intrinsic features carry through subdivision."""
    n, F = mesh.n_verts, mesh.n_faces
    mesh.vattr["tv"] = np.ones(n)  # original vertices
    mesh.vattr["te"] = np.ones(n)  # vertices lying on original edges
    tags = np.zeros(F, dtype=np.int64)
    lock = np.zeros(n)
    for bit, g in enumerate(design.groups[:MAX_GROUPS]):
        if not g["enabled"]:
            continue
        faces = np.array([f for f in g["faces"] if 0 <= f < F], dtype=np.int64)
        tags[faces] |= np.int64(1) << np.int64(bit)
        if g["lock"] > 0:
            verts = {v for v in g["verts"] if 0 <= v < n}
            if g["lock_faces"]:
                for f in faces:
                    verts.update(mesh.face_idx[mesh.face_ptr[f]:mesh.face_ptr[f + 1]].tolist())
            idx = np.array(sorted(verts), dtype=np.int64)
            lock[idx] = np.maximum(lock[idx], g["lock"])
    mesh.fattr["tags"] = tags
    if np.any(lock):
        mesh.vattr["lock"] = lock


def apply_group_rules(design, mesh, level: int, W: dict) -> dict:
    groups = [(bit, g) for bit, g in enumerate(design.groups[:MAX_GROUPS])
              if g["enabled"] and any(r["from"] <= level + 1 <= r["to"] for r in g["rules"])]
    if not groups or "tags" not in mesh.fattr:
        return W
    F = mesh.n_faces
    W = {n: np.broadcast_to(np.asarray(v, float), (F,)).copy() for n, v in W.items()}
    tags = mesh.fattr["tags"]
    for bit, g in groups:
        sel = (tags >> np.int64(bit)) & 1 == 1
        if not np.any(sel):
            continue
        for r in g["rules"]:
            if not (r["from"] <= level + 1 <= r["to"]) or r["weight"] not in W:
                continue
            w = W[r["weight"]]
            w[sel] = w[sel] * r["value"] if r["op"] == "scale" else w[sel] + r["value"]
    return W


def group_signature(design, level: int):
    out = []
    for g in design.groups:
        if g["enabled"]:
            out.append([r for r in g["rules"] if r["from"] <= level + 1 <= r["to"]])
    return out if any(out) else None


def base_signature(design):
    """What groups contribute to the decorated input mesh (tags and locks)."""
    return [[g["faces"], g["verts"], g["lock"], g["lock_faces"], g["enabled"]] for g in design.groups] or None


def signature(design, level: int):
    rules = _active_rules(design, level)
    motifs = {k: v for k, v in design.motifs.items() if float(v) != 0.0}
    return {"rules": rules, "motifs": motifs} if rules or motifs else None


# --------------------------------------------------------- selection helpers
AXIS_VECTORS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def select_by_normal(mesh, direction: str, max_angle_deg: float = 30.0) -> np.ndarray:
    d = np.array(AXIS_VECTORS[direction], float)
    return np.nonzero(mesh.face_normal @ d >= np.cos(np.radians(max_angle_deg)))[0]


def select_by_height(mesh, axis: int, lo: float, hi: float, faces: bool = True) -> np.ndarray:
    """Faces (or vertices) whose centroid lies in the [lo, hi] band (fractions of the bounding box)."""
    P = mesh.face_centroid if faces else mesh.V
    vmin, vmax = mesh.V[:, axis].min(), mesh.V[:, axis].max()
    t = (P[:, axis] - vmin) / max(vmax - vmin, 1e-12)
    return np.nonzero((t >= lo - 1e-9) & (t <= hi + 1e-9))[0]


def select_every_kth(n: int, k: int, start: int = 0) -> np.ndarray:
    return np.arange(start % max(k, 1), n, max(k, 1))


def select_by_motif(mesh, label: str) -> np.ndarray:
    return np.nonzero(mesh.motif_codes() == parse_label(label))[0]


def select_edge_verts(mesh, edge: int) -> np.ndarray:
    return mesh.edge_verts[edge]

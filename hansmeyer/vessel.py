"""Vessel: a thin-walled lithophane sphere, smooth outside, the subdivision relief inside (our extension).

The subdivided form (usually the "Sphere with opening" base) becomes the *inner* face of a shell:

  * the outer face is an exact sphere of the given diameter, through every vertex's rest direction
    (where it sat on the input mesh), so it stays perfectly smooth whatever the schedule does
  * the form's relief - how far the schedule's weights moved each vertex in or out, measured against the
    same schedule run with zero weights (plain subdivision) - becomes the wall: the outermost `glow` share
    of the relief is pressed against the shell at `min_wall_mm` (windows that glow when lit from inside) and
    everything else is that much thicker, at true proportion times `depth`, up to `max_wall_mm`. Weight
    changes act on the vessel as they do on any form
  * every boundary loop of the form gets a rim joining the two faces; the top opening's rim is made an
    exact, flat circle of `rim_mm` wall, so the vessel can be printed upside down standing on it

The result is one closed, consistently oriented mesh with no self-intersections (inner and outer
faces share their directions), so it can be exported exactly, without a voxel remesh.
Wall thicknesses are in millimetres at the vessel's printed diameter.
"""

from __future__ import annotations

import copy

import numpy as np

from .mesh import PolyMesh

DEFAULT_VESSEL = {
    "enabled": False,
    "diameter_mm": 120.0,  # printed outer diameter
    "min_wall_mm": 0.8,  # thinnest wall: where the relief reaches out furthest (brightest)
    "max_wall_mm": 12.0,  # deepest the relief may reach into the sphere (a cap on the wall)
    "depth": 1.0,  # relief depth: 1 = the form's true proportions, 2 = twice as deep
    "glow": 12.0,  # % of the inner surface pressed against the thinnest wall: the glowing windows
    "rim_mm": 2.5,  # wall at the opening: a solid ring to print standing on
    "rim_band": 10.0,  # degrees over which the relief fades into the rim
    "invert": False,  # turn the relief inside out (swap bright and dark)
}


def normalize(v: dict | None) -> dict:
    out = copy.deepcopy(DEFAULT_VESSEL)
    out.update({k: val for k, val in (v or {}).items() if k in DEFAULT_VESSEL})
    out["min_wall_mm"] = max(float(out["min_wall_mm"]), 0.1)
    out["max_wall_mm"] = max(float(out["max_wall_mm"]), out["min_wall_mm"])
    out["diameter_mm"] = max(float(out["diameter_mm"]), 1.0)
    return out


def active(design) -> bool:
    return bool(design.vessel.get("enabled"))


def _boundary_loops(mesh: PolyMesh) -> list[np.ndarray]:
    """Boundary half-edges grouped into closed loops (in face orientation)."""
    hb = np.nonzero(mesh.he_twin < 0)[0]
    if not len(hb):
        return []
    start = {int(mesh.he_from[h]): int(h) for h in hb}
    seen, loops = set(), []
    for h0 in hb:
        h0 = int(h0)
        if h0 in seen:
            continue
        loop, h = [], h0
        while h not in seen:
            seen.add(h)
            loop.append(h)
            h = start.get(int(mesh.he_to[h]))
            if h is None:  # open chain (pinched boundary): stop here
                break
        loops.append(np.array(loop))
    return loops


def _smooth(values: np.ndarray, mesh: PolyMesh, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return values
    from scipy.sparse import coo_matrix

    a, b = mesh.he_from, mesh.he_to
    n = mesh.n_verts
    A = coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    for _ in range(iterations):
        values = 0.5 * values + 0.5 * (A @ values) / deg
    return values


def plain_design(design):
    """The same input and schedule with all weights zero and no fields: plain subdivision. Its vertices
    match the form's one to one (merging aside), so the difference is exactly what the weights did."""
    from .schedule import Design, IterationSpec

    return Design(base=dict(design.base), extrusion=design.extrusion, boundary=design.boundary,
                  fade_rows=design.fade_rows, iterations=[IterationSpec(it.scheme, {}) for it in design.iterations],
                  groups=[dict(g) for g in design.groups])


def relief_height(relief: PolyMesh, plain: PolyMesh | None = None) -> np.ndarray:
    """How far (model units, + outward) each vertex sits from the plain subdivision, along its rest direction."""
    rest = relief.vattr.get("rest", relief.V)
    d = rest / np.maximum(np.linalg.norm(rest, axis=1, keepdims=True), 1e-12)
    h = np.einsum("ij,ij->i", relief.V, d)
    if plain is not None and plain.n_verts == relief.n_verts:
        return h - np.einsum("ij,ij->i", plain.V, d)
    return h - _smooth(h, relief, 40)  # topology changed (vertex merging): use the smoothed form instead


def wall_thickness(design, relief: PolyMesh, plain: PolyMesh | None = None) -> np.ndarray:
    """Wall thickness (mm) at every vertex of the relief, before the rim blend."""
    v = normalize(design.vessel)
    h = relief_height(relief, plain)
    if v["invert"]:
        h = -h
    top = np.percentile(h, 100.0 - min(max(float(v["glow"]), 0.5), 90.0))  # everything above is a window
    t = v["min_wall_mm"] + float(v["depth"]) * (top - h) * 0.5 * v["diameter_mm"]
    return np.clip(t, v["min_wall_mm"], v["max_wall_mm"])


def build(design, relief: PolyMesh, plain: PolyMesh | None = None) -> PolyMesh:
    v = normalize(design.vessel)
    unit = 2.0 / v["diameter_mm"]  # model units per mm: the outer sphere has radius 1
    rest = relief.vattr.get("rest", relief.V)
    d = rest / np.maximum(np.linalg.norm(rest, axis=1, keepdims=True), 1e-12)
    t = wall_thickness(design, relief, plain)

    loops = _boundary_loops(relief)
    top = max(loops, key=lambda lp: d[relief.he_from[lp], 1].mean()) if loops else None
    if top is not None:
        rim_v = relief.he_from[top]
        theta_rim = float(np.arccos(np.clip(d[rim_v, 1], -1, 1)).mean())
        if design.base.get("shape") == "sphere_open":  # the exact opening of the input sphere
            theta_rim = float(np.radians(min(max(float(design.base.get("opening", 40.0)), 5.0), 150.0)))
        theta = np.arccos(np.clip(d[:, 1], -1, 1))
        b = np.clip((theta - theta_rim) / np.radians(max(v["rim_band"], 0.1)), 0.0, 1.0)
        b = b * b * (3 - 2 * b)
        t = v["rim_mm"] + b * (t - v["rim_mm"])
    t = np.minimum(t, 0.95 / unit)  # never through the centre

    n = relief.n_verts
    outer = d.copy()
    inner = d * (1.0 - t * unit)[:, None]
    if top is not None:  # an exact, flat, circular rim to stand on
        ang = np.arctan2(d[rim_v, 2], d[rim_v, 0])
        s, c = np.sin(theta_rim), np.cos(theta_rim)
        ring = np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], 1)
        outer[rim_v] = ring * s + [0.0, c, 0.0]
        inner[rim_v] = ring * max(s - v["rim_mm"] * unit, 1e-3) + [0.0, c, 0.0]

    # faces: outer as the form, inner reversed, and a quad strip joining them along every boundary loop
    ptr, idx = relief.face_ptr, relief.face_idx
    size = np.diff(ptr)
    fid = np.repeat(np.arange(len(size)), size)
    rev = idx[2 * ptr[fid] + size[fid] - 1 - np.arange(len(idx))] + n
    rims = []
    for lp in loops:
        a, bb = relief.he_from[lp], relief.he_to[lp]
        rims.append(np.stack([bb, a, a + n, bb + n], 1))
    rims = np.concatenate(rims) if rims else np.zeros((0, 4), np.int64)
    face_idx = np.concatenate([idx, rev, rims.reshape(-1)])
    face_ptr = np.concatenate([ptr, ptr[-1] + ptr[1:], 2 * ptr[-1] + 4 * np.arange(1, len(rims) + 1)])
    V = np.vstack([outer, inner])
    m = PolyMesh(V, face_ptr, face_idx)
    m.vattr["rest"] = np.vstack([rest, rest])
    m.vattr["wall_mm"] = np.concatenate([t, t])
    m.vattr["vessel_outer"] = np.r_[np.ones(n), np.zeros(n)]
    m.info["vessel"] = {"diameter_mm": v["diameter_mm"], "opening_deg": float(np.degrees(theta_rim)) if top is not None
                        else None, "wall_mm": (float(t.min()), float(t.max())), "loops": len(loops)}
    return m


def light(mesh: PolyMesh, absorb: float = 1.0) -> np.ndarray:
    """Per-face brightness (0..1) of the lit vessel seen from outside: light through PLA falls off
    roughly as exp(-absorb * thickness_mm), normalised between the thickest and thinnest wall."""
    w = mesh.vattr["wall_mm"]
    tf = mesh.face_reduce(w[mesh.face_idx]) / mesh.face_size
    lo, hi = float(w.min()), float(w.max())
    i = np.exp(-absorb * tf)
    return np.clip((i - np.exp(-absorb * hi)) / max(np.exp(-absorb * lo) - np.exp(-absorb * hi), 1e-12), 0, 1)

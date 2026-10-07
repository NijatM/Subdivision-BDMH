"""Extended Catmull-Clark subdivision (Hansmeyer 2010, eq. 1-4).

Weights may be scalars or per-face arrays (non-uniform weights). Edge weights are
the mean of the two adjacent faces, vertex weights the mean of incident faces.
With all weights at zero this is exactly standard Catmull-Clark (incl. the
standard boundary rules), which the tests verify against a reference.

Boundaries: by default the standard (smooth) boundary rules apply. With
``lock_boundary`` boundary vertices stay fixed and boundary edge points are
plain midpoints without extrusion, so the boundary polyline never changes and
identical open panels tile seamlessly. ``vmask`` (per-vertex, 0..1) scales all
extrusion — used to fade relief out towards a locked boundary.

Locking (paper Fig. 9): vertices in ``vlock`` keep their position; an edge with both
ends locked gets its plain midpoint and a face with all corners locked its centroid,
so locked edges stay sharp creases. Motif attraction (eq. 10-11): with a per-vertex
attractor/deflector value ``vU``, face and edge points are pulled toward (U > 0) or
pushed away from (U < 0) their vertices, scaled by w6 / w7.

Extension beyond the paper: eq. 1 and 3 are applied to n-gons and arbitrary
valence (the paper states them for quads); eq. 4 applies to quads that have the
corner/edge/face/edge provenance pattern produced by a previous CC step.
"""

from __future__ import annotations

import numpy as np

from .mesh import FCLASS_CC, VTYPE_CORNER, VTYPE_EDGE, VTYPE_FACE, PolyMesh, Topology, scatter_add


def _per_face(W: dict, name: str, n: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(W.get(name, 0.0), dtype=float), (n,))


def face_points(m: PolyMesh, W: dict) -> np.ndarray:
    """Eq. 1 (centroid) and, where the provenance pattern allows, eq. 4 — before extrusion."""
    F = m.n_faces
    Fp = m.face_centroid.copy()
    w3 = _per_face(W, "w3", F)
    w4 = _per_face(W, "w4", F)
    if not (np.any(w3) or np.any(w4)):
        return Fp
    quads = np.nonzero(m.face_size == 4)[0]
    if len(quads) == 0:
        return Fp
    verts = m.face_idx[m.face_ptr[quads][:, None] + np.arange(4)]
    t = m.vtype[verts]
    for r in range(4):
        a, b, c, d = r, (r + 1) % 4, (r + 2) % 4, (r + 3) % 4
        ok = (t[:, a] == VTYPE_CORNER) & (t[:, c] == VTYPE_FACE) & (t[:, b] == VTYPE_EDGE) & (t[:, d] == VTYPE_EDGE)
        if not np.any(ok):
            continue
        q = quads[ok]
        vv = verts[ok]
        P_v, P_e1, P_f, P_e2 = (m.V[vv[:, i]] for i in (a, b, c, d))
        w3q, w4q = w3[q][:, None], w4[q][:, None]
        Fp[q] = ((P_v * (1 + w3q) + P_f * (1 - w3q)) * (1 + w4q) + (P_e1 + P_e2) * (1 - w4q)) / 4.0
    return Fp


def subdivide(
    m: PolyMesh,
    W: dict,
    relative: bool = True,
    lock_boundary: bool = False,
    vmask: np.ndarray | None = None,
    vlock: np.ndarray | None = None,
    vU: np.ndarray | None = None,
) -> PolyMesh:
    N, E, F = m.n_verts, m.n_edges, m.n_faces
    V = m.V
    ev = m.edge_verts
    if vlock is not None and not np.any(vlock):
        vlock = None
    if vlock is not None:
        vlock = np.asarray(vlock, bool)
        lock_e = vlock[ev[:, 0]] & vlock[ev[:, 1]]
        lock_f = np.minimum.reduceat(vlock[m.face_idx].astype(np.int8), m.face_ptr[:-1]).astype(bool)
    if vmask is None:
        mask_f = mask_e = mask_v = None
    else:
        mask_v = np.asarray(vmask, dtype=float)
        mask_f = m.face_reduce(mask_v[m.face_idx]) / m.face_size
        mask_e = 0.5 * (mask_v[ev[:, 0]] + mask_v[ev[:, 1]])

    w_f = _per_face(W, "w_f", F)
    w_e_f = _per_face(W, "w_e", F)
    w_c_f = _per_face(W, "w_c", F)
    w1_f = _per_face(W, "w1", F)
    w2_f = _per_face(W, "w2", F)

    scale_f = m.face_scale if relative else np.ones(F)

    # ---- eq. 1 / 4: face points
    Fp = face_points(m, W)
    if np.any(w_f):
        amount = w_f * scale_f if mask_f is None else w_f * scale_f * mask_f
        Fp += m.face_normal * amount[:, None]
    if vlock is not None:
        Fp[lock_f] = m.face_centroid[lock_f]

    # ---- eq. 2: edge points
    ef = m.edge_faces
    Pa, Pb = V[ev[:, 0]], V[ev[:, 1]]
    w1 = m.edge_mean(w1_f)[:, None]
    inner = ef[:, 1] >= 0
    Ep = 0.5 * (Pa + Pb)  # boundary rule: edge midpoint
    Fsum = Fp[ef[inner, 0]] + Fp[ef[inner, 1]]
    Ep[inner] = (Fsum * (1 + w1[inner]) + (Pa[inner] + Pb[inner]) * (1 - w1[inner])) / 4.0
    w_e = m.edge_mean(w_e_f)
    if np.any(w_e):
        amount = w_e * m.edge_mean(scale_f)
        if mask_e is not None:
            amount = amount * mask_e
        if lock_boundary:
            amount = np.where(inner, amount, 0.0)
        Ep += m.edge_normal * amount[:, None]
    if vlock is not None:
        Ep[lock_e] = 0.5 * (Pa[lock_e] + Pb[lock_e])  # locked edges stay straight creases

    # ---- eq. 3: corner points
    val = m.valence.astype(float)
    Fbar = m.vert_mean(Fp)
    mid = 0.5 * (Pa + Pb)
    Esum = scatter_add(ev.T.reshape(-1), np.concatenate([mid, mid]), N)
    Ebar = Esum / np.maximum(val, 1)[:, None]
    w2 = m.vert_mean(w2_f)[:, None]
    i = val[:, None]
    Cp = (Fbar * (1 + w2) + 2 * Ebar * (1 - w2 / 2) + (i - 3) * V) / np.maximum(i, 1)

    bnd = m.vert_is_boundary
    if np.any(bnd):
        # standard boundary rule: cubic B-spline along the boundary, (P- + 6P + P+) / 8
        be = ev[m.edge_is_boundary]
        nb_sum = scatter_add(be.T.reshape(-1), np.concatenate([V[be[:, 1]], V[be[:, 0]]]), N)
        nb_cnt = np.bincount(be.ravel(), minlength=N)
        regular = bnd & (nb_cnt == 2) & (not lock_boundary)
        Cp[regular] = (nb_sum[regular] + 6 * V[regular]) / 8.0
        fixed = bnd & ~regular  # locked boundary, corners, bow-ties: stay put
        Cp[fixed] = V[fixed]
    w_c = m.vert_mean(w_c_f)
    if np.any(w_c):
        amount = w_c * m.vert_mean(scale_f)
        if mask_v is not None:
            amount = amount * mask_v
        if lock_boundary:
            amount = np.where(bnd, 0.0, amount)
        Cp += m.vert_normal * amount[:, None]
    if vlock is not None:
        Cp[vlock] = V[vlock]

    # ---- eq. 10 / 11: motif attraction, applied after eq. 1-2
    if vU is not None:
        U = np.asarray(vU, float)
        w6 = _per_face(W, "w6", F)
        w7 = m.edge_mean(_per_face(W, "w7", F))
        if np.any(w6):
            pull = m.face_reduce((V[m.he_from] - Fp[m.he_face]) * U[m.he_from][:, None])
            att = w6[:, None] * pull
            if vlock is not None:
                att[lock_f] = 0.0
            Fp = Fp + att
        if np.any(w7):
            att = w7[:, None] * ((Pa - Ep) * U[ev[:, 0]][:, None] + (Pb - Ep) * U[ev[:, 1]][:, None])
            if vlock is not None:
                att[lock_e] = 0.0
            Ep = Ep + att

    # ---- topology: one quad per half-edge (the same for every run from the same input: built once)
    topo = m.topo.child("cc", lambda: child_topology(m))
    vtype, fclass = topo.extra
    return PolyMesh(
        np.concatenate([Cp, Ep, Fp]),
        topo.face_ptr,
        topo.face_idx,
        vtype=vtype,
        fclass=fclass,
        vattr={k: propagate_attr(m, a, k) for k, a in m.vattr.items()},
        fattr={k: a[m.he_face] for k, a in m.fattr.items()},  # each child quad inherits its parent face
        topo=topo,
    )


def child_topology(m: PolyMesh) -> Topology:
    """Connectivity of the subdivided mesh (corners, then edge points, then face points); extra = (vtype, fclass)."""
    N, E, F, H = m.n_verts, m.n_edges, m.n_faces, m.n_halfedges
    quads = np.stack([m.he_from, N + m.he_edge, N + E + m.he_face, N + m.he_edge[m.he_prev]], axis=1)
    vtype = np.concatenate([np.full(N, VTYPE_CORNER), np.full(E, VTYPE_EDGE), np.full(F, VTYPE_FACE)]).astype(np.int8)
    return Topology(np.arange(0, 4 * H + 1, 4, dtype=np.int64), quads.reshape(-1), N + E + F,
                    extra=(vtype, np.full(H, FCLASS_CC, dtype=np.int8)))


def propagate_attr(m: PolyMesh, a: np.ndarray, name: str = "") -> np.ndarray:
    """Carry a per-vertex attribute to the next level by plain (unweighted) averaging:
    corners keep their value, edge points take the endpoint mean, face points the face mean.

    Special attributes: "tv" (original vertex) only survives on corners; "te" (on an
    original edge) also on edge points of such edges; "lock" (iterations to stay
    locked) passes to edge/face points only if all their parents are locked."""
    ev = m.edge_verts
    if name == "tv":
        return np.concatenate([a, np.zeros(m.n_edges), np.zeros(m.n_faces)])
    if name == "te":
        return np.concatenate([a, a[ev[:, 0]] * a[ev[:, 1]], np.zeros(m.n_faces)])
    if name == "lock":
        edge = np.where((a[ev[:, 0]] > 0) & (a[ev[:, 1]] > 0), np.minimum(a[ev[:, 0]], a[ev[:, 1]]), 0.0)
        face = np.minimum.reduceat(a[m.face_idx], m.face_ptr[:-1])
        return np.concatenate([a, edge, face])
    shape = (-1,) + (1,) * (a.ndim - 1)
    edge = 0.5 * (a[ev[:, 0]] + a[ev[:, 1]])
    face = m.face_reduce(a[m.face_idx]) / m.face_size.reshape(shape)
    return np.concatenate([a, edge, face])


def predicted_faces(m: PolyMesh) -> int:
    return m.n_halfedges

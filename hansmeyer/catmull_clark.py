"""Extended Catmull-Clark subdivision (Hansmeyer 2010, eq. 1-4).

Weights may be scalars or per-face arrays (non-uniform weights). Edge weights are
the mean of the two adjacent faces, vertex weights the mean of incident faces.
With all weights at zero this is exactly standard Catmull-Clark (incl. the
standard boundary rules), which the tests verify against a reference.

Extension beyond the paper: eq. 1 and 3 are applied to n-gons and arbitrary
valence (the paper states them for quads); eq. 4 applies to quads that have the
corner/edge/face/edge provenance pattern produced by a previous CC step.
"""

from __future__ import annotations

import numpy as np

from .mesh import FCLASS_CC, VTYPE_CORNER, VTYPE_EDGE, VTYPE_FACE, PolyMesh


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


def subdivide(m: PolyMesh, W: dict, relative: bool = True) -> PolyMesh:
    N, E, F = m.n_verts, m.n_edges, m.n_faces
    V = m.V

    w_f = _per_face(W, "w_f", F)
    w_e_f = _per_face(W, "w_e", F)
    w_c_f = _per_face(W, "w_c", F)
    w1_f = _per_face(W, "w1", F)
    w2_f = _per_face(W, "w2", F)

    scale_f = m.face_scale if relative else np.ones(F)

    # ---- eq. 1 / 4: face points
    Fp = face_points(m, W)
    if np.any(w_f):
        Fp += m.face_normal * (w_f * scale_f)[:, None]

    # ---- eq. 2: edge points
    ev, ef = m.edge_verts, m.edge_faces
    Pa, Pb = V[ev[:, 0]], V[ev[:, 1]]
    w1 = m.edge_mean(w1_f)[:, None]
    inner = ef[:, 1] >= 0
    Ep = 0.5 * (Pa + Pb)  # boundary rule: edge midpoint
    Fsum = Fp[ef[inner, 0]] + Fp[ef[inner, 1]]
    Ep[inner] = (Fsum * (1 + w1[inner]) + (Pa[inner] + Pb[inner]) * (1 - w1[inner])) / 4.0
    w_e = m.edge_mean(w_e_f)
    if np.any(w_e):
        scale_e = m.edge_mean(scale_f)
        Ep += m.edge_normal * (w_e * scale_e)[:, None]

    # ---- eq. 3: corner points
    val = m.valence.astype(float)
    Fbar = m.vert_mean(Fp)
    mid = 0.5 * (Pa + Pb)
    Esum = np.zeros_like(V)
    np.add.at(Esum, ev[:, 0], mid)
    np.add.at(Esum, ev[:, 1], mid)
    Ebar = Esum / np.maximum(val, 1)[:, None]
    w2 = m.vert_mean(w2_f)[:, None]
    i = val[:, None]
    Cp = (Fbar * (1 + w2) + 2 * Ebar * (1 - w2 / 2) + (i - 3) * V) / np.maximum(i, 1)

    bnd = m.vert_is_boundary
    if np.any(bnd):
        # standard boundary rule: cubic B-spline along the boundary, (P- + 6P + P+) / 8
        be = ev[m.edge_is_boundary]
        nb_sum = np.zeros_like(V)
        nb_cnt = np.zeros(N)
        np.add.at(nb_sum, be[:, 0], V[be[:, 1]])
        np.add.at(nb_sum, be[:, 1], V[be[:, 0]])
        np.add.at(nb_cnt, be.ravel(), 1)
        regular = bnd & (nb_cnt == 2)
        Cp[regular] = (nb_sum[regular] + 6 * V[regular]) / 8.0
        Cp[bnd & ~regular] = V[bnd & ~regular]  # corners / bow-ties stay put
    w_c = m.vert_mean(w_c_f)
    if np.any(w_c):
        scale_v = m.vert_mean(scale_f)
        Cp += m.vert_normal * (w_c * scale_v)[:, None]

    # ---- topology: one quad per half-edge
    h = np.arange(m.n_halfedges)
    quads = np.stack(
        [m.he_from[h], N + m.he_edge[h], N + E + m.he_face[h], N + m.he_edge[m.he_prev[h]]],
        axis=1,
    )
    newV = np.concatenate([Cp, Ep, Fp])
    vtype = np.concatenate(
        [np.full(N, VTYPE_CORNER), np.full(E, VTYPE_EDGE), np.full(F, VTYPE_FACE)]
    ).astype(np.int8)
    H = m.n_halfedges
    return PolyMesh(
        newV,
        np.arange(0, 4 * H + 1, 4, dtype=np.int64),
        quads.reshape(-1),
        vtype=vtype,
        fclass=np.full(H, FCLASS_CC, dtype=np.int8),
    )


def predicted_faces(m: PolyMesh) -> int:
    return m.n_halfedges

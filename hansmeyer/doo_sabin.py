"""Extended Doo-Sabin subdivision (Hansmeyer 2010, eq. 5-6).

Each face's corner i gets a new point  sum_d alpha_d(w1) * P_(i+d)  + n_f * w_f.
The standard Doo-Sabin masks are
    alpha_0 = (k+5)/(4k),   alpha_d = (3 + 2 cos(2 pi d / k)) / (4k).
w1 moves  w1 * (k-2)/k  of weight from the two adjacent vertices onto the
corner's own vertex. For k = 4 and k = 3 this reproduces the paper's eq. 5 and 6
exactly; other k is our generalisation.

Faces are classed as face-, edge- or vertex-derived after a DS step, and each
class can carry its own (w1, w_f) — as the paper proposes.

Open meshes: Doo-Sabin has no boundary rule; boundary edges and vertices get
no E-/V-faces, so the boundary recedes. It therefore cannot keep a locked
(tileable) boundary — the app warns about that combination.
"""

from __future__ import annotations

import numpy as np

from .mesh import (
    FCLASS_DS_EDGE,
    FCLASS_DS_FACE,
    FCLASS_DS_VERT,
    VTYPE_DS,
    PolyMesh,
)


def ds_alphas(k: int, w1: np.ndarray) -> np.ndarray:
    """(n, k) mask weights for faces of size k; column d multiplies the vertex d steps ahead."""
    d = np.arange(k)
    base = (3 + 2 * np.cos(2 * np.pi * d / k)) / (4 * k)
    base[0] = (k + 5) / (4 * k)
    a = np.tile(base, (len(w1), 1))
    shift = w1 * (k - 2) / k
    a[:, 0] += shift
    a[:, 1] -= shift / 2
    a[:, k - 1] -= shift / 2
    return a


def _class_weights(m: PolyMesh, W: dict) -> tuple[np.ndarray, np.ndarray]:
    F = m.n_faces

    def get(name):
        return np.broadcast_to(np.asarray(W.get(name, 0.0), dtype=float), (F,))

    fc = m.fclass
    is_e, is_v = fc == FCLASS_DS_EDGE, fc == FCLASS_DS_VERT
    w1 = np.where(is_e, get("ds_w1_edge"), np.where(is_v, get("ds_w1_vert"), get("ds_w1_face")))
    wf = np.where(is_e, get("ds_wf_edge"), np.where(is_v, get("ds_wf_vert"), get("ds_wf_face")))
    return w1, wf


def _vertex_fans(m: PolyMesh) -> list[np.ndarray]:
    """Closed fans of outgoing half-edges around interior vertices, grouped by size.

    rot(h) = twin(prev(h)) steps counter-clockwise around the vertex, so each fan
    lists the corners in the order of a correctly oriented polygon.
    """
    twin, prev = m.he_twin, m.he_prev
    interior = ~m.vert_is_boundary
    start = np.full(m.n_verts, -1, dtype=np.int64)
    # first outgoing half-edge per vertex
    order = np.argsort(m.he_from, kind="stable")
    first = np.ones(len(order), dtype=bool)
    first[1:] = m.he_from[order][1:] != m.he_from[order][:-1]
    start[m.he_from[order][first]] = order[first]
    verts = np.nonzero(interior & (start >= 0))[0]
    counts = m.vert_face_count[verts]
    fans = []
    for k in np.unique(counts):
        if k < 3:
            continue
        vs = verts[counts == k]
        cols = [start[vs]]
        for _ in range(k - 1):
            cols.append(np.where(cols[-1] >= 0, twin[prev[np.maximum(cols[-1], 0)]], -1))
        fan = np.stack(cols, axis=1)
        # keep only single closed fans (skips bow-tie / non-manifold vertices)
        last = fan[:, -1]
        closes = np.all(fan >= 0, axis=1) & (twin[prev[np.maximum(last, 0)]] == fan[:, 0])
        fans.append(fan[closes])
    return [f for f in fans if len(f)]


def corner_values(m: PolyMesh, X: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """Apply the (extended) Doo-Sabin masks to a per-vertex array X -> one value per half-edge."""
    H = m.n_halfedges
    out_all = np.empty((H,) + X.shape[1:])
    sizes = m.face_size
    for k in np.unique(sizes):
        sel = np.nonzero(sizes == k)[0]
        slots = m.face_ptr[sel][:, None] + np.arange(k)  # (n, k) half-edge ids
        P = X[m.face_idx[slots]]  # (n, k, ...)
        a = ds_alphas(k, w1[sel])  # (n, k)
        a = a.reshape(a.shape + (1,) * (P.ndim - 2))
        out = np.zeros_like(P)
        for d in range(k):
            out += a[:, d:d + 1] * np.roll(P, -d, axis=1)
        out_all[slots] = out
    return out_all


def subdivide(
    m: PolyMesh,
    W: dict,
    relative: bool = True,
    lock_boundary: bool = False,
    vmask: np.ndarray | None = None,
    vlock: np.ndarray | None = None,
    vU: np.ndarray | None = None,
) -> PolyMesh:
    """Locks and motif values are Catmull-Clark features; Doo-Sabin ignores them but
    passes each vertex's lock on to the corners cut from it."""
    H = m.n_halfedges
    w1, wf = _class_weights(m, W)
    scale_f = m.face_scale if relative else np.ones(m.n_faces)
    sizes = m.face_size

    # ---- new corner points: one per half-edge (face corner)
    newV = corner_values(m, m.V, w1)
    if np.any(wf):
        amount = wf * scale_f
        if vmask is not None:
            amount = amount * (m.face_reduce(np.asarray(vmask, float)[m.face_idx]) / sizes)
        newV += (m.face_normal * amount[:, None])[m.he_face]
    zero = np.zeros(m.n_faces)
    vattr = {}
    for k, a in m.vattr.items():
        if k == "lock":
            vattr[k] = a[m.he_from]
        elif k in ("tv", "te"):
            vattr[k] = np.zeros(H)
        else:
            vattr[k] = corner_values(m, a, zero)

    # ---- faces: F-faces, E-faces (interior edges), V-faces (interior vertices)
    nxt, twin = m.he_next, m.he_twin
    eh = m.edge_he
    eh = eh[twin[eh] >= 0]
    et = twin[eh]
    e_quads = np.stack([nxt[et], et, nxt[eh], eh], axis=1)
    fans = _vertex_fans(m)

    parts_idx = [np.arange(H), e_quads.reshape(-1)] + [f.reshape(-1) for f in fans]
    parts_size = [sizes, np.full(len(e_quads), 4)] + [np.full(len(f), f.shape[1]) for f in fans]
    face_idx = np.concatenate(parts_idx)
    face_size = np.concatenate(parts_size)
    face_ptr = np.concatenate([[0], np.cumsum(face_size)])
    n_v = sum(len(f) for f in fans)
    fclass = np.concatenate(
        [
            np.full(m.n_faces, FCLASS_DS_FACE),
            np.full(len(e_quads), FCLASS_DS_EDGE),
            np.full(n_v, FCLASS_DS_VERT),
        ]
    ).astype(np.int8)
    fattr = {}
    for k, a in m.fattr.items():
        combine = np.bitwise_or if np.issubdtype(a.dtype, np.integer) else np.maximum
        parts = [a, combine(a[m.he_face[eh]], a[m.he_face[et]])]
        for fan in fans:
            parts.append(combine.reduce(a[m.he_face[fan]], axis=1))
        fattr[k] = np.concatenate(parts)
    return PolyMesh(newV, face_ptr, face_idx, vtype=np.full(H, VTYPE_DS, dtype=np.int8), fclass=fclass,
                    vattr=vattr, fattr=fattr)


def predicted_faces(m: PolyMesh) -> int:
    interior_edges = int(np.sum(~m.edge_is_boundary))
    interior_verts = int(np.sum(~m.vert_is_boundary))
    return m.n_faces + interior_edges + interior_verts

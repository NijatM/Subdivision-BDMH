"""Vertex merging and porosity (Hansmeyer 2010, discussion of motifs):

  "if two vertices produced in the exact same position are treated as a single vertex,
   then the topology of the mesh changes ... These constellations can be intentionally
   invoked by specifying a minimum distance beneath which proximate vertices are joined.
   ... when limiting the joined vertex's maximum valence: as edges and corresponding
   faces cannot form, the mesh assumes a distinct porosity."

After an iteration, pairs of vertices closer than `distance` are welded — but only
vertices that do not already share a face, i.e. parts of the surface that have grown
into contact (welding neighbours would just collapse edges). With a relative distance
the threshold of a pair is distance x the smaller of the two local edge lengths, so
dense, crumpled regions are not swallowed. Each vertex joins at most one partner
(nearest pairs first), like the paper's "two vertices ... treated as a single vertex".
Faces that collapse are
removed; where a welded vertex would exceed `max_valence`, the faces that joined it are
not formed (holes). A cleanup pass then removes faces until every edge is manifold and
consistently oriented, so the next iteration can subdivide the result.
"""

from __future__ import annotations

import numpy as np

from .mesh import PolyMesh

DEFAULT_MERGE = {"enabled": False, "distance": 0.15, "relative": True, "max_valence": 0, "from": 1, "to": 10}

_OFFSETS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64)


def normalize(m: dict | None) -> dict:
    out = dict(DEFAULT_MERGE)
    out.update({k: v for k, v in (m or {}).items() if k in DEFAULT_MERGE})
    return out


def active(design, level: int) -> bool:
    m = design.merge
    return bool(m["enabled"]) and m["from"] <= level + 1 <= m["to"]


def signature(design, level: int):
    return dict(design.merge) if active(design, level) else None


def _encode(cells: np.ndarray) -> np.ndarray:
    c = cells + (1 << 20)
    return (c[..., 0] << 42) | (c[..., 1] << 21) | c[..., 2]


def close_pairs(V: np.ndarray, thr: float, cap: int = 48) -> np.ndarray:
    """Vertex pairs (i < j) closer than thr, via a uniform grid of cell size thr.
    At most `cap` points per neighbouring cell are examined (bounds the cost in dense clumps)."""
    cells = np.floor(V / thr).astype(np.int64)
    keys = _encode(cells)
    order = np.argsort(keys, kind="stable")
    skeys = keys[order]
    ukeys, start, count = np.unique(skeys, return_index=True, return_counts=True)
    pairs = []
    for off in _OFFSETS:
        nk = _encode(cells + off)
        pos = np.searchsorted(ukeys, nk)
        pos = np.minimum(pos, len(ukeys) - 1)
        hit = ukeys[pos] == nk
        if not np.any(hit):
            continue
        src = np.nonzero(hit)[0]
        st, ct = start[pos[hit]], count[pos[hit]]
        for j in range(min(int(ct.max()), cap)):
            ok = j < ct
            a, b = src[ok], order[st[ok] + j]
            keep = a < b
            pairs.append(np.stack([a[keep], b[keep]], 1))
    if not pairs:
        return np.zeros((0, 2), np.int64)
    P = np.unique(np.concatenate(pairs), axis=0)
    d = np.linalg.norm(V[P[:, 0]] - V[P[:, 1]], axis=1)
    return P[d < thr]


def _share_a_face(mesh: PolyMesh, pairs: np.ndarray) -> np.ndarray:
    """True for vertex pairs that are corners of a common face (topological neighbours)."""
    n = mesh.n_verts
    keys = []
    for k in np.unique(mesh.face_size):
        sel = np.nonzero(mesh.face_size == k)[0]
        F = mesh.face_idx[mesh.face_ptr[sel][:, None] + np.arange(k)]
        for i in range(k):
            for j in range(i + 1, k):
                a, b = np.minimum(F[:, i], F[:, j]), np.maximum(F[:, i], F[:, j])
                keys.append(a * n + b)
    return np.isin(pairs[:, 0] * n + pairs[:, 1], np.concatenate(keys))


def _match(n: int, pairs: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Greedy matching, nearest pairs first: each vertex joins at most one partner.
    Returns labels (the smaller index of each matched pair)."""
    lab = np.arange(n)
    used = np.zeros(n, bool)
    for k in np.argsort(dist, kind="stable"):
        a, b = pairs[k]
        if not used[a] and not used[b]:
            used[a] = used[b] = True
            lab[b] = lab[a] = min(a, b)
    return lab


def make_manifold(faces: list, keep: np.ndarray) -> np.ndarray:
    """Drop faces until every edge has at most one face per direction (manifold, oriented).
    `faces` only needs to contain the faces that can conflict (those touching welded vertices)."""
    keep = keep.copy()
    for _ in range(10):
        owner = {}
        bad = set()
        for fi in np.nonzero(keep)[0]:
            f = faces[fi]
            k = len(f)
            for i in range(k):
                a, b = int(f[i]), int(f[(i + 1) % k])
                if (a, b) in owner:
                    bad.add(fi)  # this direction is taken: drop the later face
                else:
                    owner[(a, b)] = fi
        if not bad:
            return keep
        keep[list(bad)] = False
    return keep


def merge_vertices(mesh: PolyMesh, distance: float, max_valence: int = 0,
                   local_scale: np.ndarray | None = None) -> tuple[PolyMesh, dict]:
    """Weld vertex pairs closer than `distance` (x the pair's smaller local_scale, if given)."""
    stats = {"merged": 0, "dropped": 0}
    if local_scale is None:
        pairs = close_pairs(mesh.V, distance)
        limit = np.full(len(pairs), distance)
    else:
        pairs = close_pairs(mesh.V, distance * float(np.percentile(local_scale, 90)))
        limit = distance * np.minimum(local_scale[pairs[:, 0]], local_scale[pairs[:, 1]])
    if len(pairs):
        d = np.linalg.norm(mesh.V[pairs[:, 0]] - mesh.V[pairs[:, 1]], axis=1)
        ok = (d < limit) & ~_share_a_face(mesh, pairs)
        pairs, d = pairs[ok], d[ok]
    if len(pairs) == 0:
        return mesh, stats
    lab = _match(mesh.n_verts, pairs, d)
    reps, inv, sizes = np.unique(lab, return_inverse=True, return_counts=True)
    stats["merged"] = int(mesh.n_verts - len(reps))

    # merged positions / attributes = cluster means (locks: the longest lock wins)
    def cmean(x):
        acc = np.zeros((len(reps),) + x.shape[1:])
        np.add.at(acc, inv, x)
        return acc / sizes.reshape((-1,) + (1,) * (x.ndim - 1))

    def cmax(x):
        acc = np.full(len(reps), -np.inf)
        np.maximum.at(acc, inv, x)
        return acc

    newV = cmean(mesh.V)
    new_vattr = {k: (cmax(a) if k == "lock" else cmean(a)) for k, a in mesh.vattr.items()}

    # only faces touching a welded vertex can change; the rest are remapped in bulk
    welded = sizes[inv] > 1  # per old vertex
    face_aff = np.add.reduceat(welded[mesh.face_idx].astype(np.int64), mesh.face_ptr[:-1]) > 0
    aff = np.nonzero(face_aff)[0]
    faces, keep = {}, {}
    for fi in aff:
        old = mesh.face_idx[mesh.face_ptr[fi]:mesh.face_ptr[fi + 1]]
        g = inv[old]
        g = g[g != np.roll(g, 1)]  # drop consecutive repeats (collapsed edges)
        faces[fi] = g
        keep[fi] = len(g) >= 3 and len(np.unique(g)) == len(g)

    if max_valence and max_valence > 0:
        edges = set()
        for fi in aff:
            if keep[fi]:
                f = faces[fi]
                for i in range(len(f)):
                    a, b = int(f[i]), int(f[(i + 1) % len(f)])
                    edges.add((min(a, b), max(a, b)))
        val = np.zeros(len(reps), np.int64)
        for a, b in edges:
            val[a] += 1
            val[b] += 1
        over = val > max_valence
        is_rep = np.zeros(mesh.n_verts, bool)
        is_rep[reps] = True
        for fi in aff:
            old = mesh.face_idx[mesh.face_ptr[fi]:mesh.face_ptr[fi + 1]]
            # the face reached an over-valence vertex through a non-representative member: not formed
            if keep[fi] and np.any(over[inv[old]] & welded[old] & ~is_rep[old]):
                keep[fi] = False

    aff_list = [faces[fi] for fi in aff]
    kmask = make_manifold(aff_list, np.array([keep[fi] for fi in aff], bool))
    stats["dropped"] = int(np.sum(~kmask))

    # assemble: untouched faces (vectorised) then the surviving affected faces
    un = np.nonzero(~face_aff)[0]
    he_un = np.repeat(~face_aff, mesh.face_size)
    idx_un = inv[mesh.face_idx[he_un]]
    kept_aff = aff[kmask]
    parts = [idx_un] + [faces[fi] for fi in kept_aff]
    fsizes = np.concatenate([mesh.face_size[un], [len(faces[fi]) for fi in kept_aff]]).astype(np.int64)
    order = np.concatenate([un, kept_aff]).astype(np.int64)
    idx = np.concatenate(parts).astype(np.int64)
    used = np.unique(idx)
    remap = -np.ones(len(reps), np.int64)
    remap[used] = np.arange(len(used))
    out = PolyMesh(
        newV[used],
        np.concatenate([[0], np.cumsum(fsizes)]),
        remap[idx],
        vtype=mesh.vtype[reps][used],
        fclass=mesh.fclass[order],
        vattr={k: v[used] for k, v in new_vattr.items()},
        fattr={k: v[order] for k, v in mesh.fattr.items()},
        info={**mesh.info, "merge": stats},
    )
    return out, stats


def maybe_merge(design, mesh: PolyMesh, level: int) -> PolyMesh:
    if not active(design, level):
        return mesh
    m = design.merge
    scale = mesh.vert_mean(mesh.face_scale) if m["relative"] else None
    out, stats = merge_vertices(mesh, max(float(m["distance"]), 1e-12), int(m["max_valence"]), scale)
    if out.n_faces < 0.5 * mesh.n_faces:  # safety net: never let merging eat the form
        return type(mesh)(mesh.V, mesh.face_ptr, mesh.face_idx, vtype=mesh.vtype, fclass=mesh.fclass,
                          vattr=mesh.vattr, fattr=mesh.fattr,
                          info={**mesh.info, "merge": {**stats, "skipped": True}})
    return out

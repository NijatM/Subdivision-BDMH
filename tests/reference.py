"""Deliberately naive, loop-based textbook Catmull-Clark and Doo-Sabin.

Written independently of the vectorised engine (dicts and loops, no shared
code) so the tests can check that zero weights give standard subdivision.
Meshes are (vertices: list[tuple], faces: list[list[int]]).
"""

from __future__ import annotations

import math


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _scale(a, s):
    return tuple(x * s for x in a)


def _avg(points):
    acc = (0.0, 0.0, 0.0)
    for p in points:
        acc = _add(acc, p)
    return _scale(acc, 1.0 / len(points))


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def catmull_clark(verts, faces):
    face_pt = [_avg([verts[v] for v in f]) for f in faces]

    edge_faces = {}
    for fi, f in enumerate(faces):
        for i, a in enumerate(f):
            b = f[(i + 1) % len(f)]
            edge_faces.setdefault(_edge_key(a, b), []).append(fi)

    edge_pt = {}
    for (a, b), fs in edge_faces.items():
        if len(fs) == 2:
            edge_pt[(a, b)] = _avg([verts[a], verts[b], face_pt[fs[0]], face_pt[fs[1]]])
        else:
            edge_pt[(a, b)] = _avg([verts[a], verts[b]])

    v_faces = {i: [] for i in range(len(verts))}
    for fi, f in enumerate(faces):
        for v in f:
            v_faces[v].append(fi)
    v_edges = {i: [] for i in range(len(verts))}
    for (a, b) in edge_faces:
        v_edges[a].append((a, b))
        v_edges[b].append((a, b))

    new_v = []
    for i, p in enumerate(verts):
        bnd = [e for e in v_edges[i] if len(edge_faces[e]) == 1]
        if bnd:
            if len(bnd) == 2:
                nb = [e[0] if e[1] == i else e[1] for e in bnd]
                new_v.append(_scale(_add(_add(verts[nb[0]], verts[nb[1]]), _scale(p, 6)), 1 / 8))
            else:
                new_v.append(p)
            continue
        n = len(v_edges[i])
        Fa = _avg([face_pt[f] for f in v_faces[i]])
        R = _avg([_avg([verts[a], verts[b]]) for (a, b) in v_edges[i]])
        new_v.append(_scale(_add(_add(Fa, _scale(R, 2)), _scale(p, n - 3)), 1 / n))

    out_v = list(new_v)
    e_index = {}
    for e, pt in edge_pt.items():
        e_index[e] = len(out_v)
        out_v.append(pt)
    f_index = []
    for pt in face_pt:
        f_index.append(len(out_v))
        out_v.append(pt)

    out_f = []
    for fi, f in enumerate(faces):
        k = len(f)
        for i, v in enumerate(f):
            nxt, prv = f[(i + 1) % k], f[(i - 1) % k]
            out_f.append([v, e_index[_edge_key(v, nxt)], f_index[fi], e_index[_edge_key(prv, v)]])
    return out_v, out_f


def doo_sabin(verts, faces):
    corner = {}  # (face, vertex) -> new vertex index
    out_v = []
    for fi, f in enumerate(faces):
        k = len(f)
        for i, v in enumerate(f):
            p = (0.0, 0.0, 0.0)
            for j in range(k):
                d = (j - i) % k
                if d == 0:
                    a = (k + 5) / (4 * k)
                else:
                    a = (3 + 2 * math.cos(2 * math.pi * d / k)) / (4 * k)
                p = _add(p, _scale(verts[f[j]], a))
            corner[(fi, v)] = len(out_v)
            out_v.append(p)

    out_f = [[corner[(fi, v)] for v in f] for fi, f in enumerate(faces)]

    # directed edge -> face
    dir_face = {}
    for fi, f in enumerate(faces):
        for i, a in enumerate(f):
            dir_face[(a, f[(i + 1) % len(f)])] = fi

    done = set()
    for (a, b), f1 in dir_face.items():
        if (b, a) not in dir_face or _edge_key(a, b) in done:
            continue
        done.add(_edge_key(a, b))
        f2 = dir_face[(b, a)]
        out_f.append([corner[(f2, a)], corner[(f2, b)], corner[(f1, b)], corner[(f1, a)]])

    # vertex faces: walk around each interior vertex
    for v in range(len(verts)):
        start = next(((a, b) for (a, b) in dir_face if a == v), None)
        if start is None:
            continue
        ring, he, ok = [], start, True
        while True:
            f = dir_face[he]
            ring.append(corner[(f, v)])
            # previous vertex of v in face f
            fv = faces[f]
            prev_v = fv[(fv.index(v) - 1) % len(fv)]
            nxt = (v, prev_v)
            if nxt not in dir_face:
                ok = False
                break
            he = nxt
            if he == start:
                break
        if ok and len(ring) >= 3:
            out_f.append(ring)
    return out_v, out_f

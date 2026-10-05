"""Mesh import / export: OBJ in (with cleanup), OBJ / STL / PLY out."""

from __future__ import annotations

import struct
from collections import defaultdict, deque

import numpy as np

from .mesh import PolyMesh


class MeshImportError(ValueError):
    pass


# ------------------------------------------------------------------- import
def read_obj(path: str) -> tuple[np.ndarray, list[list[int]]]:
    """Positions and polygon faces from an OBJ (v / f only; handles v/vt/vn and negative indices)."""
    verts, faces = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idx = []
                for p in parts[1:]:
                    i = int(p.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                faces.append(idx)
    if not verts or not faces:
        raise MeshImportError(f"{path}: no vertices/faces found")
    return np.array(verts, float), faces


def clean_mesh(V: np.ndarray, faces: list[list[int]], weld: bool = True):
    """Weld duplicate vertices, drop degenerate/duplicate faces and unused vertices."""
    V = np.asarray(V, float)
    if weld:
        diag = np.linalg.norm(V.max(0) - V.min(0)) or 1.0
        keys = np.round(V / (diag * 1e-7)).astype(np.int64)
        _, first, remap = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        remap = remap.reshape(-1)
        V = V[first]
        faces = [[int(remap[i]) for i in f] for f in faces]
    out, seen = [], set()
    for f in faces:
        g = [v for i, v in enumerate(f) if v != f[i - 1]]  # drop repeated consecutive indices
        if len(g) < 3 or len(set(g)) != len(g):
            continue
        key = tuple(sorted(g))
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    used = np.unique(np.concatenate([np.asarray(f) for f in out])) if out else np.zeros(0, int)
    new_index = -np.ones(len(V), dtype=np.int64)
    new_index[used] = np.arange(len(used))
    return V[used], [[int(new_index[v]) for v in f] for f in out]


def orient_faces(V: np.ndarray, faces: list[list[int]]) -> list[list[int]]:
    """Make winding consistent per connected component; closed components face outward.

    Raises MeshImportError for non-manifold edges (shared by >2 faces) or
    non-orientable surfaces (e.g. a Moebius strip)."""
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for i, a in enumerate(f):
            b = f[(i + 1) % len(f)]
            edge_faces[(min(a, b), max(a, b))].append(fi)
    bad = [e for e, fs in edge_faces.items() if len(fs) > 2]
    if bad:
        raise MeshImportError(
            f"non-manifold mesh: {len(bad)} edge(s) are shared by more than two faces "
            "(e.g. T-junction walls or touching parts) — clean it up in Rhino/Blender first"
        )

    def directed(f):
        return {(f[i], f[(i + 1) % len(f)]) for i in range(len(f))}

    faces = [list(f) for f in faces]
    done = np.zeros(len(faces), bool)
    for seed in range(len(faces)):
        if done[seed]:
            continue
        comp, queue = [seed], deque([seed])
        done[seed] = True
        while queue:
            fi = queue.popleft()
            f = faces[fi]
            for i, a in enumerate(f):
                b = f[(i + 1) % len(f)]
                for gj in edge_faces[(min(a, b), max(a, b))]:
                    if gj == fi:
                        continue
                    same_dir = (a, b) in directed(faces[gj])
                    if done[gj]:
                        if same_dir:
                            raise MeshImportError("non-orientable surface (Moebius-like) — cannot be subdivided")
                        continue
                    if same_dir:
                        faces[gj].reverse()
                    done[gj] = True
                    comp.append(gj)
                    queue.append(gj)
        # closed component -> outward (positive signed volume)
        comp_edges = defaultdict(int)
        for fi in comp:
            f = faces[fi]
            for i, a in enumerate(f):
                b = f[(i + 1) % len(f)]
                comp_edges[(min(a, b), max(a, b))] += 1
        if all(c == 2 for c in comp_edges.values()):
            vol = 0.0
            for fi in comp:
                p = V[faces[fi]]
                for i in range(1, len(p) - 1):
                    vol += np.dot(p[0], np.cross(p[i], p[i + 1]))
            if vol < 0:
                for fi in comp:
                    faces[fi].reverse()
    return faces


def load_obj(path: str, normalize: bool = True, weld: bool = True, radius: float = 3**0.5) -> PolyMesh:
    V, faces = read_obj(path)
    V, faces = clean_mesh(V, faces, weld=weld)
    if not faces:
        raise MeshImportError(f"{path}: no valid faces after cleanup")
    faces = orient_faces(V, faces)
    if normalize:
        center = 0.5 * (V.min(0) + V.max(0))
        V = V - center
        r = np.linalg.norm(V, axis=1).max()
        if r > 0:
            V = V * (radius / r)
    m = PolyMesh.from_faces(V, faces)
    _ = m.he_twin  # validate
    return m


# ------------------------------------------------------------------- export
def write_obj(path: str, m: PolyMesh) -> None:
    with open(path, "w", newline="\n") as fh:
        fh.write("# Hansmeyer subdivision engine\n")
        np.savetxt(fh, m.V, fmt="v %.6f %.6f %.6f")
        sizes = m.face_size
        for k in np.unique(sizes):
            sel = np.nonzero(sizes == k)[0]
            idx = m.face_idx[m.face_ptr[sel][:, None] + np.arange(k)] + 1
            np.savetxt(fh, idx, fmt="f" + " %d" * k)


def write_stl(path: str, m: PolyMesh) -> None:
    """Binary STL (fan-triangulated)."""
    tri = m.triangles()
    p = m.V[tri].astype(np.float32)  # (T, 3, 3)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, p
    with open(path, "wb") as fh:
        fh.write(b"Hansmeyer subdivision engine".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tri)))
        fh.write(rec.tobytes())


def write_ply(path: str, m: PolyMesh) -> None:
    """Binary little-endian PLY with the original polygons (quads stay quads)."""
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment Hansmeyer subdivision engine\n"
        f"element vertex {m.n_verts}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {m.n_faces}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(m.V.astype("<f4").tobytes())
        sizes = m.face_size
        if m.n_faces and np.all(sizes == sizes[0]):
            k = int(sizes[0])
            rec = np.zeros(m.n_faces, dtype=[("n", "u1"), ("i", "<i4", k)])
            rec["n"], rec["i"] = k, m.face_idx.reshape(-1, k)
            fh.write(rec.tobytes())
        else:
            for f in m.faces_list():
                fh.write(struct.pack("<B", len(f)) + np.asarray(f, "<i4").tobytes())


EXPORTERS = {"obj": write_obj, "stl": write_stl, "ply": write_ply}


def export(path: str, m: PolyMesh) -> None:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext not in EXPORTERS:
        raise ValueError(f"unsupported export format .{ext}")
    EXPORTERS[ext](path, m)

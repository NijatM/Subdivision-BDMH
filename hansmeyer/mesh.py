"""General polygon mesh with vectorised half-edge connectivity.

Faces are stored in CSR form: ``face_ptr`` (F+1) indexes into ``face_idx`` (H),
so face ``f`` has vertices ``face_idx[face_ptr[f]:face_ptr[f+1]]`` in
counter-clockwise order (normals point outward by the right-hand rule).
Every slot of ``face_idx`` is one half-edge going from that vertex to the next
vertex of the same face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

# Vertex provenance: how a vertex was created in the most recent iteration.
VTYPE_BASE = -1  # vertex of the input mesh
VTYPE_CORNER = 0  # Catmull-Clark corner (former vertex) point
VTYPE_EDGE = 1  # Catmull-Clark edge point
VTYPE_FACE = 2  # Catmull-Clark face point
VTYPE_DS = 3  # Doo-Sabin corner

# Face provenance: what a face was created from in the most recent iteration.
FCLASS_BASE = -1  # face of the input mesh
FCLASS_CC = 0  # Catmull-Clark sub-quad
FCLASS_DS_FACE = 1  # Doo-Sabin face derived from an old face
FCLASS_DS_EDGE = 2  # Doo-Sabin face derived from an old edge
FCLASS_DS_VERT = 3  # Doo-Sabin face derived from an old vertex

_EPS = 1e-12


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, _EPS)


@dataclass(eq=False)
class PolyMesh:
    V: np.ndarray  # (N, 3) float64 positions
    face_ptr: np.ndarray  # (F+1,) int64
    face_idx: np.ndarray  # (H,) int64
    vtype: np.ndarray = field(default=None)  # (N,) int8 provenance
    fclass: np.ndarray = field(default=None)  # (F,) int8 provenance
    vattr: dict = field(default_factory=dict)  # per-vertex attributes carried through subdivision
    fattr: dict = field(default_factory=dict)  # per-face attributes (e.g. group tag bits), inherited by child faces
    info: dict = field(default_factory=dict)  # per-level notes (e.g. merge statistics)

    def __post_init__(self):
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        self.face_ptr = np.asarray(self.face_ptr, dtype=np.int64)
        self.face_idx = np.asarray(self.face_idx, dtype=np.int64)
        if self.vtype is None:
            self.vtype = np.full(len(self.V), VTYPE_BASE, dtype=np.int8)
        if self.fclass is None:
            self.fclass = np.full(self.n_faces, FCLASS_BASE, dtype=np.int8)
        self.vattr = {k: np.asarray(v, dtype=float) for k, v in self.vattr.items()}
        self.fattr = {k: np.asarray(v) for k, v in self.fattr.items()}

    # ------------------------------------------------------------------ build
    @classmethod
    def from_faces(cls, V, faces) -> "PolyMesh":
        """Build from positions and a list of faces (lists of vertex indices) or an (F, k) array."""
        faces = [np.asarray(f, dtype=np.int64) for f in faces]
        sizes = np.array([len(f) for f in faces], dtype=np.int64)
        ptr = np.concatenate([[0], np.cumsum(sizes)])
        idx = np.concatenate(faces) if faces else np.zeros(0, np.int64)
        return cls(np.asarray(V, float), ptr, idx)

    # ------------------------------------------------------------- basic info
    @property
    def n_verts(self) -> int:
        return len(self.V)

    @property
    def n_faces(self) -> int:
        return len(self.face_ptr) - 1

    @property
    def n_halfedges(self) -> int:
        return len(self.face_idx)

    @cached_property
    def face_size(self) -> np.ndarray:
        return np.diff(self.face_ptr)

    def faces_list(self) -> list[np.ndarray]:
        return [self.face_idx[a:b] for a, b in zip(self.face_ptr[:-1], self.face_ptr[1:])]

    def is_all_quads(self) -> bool:
        return self.n_faces > 0 and bool(np.all(self.face_size == 4))

    # ------------------------------------------------------------- half-edges
    @cached_property
    def he_face(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_faces), self.face_size)

    @cached_property
    def he_next(self) -> np.ndarray:
        h = np.arange(self.n_halfedges)
        nxt = h + 1
        ends = self.face_ptr[1:] - 1
        nxt[ends] = self.face_ptr[:-1]
        return nxt

    @cached_property
    def he_prev(self) -> np.ndarray:
        prv = np.empty(self.n_halfedges, dtype=np.int64)
        prv[self.he_next] = np.arange(self.n_halfedges)
        return prv

    @property
    def he_from(self) -> np.ndarray:
        return self.face_idx

    @cached_property
    def he_to(self) -> np.ndarray:
        return self.face_idx[self.he_next]

    @cached_property
    def _edge_data(self):
        a, b = self.he_from, self.he_to
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo * self.n_verts + hi
        uniq, he_edge, counts = np.unique(key, return_inverse=True, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("non-manifold mesh: an edge is shared by more than two faces")
        # twin: the other half-edge with the same undirected key
        order = np.argsort(he_edge, kind="stable")
        sorted_edges = he_edge[order]
        twin = np.full(self.n_halfedges, -1, dtype=np.int64)
        same = sorted_edges[:-1] == sorted_edges[1:]
        i = np.nonzero(same)[0]
        twin[order[i]] = order[i + 1]
        twin[order[i + 1]] = order[i]
        # consistent orientation: twins must run in opposite directions
        has = twin >= 0
        if np.any(self.he_from[has] == self.he_from[twin[has]]):
            raise ValueError("inconsistently oriented mesh: neighbouring faces disagree on winding")
        # one representative half-edge per undirected edge (the lower index)
        n_edges = len(uniq)
        edge_he = np.full(n_edges, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(edge_he, he_edge, np.arange(self.n_halfedges))
        return he_edge.astype(np.int64), twin, edge_he, n_edges

    @property
    def he_edge(self) -> np.ndarray:
        return self._edge_data[0]

    @property
    def he_twin(self) -> np.ndarray:
        return self._edge_data[1]

    @property
    def edge_he(self) -> np.ndarray:
        """A representative half-edge for each undirected edge."""
        return self._edge_data[2]

    @property
    def n_edges(self) -> int:
        return self._edge_data[3]

    @cached_property
    def edge_verts(self) -> np.ndarray:
        h = self.edge_he
        return np.stack([self.he_from[h], self.he_to[h]], axis=1)

    @cached_property
    def edge_faces(self) -> np.ndarray:
        """(E, 2) adjacent faces; column 1 is -1 on boundary edges."""
        h = self.edge_he
        t = self.he_twin[h]
        f2 = np.where(t >= 0, self.he_face[np.maximum(t, 0)], -1)
        return np.stack([self.he_face[h], f2], axis=1)

    @cached_property
    def edge_is_boundary(self) -> np.ndarray:
        return self.he_twin[self.edge_he] < 0

    @cached_property
    def vert_is_boundary(self) -> np.ndarray:
        out = np.zeros(self.n_verts, dtype=bool)
        ev = self.edge_verts[self.edge_is_boundary]
        out[ev.ravel()] = True
        return out

    @cached_property
    def valence(self) -> np.ndarray:
        """Number of incident edges per vertex."""
        return np.bincount(self.edge_verts.ravel(), minlength=self.n_verts)

    @cached_property
    def vert_face_count(self) -> np.ndarray:
        return np.bincount(self.he_from, minlength=self.n_verts)

    def euler_characteristic(self) -> int:
        return self.n_verts - self.n_edges + self.n_faces

    def boundary_distance(self) -> np.ndarray:
        """Edge-hop distance from each vertex to the nearest boundary vertex (inf on closed meshes)."""
        return self.hop_distance(self.vert_is_boundary)

    def hop_distance(self, seeds: np.ndarray) -> np.ndarray:
        """Edge-hop (topological) distance from each vertex to the nearest seed vertex."""
        dist = np.full(self.n_verts, np.inf)
        dist[np.asarray(seeds, bool)] = 0.0
        if not np.any(seeds):
            return dist
        a, b = self.edge_verts[:, 0], self.edge_verts[:, 1]
        while True:
            new = dist.copy()
            np.minimum.at(new, a, dist[b] + 1)
            np.minimum.at(new, b, dist[a] + 1)
            if np.array_equal(new, dist):
                return dist
            dist = new

    # --------------------------------------------------------------- geometry
    def face_reduce(self, per_he: np.ndarray) -> np.ndarray:
        """Sum a per-half-edge quantity over each face."""
        return np.add.reduceat(per_he, self.face_ptr[:-1], axis=0)

    @cached_property
    def face_centroid(self) -> np.ndarray:
        return self.face_reduce(self.V[self.face_idx]) / self.face_size[:, None]

    @cached_property
    def face_area_vec(self) -> np.ndarray:
        """Newell area vector (direction = normal, length = area) — robust for non-planar n-gons."""
        p, q = self.V[self.he_from], self.V[self.he_to]
        return 0.5 * self.face_reduce(np.cross(p, q))

    @cached_property
    def face_normal(self) -> np.ndarray:
        return _normalize(self.face_area_vec)

    @cached_property
    def edge_length(self) -> np.ndarray:
        ev = self.edge_verts
        return np.linalg.norm(self.V[ev[:, 1]] - self.V[ev[:, 0]], axis=1)

    @cached_property
    def face_scale(self) -> np.ndarray:
        """Mean edge length of each face (the local length unit for relative extrusion)."""
        return self.face_reduce(self.edge_length[self.he_edge]) / self.face_size

    @cached_property
    def edge_normal(self) -> np.ndarray:
        """Average of the adjacent face normals (paper: n_e)."""
        ef = self.edge_faces
        n = self.face_normal[ef[:, 0]].copy()
        inner = ef[:, 1] >= 0
        n[inner] += self.face_normal[ef[inner, 1]]
        return _normalize(n)

    @cached_property
    def vert_normal(self) -> np.ndarray:
        """Area-weighted average of incident face normals (paper: n_p)."""
        n = np.zeros_like(self.V)
        np.add.at(n, self.he_from, self.face_area_vec[self.he_face])
        return _normalize(n)

    def edge_mean(self, per_face: np.ndarray) -> np.ndarray:
        """Average a per-face quantity onto edges."""
        ef = self.edge_faces
        out = per_face[ef[:, 0]].astype(float)
        inner = ef[:, 1] >= 0
        out[inner] = 0.5 * (out[inner] + per_face[ef[inner, 1]])
        return out

    def vert_mean(self, per_face: np.ndarray) -> np.ndarray:
        """Average a per-face quantity onto vertices."""
        acc = np.zeros((self.n_verts,) + per_face.shape[1:])
        np.add.at(acc, self.he_from, per_face[self.he_face])
        cnt = np.maximum(self.vert_face_count, 1).reshape((-1,) + (1,) * (per_face.ndim - 1))
        return acc / cnt

    def motif_codes(self) -> np.ndarray:
        """Vertex motif (paper Fig. 6) encoded as faces*100 + edges, e.g. 404 = "4F4E"."""
        return self.vert_face_count * 100 + self.valence

    @cached_property
    def face_planarity(self) -> np.ndarray:
        """Largest distance of a face's vertices from its mean plane, relative to the face size (0 = planar)."""
        d = np.abs(np.einsum("ij,ij->i", self.V[self.he_from] - self.face_centroid[self.he_face],
                             self.face_normal[self.he_face]))
        return np.maximum.reduceat(d, self.face_ptr[:-1]) / np.maximum(self.face_scale, _EPS)

    @cached_property
    def face_bend(self) -> np.ndarray:
        """Mean (1 - cos) of the dihedral angles to neighbouring faces: 0 = flat, 1 = folded at 90 deg."""
        tw = self.he_twin
        has = tw >= 0
        n = self.face_normal
        val = np.zeros(self.n_halfedges)
        val[has] = 1.0 - np.einsum("ij,ij->i", n[self.he_face[has]], n[self.he_face[tw[has]]])
        cnt = np.add.reduceat(has.astype(float), self.face_ptr[:-1])
        return np.add.reduceat(val, self.face_ptr[:-1]) / np.maximum(cnt, 1)

    def signed_volume(self) -> float:
        """Divergence-theorem volume; positive for a closed, outward-oriented mesh."""
        return float(np.sum(self.face_area_vec * self.face_centroid) / 3.0)

    # --------------------------------------------------------------- display
    def triangles(self) -> np.ndarray:
        """Fan triangulation (for display / STL export)."""
        sizes = self.face_size
        n_tri = sizes - 2
        starts = np.repeat(self.face_ptr[:-1], n_tri)
        local = np.arange(n_tri.sum()) - np.repeat(np.cumsum(n_tri) - n_tri, n_tri)
        return np.stack(
            [self.face_idx[starts], self.face_idx[starts + local + 1], self.face_idx[starts + local + 2]],
            axis=1,
        )

    def display_faces(self) -> np.ndarray:
        """Quads as an (F,4) array when possible (clean wireframe), else triangles."""
        if self.is_all_quads():
            return self.face_idx.reshape(-1, 4)
        if self.n_faces and np.all(self.face_size == 3):
            return self.face_idx.reshape(-1, 3)
        return self.triangles()

    # -------------------------------------------------------------------- io
    def write_obj(self, path: str) -> None:
        from .meshio import write_obj

        write_obj(path, self)

"""Base (input) meshes.

Every shape is registered with its parameters so the UI can build sliders
automatically. A base spec is a dict: {"shape": name, **params}.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field

import numpy as np

from .mesh import PolyMesh

PHI = (1 + 5**0.5) / 2
UNIT_RADIUS = 3**0.5  # circumradius of the default cube, used for all solids


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    lo: float
    hi: float
    default: float
    integer: bool = False
    help: str = ""


@dataclass(frozen=True)
class ShapeDef:
    name: str
    label: str
    build: callable
    params: tuple = ()
    view: str = "diagonal"
    help: str = ""
    extra_defaults: dict = field(default_factory=dict)  # non-slider params (e.g. obj path)


# ------------------------------------------------------------------ helpers
def _orient_convex(V: np.ndarray, faces: list) -> list:
    """Wind faces of a convex solid centred at the origin so normals point outward."""
    out = []
    for f in faces:
        p = V[list(f)]
        n = np.zeros(3)
        for i in range(len(p)):
            n += np.cross(p[i], p[(i + 1) % len(p)])
        out.append(list(f) if np.dot(n, p.mean(0)) > 0 else list(f)[::-1])
    return out


def _triangles_by_edge_length(V: np.ndarray) -> list:
    """All vertex triples whose three sides equal the shortest vertex distance (regular deltahedra)."""
    D = np.linalg.norm(V[:, None] - V[None], axis=-1)
    L = D[D > 1e-9].min()
    adj = np.abs(D - L) < 1e-6 * L
    return [t for t in itertools.combinations(range(len(V)), 3) if adj[t[0], t[1]] and adj[t[1], t[2]] and adj[t[0], t[2]]]


def _scale_to_radius(V: np.ndarray, radius: float) -> np.ndarray:
    return V * (radius / np.linalg.norm(V, axis=1).max())


def dual(m: PolyMesh) -> PolyMesh:
    """Dual mesh of a closed manifold: face centroids become vertices, vertex fans become faces."""
    from .doo_sabin import _vertex_fans

    faces = []
    for fan in _vertex_fans(m):
        faces.extend(m.he_face[fan].tolist())
    return PolyMesh.from_faces(m.face_centroid, faces)


# ------------------------------------------------------------ platonic solids
def tetrahedron(radius: float = UNIT_RADIUS) -> PolyMesh:
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    V = _scale_to_radius(V, radius)
    return PolyMesh.from_faces(V, _orient_convex(V, list(itertools.combinations(range(4), 3))))


def cube(size: float = 2.0) -> PolyMesh:
    h = size / 2
    V = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return PolyMesh.from_faces(V, F)


def octahedron(radius: float = UNIT_RADIUS) -> PolyMesh:
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * radius
    return PolyMesh.from_faces(V, _orient_convex(V, _triangles_by_edge_length(V)))


def icosahedron(radius: float = UNIT_RADIUS) -> PolyMesh:
    V = []
    for a, b in itertools.product((-1, 1), repeat=2):
        V += [[0, a, b * PHI], [a, b * PHI, 0], [b * PHI, 0, a]]
    V = _scale_to_radius(np.array(V, float), radius)
    return PolyMesh.from_faces(V, _orient_convex(V, _triangles_by_edge_length(V)))


def dodecahedron(radius: float = UNIT_RADIUS) -> PolyMesh:
    d = dual(icosahedron())
    return PolyMesh.from_faces(_scale_to_radius(d.V, radius), d.faces_list())


# --------------------------------------------------------------- lathe solids
def lathe(profile: list, sides: int, twist: float = 0.0) -> PolyMesh:
    """Revolve a profile [(apothem r, height y), ...] (bottom to top) into a closed prism-like solid.

    r is the distance from the axis to the middle of each side, so sides=4 gives a
    true square cross-section with faces aligned to the x/z axes. Caps are single polygons.
    """
    sides = int(sides)
    prof = np.asarray(profile, float)
    m = len(prof)
    corner = 1.0 / np.cos(np.pi / sides)
    V = []
    for i, (r, y) in enumerate(prof):
        t = twist * (y - prof[0, 1]) / max(prof[-1, 1] - prof[0, 1], 1e-9)
        for j in range(sides):
            th = np.pi / sides + 2 * np.pi * j / sides + t
            V.append([r * corner * np.cos(th), y, r * corner * np.sin(th)])
    idx = lambda i, j: i * sides + (j % sides)  # noqa: E731
    F = []
    for i in range(m - 1):
        for j in range(sides):
            F.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
    F.append([idx(0, j) for j in range(sides)])  # bottom cap (normal -y)
    F.append([idx(m - 1, j) for j in reversed(range(sides))])  # top cap (normal +y)
    return PolyMesh.from_faces(np.array(V), F)


def column_box(height=6.0, shaft_width=0.6, base_width=1.0, base_height=0.5, shaft_segments=1) -> PolyMesh:
    """Paper Fig. 4: an elongated cube on a base (square cross-section)."""
    prof = [(base_width, 0.0), (base_width, base_height)]
    for t in np.linspace(0, 1, int(shaft_segments) + 1):
        prof.append((shaft_width, base_height + t * (height - base_height)))
    return _centre_y(lathe(prof, 4))


def column_classic(sides=8, height=6.0, base_width=0.95, base_height=0.45, shaft_width=0.55,
                   entasis=0.08, taper=0.12, capital_width=1.0, capital_height=0.8, shaft_rings=4) -> PolyMesh:
    """Paper Fig. 7 style: a low-poly column whose topography already carries base, shaft
    (with entasis and taper), echinus and abacus."""
    top_shaft = height - capital_height
    prof = [(base_width, 0.0), (base_width, base_height)]
    for t in np.linspace(0, 1, int(shaft_rings) + 1):
        r = shaft_width * (1 + entasis * np.sin(np.pi * t)) * (1 - taper * t)
        prof.append((r, base_height + t * (top_shaft - base_height)))
    r_top = prof[-1][0]
    prof.append(((r_top + capital_width) / 2, top_shaft + 0.45 * capital_height))  # echinus
    prof.append((capital_width, top_shaft + 0.7 * capital_height))  # abacus
    prof.append((capital_width, height))
    return _centre_y(lathe(prof, int(sides)))


def _centre_y(m: PolyMesh) -> PolyMesh:
    V = m.V.copy()
    V[:, 1] -= 0.5 * (V[:, 1].min() + V[:, 1].max())
    return PolyMesh.from_faces(V, m.faces_list())


# --------------------------------------------------------------------- panel
def panel(nx=4, ny=4, width=2.0, height=2.0, bend=0.0, saddle=0.0) -> PolyMesh:
    """Open quad grid in the x/y plane facing +z — a relief tile. Open boundary."""
    nx, ny = int(nx), int(ny)
    xs = np.linspace(-width / 2, width / 2, nx + 1)
    ys = np.linspace(-height / 2, height / 2, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    u, v = 2 * X / width, 2 * Y / height
    Z = 0.5 * bend * (1 - u**2) * width / 2 + 0.5 * saddle * (u**2 - v**2) * width / 2
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    F = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            F.append([a, a + 1, a + nx + 2, a + nx + 1])
    return PolyMesh.from_faces(V, F)


# ----------------------------------------------------------------------- obj
def obj(path: str = "", normalize: bool = True, weld: bool = True, root: str | None = None) -> PolyMesh:
    from .meshio import load_obj

    if not path:
        raise ValueError("choose an .obj file to import")
    full = path if os.path.isabs(path) or root is None else os.path.join(root, path)
    return load_obj(full, normalize=bool(normalize), weld=bool(weld))


# ------------------------------------------------------------------ registry
_R = Param("radius", "radius", 0.5, 4.0, UNIT_RADIUS)
SHAPES: dict[str, ShapeDef] = {
    s.name: s
    for s in [
        ShapeDef("cube", "Cube", cube, (Param("size", "size", 0.5, 4.0, 2.0),), "diagonal", "Paper Fig. 3 input."),
        ShapeDef("tetrahedron", "Tetrahedron", tetrahedron, (_R,), "three_quarter", "Platonic solid, 4 triangles."),
        ShapeDef("octahedron", "Octahedron", octahedron, (_R,), "diagonal", "Platonic solid, 8 triangles."),
        ShapeDef("dodecahedron", "Dodecahedron", dodecahedron, (_R,), "diagonal", "Platonic solid, 12 pentagons."),
        ShapeDef("icosahedron", "Icosahedron", icosahedron, (_R,), "diagonal", "Platonic solid, 20 triangles."),
        ShapeDef(
            "column_box", "Column: box on base (Fig. 4)", column_box,
            (
                Param("height", "height", 2.0, 12.0, 6.0),
                Param("shaft_width", "shaft half-width", 0.2, 2.0, 0.6),
                Param("base_width", "base half-width", 0.2, 3.0, 1.0),
                Param("base_height", "base height", 0.1, 3.0, 0.5),
                Param("shaft_segments", "shaft segments", 1, 12, 1, True, "Rings along the shaft (paper: 1)."),
            ),
            "three_quarter", "Uniform input mesh of paper Fig. 4: an elongated cube and a base.",
        ),
        ShapeDef(
            "column_classic", "Column: classic (Fig. 7)", column_classic,
            (
                Param("sides", "sides", 3, 24, 8, True),
                Param("height", "height", 2.0, 12.0, 6.0),
                Param("base_width", "base half-width", 0.2, 3.0, 0.95),
                Param("base_height", "base height", 0.1, 2.0, 0.45),
                Param("shaft_width", "shaft half-width", 0.2, 2.0, 0.55),
                Param("entasis", "entasis", 0.0, 0.4, 0.08, help="Swelling of the shaft (Doric entasis)."),
                Param("taper", "taper", 0.0, 0.6, 0.12),
                Param("capital_width", "capital half-width", 0.2, 3.0, 1.0),
                Param("capital_height", "capital height", 0.2, 3.0, 0.8),
                Param("shaft_rings", "shaft rings", 1, 16, 4, True),
            ),
            "three_quarter", "Differentiated input like paper Fig. 7: base, shaft with entasis, echinus, abacus.",
        ),
        ShapeDef(
            "panel", "Panel (open relief tile)", panel,
            (
                Param("nx", "cells x", 1, 24, 4, True),
                Param("ny", "cells y", 1, 24, 4, True),
                Param("width", "width", 0.5, 6.0, 2.0),
                Param("height", "height", 0.5, 6.0, 2.0),
                Param("bend", "bend", -1.0, 1.0, 0.0, help="Cylindrical curvature."),
                Param("saddle", "saddle", -1.0, 1.0, 0.0, help="Hyperbolic (saddle) curvature."),
            ),
            "three_quarter", "Open quad grid facing +z. Use Locked boundary to keep it tileable.",
        ),
        ShapeDef(
            "obj", "Import OBJ", obj, (), "three_quarter",
            "Any polygon mesh (quads recommended). Welded, re-oriented and scaled to fit.",
            {"path": "", "normalize": True, "weld": True},
        ),
    ]
}


def default_spec(name: str) -> dict:
    s = SHAPES[name]
    spec = {"shape": name, **{p.name: p.default for p in s.params}, **s.extra_defaults}
    return spec


def make_base(spec: dict, root: str | None = None) -> PolyMesh:
    spec = dict(spec)
    name = spec.pop("shape", "cube")
    if name not in SHAPES:
        raise ValueError(f"unknown base shape {name!r}; available: {', '.join(SHAPES)}")
    s = SHAPES[name]
    kwargs = {p.name: p.default for p in s.params}
    kwargs.update(s.extra_defaults)
    kwargs.update({k: v for k, v in spec.items() if k in kwargs})
    for p in s.params:
        if p.integer:
            kwargs[p.name] = int(round(kwargs[p.name]))
    if name == "obj":
        kwargs["root"] = root
    return s.build(**kwargs)

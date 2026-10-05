"""Base (input) meshes. Milestone 2 adds the Platonic solids, columns, panel and OBJ import."""

from __future__ import annotations

import numpy as np

from .mesh import PolyMesh


def cube(size: float = 2.0) -> PolyMesh:
    h = size / 2
    V = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return PolyMesh.from_faces(V, F)


SHAPES = {
    "cube": cube,
}


def make_base(spec: dict) -> PolyMesh:
    spec = dict(spec)
    name = spec.pop("shape", "cube")
    if name not in SHAPES:
        raise ValueError(f"unknown base shape {name!r}; available: {', '.join(SHAPES)}")
    return SHAPES[name](**spec)

"""Runs a Design's schedule with per-iteration caching and a face budget."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import attractors, catmull_clark, doo_sabin
from .mesh import PolyMesh
from .schedule import Design
from .shapes import make_base

DEFAULT_FACE_BUDGET = 2_000_000

SCHEME_MODULES = {"cc": catmull_clark, "ds": doo_sabin}


def boundary_fade(dist: np.ndarray, rows: float) -> np.ndarray:
    """0 on the boundary, smoothly rising to 1 at `rows` base-mesh rows inside."""
    if rows <= 0:
        return (dist > 0).astype(float)
    t = np.clip(dist / rows, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def uniform_weights(mesh: PolyMesh, level: int, spec) -> dict:
    """Weight provider for uniform (per-iteration only) weights."""
    return spec.weights


@dataclass
class RunResult:
    mesh: PolyMesh
    depth_reached: int
    depth_requested: int
    capped_by_budget: bool
    seconds: float
    cache_hits: int


class Pipeline:
    def __init__(self, face_budget: int = DEFAULT_FACE_BUDGET, max_cached: int = 32, root: str | None = None):
        self.root = root  # resolves relative OBJ paths
        self.face_budget = face_budget
        self.max_cached = max_cached
        self._cache: OrderedDict[str, PolyMesh] = OrderedDict()

    def _get(self, key):
        m = self._cache.get(key)
        if m is not None:
            self._cache.move_to_end(key)
        return m

    def _put(self, key, mesh):
        self._cache[key] = mesh
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    def base(self, design: Design) -> PolyMesh:
        """The (cached) input mesh of a design."""
        key = design.base_key(self.root)
        mesh = self._get(key)
        if mesh is None:
            mesh = make_base(design.base, self.root)
            mesh.vattr["rest"] = mesh.V.copy()  # input-mesh position, carried through subdivision
            if design.boundary == "locked":
                mesh.vattr["bdist"] = mesh.boundary_distance()
            self._put(key, mesh)
        return mesh

    def run(self, design: Design, depth: int, weight_provider=None) -> RunResult:
        """weight_provider(mesh, level, spec) -> weights; default: the design's attractor field."""
        t0 = time.perf_counter()
        if weight_provider is None:
            weight_provider = attractors.make_provider(design)
        mesh = self.base(design)
        relative = design.extrusion == "relative"
        lock = design.boundary == "locked"
        keys = design.level_keys(depth, self.root)
        hits, reached, capped = 0, 0, False
        for level, (spec, key) in enumerate(zip(design.iterations[:depth], keys)):
            cached = self._get(key)
            if cached is not None:
                mesh, hits, reached = cached, hits + 1, level + 1
                continue
            module = SCHEME_MODULES[spec.scheme]
            if module.predicted_faces(mesh) > self.face_budget:
                capped = True
                break
            vmask = None
            if lock and "bdist" in mesh.vattr and np.isfinite(mesh.vattr["bdist"]).any():
                vmask = boundary_fade(mesh.vattr["bdist"], design.fade_rows)
            mesh = module.subdivide(
                mesh, weight_provider(mesh, level, spec), relative=relative, lock_boundary=lock, vmask=vmask
            )
            self._put(key, mesh)
            reached = level + 1
        return RunResult(mesh, reached, depth, capped, time.perf_counter() - t0, hits)

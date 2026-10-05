"""Runs a Design's schedule with per-iteration caching and a face budget."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from . import catmull_clark, doo_sabin
from .mesh import PolyMesh
from .schedule import Design
from .shapes import make_base

DEFAULT_FACE_BUDGET = 2_000_000

SCHEME_MODULES = {"cc": catmull_clark, "ds": doo_sabin}


def uniform_weights(mesh: PolyMesh, level: int, spec) -> dict:
    """Weight provider for uniform (per-iteration only) weights. Attractors and
    function fields (milestones 3-4) will return per-face arrays here instead."""
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
    def __init__(self, face_budget: int = DEFAULT_FACE_BUDGET, max_cached: int = 32):
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

    def run(self, design: Design, depth: int, weight_provider=uniform_weights) -> RunResult:
        t0 = time.perf_counter()
        base_key = design.base_key()
        mesh = self._get(base_key)
        if mesh is None:
            mesh = make_base(design.base)
            self._put(base_key, mesh)
        relative = design.extrusion == "relative"
        keys = design.level_keys(depth)
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
            mesh = module.subdivide(mesh, weight_provider(mesh, level, spec), relative=relative)
            self._put(key, mesh)
            reached = level + 1
        return RunResult(mesh, reached, depth, capped, time.perf_counter() - t0, hits)

"""Runs a Design's schedule with per-iteration caching and a face budget."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import attractors, catmull_clark, doo_sabin, intrinsic, layers, merge, vessel
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


def make_provider(design):
    """The design's weights for one iteration: the iteration's schedule, then attractors (eq. 8-9),
    group rules, intrinsic rules and function layers, each refining the previous result."""
    att = attractors.make_provider(design)

    def provider(mesh, level, spec):
        W = att(mesh, level, spec)
        W = intrinsic.apply_group_rules(design, mesh, level, W)
        W = intrinsic.apply_rules(design, mesh, level, W)
        return layers.apply_weight_layers(design, mesh, level, W)

    return provider


@dataclass
class RunResult:
    mesh: PolyMesh
    depth_reached: int
    depth_requested: int
    capped_by_budget: bool
    seconds: float
    cache_hits: int
    relief: PolyMesh | None = None  # with a vessel: the subdivided form that became its inner face


class Pipeline:
    def __init__(self, face_budget: int = DEFAULT_FACE_BUDGET, max_cached: int = 32, root: str | None = None,
                 max_bytes: int = 1_000_000_000):
        self.root = root  # resolves relative OBJ paths
        self.face_budget = face_budget
        self.max_cached = max_cached
        self.max_bytes = max_bytes  # the level cache also stays under this much memory
        self._cache: OrderedDict[str, PolyMesh] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._bytes = 0

    def _get(self, key):
        m = self._cache.get(key)
        if m is not None:
            self._cache.move_to_end(key)
        return m

    def _put(self, key, mesh):
        if key in self._cache:
            self._bytes -= self._sizes.pop(key)
        self._cache[key] = mesh
        self._cache.move_to_end(key)
        self._sizes[key] = mesh.nbytes() + mesh.face_idx.nbytes
        self._bytes += self._sizes[key]
        while len(self._cache) > 1 and (len(self._cache) > self.max_cached or self._bytes > self.max_bytes):
            old, _ = self._cache.popitem(last=False)
            self._bytes -= self._sizes.pop(old)

    def clear(self):
        self._cache.clear()
        self._sizes.clear()
        self._bytes = 0

    def base(self, design: Design) -> PolyMesh:
        """The (cached) input mesh of a design."""
        key = design.base_key(self.root)
        mesh = self._get(key)
        if mesh is None:
            mesh = make_base(design.base, self.root).share_topology()  # same faces, new positions: tables reused
            mesh.vattr["rest"] = mesh.V.copy()  # input-mesh position, carried through subdivision
            intrinsic.decorate_base(design, mesh)  # tags, locks, original vertex/edge markers
            if design.boundary == "locked":
                mesh.vattr["bdist"] = mesh.boundary_distance()
            self._put(key, mesh)
        return mesh

    def run(self, design: Design, depth: int, weight_provider=None) -> RunResult:
        """weight_provider(mesh, level, spec) -> weights; default: the design's attractor field."""
        t0 = time.perf_counter()
        if weight_provider is None:
            weight_provider = make_provider(design)
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
            vlock = mesh.vattr["lock"] > level if "lock" in mesh.vattr else None
            vU = intrinsic.motif_U(design, mesh) if spec.scheme == "cc" else None
            mesh = module.subdivide(
                mesh, weight_provider(mesh, level, spec), relative=relative, lock_boundary=lock, vmask=vmask,
                vlock=vlock, vU=vU,
            )
            if lock and "bdist" in mesh.vattr:
                vmask = boundary_fade(mesh.vattr["bdist"], design.fade_rows)
            mesh = layers.post_process(design, mesh, level, relative, vmask)
            mesh = merge.maybe_merge(design, mesh, level)
            self._put(key, mesh)
            reached = level + 1
        relief = None
        if vessel.active(design) and mesh.n_faces:
            plain = self.run(vessel.plain_design(design), reached).mesh if reached else None
            relief, mesh = mesh, vessel.build(design, mesh, plain)
        return RunResult(mesh, reached, depth, capped, time.perf_counter() - t0, hits, relief)

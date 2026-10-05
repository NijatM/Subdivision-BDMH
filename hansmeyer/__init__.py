"""Hansmeyer-style generative subdivision engine (extended Catmull-Clark / Doo-Sabin)."""

from .mesh import PolyMesh
from .pipeline import Pipeline, RunResult
from .schedule import CC_WEIGHTS, DS_WEIGHTS, MAX_ITERATIONS, Design, IterationSpec
from .shapes import SHAPES, default_spec, make_base
from . import attractors, catmull_clark, doo_sabin, meshio

__all__ = [
    "PolyMesh",
    "Pipeline",
    "RunResult",
    "Design",
    "IterationSpec",
    "CC_WEIGHTS",
    "DS_WEIGHTS",
    "MAX_ITERATIONS",
    "SHAPES",
    "make_base",
    "default_spec",
    "meshio",
    "attractors",
    "catmull_clark",
    "doo_sabin",
]

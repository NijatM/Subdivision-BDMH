"""Weight definitions, per-iteration schedule and the full design state (saved as JSON presets)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

MAX_ITERATIONS = 10


@dataclass(frozen=True)
class WeightDef:
    name: str
    label: str
    lo: float
    hi: float
    help: str


# Extended Catmull-Clark (Hansmeyer 2010, eq. 1-4). All zero == standard Catmull-Clark.
CC_WEIGHTS = [
    WeightDef("w_f", "w_f  face extrude", -2.0, 2.0, "Eq.1: push face points along the face normal."),
    WeightDef("w_e", "w_e  edge extrude", -2.0, 2.0, "Eq.2: push edge points along the edge normal."),
    WeightDef("w_c", "w_c  corner extrude", -2.0, 2.0, "Eq.3: push corner points along the vertex normal."),
    WeightDef("w1", "w1   edge bias", -2.0, 2.0, "Eq.2: + pulls edge points toward face points, - toward the edge."),
    WeightDef("w2", "w2   corner bias", -2.0, 2.0, "Eq.3: + pulls corners toward face points, - toward edge midpoints."),
    WeightDef("w3", "w3   V/F bias", -2.0, 2.0, "Eq.4 (iter 2+): face point bias toward the old-corner vs old-face vertex."),
    WeightDef("w4", "w4   diag/edge bias", -2.0, 2.0, "Eq.4 (iter 2+): face point bias toward the diagonal vs the edge vertices."),
]

# Extended Doo-Sabin (eq. 5-6), with separate weights per face class (face-, edge-, vertex-derived).
DS_WEIGHTS = [
    WeightDef("ds_w1_face", "w1   face-faces", -2.0, 2.0, "Eq.5/6: corner pull toward its own vertex (faces from old faces)."),
    WeightDef("ds_wf_face", "w_f  face-faces", -2.0, 2.0, "Extrude new corners along the face normal (faces from old faces)."),
    WeightDef("ds_w1_edge", "w1   edge-faces", -2.0, 2.0, "As above, for faces created from old edges (after a DS step)."),
    WeightDef("ds_wf_edge", "w_f  edge-faces", -2.0, 2.0, "As above, for faces created from old edges (after a DS step)."),
    WeightDef("ds_w1_vert", "w1   vertex-faces", -2.0, 2.0, "As above, for faces created from old vertices (after a DS step)."),
    WeightDef("ds_wf_vert", "w_f  vertex-faces", -2.0, 2.0, "As above, for faces created from old vertices (after a DS step)."),
]

ALL_WEIGHTS = CC_WEIGHTS + DS_WEIGHTS
WEIGHT_NAMES = [w.name for w in ALL_WEIGHTS]
SCHEMES = ("cc", "ds")


def zero_weights() -> dict[str, float]:
    return {n: 0.0 for n in WEIGHT_NAMES}


@dataclass
class IterationSpec:
    scheme: str = "cc"
    weights: dict = field(default_factory=zero_weights)

    def __post_init__(self):
        if self.scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {self.scheme!r}")
        merged = zero_weights()
        merged.update({k: float(v) for k, v in self.weights.items() if k in merged})
        self.weights = merged


@dataclass
class Design:
    """Everything needed to reproduce a form. Serialised as a preset."""

    name: str = "Untitled"
    description: str = ""
    base: dict = field(default_factory=lambda: {"shape": "cube"})
    extrusion: str = "relative"  # "relative" (w x local edge length) or "absolute" (paper-literal)
    boundary: str = "smooth"  # open meshes: "smooth" (standard rules) or "locked" (fixed, tileable)
    fade_rows: float = 2.0  # locked: extrusion fades in over this many base-mesh rows from the boundary
    preview_depth: int = 5
    full_depth: int = 8
    iterations: list = field(default_factory=lambda: [IterationSpec() for _ in range(MAX_ITERATIONS)])
    view: str = "diagonal"

    def __post_init__(self):
        its = [i if isinstance(i, IterationSpec) else IterationSpec(**i) for i in self.iterations]
        its += [IterationSpec() for _ in range(MAX_ITERATIONS - len(its))]
        self.iterations = its[:MAX_ITERATIONS]
        if self.extrusion not in ("relative", "absolute"):
            raise ValueError("extrusion must be 'relative' or 'absolute'")
        if self.boundary not in ("smooth", "locked"):
            raise ValueError("boundary must be 'smooth' or 'locked'")

    # ------------------------------------------------------------- caching key
    def base_key(self, root: str | None = None) -> str:
        key = {"base": self.base, "extrusion": self.extrusion, "boundary": self.boundary, "fade": self.fade_rows}
        path = self.base.get("path") if self.base.get("shape") == "obj" else None
        if path:  # re-import when the file changes on disk
            full = path if os.path.isabs(path) or root is None else os.path.join(root, path)
            key["mtime"] = os.path.getmtime(full) if os.path.exists(full) else None
        return _digest(key)

    def level_keys(self, depth: int, root: str | None = None) -> list[str]:
        """Key for the mesh after each iteration: changing iteration k only invalidates levels >= k."""
        keys, prev = [], self.base_key(root)
        for spec in self.iterations[:depth]:
            prev = _digest({"prev": prev, "it": asdict(spec)})
            keys.append(prev)
        return keys

    # ------------------------------------------------------------------- json
    def to_dict(self) -> dict:
        d = asdict(self)
        # keep presets readable: drop all-zero weights
        for it in d["iterations"]:
            it["weights"] = {k: v for k, v in it["weights"].items() if v != 0.0}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Design":
        d = copy.deepcopy(d)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Design":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def copy(self) -> "Design":
        return Design.from_dict(self.to_dict())


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()

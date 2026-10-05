"""Function library for the layer stack (our extension — not part of Hansmeyer's method).

Two kinds of functions:
  * fields  f(p, **params) -> (N,) values, roughly in [-1, 1]
      used to drive weights or to displace the surface
  * folds   g(p, **params) -> (N, 3) positions
      used to deform the mesh between iterations, or to fold the domain of a field
      (repeated folding gives fractal, Mandelbox-like patterns)

Register new ones with the decorators — this is also the plug-in API used by the
files in the project's functions/ folder:

    from hansmeyer.functions import field

    @field("Ripples", family="custom", frequency=(0.1, 10.0, 2.0), help="Concentric waves.")
    def ripples(p, frequency):
        return np.sin(2 * np.pi * frequency * np.linalg.norm(p, axis=1))

A parameter is (lo, hi, default) or (lo, hi, default, "int").
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import noise


@dataclass(frozen=True)
class FuncParam:
    name: str
    lo: float
    hi: float
    default: float
    integer: bool = False


@dataclass(frozen=True)
class FuncDef:
    name: str
    label: str
    kind: str  # "field" | "fold"
    family: str
    fn: callable
    params: tuple = ()
    help: str = ""
    source: str = ""  # plug-in file (empty for built-ins)
    version: float = 0.0  # plug-in file mtime: changes invalidate cached results

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def call(self, p: np.ndarray, params: dict) -> np.ndarray:
        kw = self.defaults()
        kw.update({k: v for k, v in params.items() if k in kw})
        for prm in self.params:
            if prm.integer:
                kw[prm.name] = int(round(kw[prm.name]))
        return self.fn(np.asarray(p, float), **kw)


REGISTRY: dict[str, FuncDef] = {}
PLUGIN_ERRORS: list[str] = []
_LOADING = {"source": "", "version": 0.0}


def _parse_params(params: dict) -> tuple:
    out = []
    for name, spec in params.items():
        if isinstance(spec, FuncParam):
            out.append(spec)
            continue
        lo, hi, default = spec[:3]
        out.append(FuncParam(name, float(lo), float(hi), float(default), len(spec) > 3 and spec[3] == "int"))
    return tuple(out)


def _register(kind, name, label, family, help, params):
    def deco(fn):
        key = name.lower().replace(" ", "_")
        REGISTRY[key] = FuncDef(key, label or name, kind, family, fn, _parse_params(params), help,
                                _LOADING["source"], _LOADING["version"])
        return fn

    return deco


def field(name: str, label: str | None = None, family: str = "custom", help: str = "", **params):
    """Register a scalar field f(p, **params) -> (N,) values in about [-1, 1]."""
    return _register("field", name, label, family, help, params)


def fold(name: str, label: str | None = None, family: str = "custom", help: str = "", **params):
    """Register a fold / deformation g(p, **params) -> (N, 3)."""
    return _register("fold", name, label, family, help, params)


def fields() -> list[FuncDef]:
    return [f for f in REGISTRY.values() if f.kind == "field"]


def folds() -> list[FuncDef]:
    return [f for f in REGISTRY.values() if f.kind == "fold"]


# =====================================================================  TPMS
TAU = 2 * np.pi


def _tpms_coords(p, frequency, phase):
    q = TAU * (frequency * p + phase)
    return q[:, 0], q[:, 1], q[:, 2]


@field("gyroid", "Gyroid", "TPMS", "Triply periodic minimal surface: labyrinthine, coral-like channels.",
       frequency=(0.05, 6.0, 1.0), phase=(0.0, 1.0, 0.0))
def gyroid(p, frequency, phase):
    x, y, z = _tpms_coords(p, frequency, phase)
    return (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)) / 1.5


@field("schwarz_p", "Schwarz P", "TPMS", "Primitive surface: a lattice of rounded cells.",
       frequency=(0.05, 6.0, 1.0), phase=(0.0, 1.0, 0.0))
def schwarz_p(p, frequency, phase):
    x, y, z = _tpms_coords(p, frequency, phase)
    return (np.cos(x) + np.cos(y) + np.cos(z)) / 3.0


@field("schwarz_d", "Schwarz D (diamond)", "TPMS", "Diamond surface: tetrahedral, crystalline channels.",
       frequency=(0.05, 6.0, 1.0), phase=(0.0, 1.0, 0.0))
def schwarz_d(p, frequency, phase):
    x, y, z = _tpms_coords(p, frequency, phase)
    s, c = np.sin, np.cos
    return (s(x) * s(y) * s(z) + s(x) * c(y) * c(z) + c(x) * s(y) * c(z) + c(x) * c(y) * s(z)) / 1.42


@field("neovius", "Neovius", "TPMS", "Neovius surface: cubic cells with tubular necks.",
       frequency=(0.05, 6.0, 1.0), phase=(0.0, 1.0, 0.0))
def neovius(p, frequency, phase):
    x, y, z = _tpms_coords(p, frequency, phase)
    return (3 * (np.cos(x) + np.cos(y) + np.cos(z)) + 4 * np.cos(x) * np.cos(y) * np.cos(z)) / 13.0


# ==================================================================== noise
@field("perlin", "Perlin noise", "noise", "Smooth random variation.",
       frequency=(0.05, 12.0, 1.5), seed=(0, 99, 0, "int"))
def perlin_field(p, frequency, seed):
    return np.clip(noise.perlin(p * frequency, seed) * 1.4, -1, 1)


@field("fbm", "fBm (fractal noise)", "noise", "Several octaves of noise: detail at every scale.",
       frequency=(0.05, 12.0, 1.5), octaves=(1, 8, 4, "int"), gain=(0.1, 0.9, 0.5), seed=(0, 99, 0, "int"))
def fbm_field(p, frequency, octaves, gain, seed):
    return np.clip(noise.fbm(p * frequency, octaves, 2.0, gain, seed) * 1.6, -1, 1)


@field("ridged", "Ridged noise", "noise", "Sharp crests and valleys, mountain-like.",
       frequency=(0.05, 12.0, 1.5), octaves=(1, 8, 4, "int"), gain=(0.1, 0.9, 0.5), seed=(0, 99, 0, "int"))
def ridged_field(p, frequency, octaves, gain, seed):
    return noise.ridged(p * frequency, octaves, 2.0, gain, seed)


@field("warped", "Domain-warped noise", "noise", "Noise fed by noise: marbled, flowing patterns.",
       frequency=(0.05, 12.0, 1.0), warp=(0.0, 4.0, 1.5), seed=(0, 99, 0, "int"))
def warped_field(p, frequency, warp, seed):
    return np.clip(noise.warped(p * frequency, warp, 4, seed) * 1.8, -1, 1)


@field("worley", "Worley / Voronoi cells", "noise",
       "Cellular pattern. mode 0: cell centres high; mode 1: cell borders high (F2 - F1).",
       frequency=(0.1, 12.0, 2.0), mode=(0, 1, 0, "int"), jitter=(0.0, 1.0, 1.0), seed=(0, 99, 0, "int"))
def worley_field(p, frequency, mode, jitter, seed):
    f1, f2 = noise.worley(p * frequency, seed, jitter)
    if mode == 0:
        return np.clip(1.0 - 2.2 * f1, -1, 1)
    return np.clip(1.0 - 6.0 * (f2 - f1), -1, 1)


# ================================================================= analytic
def _spherical(p):
    r = np.linalg.norm(p, axis=1)
    rr = np.maximum(r, 1e-12)
    theta = np.arccos(np.clip(p[:, 1] / rr, -1, 1))  # polar angle from +y
    phi = np.arctan2(p[:, 2], p[:, 0])  # azimuth around y
    return r, theta, phi


def _assoc_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.maximum(1 - x * x, 0))
        fact = 1.0
        for _ in range(m):
            pmm = -pmm * fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pmmp1


def _real_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    P = _assoc_legendre(l, abs(m), np.cos(theta))
    if m > 0:
        return P * np.cos(m * phi)
    if m < 0:
        return P * np.sin(-m * phi)
    return P


@lru_cache(maxsize=128)
def _sh_scale(l: int, m: int) -> float:
    u = np.linspace(0, 1, 160)
    th, ph = np.meshgrid(np.arccos(1 - 2 * u), np.linspace(-np.pi, np.pi, 160))
    return float(np.abs(_real_sh(l, m, th.ravel(), ph.ravel())).max()) or 1.0


@field("spherical_harmonic", "Spherical harmonic", "analytic",
       "Real spherical harmonic Y(l, m) around the y axis: lobed, symmetric patterns.",
       l=(0, 10, 4, "int"), m=(-10, 10, 3, "int"))
def spherical_harmonic(p, l, m):
    m = max(-l, min(l, m))
    _, th, ph = _spherical(p)
    return _real_sh(l, m, th, ph) / _sh_scale(l, m)


def _sf(angle, m, n1, n2, n3):
    t = m * angle / 4
    return (np.abs(np.cos(t)) ** n2 + np.abs(np.sin(t)) ** n3) ** (-1.0 / max(n1, 1e-6))


@field("constant", "Constant", "analytic", "1 everywhere: with a mask, applies amplitude + offset only where "
       "the mask is (e.g. a weight boost on upward-facing surfaces).")
def constant_field(p):
    return np.ones(len(p))


@field("superformula", "Superformula (Gielis)", "analytic",
       "Positive inside a Gielis superformula shape, negative outside: displace to morph toward it.",
       m=(0, 16, 6, "int"), n1=(0.1, 20.0, 3.0), n2=(0.1, 20.0, 6.0), n3=(0.1, 20.0, 6.0), size=(0.2, 6.0, 1.8))
def superformula(p, m, n1, n2, n3, size):
    r, th, ph = _spherical(p)
    rho = np.minimum(_sf(ph, m, n1, n2, n3) * _sf(th - np.pi / 2, m, n1, n2, n3), 10.0)
    return np.clip((size * rho - r) / size, -1, 1)


@field("superquadric", "Superquadric", "analytic",
       "Positive inside a superquadric (exponent 2 = sphere, high = cube, <1 = star), negative outside.",
       exponent=(0.3, 12.0, 4.0), size=(0.2, 6.0, 1.6))
def superquadric(p, exponent, size):
    r = np.linalg.norm(p, axis=1)
    d = p / np.maximum(r, 1e-12)[:, None]
    rho = size / np.maximum((np.abs(d) ** exponent).sum(1), 1e-12) ** (1.0 / exponent)
    return np.clip((rho - r) / size, -1, 1)


@field("rose", "Rose (k-fold radial)", "analytic",
       "cos(k * angle) around the y axis: flutes, petals and ribs; twist turns them into spirals.",
       k=(1, 32, 8, "int"), twist=(-6.0, 6.0, 0.0), axial=(0.0, 6.0, 0.0))
def rose(p, k, twist, axial):
    ang = np.arctan2(p[:, 2], p[:, 0]) + twist * p[:, 1]
    out = np.cos(k * ang)
    if axial > 0:
        out = out * np.cos(TAU * axial * p[:, 1] / 2)
    return out


# ==================================================================== folds
@fold("box_fold", "Box fold", "fold", "Mandelbox box fold: reflects points beyond +/- limit back inside.",
      limit=(0.1, 4.0, 1.0))
def box_fold(p, limit):
    return np.clip(p, -limit, limit) * 2 - p


@fold("sphere_fold", "Sphere fold", "fold",
      "Mandelbox sphere fold: inverts points inside the fixed radius (scales up the core).",
      min_radius=(0.05, 2.0, 0.5), fixed_radius=(0.1, 4.0, 1.0))
def sphere_fold(p, min_radius, fixed_radius):
    r2 = (p * p).sum(1)
    mr2, fr2 = min_radius**2, max(fixed_radius**2, min_radius**2 + 1e-9)
    k = np.where(r2 < mr2, fr2 / mr2, np.where(r2 < fr2, fr2 / np.maximum(r2, 1e-12), 1.0))
    return p * k[:, None]


@fold("kaleido", "Kaleidoscopic mirror", "fold", "Mirrors space into k wedges around the y axis (k-fold symmetry).",
      k=(2, 24, 6, "int"), rotation=(0.0, 1.0, 0.0))
def kaleido(p, k, rotation):
    r = np.hypot(p[:, 0], p[:, 2])
    ang = np.arctan2(p[:, 2], p[:, 0]) - rotation * TAU / k
    sector = TAU / k
    a = np.mod(ang, sector)
    a = np.where(a > sector / 2, sector - a, a) + rotation * TAU / k
    return np.stack([r * np.cos(a), p[:, 1], r * np.sin(a)], 1)


@fold("mandelbox", "Mandelbox step", "fold",
      "One Mandelbox iteration: box fold, sphere fold, then scale (use 'repeat' in a field's domain).",
      scale=(-3.0, 3.0, 2.0), limit=(0.1, 4.0, 1.0), min_radius=(0.05, 2.0, 0.5), fixed_radius=(0.1, 4.0, 1.0))
def mandelbox(p, scale, limit, min_radius, fixed_radius):
    return scale * sphere_fold(box_fold(p, limit), min_radius, fixed_radius)


# ============================================================== evaluation
def fold_domain(p: np.ndarray, domain: dict | None) -> np.ndarray:
    """Repeatedly fold the input of a field: p <- scale * fold(p) + c * p0."""
    if not domain or domain.get("fold", "none") in ("none", "", None):
        return p
    fd = REGISTRY.get(domain["fold"])
    if fd is None or fd.kind != "fold":
        return p
    q, p0 = p.copy(), p
    scale, c = float(domain.get("scale", 1.0)), float(domain.get("c", 0.0))
    for _ in range(int(domain.get("repeat", 1))):
        q = scale * fd.call(q, domain.get("params", {})) + c * p0
        q = np.clip(q, -1e6, 1e6)
    return q


def evaluate(name: str, p: np.ndarray, params: dict, domain: dict | None = None) -> np.ndarray:
    fd = REGISTRY.get(name)
    if fd is None:
        raise KeyError(f"unknown function {name!r} (missing plug-in?)")
    q = fold_domain(np.asarray(p, float), domain) if fd.kind == "field" else np.asarray(p, float)
    out = np.asarray(fd.call(q, params), float)
    expected = (len(p),) if fd.kind == "field" else (len(p), 3)
    if out.shape != expected:
        raise ValueError(f"function {name!r} returned shape {out.shape}, expected {expected}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ================================================================== plug-ins
def load_plugins(folder: str) -> list[str]:
    """Import every functions/*.py so its decorators register new functions. Returns error messages."""
    PLUGIN_ERRORS.clear()
    for name in [k for k, f in REGISTRY.items() if f.source]:  # drop previous plug-in registrations
        del REGISTRY[name]
    if not os.path.isdir(folder):
        return []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(folder, fname)
        _LOADING.update(source=path, version=os.path.getmtime(path))
        try:
            mod_name = f"hansmeyer_plugin_{os.path.splitext(fname)[0]}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        except Exception:
            PLUGIN_ERRORS.append(f"{fname}: {traceback.format_exc(limit=2).strip().splitlines()[-1]}")
        finally:
            _LOADING.update(source="", version=0.0)
    return list(PLUGIN_ERRORS)


def signature(name: str):
    fd = REGISTRY.get(name)
    return None if fd is None else [fd.name, fd.version]


__all__ = ["field", "fold", "fields", "folds", "evaluate", "fold_domain", "load_plugins", "REGISTRY", "FuncDef",
           "FuncParam"]

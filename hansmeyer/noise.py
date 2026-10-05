"""Vectorised 3D noise: improved Perlin, fBm, ridged, domain-warped and Worley (cellular)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def _perm(seed: int) -> np.ndarray:
    p = np.random.default_rng(seed).permutation(256)
    return np.concatenate([p, p]).astype(np.int64)


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _grad(h, x, y, z):
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)


def perlin(p: np.ndarray, seed: int = 0) -> np.ndarray:
    """Ken Perlin's improved noise, roughly in [-1, 1]."""
    p = np.asarray(p, float)
    perm = _perm(int(seed))
    fl = np.floor(p)
    i = fl.astype(np.int64) & 255
    f = p - fl
    u, v, w = _fade(f[:, 0]), _fade(f[:, 1]), _fade(f[:, 2])
    X, Y, Z = i[:, 0], i[:, 1], i[:, 2]
    x, y, z = f[:, 0], f[:, 1], f[:, 2]
    A, B = perm[X] + Y, perm[X + 1] + Y
    AA, AB, BA, BB = perm[A] + Z, perm[A + 1] + Z, perm[B] + Z, perm[B + 1] + Z

    def lerp(t, a, b):
        return a + t * (b - a)

    return lerp(w,
                lerp(v, lerp(u, _grad(perm[AA], x, y, z), _grad(perm[BA], x - 1, y, z)),
                     lerp(u, _grad(perm[AB], x, y - 1, z), _grad(perm[BB], x - 1, y - 1, z))),
                lerp(v, lerp(u, _grad(perm[AA + 1], x, y, z - 1), _grad(perm[BA + 1], x - 1, y, z - 1)),
                     lerp(u, _grad(perm[AB + 1], x, y - 1, z - 1), _grad(perm[BB + 1], x - 1, y - 1, z - 1))))


def fbm(p: np.ndarray, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5, seed: int = 0) -> np.ndarray:
    total, amp, norm = np.zeros(len(p)), 1.0, 0.0
    q = np.asarray(p, float)
    for o in range(int(octaves)):
        total += amp * perlin(q, seed + o)
        norm += amp
        amp *= gain
        q = q * lacunarity
    return total / max(norm, 1e-9)


def ridged(p: np.ndarray, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5, seed: int = 0) -> np.ndarray:
    """Sharp ridges: 1 - |noise| per octave, mapped to [-1, 1]."""
    total, amp, norm = np.zeros(len(p)), 1.0, 0.0
    q = np.asarray(p, float)
    for o in range(int(octaves)):
        total += amp * (1.0 - np.abs(perlin(q, seed + o))) ** 2
        norm += amp
        amp *= gain
        q = q * lacunarity
    return 2.0 * total / max(norm, 1e-9) - 1.0


def warped(p: np.ndarray, warp: float = 1.0, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Domain-warped fBm (noise fed by noise): marbled, flowing patterns."""
    q = np.stack([fbm(p + o, octaves, seed=seed + 7) for o in ((0, 0, 0), (5.2, 1.3, 2.8), (1.7, 9.2, 4.1))], 1)
    return fbm(p + warp * q, octaves, seed=seed)


def _hash3(cells: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic pseudo-random points in [0,1)^3 per integer cell."""
    c = cells.astype(np.uint64)
    h = (c[..., 0] * np.uint64(73856093)) ^ (c[..., 1] * np.uint64(19349663)) ^ (c[..., 2] * np.uint64(83492791))
    h = h ^ np.uint64(seed * 2654435761 % (2**32))
    out = []
    for k in range(3):
        h = (h * np.uint64(6364136223846793005) + np.uint64(1442695040888963407 + k))
        out.append(((h >> np.uint64(33)) & np.uint64(0xFFFFFF)).astype(float) / float(0x1000000))
    return np.stack(out, -1)


def worley(p: np.ndarray, seed: int = 0, jitter: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Distances to the nearest and second-nearest feature point (F1, F2) of a jittered grid."""
    p = np.asarray(p, float)
    base = np.floor(p).astype(np.int64)
    offs = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    cells = base[:, None, :] + offs[None]  # (n, 27, 3)
    feat = cells + 0.5 + jitter * (_hash3(cells, seed) - 0.5)
    d = np.linalg.norm(feat - p[:, None, :], axis=-1)
    part = np.partition(d, 1, axis=1)
    return part[:, 0], part[:, 1]

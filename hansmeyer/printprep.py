"""Print preparation: turn any form into watertight solids for FDM printing.

Subdivision forms self-intersect, and open or porous ones have no inside at all,
so they cannot go to a slicer directly. This rebuilds them through a voxel grid:

  1. the form is turned so the chosen model direction points up, scaled to millimetres,
     and its surface is sampled densely and rasterised into a voxel shell
  2. the shell is closed (one-voxel dilation) and everything it encloses is filled,
     which unions all self-intersecting parts into one solid
  3. the shell is thickened to a minimum wall, so open skins (panels, porous forms)
     become printable; reliefs can instead be filled down to a flat base
  4. optionally the undersides are made self-supporting (material is added below every overhang
     as a smooth keel no flatter than a chosen angle) and the bottom is trimmed to a flat foot
  5. optionally the solid is cut by axis-aligned planes into 2, 4 or 8 parts, with
     matching holes for alignment pins drilled into every joint
  6. marching cubes on the softened voxel field extracts a closed, outward-oriented surface
     per part, which is smoothed (Taubin: no shrinking) while cut faces stay exactly flat
  7. each part is turned to need the least support and placed on the build plate
  8. features thinner than about two nozzle widths are flagged: they would not print

Overhangs are judged like a slicer's support "threshold angle": a downward-facing surface
whose slope from the horizontal is below the threshold needs support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .mesh import PolyMesh

UP_KEYS = ("x", "-x", "y", "-y", "z", "-z")
SIDE_NAMES = (("left", "right"), ("front", "back"), ("bottom", "top"))  # low / high side of a cut, per axis
PLATE_MM = 0.4  # surfaces this close to the build plate rest on it (never need support)


@dataclass
class PrintSettings:
    size_mm: float = 100.0  # size of the assembled object, measured along size_axis
    size_axis: str = "longest"  # "longest" | "width" (printer X) | "depth" (printer Y) | "height" (printer Z)
    voxel_mm: float = 0.3  # voxel edge length (the print model's resolution)
    wall_mm: float = 0.8  # minimum wall thickness (2 x a 0.4 mm nozzle)
    up: object = "y"  # model direction that points up on the printer: "x", "-x", "y", "-y", "z", "-z"
    base: bool = False  # fill every column down to a flat base (for reliefs)
    base_mm: float = 2.0  # base thickness below the lowest point of the form
    smooth: int = 8  # Taubin smoothing iterations (0 = raw voxel surface)
    soften: float = 0.7  # blur of the voxel solid (in voxels) before extraction: removes stair-steps
    nozzle_mm: float = 0.4
    max_grid: int = 600  # cap on voxels along the longest axis (memory / time guard)
    detail: int = 1  # 1 = full, 2 = half (a 2x coarser output surface: ~4x fewer triangles)
    undersides: float = 0.0  # > 0: fill below overhangs so no downward surface is flatter than this (deg)
    foot_mm: float = 0.0  # trim this much off the bottom so the print stands on a flat foot
    cut_x: float | None = None  # cut plane across the width at this fraction of it (None = no cut)
    cut_y: float | None = None  # ... across the depth
    cut_z: float | None = None  # ... across the height; a negative value cuts at the widest section
    part_up: str = "auto"  # cut parts print "auto" (turned for least support) or "assembly" (as assembled)
    pins: bool = True  # drill matching holes for alignment pins into every joint
    pin_mm: float = 2.0  # hole diameter (2.0 mm takes 1.75 mm filament as a dowel)
    pin_depth_mm: float = 5.0  # hole depth into each part


@dataclass
class PrintPart:
    name: str  # which side of each cut, e.g. "left front"; "" when the form is not cut
    up: str  # the assembled object's direction that points up when this part is printed
    v0: int  # this part's vertices are PrintResult.V[v0:v1], its triangles PrintResult.F[f0:f1]
    v1: int
    f0: int
    f1: int
    R: np.ndarray  # rotation assembled -> printed
    t: np.ndarray  # printed = assembled @ R.T + t (standing on its own plate, centred)
    shift: np.ndarray  # extra offset of this part in the combined layout (PrintResult.V)
    side: tuple  # -1 / 0 / +1 per axis: the part's side of each cut (0 = that axis is not cut)
    dims_mm: tuple  # printed size
    shells: int  # separate solids inside this part (1 unless a cut left loose pieces)


@dataclass
class PrintResult:
    V: np.ndarray  # (n, 3) every part on the build plate, side by side, in mm, Z up
    F: np.ndarray  # (m, 3) triangles, outward oriented
    voxel_mm: float
    grid: tuple
    dims_mm: tuple  # size of the assembled object
    volume_cm3: float
    thin_fraction: float  # share of the solid thinner than ~2 nozzle widths
    thin_vertices: np.ndarray  # (n,) bool: vertices in thin regions
    vertex_nz: np.ndarray  # (n,) printed Z of the smoothed vertex normals (for the overhang check)
    parts: list = field(default_factory=list)
    cuts: list = field(default_factory=list)  # (axis, fraction of the occupied extent) of each cut plane
    pins: int = 0
    notes: list = field(default_factory=list)
    seconds: float = 0.0
    _face_area: np.ndarray | None = field(default=None, repr=False)

    def mesh(self) -> PolyMesh:
        return _polymesh(self.V, self.F)

    def part_arrays(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """One part as printed on its own plate."""
        p = self.parts[i]
        return self.V[p.v0:p.v1] - p.shift, self.F[p.f0:p.f1] - p.v0

    def part_mesh(self, i: int) -> PolyMesh:
        return _polymesh(*self.part_arrays(i))

    def assembled(self, explode_mm: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All parts back in their assembled positions, pushed apart by explode_mm along every cut.
        Returns vertices (Z up), triangles and the part index of every vertex."""
        Vs, owner = [], []
        for i, p in enumerate(self.parts):
            Vp, _ = self.part_arrays(i)
            Va = (Vp - p.t) @ p.R  # R is orthonormal: its inverse is its transpose
            Vs.append(Va + explode_mm * 0.5 * np.asarray(p.side, float))
            owner.append(np.full(len(Va), i))
        return np.concatenate(Vs), self.F, np.concatenate(owner)

    def face_area(self) -> np.ndarray:
        if self._face_area is None:
            tri = self.V[self.F].astype(np.float64)
            self._face_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        return self._face_area

    def overhang(self, threshold_deg: float) -> tuple[np.ndarray, list, float]:
        """Where a slicer with this support threshold angle would add support.
        Returns (vertex mask, overhang area of each part in cm2, share of the whole surface)."""
        mask = overhang_mask(self.vertex_nz, self.V[:, 2], threshold_deg)
        area = self.face_area()
        a = np.where(mask[self.F].sum(1) >= 2, area, 0.0)
        per = [float(a[p.f0:p.f1].sum()) / 100.0 for p in self.parts]
        return mask, per, float(a.sum() / max(area.sum(), 1e-12))

    def mass_g(self, infill: float = 0.2, wall_share: float = 0.35, density: float = 1.24) -> float:
        """Rough PLA estimate: walls solid, the rest at `infill`."""
        solid_share = wall_share + (1 - wall_share) * infill
        return self.volume_cm3 * min(solid_share, 1.0) * density


def _polymesh(V, F) -> PolyMesh:
    n = len(F)
    return PolyMesh(np.asarray(V, float), np.arange(0, 3 * n + 1, 3), np.asarray(F).reshape(-1))


def overhang_mask(nz: np.ndarray, z: np.ndarray, threshold_deg: float) -> np.ndarray:
    """Downward-facing surface flatter than threshold_deg (slope from the horizontal), off the plate."""
    if threshold_deg <= 0:
        return np.zeros(len(nz), bool)
    return (-nz > np.cos(np.radians(threshold_deg)) - 1e-9) & (z > PLATE_MM)


# --------------------------------------------------------------- orientation
def up_vector(up) -> np.ndarray:
    if isinstance(up, str):
        sign = -1.0 if up.startswith("-") else 1.0
        v = np.zeros(3)
        v["xyz".index(up.lstrip("+-"))] = sign
        return v
    v = np.asarray(up, float)
    return v / np.linalg.norm(v)


def up_rotation(up) -> np.ndarray:
    """Smallest rotation taking the model direction `up` to the printer's +Z (never a mirror)."""
    u = up_vector(up)
    c = float(u[2])
    if c < -1 + 1e-9:  # upside down: half turn about X
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(u, [0.0, 0.0, 1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def to_print_coords(V: np.ndarray, up) -> np.ndarray:
    """Rotate so the chosen model direction becomes +Z (right-handed)."""
    return np.asarray(V, float) @ up_rotation(up).T


def vertex_normals(V: np.ndarray, F: np.ndarray, smooth: int = 0, A=None) -> np.ndarray:
    """Area-weighted vertex normals, optionally averaged over `smooth` rings of neighbours
    (so voxel stair-steps do not read as overhangs)."""
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.stack([sum(np.bincount(F[:, k], fn[:, c], len(V)) for k in range(3)) for c in range(3)], 1)
    if smooth > 0:
        A = _adjacency(F, len(V))[0] if A is None else A
        for _ in range(smooth):
            vn = vn + A @ vn
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def overhang_scores(V: np.ndarray, F: np.ndarray, threshold_deg: float = 45.0, vn=None) -> dict:
    """For each of the six axis directions as 'up': (overhang area, area resting on the plate)."""
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = vertex_normals(V, F, 2) if vn is None else vn
    n = n[F].sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    c = np.cos(np.radians(threshold_deg))
    out = {}
    for key in UP_KEYS:
        d = up_vector(key)
        h = V @ d
        resting = h[F].max(1) < h.min() + PLATE_MM
        nd = n @ d
        out[key] = (float(area[(nd < -c) & ~resting].sum()), float(area[resting & (nd < -0.9)].sum()))
    return out


def best_up(V: np.ndarray, F: np.ndarray, threshold_deg: float = 45.0, vn=None) -> str:
    """The axis direction to point up that needs the least support (a big flat base breaks ties)."""
    scores = overhang_scores(V, F, threshold_deg, vn)
    return min(UP_KEYS, key=lambda k: scores[k][0] - 0.5 * scores[k][1])


# ------------------------------------------------------------------ geometry
def scale_for(mesh: PolyMesh, s: PrintSettings) -> tuple[float, np.ndarray]:
    """Model -> mm scale for the requested size, and the assembled size (W, D, H) in mm."""
    ext = np.ptp(to_print_coords(mesh.V, s.up), axis=0)
    ref = {"width": ext[0], "depth": ext[1], "height": ext[2]}.get(s.size_axis, ext.max())
    if ref < 1e-6 * max(ext.max(), 1e-12):  # e.g. the height of a flat sheet: fall back to the longest side
        ref = ext.max()
    scale = s.size_mm / max(ref, 1e-12)
    return scale, ext * scale


def grid_for(mesh: PolyMesh, s: PrintSettings):
    """Scale factor, voxel size and grid shape for a mesh (used for estimates before running)."""
    scale, dims = scale_for(mesh, s)
    return (scale,) + grid_for_dims(dims, s)


def grid_for_dims(dims: np.ndarray, s: PrintSettings):
    """Voxel size, padding and grid shape for an assembled size (W, D, H) in mm."""
    voxel = max(s.voxel_mm, dims.max() / s.max_grid)
    pad = int(np.ceil(s.wall_mm / voxel)) + 3
    shape = np.ceil(dims / voxel).astype(int) + 2 * pad + 1
    if s.base:
        shape[2] += int(np.ceil(s.base_mm / voxel))
    return voxel, pad, tuple(int(x) for x in shape)


def finest_voxel(mesh: PolyMesh, s: PrintSettings) -> float:
    """The smallest voxel the grid cap allows at this size."""
    return float(scale_for(mesh, s)[1].max() / s.max_grid)


def _barycentric(n: int) -> np.ndarray:
    """Barycentric sample grid with n steps per edge: (k, 3) weights."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = i + j <= n
    i, j = i[keep], j[keep]
    return np.stack([n - i - j, i, j], 1) / n


def _rasterise(tris: np.ndarray, origin, voxel, shape, budget: int = 4_000_000) -> np.ndarray:
    """Mark every voxel touched by the triangles: each triangle is sampled on a barycentric grid
    fine enough (spacing < 0.7 voxel) that consecutive samples never skip a voxel."""
    grid = np.zeros(shape, bool)
    hi = np.array(shape) - 1
    e = np.stack([np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1), np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1),
                  np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)], 1).max(1)
    steps = np.maximum(np.ceil(e / (0.7 * voxel)).astype(np.int64), 1)
    for n in np.unique(steps):
        B = _barycentric(int(n))  # (k, 3)
        sel = np.nonzero(steps == n)[0]
        per = max(budget // len(B), 1)
        for s in range(0, len(sel), per):
            T = tris[sel[s:s + per]]  # (c, 3, 3)
            pts = np.einsum("kv,cvd->ckd", B, T).reshape(-1, 3)
            _mark(grid, pts, origin, voxel, hi)
    return grid


def _mark(grid, pts, origin, voxel, hi):
    idx = np.clip(np.floor((pts - origin) / voxel).astype(np.int64), 0, hi)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True


def _ball(r: int) -> np.ndarray:
    r = max(int(r), 1)
    x, y, z = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return x * x + y * y + z * z <= r * r + 0.25


def _disk(r: float) -> np.ndarray:
    R = int(np.ceil(r))
    x, y = np.mgrid[-R:R + 1, -R:R + 1]
    return x * x + y * y <= r * r


def _adjacency(F: np.ndarray, n: int):
    """Vertex adjacency (sparse) and degree. Each edge of a closed surface is listed by both of its
    triangles, so all weights are equal (2) - no slow de-duplication needed."""
    from scipy.sparse import coo_matrix

    a, b = F.reshape(-1), np.roll(F, -1, axis=1).reshape(-1)
    A = coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    return A, np.maximum(np.asarray(A.sum(1)), 1)


def taubin(V: np.ndarray, F: np.ndarray, iterations: int, lam: float = 0.5, mu: float = -0.53,
           project=None) -> np.ndarray:
    """Taubin lambda/mu smoothing: removes voxel steps without shrinking the form.
    `project(V)` (optional) is applied in place after every step, e.g. to keep cut faces flat."""
    if iterations <= 0:
        return V
    A, deg = _adjacency(F, len(V))
    V = V.astype(float, copy=True)
    for _ in range(iterations):
        for f in (lam, mu):
            V = V + f * (A @ V / deg - V)
            if project is not None:
                project(V)
    return V


def self_supporting(solid: np.ndarray, angle_deg: float) -> np.ndarray:
    """Add material below every overhang so each layer sits within the layer under it, grown by
    one layer height / tan(angle): overhangs become keels sloping at `angle` from the horizontal,
    horizontal holes get pointed (teardrop) tops, and upward-facing surfaces are untouched.
    Overhangs narrower than about two voxels (which slicers bridge) are left as they are."""
    out = solid.copy()
    step = 1.0 / np.tan(np.radians(min(max(angle_deg, 5.0), 85.0)))  # horizontal shrink per layer, in voxels
    carry, above = 0.0, None
    for z in range(out.shape[2] - 1, -1, -1):
        if above is not None:
            carry += step
            r = int(carry)
            carry -= r
            if r == 0:
                out[:, :, z] |= above
            else:
                rows, cols = np.nonzero(above.any(1))[0], np.nonzero(above.any(0))[0]
                if len(rows):
                    r0, r1 = max(rows[0] - 1, 0), rows[-1] + 2
                    c0, c1 = max(cols[0] - 1, 0), cols[-1] + 2
                    keep = ndimage.distance_transform_edt(np.pad(above[r0:r1, c0:c1], 1))[1:-1, 1:-1] > r
                    out[r0:r1, c0:c1, z] |= keep
        above = out[:, :, z] if out[:, :, z].any() else None
        if above is None:
            carry = 0.0
    return out


# ------------------------------------------------------------------- cutting
def _cut_indices(solid: np.ndarray, s: PrintSettings, step: int) -> tuple[list, list]:
    """[(axis, k)]: cut between voxel layers k-1 and k (aligned to the output step), and
    [(axis, fraction)]: where each cut sits within the solid's extent."""
    cuts, fracs = [], []
    for a, f in enumerate((s.cut_x, s.cut_y, s.cut_z)):
        if f is None:
            continue
        prof = solid.sum(axis=tuple(i for i in range(3) if i != a)).astype(float)
        occ = np.nonzero(prof)[0]
        if len(occ) < 4 * step:
            continue
        lo, hi = int(occ[0]), int(occ[-1]) + 1
        if f < 0:  # the widest cross-section, away from the ends: the biggest flat face for both parts
            sm = ndimage.uniform_filter1d(prof, size=max(3, (hi - lo) // 50))
            i0, i1 = lo + (hi - lo) // 6, hi - (hi - lo) // 6
            k = i0 + int(np.argmax(sm[i0:i1]))
        else:
            k = int(round(lo + min(max(f, 0.0), 1.0) * (hi - lo)))
        k = int(round(k / step)) * step
        k = min(max(k, (lo // step + 1) * step), (hi - 1) // step * step)
        cuts.append((a, k))
        fracs.append((a, (k - lo) / (hi - lo)))
    return cuts, fracs


def _drill_pins(solid: np.ndarray, cuts, voxel: float, s: PrintSettings, notes: list) -> int:
    """Matching blind holes across every joint, placed where the material is thickest
    (up to two per joint, each with at least a wall of material around it)."""
    r = 0.5 * s.pin_mm / voxel
    if r < 1.5:
        notes.append(f"alignment holes skipped: {s.pin_mm:.1f} mm is too small for the {voxel:.2f} mm voxel")
        return 0
    need = r + max(1.2, 3 * s.nozzle_mm) / voxel  # hole + surrounding wall, in voxels
    full_depth = max(int(round(s.pin_depth_mm / voxel)), 1)
    # the surface is later pulled back half a voxel (for rasterised surfaces), which widens holes: carve them smaller
    hole, keep_out = _disk(r - 0.5), _disk(need)
    count = 0
    for a, k in cuts:
        S = np.moveaxis(solid, a, -1)  # a view: (other axis 1, other axis 2, cut axis)
        others = [i for i in range(3) if i != a]
        sec = S[:, :, k - 1] & S[:, :, k]  # material on both sides of the cut
        bounds = []
        for j, b in enumerate(others):  # other cuts split this face into separate joints
            ks = [kk for aa, kk in cuts if aa == b]
            bounds.append([0] + ks + [sec.shape[j]])
            for kk in ks:
                if j == 0:
                    sec[kk - 1:kk + 1, :] = False
                else:
                    sec[:, kk - 1:kk + 1] = False
        edt = ndimage.distance_transform_edt(sec)
        for b0, b1 in zip(bounds[0][:-1], bounds[0][1:]):
            for c0, c1 in zip(bounds[1][:-1], bounds[1][1:]):
                D = edt[b0:b1, c0:c1].copy()
                spacing = max(8 * r, 0.35 * max(D.shape))
                ii, jj = np.indices(D.shape)
                placed = 0
                for _ in range(40):
                    if placed == 2:
                        break
                    flat = int(np.argmax(D))
                    if D.flat[flat] < need:
                        break
                    pi, pj = np.unravel_index(flat, D.shape)
                    pb, pc = pi + b0, pj + c0
                    depth = next((d for d in (full_depth, full_depth // 2) if d >= 2 and _fits(S, pb, pc, k, d, keep_out)), 0)
                    if depth:
                        _stamp(S, pb, pc, slice(k - depth, k + depth), hole)
                        placed += 1
                        D[(ii - pi) ** 2 + (jj - pj) ** 2 < spacing ** 2] = 0
                    else:
                        D[(ii - pi) ** 2 + (jj - pj) ** 2 < need ** 2] = 0
                count += placed
    return count


def _window(S, pb, pc, disk):
    R = disk.shape[0] // 2
    b0, c0 = pb - R, pc - R
    if b0 < 0 or c0 < 0 or b0 + disk.shape[0] > S.shape[0] or c0 + disk.shape[1] > S.shape[1]:
        return None
    return slice(b0, b0 + disk.shape[0]), slice(c0, c0 + disk.shape[1])


def _fits(S, pb, pc, k, depth, keep_out) -> bool:
    w = _window(S, pb, pc, keep_out)
    if w is None or k - depth < 0 or k + depth > S.shape[2]:
        return False
    block = S[w[0], w[1], k - depth:k + depth]
    return bool(block[keep_out].all())


def _stamp(S, pb, pc, zs, disk):
    w = _window(S, pb, pc, disk)
    S[w[0], w[1], zs] &= ~disk[:, :, None]


def _boxes(shape, cuts):
    """The parts' voxel boxes: every combination of the two sides of each cut."""
    spans = []
    for a in range(3):
        ks = [k for aa, k in cuts if aa == a]
        spans.append([(0, ks[0], -1), (ks[0], shape[a], 1)] if ks else [(0, shape[a], 0)])
    for sx in spans[0]:
        for sy in spans[1]:
            for sz in spans[2]:
                yield (sx, sy, sz)


# ------------------------------------------------------------ surface output
def _extract(sub: np.ndarray, origin: np.ndarray, voxel: float, step: int, s: PrintSettings, planes):
    """Closed surface of one voxel block (mm). Faces on the flat `planes` [(axis, layer, side)] - cuts
    and the foot, at voxel `layer` of this block, with the material on the `side` (+1 above, -1 below) -
    stay exactly on them and keep their full area, so mating parts meet without a gap."""
    from skimage.measure import marching_cubes

    vox = voxel
    if step > 1:
        # lower detail = a coarser volume (a coarse voxel is solid if any of its fine voxels is), not a
        # coarser marching-cubes step: stepping skips thin parts and leaves non-manifold edges
        pad_to = [(-n) % step for n in sub.shape]
        c = np.pad(sub, [(0, p) for p in pad_to])
        nx, ny, nz = (n // step for n in c.shape)
        sub = c.reshape(nx, step, ny, step, nz, step).any(axis=(1, 3, 5))
        vox = voxel * step
    P = 3  # empty border, wide enough for the blur
    vol = np.pad(sub, P).astype(np.float32)
    # the two voxel layers either side of each flat face, in the padded (output) grid
    faces = [(a, P + k // step + (0 if side > 0 else -1), P + k // step + (-1 if side > 0 else 0))
             for a, k, side in planes]
    if s.soften > 0:
        keep = [(a, np.take(vol, i, axis=a).copy(), o) for a, i, o in faces]
        ndimage.gaussian_filter(vol, s.soften, output=vol, mode="constant", cval=0.0)
        for (a, layer, o), (_, i, _) in zip(keep, faces):  # blurring would round the flat faces off
            idx = [slice(None)] * 3
            idx[a] = i
            vol[tuple(idx)] = layer
            idx[a] = o
            vol[tuple(idx)] = 0.0
    if vol.max() <= 0.5:
        return None  # a sliver that vanishes at this resolution
    V, F, _, _ = marching_cubes(vol, level=0.5, spacing=(vox,) * 3)
    del vol
    V = V.astype(np.float64) + origin + (0.5 - P) * vox  # samples sit at voxel centres
    tri = V[F]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        F = F[:, ::-1]  # make the winding outward before using it
    on = []
    for a, k, side in planes:
        c = origin[a] + k * voxel
        on.append((a, c, side, np.abs(V[:, a] - c) < 1e-3 * vox))
    free = np.ones(len(V), bool)
    for *_, m in on:
        free &= ~m
    # surface voxels reach on average half a voxel beyond the true surface: pull it back in
    vn = vertex_normals(V, F)
    V[free] -= 0.5 * vox * vn[free]

    def project(V):
        for a, c, side, m in on:
            V[m, a] = c
            V[:, a] = np.maximum(V[:, a], c) if side > 0 else np.minimum(V[:, a], c)

    project(V)
    return taubin(V, F, s.smooth, project=project if on else None), F.astype(np.int64)


def _shells(A, V, F) -> int:
    """Separate solid pieces (closed shells enclosing material; internal cavities don't count)."""
    from scipy.sparse.csgraph import connected_components

    k, label = connected_components(A, directed=False)
    if k == 1:
        return 1
    tri = V[F]
    vol = np.bincount(label[F[:, 0]], np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])), k)
    return int(max((vol > 0).sum(), 1))


def _settle(V: np.ndarray, voxel: float) -> None:
    """If the part rests on a flat face, lift the few rim vertices that smoothing pushed a hair below
    it, so the whole face (not one stray vertex) sits on the build plate."""
    z = V[:, 2]
    near = z < z.min() + 0.5 * voxel
    vals, counts = np.unique(np.round(z[near], 4), return_counts=True)
    if len(counts) and counts.max() >= 20:
        V[:, 2] = np.maximum(z, vals[np.argmax(counts)])


def _layout(sizes, gap: float = 10.0) -> list[np.ndarray]:
    """xy offsets placing footprints (w, d) in a near-square grid, centred on the origin."""
    n = len(sizes)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cw = [max([sizes[i][0] for i in range(c, n, cols)]) for c in range(cols)]
    rd = [max([sizes[i][1] for i in range(r * cols, min((r + 1) * cols, n))]) for r in range(rows)]
    xs = np.cumsum([0] + [w + gap for w in cw[:-1]]) + 0.5 * np.array(cw)
    ys = -(np.cumsum([0] + [d + gap for d in rd[:-1]]) + 0.5 * np.array(rd))
    xs -= 0.5 * (sum(cw) + gap * (cols - 1))
    ys += 0.5 * (sum(rd) + gap * (rows - 1))
    return [np.array([xs[i % cols], ys[i // cols], 0.0]) for i in range(n)]


# ---------------------------------------------------------------- pipeline
def prepare(mesh: PolyMesh, s: PrintSettings, progress=None) -> PrintResult:
    t0 = time.perf_counter()
    say = progress or (lambda msg: None)
    notes = []
    scale, voxel, pad, shape = grid_for(mesh, s)
    if voxel > s.voxel_mm + 1e-12:
        notes.append(f"voxel raised to {voxel:.3f} mm (grid capped at {s.max_grid} per axis)")

    P = to_print_coords(mesh.V, s.up) * scale
    lo = P.min(0)
    origin = lo - (pad + 0.5) * voxel  # half-voxel offset: flat faces never sit on voxel boundaries
    if s.base:
        origin[2] -= s.base_mm
    tris = P[mesh.triangles()]

    say("rasterising surface")
    surface = _rasterise(tris, origin, voxel, shape)  # voxels the surface passes through
    cube3 = np.ones((3, 3, 3), bool)
    sealed = ndimage.binary_dilation(surface, structure=cube3)  # seal diagonal gaps for the fill

    say("filling enclosed volume")
    filled = ndimage.binary_fill_holes(sealed)
    if not s.base and filled.sum() <= sealed.sum() * 1.01:
        notes.append("no enclosed volume (open or porous form): printing the skin at the wall thickness")
    # undo the sealing layer so the outside sits on the real surface (keeps the requested size)
    solid = ndimage.binary_erosion(filled, structure=cube3) | surface
    # thicken only open skins: surface that does not bound an enclosed interior (closed parts are
    # already solid, and thickening them would inflate the print beyond the requested size)
    r_wall = int(round((s.wall_mm / voxel - 1) / 2))  # skin = surface voxel + r on either side
    if r_wall >= 1:
        interior = filled & ~sealed
        # surface bands can be ~3 voxels thick where they cross the grid obliquely: allow 4
        open_skin = surface & ~ndimage.binary_dilation(interior, structure=cube3, iterations=4)
        if open_skin.any():
            say("thickening open skins")
            solid |= ndimage.binary_dilation(open_skin, structure=_ball(r_wall))
        del interior, open_skin
    del surface, sealed, filled
    if s.base:
        say("filling down to the base")
        # every voxel below solid material becomes solid; the plate bottom sits base_mm below the form
        solid |= np.flip(np.logical_or.accumulate(np.flip(solid, axis=2), axis=2), axis=2)
        solid[:, :, :pad] = False

    if s.undersides > 0:
        say("making the undersides self-supporting")
        solid = ndimage.binary_fill_holes(self_supporting(solid, s.undersides))  # no sealed air pockets
    step = max(int(s.detail), 1)
    foot = None
    if s.foot_mm > 0:
        occ = np.nonzero(solid.any(axis=(0, 1)))[0]
        if len(occ):
            k = int(occ[0] + max(round(s.foot_mm / voxel), 1))
            k = int(np.ceil(k / step)) * step
            solid[:, :, :k] = False
            foot = k
    cuts, cut_fracs = _cut_indices(solid, s, step)
    pins = 0
    if cuts and s.pins:
        say("drilling alignment holes")
        pins = _drill_pins(solid, cuts, voxel, s, notes)

    say("checking printable thickness")
    r_noz = max(int(s.nozzle_mm / voxel), 1)  # opening removes parts thinner than ~2 nozzle widths
    thin = solid & ~ndimage.binary_opening(solid, structure=_ball(r_noz))
    thin_fraction = float(thin.sum()) / max(int(solid.sum()), 1)
    thin_near = ndimage.binary_dilation(thin, iterations=2)
    del thin

    boxes = [b for b in _boxes(solid.shape, cuts) if solid[tuple(slice(b0, b1) for b0, b1, _ in b)].any()]
    raw = []
    for i, box in enumerate(boxes):
        say(f"extracting surface{f' (part {i + 1} of {len(boxes)})' if len(boxes) > 1 else ''}")
        sl = tuple(slice(b0, b1) for b0, b1, _ in box)
        start = np.array([b0 for b0, _, _ in box])
        planes = []  # (axis, voxel layer within this box, side the material is on)
        for a, k in cuts:
            if box[a][0] == k:
                planes.append((a, 0, 1))
            elif box[a][1] == k:
                planes.append((a, k - box[a][0], -1))
        if foot is not None and box[2][0] == 0:
            planes.append((2, foot, 1))  # the flat foot stays exactly flat
        out = _extract(solid[sl], origin + start * voxel, voxel, step, s, planes)
        if out is None:
            notes.append("a sliver too thin for this resolution was left out")
            continue
        Va, F = out
        vi = np.floor((Va - origin) / voxel).astype(np.int64) - start
        vi = np.clip(vi, 0, np.array(solid[sl].shape) - 1)
        tv = thin_near[sl][vi[:, 0], vi[:, 1], vi[:, 2]]
        side = tuple(sd for _, _, sd in box)
        name = " ".join(SIDE_NAMES[a][(sd + 1) // 2] for a, sd in enumerate(side) if sd)
        raw.append((Va, F, tv, side, name))
    del solid, thin_near

    say("placing parts on the plate")
    parts, Vs, Fs, thins, nzs = [], [], [], [], []
    for Va, F, tv, side, name in raw:
        A = _adjacency(F, len(Va))[0]
        vn = vertex_normals(Va, F, 2, A)
        up = "z"
        if cuts and s.part_up == "auto":
            up = best_up(Va, F, 45.0, vn)
        R = up_rotation(up)
        Vp = Va @ R.T
        _settle(Vp, voxel)
        t = -np.array([0.5 * (Vp[:, 0].min() + Vp[:, 0].max()), 0.5 * (Vp[:, 1].min() + Vp[:, 1].max()),
                       Vp[:, 2].min()])
        Vp += t
        parts.append(PrintPart(name, up, 0, len(Vp), 0, len(F), R, t, np.zeros(3), side,
                               tuple(float(x) for x in np.ptp(Vp, axis=0)), _shells(A, Vp, F)))
        Vs.append(Vp)
        Fs.append(F)
        thins.append(tv)
        nzs.append((vn @ R.T)[:, 2])
    shifts = _layout([p.dims_mm[:2] for p in parts]) if len(parts) > 1 else [np.zeros(3)]
    nv = nf = 0
    for p, Vp, F, sh in zip(parts, Vs, Fs, shifts):
        p.v0, p.v1, p.f0, p.f1, p.shift = nv, nv + len(Vp), nf, nf + len(F), sh
        nv, nf = nv + len(Vp), nf + len(F)
    V = np.concatenate([Vp + sh for Vp, sh in zip(Vs, shifts)])
    F = np.concatenate([F + p.v0 for p, F in zip(parts, Fs)])

    assembled = np.concatenate([(Vp - p.t) @ p.R for p, Vp in zip(parts, Vs)])
    tri = V[F]
    volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    for p in parts:
        if p.shells > 1:
            notes.append(f"part '{p.name}' holds {p.shells} separate pieces: glue them in place or move the cut")
    if thin_fraction > 0.02:
        notes.append(f"{100 * thin_fraction:.0f}% of the solid is thinner than ~{2 * s.nozzle_mm:.1f} mm: "
                     "fine detail will not print (scale up, or lower the iteration depth)")
    dims = tuple(float(x) for x in np.ptp(assembled, axis=0))
    return PrintResult(V.astype(np.float32), F.astype(np.int64), voxel, shape, dims, volume / 1000.0,
                       thin_fraction, np.concatenate(thins), np.concatenate(nzs).astype(np.float32), parts, cut_fracs,
                       pins, notes, time.perf_counter() - t0)


def stl_bytes(result: PrintResult, part: int | None = None) -> int:
    if part is None:
        return 84 + 50 * len(result.F)
    p = result.parts[part]
    return 84 + 50 * (p.f1 - p.f0)


def write_stl(path: str, result: PrintResult, part: int | None = None) -> dict:
    """Safe STL export of the whole result or one part (space check, temp file + rename),
    verified after writing."""
    from .meshio import atomic_write, ensure_space, validate_stl, write_stl as _write

    m = result.mesh() if part is None else result.part_mesh(part)
    ensure_space(path, stl_bytes(result, part))
    atomic_write(path, lambda tmp: _write(tmp, m))
    return validate_stl(path)


def prepare_exact(mesh: PolyMesh, s: PrintSettings, scale: float) -> PrintResult:
    """Print model straight from a mesh that is already a clean closed solid (e.g. a vessel): no voxel
    remesh, so exact spheres and flat rims stay exact. `scale` converts model units to millimetres."""
    t0 = time.perf_counter()
    notes = []
    if np.any(mesh.he_twin < 0):
        notes.append("the mesh is not closed: switch the vessel off and use the voxel print model")
    P = to_print_coords(mesh.V, s.up) * scale
    F = mesh.triangles().astype(np.int64)
    P[:, 2] -= P[:, 2].min()
    P[:, :2] -= 0.5 * (P[:, :2].min(0) + P[:, :2].max(0))
    tri = P[F]
    volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    if volume < 0:
        F, volume = F[:, ::-1].copy(), -volume
    wall = mesh.vattr.get("wall_mm")
    thin = (wall < 2 * s.nozzle_mm - 1e-6) if wall is not None else np.zeros(len(P), bool)
    thin_fraction = float(thin.mean())
    if thin_fraction > 0.02:
        notes.append(f"{100 * thin_fraction:.0f}% of the wall is thinner than {2 * s.nozzle_mm:.1f} mm "
                     "(2 nozzle widths): it may print with gaps")
    dims = tuple(float(x) for x in np.ptp(P, axis=0))
    part = PrintPart("", "z", 0, len(P), 0, len(F), np.eye(3), np.zeros(3), np.zeros(3), (0, 0, 0), dims, 1)
    return PrintResult(P.astype(np.float32), F, 0.0, (0, 0, 0), dims, volume / 1000.0, thin_fraction, thin,
                       vertex_normals(P, F)[:, 2].astype(np.float32), [part], [], 0, notes,
                       time.perf_counter() - t0)


def prepare_exact_arrays(V, face_ptr, face_idx, settings: dict, scale: float, wall=None) -> PrintResult:
    """Picklable entry point for prepare_exact."""
    m = PolyMesh(V, face_ptr, face_idx)
    if wall is not None:
        m.vattr["wall_mm"] = wall
    return prepare_exact(m, PrintSettings(**settings), scale)


def prepare_arrays(V, face_ptr, face_idx, settings: dict) -> PrintResult:
    """Picklable entry point for running print preparation in a separate process."""
    return prepare(PolyMesh(V, face_ptr, face_idx), PrintSettings(**settings))

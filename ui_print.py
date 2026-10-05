"""Output panel for app.py: watertight FDM print preparation (size, orientation, cutting into parts,
resolution, support check) and turntable renders."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from hansmeyer.printprep import (UP_KEYS, PrintSettings, best_up, finest_voxel, grid_for, prepare_arrays, scale_for,
                                 stl_bytes, up_rotation, write_stl)
from hansmeyer.view import turntable
from ui_common import WARN, toggle_button

PREVIEW = "print model"
PLANES = "cut planes"
OK = (0.45, 0.85, 0.55, 1.0)
GREY = (0.7, 0.7, 0.7, 1.0)
BASE = (0.86, 0.85, 0.82)
RED = (0.9, 0.25, 0.2)
PART_COLOURS = np.array([(0.90, 0.62, 0.30), (0.40, 0.65, 0.90), (0.55, 0.80, 0.45), (0.85, 0.45, 0.70),
                         (0.95, 0.85, 0.35), (0.50, 0.85, 0.85), (0.75, 0.60, 0.95), (0.95, 0.50, 0.45)])
NOZZLES = [0.25, 0.4, 0.6, 0.8]
PRINTERS = [("Bambu Lab A1 / P1S / X1C", (256.0, 256.0, 256.0)), ("Bambu Lab A1 mini", (180.0, 180.0, 180.0)),
            ("Prusa MK4 / MK4S", (250.0, 210.0, 220.0)), ("Prusa CORE One", (250.0, 220.0, 270.0)),
            ("Prusa MINI", (180.0, 180.0, 180.0)), ("Custom", None)]
SIZE_AXES = [("longest", "longest side"), ("height", "height (Z)"), ("width", "width (X)"), ("depth", "depth (Y)")]
QUALITY = [("Draft", 1.0, "voxel = nozzle width: fastest, smallest files"),
           ("Standard", 0.75, "a good match for FDM detail"),
           ("Fine", 0.5, "smoother surface, larger files; detail below the nozzle width still won't print")]
CUT_PRESETS = [("Whole", (None, None, None), "one piece"),
               ("2 halves", (None, None, -1.0), "one horizontal cut at the widest section: both halves print "
                                                "cut face down, with far less support"),
               ("4 quarters", (0.5, 0.5, None), "two vertical cuts through the middle: four pillars that "
                                                  "assemble into the whole, like the four corners of a cage"),
               ("8 pieces", (0.5, 0.5, -1.0), "all three cuts")]
AXIS_LABELS = ("across X (width)", "across Y (depth)", "across Z (height)")
GRID_LIMITS = [400, 600, 800, 1000]
BYTES_PER_VOXEL = 9.0  # peak memory of the voxel steps, measured on a 613^3 grid


def _ram_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):  # Windows
        return None


def _up_label(up) -> str:
    return (up if up.startswith("-") else "+" + up).upper()


def _how_printed(part) -> str:
    """Plain words for a part's print orientation."""
    a, sign = "xyz".index(part.up.lstrip("-")), (-1 if part.up.startswith("-") else 1)
    if part.side[a] == sign:  # the part's cut face along that axis sits at the bottom
        return "cut face down"
    if part.up == "z":
        return "as assembled"
    return "upside down" if part.up == "-z" else f"on its side ({_up_label(part.up)} up)"


def _bytes(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


class PrintPanel:
    def __init__(self, app):
        self.app = app
        self.s = PrintSettings()
        self.nozzle_idx = 1
        self.printer_idx = int(app.settings.get("printer", 0)) % len(PRINTERS)
        self.custom_bed = list(app.settings.get("custom_bed", [220.0, 220.0, 250.0]))
        self.job = None  # {"future", "started", "key", "settings"}
        self.executor = None  # one worker process: heavy voxel work never blocks or crashes the UI
        self.last_export = []
        self.result = None
        self.result_key = None
        self.result_settings = None
        self.ref = None  # (triangles, output resolution, seconds, voxels) of the last run, for estimates
        self.showing = False
        self.view = "layout"  # print layout | assembled
        self.explode = 15.0
        self.colour = "support"  # support | thin | parts
        self.support_deg = 30
        self._overhang = None  # (threshold, mask, per-part cm2, share)
        self._cache = {}
        self._planes_key = None
        self.defaults_for_shape_pending = True  # choose up axis / base once the input mesh exists
        self.tt = {"frames": 72, "seconds": 6.0, "size": 640, "elevation": 20.0, "format": 0}

    # ------------------------------------------------------------- helpers
    def _design_key(self):
        d = self.app.design
        keys = d.level_keys(d.full_depth, self.app.root)
        return (keys[-1] if keys else d.base_key(self.app.root), tuple(vars(self.s).items()))

    def defaults_for_shape(self):
        """Panels print lying flat with a solid base; everything else stands up."""
        panel = self.app.base_mesh is not None and np.any(self.app.base_mesh.vert_is_boundary)
        self.s.up = "z" if panel else "y"
        self.s.base = bool(panel)

    def bed(self) -> np.ndarray:
        dims = PRINTERS[self.printer_idx][1]
        return np.array(dims if dims is not None else self.custom_bed, float)

    def n_parts(self) -> int:
        return 2 ** sum(f is not None for f in (self.s.cut_x, self.s.cut_y, self.s.cut_z))

    def _mesh(self):
        return self.app.result.mesh if self.app.result is not None else None

    def _cached(self, name, key, fn):
        hit = self._cache.get(name)
        if hit is None or hit[0] != key:
            hit = (key, fn())
            self._cache[name] = hit
        return hit[1]

    def dims_now(self):
        """Assembled size (W, D, H) in mm for the current settings, without running anything."""
        m = self._mesh()
        if m is None:
            return None
        ext = self._cached("ext", (id(m), str(self.s.up)), lambda: np.ptp(m.V @ up_rotation(self.s.up).T, axis=0))
        ref = {"width": ext[0], "depth": ext[1], "height": ext[2]}.get(self.s.size_axis, ext.max())
        if ref < 1e-6 * ext.max():
            ref = ext.max()
        return ext * (self.s.size_mm / max(ref, 1e-12))

    def part_dims_estimate(self, dims):
        """Largest part before running: the result's parts if it matches, else the cut fractions."""
        r, rs = self.result, self.result_settings
        same_cuts = rs is not None and all(rs[k] == getattr(self.s, k) for k in ("cut_x", "cut_y", "cut_z", "up"))
        if r is not None and same_cuts and len(r.parts) > 1:
            k = dims.max() / max(max(r.dims_mm), 1e-12)
            return max((np.array(p.dims_mm) * k for p in r.parts), key=lambda d: d.prod())
        fr = [1.0 if f is None else (0.6 if f < 0 else max(f, 1 - f)) for f in (self.s.cut_x, self.s.cut_y, self.s.cut_z)]
        return dims * np.array(fr)

    def fits(self, dims, any_way: bool) -> bool:
        """Whole objects keep their up direction (they may turn on the plate); cut parts may lie any way."""
        bed = self.bed()
        if any_way:
            return bool(np.all(np.sort(dims) <= np.sort(bed) + 1e-6))
        return bool(np.all(np.sort(dims[:2]) <= np.sort(bed[:2]) + 1e-6) and dims[2] <= bed[2] + 1e-6)

    def fit_to_bed(self):
        dims = self.dims_now()
        if dims is None:
            return
        bed = self.bed() - 4.0  # a little margin
        if self.n_parts() == 1:
            xy, bxy = np.sort(dims[:2]), np.sort(bed[:2])
            k = min(bxy[0] / max(xy[0], 1e-9), bxy[1] / max(xy[1], 1e-9), bed[2] / max(dims[2], 1e-9))
        else:
            k = float(np.min(np.sort(bed) / np.maximum(np.sort(self.part_dims_estimate(dims)), 1e-9)))
        self.s.size_mm = float(np.floor(self.s.size_mm * k * 2) / 2)

    def effective_voxel(self) -> float:
        m = self._mesh()
        return max(self.s.voxel_mm, finest_voxel(m, self.s)) if m is not None else self.s.voxel_mm

    def estimate(self):
        """(triangles, STL bytes, RAM bytes, seconds or None) for the current settings."""
        m = self._mesh()
        if m is None:
            return None
        s = self.s
        _, voxel, _, shape = grid_for(m, s)
        res = float(self.dims_now().max()) / (voxel * s.detail)  # output cells along the longest side
        nvox = float(np.prod(shape))
        if self.ref is not None:
            tris = self.ref[0] * (res / self.ref[1]) ** 2
            secs = self.ref[2] * nvox / self.ref[3]
        else:
            def area():
                T = m.V[m.triangles()]
                return float(0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1).sum())
            a = self._cached("area", id(m), area)
            scale, _ = scale_for(m, s)
            tris = 1.6 * a * scale ** 2 / (voxel * s.detail) ** 2  # marching cubes: ~2.7 per cell, minus hidden folds
            secs = None
        return tris, 84 + 50 * tris, nvox * BYTES_PER_VOXEL, secs

    def show_form(self):
        ps.remove_surface_mesh(PREVIEW, error_if_absent=False)
        if ps.has_surface_mesh("form"):
            ps.get_surface_mesh("form").set_enabled(True)
        self.showing = False
        self._planes_key = None

    # ------------------------------------------------------- print preview
    def overhang(self):
        r = self.result
        if self._overhang is None or self._overhang[0] != self.support_deg:
            self._overhang = (self.support_deg, *r.overhang(self.support_deg))
        return self._overhang[1:]

    def _owner(self):
        r = self.result
        return np.repeat(np.arange(len(r.parts)), [p.v1 - p.v0 for p in r.parts])

    def _colours(self):
        r = self.result
        if self.colour == "parts":
            return PART_COLOURS[self._owner() % len(PART_COLOURS)]
        col = np.tile(BASE, (len(r.V), 1))
        col[self.overhang()[0] if self.colour == "support" else r.thin_vertices] = RED
        return col

    def show_print(self, reset_camera: bool = True):
        r = self.result
        if r is None:
            return
        assembled = self.view == "assembled" and len(r.parts) > 1
        V, F = (r.assembled(self.explode)[:2]) if assembled else (r.V, r.F)
        sm = ps.register_surface_mesh(PREVIEW, V, F, color=BASE, material="clay", smooth_shade=True,
                                      back_face_policy="custom", back_face_color=(0.6, 0.2, 0.2))
        sm.add_color_quantity("print colours", self._colours(), enabled=True)
        if ps.has_surface_mesh("form"):
            ps.get_surface_mesh("form").set_enabled(False)
        ps.remove_surface_mesh(PLANES, error_if_absent=False)
        self._planes_key = None
        self.showing = True
        if reset_camera:
            c = 0.5 * (V.min(0) + V.max(0))
            ext = float(np.linalg.norm(V.max(0) - V.min(0)))
            ps.set_up_dir("z_up")
            ps.look_at(tuple(c + np.array([1.0, -1.4, 0.9]) * ext * 0.8), tuple(c))

    def recolour(self):
        if self.showing and self.result is not None and ps.has_surface_mesh(PREVIEW):
            ps.get_surface_mesh(PREVIEW).add_color_quantity("print colours", self._colours(), enabled=True)

    def update_planes(self, visible: bool):
        """Translucent cut planes over the form, so cuts can be placed before preparing."""
        m, s = self._mesh(), self.s
        cuts = (s.cut_x, s.cut_y, s.cut_z)
        visible = visible and m is not None and not self.showing and any(f is not None for f in cuts)
        widest = self._widest_fraction()
        key = (visible, id(m), str(s.up), cuts, widest)
        if key == self._planes_key:
            return
        self._planes_key = key
        ps.remove_surface_mesh(PLANES, error_if_absent=False)
        if not visible:
            return
        R = up_rotation(s.up)
        P = m.V @ R.T
        lo, hi = P.min(0), P.max(0)
        pad = 0.08 * float((hi - lo).max())
        quads = []
        for a, f in enumerate(cuts):
            if f is None:
                continue
            b, c = [i for i in range(3) if i != a]
            q = np.zeros((4, 3))
            q[:, a] = lo[a] + (widest if f < 0 else f) * (hi[a] - lo[a])
            q[:, b] = [lo[b] - pad, hi[b] + pad, hi[b] + pad, lo[b] - pad]
            q[:, c] = [lo[c] - pad, lo[c] - pad, hi[c] + pad, hi[c] + pad]
            quads.append(q)
        V = np.concatenate(quads) @ R  # printer frame -> model frame
        F = np.arange(len(V)).reshape(-1, 4)
        pm = ps.register_surface_mesh(PLANES, V, F, color=(0.95, 0.55, 0.15), transparency=0.35, smooth_shade=False)
        pm.set_back_face_policy("identical")

    def _widest_fraction(self) -> float:
        """Where the last run put a 'widest section' height cut (0.5 until one ran)."""
        rs, r = self.result_settings, self.result
        if r is not None and rs is not None and rs["cut_z"] is not None and rs["cut_z"] < 0 and rs["up"] == self.s.up:
            return next((f for a, f in r.cuts if a == 2), 0.5)
        return 0.5

    # ------------------------------------------------------------ the job
    def _worker(self):
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            atexit.register(self.executor.shutdown, wait=False, cancel_futures=True)
        return self.executor

    def busy(self) -> bool:
        return self.job is not None and not self.job["future"].done()

    def start(self):
        if self.busy():
            return
        self.app.bake()  # print at full depth
        if not self.app.result:
            return
        m = self.app.result.mesh
        self.s.nozzle_mm = NOZZLES[self.nozzle_idx]
        settings = dict(vars(self.s))
        future = self._worker().submit(prepare_arrays, m.V, m.face_ptr, m.face_idx, settings)
        self.job = {"future": future, "started": time.perf_counter(), "key": self._design_key(), "settings": settings}
        self.app.error = ""

    def poll(self):
        """Per frame: pick up a finished job on the main thread (Polyscope calls must happen here)."""
        if self.defaults_for_shape_pending and self.app.base_mesh is not None and not self.app.dirty:
            self.defaults_for_shape()
            self.defaults_for_shape_pending = False
        if self.job is None or not self.job["future"].done():
            return
        job, self.job = self.job, None
        try:
            result = job["future"].result()
        except BrokenProcessPool:
            self.executor = None  # the worker died (usually out of memory): start a fresh one next time
            self.app.error = "Print prep crashed (out of memory?): use a coarser resolution or a smaller size"
            return
        except MemoryError:
            self.app.error = "Print prep ran out of memory: use a coarser resolution or a smaller size"
            return
        except Exception as e:
            self.app.error = f"Print prep failed: {type(e).__name__}: {e}"
            return
        self.result, self.result_key, self.result_settings = result, job["key"], job["settings"]
        self._overhang = None
        self.last_export = []
        res = max(result.dims_mm) / (result.voxel_mm * job["settings"]["detail"])
        self.ref = (len(result.F), res, result.seconds, float(np.prod(result.grid)))
        self.view = "layout"
        self.show_print()
        n = len(result.parts)
        self.app.status = f"Print model ready in {result.seconds:.0f}s" + (f": {n} parts" if n > 1 else "")

    def export(self):
        r = self.result
        if r is None:
            return
        name = "".join(c for c in self.app.save_name if c.isalnum() or c in "-_") or "form"
        folder = os.path.join(self.app.root, "exports")
        stem = f"{name}_print_{int(round(self.s.size_mm))}mm"
        n = len(r.parts)
        jobs = [(f"{stem}.stl", None)] if n == 1 else [
            (f"{stem}_part{i + 1}of{n}_{p.name.replace(' ', '-')}.stl", i) for i, p in enumerate(r.parts)]
        self.last_export = []
        try:
            os.makedirs(folder, exist_ok=True)
            for fname, part in jobs:
                info = write_stl(os.path.join(folder, fname), r, part)
                self.last_export.append((fname, info))
        except OSError as e:
            self.app.error = f"Export failed: {e}"
            return
        bad = [f for f, info in self.last_export if not (info["complete"] and info["watertight"])]
        if bad:
            self.app.error = f"Exported file(s) did not verify: {', '.join(bad)}"
        else:
            self.app.status = (f"Exported {n} file(s) to exports/: verified complete and watertight")

    # ------------------------------------------------------------------ ui
    def ui(self):
        self.poll()
        is_open = psim.CollapsingHeader("Print (FDM, watertight STL)")
        if is_open:
            self.print_ui()
        self.update_planes(is_open)
        if psim.CollapsingHeader("Turntable (GIF / MP4)"):
            self.turntable_ui()

    def print_ui(self):
        psim.TextWrapped("Rebuilds the form as closed solids a slicer can read: self-intersections merged, open "
                         "skins thickened, sized in mm. Cut it into parts to print with far less support.")
        self.printer_ui()
        self.size_ui()
        self.orientation_ui()
        self.cut_ui()
        self.resolution_ui()
        self.advanced_ui()
        psim.Spacing()
        if self.busy():
            psim.TextColored(WARN, f"working in a background process... {time.perf_counter() - self.job['started']:.0f}s")
        elif psim.Button("Prepare print model"):
            self.start()
        psim.SetItemTooltip("Bakes the full depth first, then runs in a separate process (the app stays responsive).")
        psim.SameLine()
        if psim.Button("Auto settings for this shape"):
            self.defaults_for_shape()
        psim.SetItemTooltip("Panels lie flat on a solid base; everything else stands up (model Y up).")
        self.result_ui()

    def printer_ui(self):
        psim.SeparatorText("Printer")
        psim.PushItemWidth(360)
        names = [n if d is None else f"{n}  -  {d[0]:.0f} x {d[1]:.0f} x {d[2]:.0f}" for n, d in PRINTERS]
        changed, self.printer_idx = psim.Combo("bed##printer", self.printer_idx, names)
        psim.PopItemWidth()
        if changed:
            self.app.remember("printer", self.printer_idx)
        if PRINTERS[self.printer_idx][1] is None:
            changed, v = psim.InputFloat3("bed W x D x H (mm)", self.custom_bed, "%.0f")
            if changed:
                self.custom_bed = [max(20.0, float(x)) for x in v]
                self.app.remember("custom_bed", self.custom_bed)
        psim.PushItemWidth(90)
        _, self.nozzle_idx = psim.Combo("nozzle", self.nozzle_idx, [f"{n} mm" for n in NOZZLES])
        psim.PopItemWidth()

    def size_ui(self):
        s = self.s
        psim.SeparatorText("Size")
        keys = [k for k, _ in SIZE_AXES]
        psim.PushItemWidth(120)
        changed, i = psim.Combo("##sizeaxis", keys.index(s.size_axis) if s.size_axis in keys else 0,
                                [label for _, label in SIZE_AXES])
        psim.PopItemWidth()
        if changed:
            s.size_axis = keys[i]
        psim.SameLine()
        psim.PushItemWidth(120)
        _, v = psim.InputFloat("mm##size", s.size_mm, 1.0, 10.0, "%.1f")
        psim.PopItemWidth()
        s.size_mm = float(min(max(v, 5.0), 2000.0))
        psim.SetItemTooltip("Type a value, or use - / + (hold Ctrl for steps of 10).")
        for mm in (50, 100, 150, 200):
            if toggle_button(f"{mm}##size{mm}", abs(s.size_mm - mm) < 1e-6):
                s.size_mm = float(mm)
            psim.SameLine()
        if psim.Button("Fit bed"):
            self.fit_to_bed()
        psim.SetItemTooltip("Largest size that fits the printer bed (cut parts are checked one by one).")
        dims = self.dims_now()
        if dims is None:
            return
        bed = self.bed()
        text = f"W {dims[0]:.1f} x D {dims[1]:.1f} x H {dims[2]:.1f} mm"
        if self.n_parts() == 1:
            ok = self.fits(dims, any_way=False)
            psim.TextColored(OK if ok else WARN, text + ("  - fits the bed" if ok else
                             f"  - bigger than the {bed[0]:.0f} x {bed[1]:.0f} x {bed[2]:.0f} bed: cut it or Fit bed"))
        else:
            psim.TextColored(GREY, text + " assembled")
            part = self.part_dims_estimate(dims)
            ok = self.fits(part, any_way=True)
            psim.TextColored(OK if ok else WARN, f"largest part ~{part[0]:.0f} x {part[1]:.0f} x {part[2]:.0f} mm"
                             + ("  - fits the bed" if ok else "  - too big for the bed: Fit bed or add a cut"))

    def orientation_ui(self):
        s = self.s
        psim.SeparatorText("Orientation")
        psim.Text("Up:")
        for key in UP_KEYS:
            psim.SameLine()
            if toggle_button(f"{_up_label(key)}##up", str(s.up) == key or (key == "y" and s.up == "+y")):
                s.up = key
        psim.SameLine()
        if psim.Button("Auto##up") and self._mesh() is not None:
            m = self._mesh()
            s.up = best_up(m.V, m.triangles(), float(self.support_deg))
            self.app.status = f"Least support with model {_up_label(s.up)} up"
        psim.SetItemTooltip("Which model axis points up on the printer. Auto picks the one with the least overhang.\n"
                            "With cuts, this sets the cut directions; each part is then turned for printing on its own.")

    def cut_ui(self):
        s = self.s
        psim.SeparatorText("Cut into parts")
        current = (s.cut_x, s.cut_y, s.cut_z)
        for i, (label, cuts, tip) in enumerate(CUT_PRESETS):
            if i:
                psim.SameLine()
            if toggle_button(f"{label}##cut", tuple(f is not None for f in current) == tuple(f is not None for f in cuts)):
                s.cut_x, s.cut_y, s.cut_z = cuts
            psim.SetItemTooltip(tip)
        names = ("cut_x", "cut_y", "cut_z")
        for a, attr in enumerate(names):
            f = getattr(s, attr)
            on = f is not None
            changed, on = psim.Checkbox(f"{AXIS_LABELS[a]}##cut{a}", on)
            if changed:
                setattr(s, attr, (-1.0 if a == 2 else 0.5) if on else None)
                f = getattr(s, attr)
            if f is None:
                continue
            psim.SameLine()
            if a == 2:
                changed, widest = psim.Checkbox("widest##cutw", f < 0)
                psim.SetItemTooltip("Cut where the form is widest: the biggest flat face for both halves to stand on.")
                if changed:
                    s.cut_z = -1.0 if widest else round(self._widest_fraction(), 2)
                if s.cut_z < 0:
                    continue
                psim.SameLine()
            psim.PushItemWidth(110)
            _, pct = psim.SliderFloat(f"position##cutp{a}", 100 * getattr(s, attr), 5.0, 95.0, "%.0f %%")
            psim.PopItemWidth()
            setattr(s, attr, pct / 100.0)
        if self.n_parts() == 1:
            return
        psim.TextDisabled(f"-> {self.n_parts()} parts. The orange planes on the form show the cuts.")
        _, s.pins = psim.Checkbox("alignment pin holes", s.pins)
        psim.SetItemTooltip("Matching blind holes in both faces of every joint, for gluing the parts in register.\n"
                            "2.0 mm takes a piece of 1.75 mm filament as the pin.")
        if s.pins:
            psim.SameLine()
            psim.PushItemWidth(70)
            _, s.pin_mm = psim.InputFloat("dia##pin", s.pin_mm, 0.0, 0.0, "%.2f")
            psim.SameLine()
            _, s.pin_depth_mm = psim.InputFloat("depth (mm)##pin", s.pin_depth_mm, 0.0, 0.0, "%.1f")
            psim.PopItemWidth()
            s.pin_mm = float(min(max(s.pin_mm, 0.5), 20.0))
            s.pin_depth_mm = float(min(max(s.pin_depth_mm, 1.0), 50.0))
        auto = s.part_up == "auto"
        if psim.RadioButton("turn each part for least support", auto):
            s.part_up = "auto"
        psim.SameLine()
        if psim.RadioButton("print as assembled", not auto):
            s.part_up = "assembly"

    def resolution_ui(self):
        s = self.s
        psim.SeparatorText("Resolution")
        nozzle = NOZZLES[self.nozzle_idx]
        finest = self.effective_voxel() if self._mesh() is None else max(finest_voxel(self._mesh(), s), 0.05)
        for i, (label, k, tip) in enumerate(QUALITY):
            v = round(k * nozzle, 3)
            if i:
                psim.SameLine()
            if toggle_button(f"{label} {v:.2f}##q", abs(s.voxel_mm - v) < 1e-6):
                s.voxel_mm = v
            psim.SetItemTooltip(f"{tip}.")
        psim.SameLine()
        if toggle_button(f"Max {finest:.2f}##q", abs(s.voxel_mm - finest) < 1e-6 or s.voxel_mm < finest - 1e-9):
            s.voxel_mm = round(finest, 3)
        psim.SetItemTooltip("The finest the grid limit allows at this size (raise 'grid limit' under Advanced).")
        eff = self.effective_voxel()
        est = self.estimate()
        line = f"detail {eff:.2f} mm"
        if est is not None:
            tris, nbytes, ram, secs = est
            line += f"  |  ~{tris / 1e6:.1f} M triangles  |  STL ~{nbytes / 1e6:.0f} MB"
            psim.Text(line)
            total = _ram_bytes()
            heavy = total is not None and ram > 0.6 * total
            psim.TextColored(WARN if heavy else GREY, f"needs ~{_bytes(ram)} RAM"
                             + (f", ~{secs / 60:.1f} min" if secs is not None and secs > 90 else
                                (f", ~{secs:.0f} s" if secs is not None else ""))
                             + ("  - close to this computer's memory" if heavy else ""))
        else:
            psim.Text(line)
        if eff > s.voxel_mm + 1e-9:
            psim.TextColored(WARN, f"Limited to {eff:.2f} mm by the grid limit ({s.max_grid} voxels along the "
                                   f"longest side). Raise it under Advanced, or print smaller.")

    def advanced_ui(self):
        s = self.s
        if not psim.TreeNode("Advanced##print"):
            return
        _, s.voxel_mm = psim.SliderFloat("voxel (mm)", s.voxel_mm, 0.05, 1.5, "%.3f")
        psim.SetItemTooltip("Exact resolution of the print model (the buttons above set this).")
        labels = []
        dims = self.dims_now()
        for g in GRID_LIMITS:
            text = f"{g}"
            if dims is not None:
                v = max(s.voxel_mm, float(dims.max()) / g)
                vox = np.prod(np.ceil(dims / v) + 12)
                text += f"  (finest {float(dims.max()) / g:.2f} mm, ~{_bytes(vox * BYTES_PER_VOXEL)} RAM)"
            labels.append(text)
        idx = GRID_LIMITS.index(s.max_grid) if s.max_grid in GRID_LIMITS else 1
        psim.PushItemWidth(260)
        _, idx = psim.Combo("grid limit", idx, labels)
        psim.PopItemWidth()
        s.max_grid = GRID_LIMITS[idx]
        psim.SetItemTooltip("Most voxels along the longest side. Higher = finer detail on big prints, more memory.")
        _, s.wall_mm = psim.SliderFloat("min wall (mm)", s.wall_mm, 0.4, 5.0, "%.2f")
        psim.SetItemTooltip("Open or porous forms are printed as a skin of this thickness (>= 2 x nozzle).")
        _, s.smooth = psim.SliderInt("smoothing", int(s.smooth), 0, 30)
        _, s.soften = psim.SliderFloat("soften voxels", s.soften, 0.0, 1.5, "%.2f")
        psim.SetItemTooltip("Blurs the voxel solid before the surface is extracted: removes stair-steps "
                            "(0 = raw voxels).")
        _, d = psim.Combo("mesh detail", 0 if s.detail == 1 else 1, ["full", "half (~4x smaller file)"])
        s.detail = 1 if d == 0 else 2
        _, s.base = psim.Checkbox("solid base (reliefs)", s.base)
        psim.SetItemTooltip("Fill everything below the form down to a flat plate: turns a relief panel into a tile.")
        if s.base:
            psim.SameLine()
            psim.PushItemWidth(90)
            _, s.base_mm = psim.SliderFloat("thickness##base", s.base_mm, 0.5, 10.0, "%.1f")
            psim.PopItemWidth()
        psim.TreePop()

    def result_ui(self):
        r = self.result
        if r is None:
            return
        stale = self.result_key != self._design_key()
        n = len(r.parts)
        psim.SeparatorText("Print model")
        psim.TextColored(WARN if stale else OK,
                         ("STALE (form or settings changed) - " if stale else "")
                         + (f"{n} parts, " if n > 1 else "")
                         + f"{r.dims_mm[0]:.1f} x {r.dims_mm[1]:.1f} x {r.dims_mm[2]:.1f} mm"
                         + (" assembled" if n > 1 else "") + f", {len(r.F):,} triangles")
        psim.Text(f"volume {r.volume_cm3:.1f} cm3  ~{r.mass_g():.0f} g PLA at 20% infill, detail {r.voxel_mm:.2f} mm"
                  + (f", {r.pins} pin holes" if n > 1 else ""))
        need, free = stl_bytes(r), shutil.disk_usage(self.app.root).free
        psim.TextColored(WARN if free < need + 64e6 else GREY, f"STL ~{need / 1e6:.0f} MB, {free / 1e9:.1f} GB free on disk")
        for note in r.notes:
            psim.TextColored(WARN, note)
        if r.thin_fraction <= 0.02:
            psim.TextColored(OK, "Thickness check passed")

        # support check
        psim.PushItemWidth(120)
        changed, self.support_deg = psim.SliderInt("support threshold angle", int(self.support_deg), 5, 85, "%d deg")
        psim.PopItemWidth()
        psim.SetItemTooltip("Same as the slicer's support 'threshold angle' (Bambu / Orca / Prusa): downward surfaces\n"
                            "flatter than this, measured from the horizontal, get support. 30 = only flat overhangs;\n"
                            "60 = steeper ones too. A higher number means MORE support, not less.")
        if changed and self.colour == "support":
            self.recolour()
        _, per, share = self.overhang()
        total = sum(per)
        psim.TextColored(OK if share < 0.03 else WARN,
                         f"~{total:.0f} cm2 ({100 * share:.1f}% of the surface) needs support (red)")
        if n == 1 and share >= 0.03:
            psim.TextWrapped("Lots of support: cut it into parts (2 halves usually helps most), or try another "
                             "up direction.")
        bed = self.bed()
        if n > 1:
            for i, p in enumerate(r.parts):
                ok = self.fits(np.array(p.dims_mm), any_way=False)
                psim.TextColored(OK if ok else WARN,
                                 f"{i + 1}. {p.name}: {p.dims_mm[0]:.0f} x {p.dims_mm[1]:.0f} x {p.dims_mm[2]:.0f} mm, "
                                 f"prints {_how_printed(p)}, support ~{per[i]:.0f} cm2"
                                 + ("" if ok else " - too big for the bed")
                                 + (f", {p.shells} loose pieces" if p.shells > 1 else ""))
        elif not self.fits(np.array(r.parts[0].dims_mm), any_way=False):
            psim.TextColored(WARN, f"Bigger than the {bed[0]:.0f} x {bed[1]:.0f} x {bed[2]:.0f} bed")

        # view
        if n > 1:
            if toggle_button("Print layout##view", self.view == "layout"):
                self.view = "layout"
                self.show_print()
            psim.SameLine()
            if toggle_button("Assembled##view", self.view == "assembled"):
                self.view = "assembled"
                self.show_print()
            if self.view == "assembled":
                psim.SameLine()
                psim.PushItemWidth(100)
                changed, self.explode = psim.SliderFloat("gap (mm)##explode", self.explode, 0.0, 60.0, "%.0f")
                psim.PopItemWidth()
                if changed and self.showing:
                    self.show_print(reset_camera=False)
        psim.Text("Colour:")
        for key, label in (("support", "needs support"), ("thin", "too thin")) + ((("parts", "parts"),) if n > 1 else ()):
            psim.SameLine()
            if toggle_button(f"{label}##colour", self.colour == key):
                self.colour = key
                self.recolour()
        if self.showing:
            if psim.Button("Show form"):
                self.show_form()
        elif psim.Button("Show print model"):
            self.show_print()
        psim.SameLine()
        if psim.Button("Export print STL" + ("s" if n > 1 else "")):
            self.export()
        psim.SetItemTooltip("Saved in exports/ (one STL per part). Load them all into the slicer together.")
        for name, info in self.last_export:
            ok = info["complete"] and info["watertight"]
            psim.TextColored(OK if ok else WARN, f"{name}: {info['triangles']:,} triangles, "
                             + ("verified complete and watertight" if ok else f"NOT valid ({info})"))

    def turntable_ui(self):
        t = self.tt
        _, t["frames"] = psim.SliderInt("frames", int(t["frames"]), 12, 240)
        _, t["seconds"] = psim.SliderFloat("seconds", t["seconds"], 1.0, 20.0, "%.1f")
        _, t["size"] = psim.SliderInt("size (px)", int(t["size"]), 256, 1600)
        _, t["elevation"] = psim.SliderFloat("elevation (deg)", t["elevation"], -60.0, 80.0, "%.0f")
        _, t["format"] = psim.Combo("format", t["format"], ["GIF", "MP4"])
        if psim.Button("Render turntable"):
            self.render_turntable()
        psim.SetItemTooltip("Orbits the camera around whatever is shown (form or print model) and saves to renders/.")

    def render_turntable(self):
        """Orbit whatever is shown: the print model (z-up, mm) or the form (y-up)."""
        if self.showing and self.result is not None:
            V, up = ps.get_surface_mesh(PREVIEW).get_vertex_positions() if ps.has_surface_mesh(PREVIEW) \
                else self.result.V, "z"
        elif self.app.result is not None:
            V, up = self.app.result.mesh.V, "y"
        else:
            return
        t = self.tt
        lo, hi = V.min(0), V.max(0)
        ext = ["gif", "mp4"][t["format"]]
        os.makedirs(os.path.join(self.app.root, "renders"), exist_ok=True)
        name = "".join(c for c in self.app.save_name if c.isalnum() or c in "-_") or "form"
        path = os.path.join(self.app.root, "renders", f"{name}_turntable.{ext}")
        t0 = time.perf_counter()
        try:
            turntable(path, 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo)), int(t["frames"]),
                      float(t["seconds"]), int(t["size"]), float(t["elevation"]), up=up)
        except OSError as e:
            self.app.error = f"Turntable failed: {e}"
            return
        self.app.status = f"Saved renders/{os.path.basename(path)} in {time.perf_counter() - t0:.1f}s"

"""Output panel for app.py: watertight FDM print preparation (size, orientation, cutting into parts,
resolution, support check) and turntable renders."""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures.process import BrokenProcessPool

import json

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from hansmeyer import background, vessel

from hansmeyer.printprep import (UP_KEYS, PrintSettings, best_up, grid_for_dims, prepare_arrays, prepare_exact_arrays,
                                 stl_bytes, to_print_coords, up_rotation, write_stl)
from hansmeyer.view import MESH_COLOR, helper, look, set_view, turntable
import ui_style as ui

PREVIEW = "print model"
PLANES = "cut planes"
BASE = MESH_COLOR
RED = (0.9, 0.25, 0.2)  # data colour: surfaces that need support / are too thin
PART_COLOURS = np.array([(0.90, 0.62, 0.30), (0.40, 0.65, 0.90), (0.55, 0.80, 0.45), (0.85, 0.45, 0.70),
                         (0.95, 0.85, 0.35), (0.50, 0.85, 0.85), (0.75, 0.60, 0.95), (0.95, 0.50, 0.45)])
NOZZLES = [0.25, 0.4, 0.6, 0.8]
PRINTERS = [("Bambu Lab A1 / P1S / X1C", (256.0, 256.0, 256.0)), ("Bambu Lab A1 mini", (180.0, 180.0, 180.0)),
            ("Prusa MK4 / MK4S", (250.0, 210.0, 220.0)), ("Prusa CORE One", (250.0, 220.0, 270.0)),
            ("Prusa MINI", (180.0, 180.0, 180.0)), ("Custom", None)]
SIZE_AXES = [("longest", "longest side"), ("height", "height (Z)"), ("width", "width (X)"), ("depth", "depth (Y)")]
QUALITY = [("draft", 1.0, "voxel = nozzle width: fastest, smallest files"),
           ("standard", 0.75, "a good match for FDM detail"),
           ("fine", 0.5, "smoother surface, larger files; detail below the nozzle width still won't print")]
CUT_PRESETS = [("whole", (None, None, None), "One piece."),
               ("2 halves", (None, None, -1.0), "One horizontal cut at the widest section: both halves print "
                                                "cut face down, with far less support."),
               ("4 quarters", (0.5, 0.5, None), "Two vertical cuts through the middle: four pillars that "
                                                  "assemble into the whole, like the four corners of a cage."),
               ("8 pieces", (0.5, 0.5, -1.0), "All three cuts.")]
AXIS_LABELS = ("across x", "across y", "across z")
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
        self.worker = background.Worker()  # its own process: heavy voxel work never blocks or crashes the UI
        self.last_export = []
        self.result = None
        self.result_key = None
        self.result_settings = None
        self.ref = None  # (triangles, output resolution, seconds, voxels) of the last run, for estimates
        self.showing = False
        self.shown_V = None  # vertices of the print model as shown (for the section view and turntables)
        self.view = "layout"  # print layout | assembled
        self.explode = 15.0
        self.colour = "support"  # support | thin | parts
        self.support_deg = 30
        self._overhang = None  # (threshold, mask, per-part cm2, share)
        self._cache = {}
        self._planes_key = None
        self.defaults_for_shape_pending = True  # choose up axis / base once the input mesh exists
        self.tt = {"frames": 72, "seconds": 6.0, "size": 640, "elevation": 20.0, "format": 0}
        self.advanced = False

    # ------------------------------------------------------------- helpers
    def _design_key(self):
        d = self.app.design
        keys = d.level_keys(d.full_depth, self.app.root)
        return (keys[-1] if keys else d.base_key(self.app.root), tuple(vars(self.s).items()),
                json.dumps(d.vessel, sort_keys=True) if vessel.active(d) else "")

    def vessel_mode(self) -> bool:
        return vessel.active(self.app.design)

    def defaults_for_shape(self):
        """Vessels print upside down on their rim, panels lie flat on a solid base, everything else stands up."""
        if self.vessel_mode():
            self.s.up, self.s.base = "-y", False
            return
        panel = self.app.base_mesh is not None and np.any(self.app.base_mesh.vert_is_boundary)
        panel = panel and self.app.design.base.get("shape") != "sphere_open"
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

    def _scale_dims(self):
        """(model -> mm scale, assembled size (W, D, H) in mm) for the current settings, as printprep.scale_for
        computes it. Asked every frame, so the form's extent is computed once per mesh and up direction."""
        m = self._mesh()
        if m is None:
            return None
        ext = self._cached("ext", (m, str(self.s.up)), lambda: np.ptp(to_print_coords(m.V, self.s.up), axis=0))
        ref = {"width": ext[0], "depth": ext[1], "height": ext[2]}.get(self.s.size_axis, ext.max())
        if ref < 1e-6 * max(ext.max(), 1e-12):
            ref = ext.max()
        scale = self.s.size_mm / max(ref, 1e-12)
        return scale, ext * scale

    def dims_now(self):
        """Assembled size (W, D, H) in mm for the current settings, without running anything."""
        sd = self._scale_dims()
        return None if sd is None else sd[1]

    def finest_voxel(self):
        """The smallest voxel the grid limit allows at this size (or None without a form)."""
        dims = self.dims_now()
        return None if dims is None else float(dims.max() / self.s.max_grid)

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
        finest = self.finest_voxel()
        return self.s.voxel_mm if finest is None else max(self.s.voxel_mm, finest)

    def estimate(self):
        """(triangles, STL bytes, RAM bytes, seconds or None) for the current settings."""
        m = self._mesh()
        if m is None:
            return None
        s = self.s
        scale, dims = self._scale_dims()
        voxel, _, shape = grid_for_dims(dims, s)
        res = float(dims.max()) / (voxel * s.detail)  # output cells along the longest side
        nvox = float(np.prod(shape))
        if self.ref is not None:
            tris = self.ref[0] * (res / self.ref[1]) ** 2
            secs = self.ref[2] * nvox / self.ref[3]
        else:
            def area():  # from a sample of ~60k faces (the estimate is rough anyway; a 1M-face form stays quick)
                if m.is_all_quads():  # the same two fan triangles per quad as the export
                    Q = m.face_idx.reshape(-1, 4)
                    P = m.V[Q[::max(1, len(Q) // 60_000) | 1]]  # (odd: not in step with the 4 children)
                    a = 0.5 * (np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
                               + np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]), axis=1))
                else:
                    T = m.triangles()
                    P = m.V[T[::max(1, len(T) // 60_000) | 1]]
                    a = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
                    Q = T
                return float(a.sum()) * len(Q) / max(len(a), 1)
            a = self._cached("area", m, area)
            tris = 1.6 * a * scale ** 2 / (voxel * s.detail) ** 2  # marching cubes: ~2.7 per cell, minus hidden folds
            secs = None
        return tris, 84 + 50 * tris, nvox * BYTES_PER_VOXEL, secs

    def _free_disk(self) -> int:
        """Free space on the project's disk, checked at most every 2 s (it is shown every frame)."""
        now = time.perf_counter()
        hit = self._cache.get("disk")
        if hit is None or now - hit[0] > 2.0:
            try:
                hit = (now, shutil.disk_usage(self.app.root).free)
            except OSError:
                hit = (now, 0)
            self._cache["disk"] = hit
        return hit[1]

    def show_form(self):
        ps.remove_surface_mesh(PREVIEW, error_if_absent=False)
        if ps.has_surface_mesh("form"):
            ps.get_surface_mesh("form").set_enabled(True)
        if self.showing and self.app.result is not None:  # the print model's camera is z-up and in mm: frame the form
            set_view(self.app.result.mesh, self.app.design.view)
        self.showing = False
        self.shown_V = None
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
        self.shown_V = np.asarray(V)
        if reset_camera:
            c = 0.5 * (V.min(0) + V.max(0))
            ext = float(np.linalg.norm(V.max(0) - V.min(0)))
            ps.set_up_dir("z_up")
            look(c + np.array([1.0, -1.4, 0.9]) * ext * 0.8, c)

    def recolour(self):
        if self.showing and self.result is not None and ps.has_surface_mesh(PREVIEW):
            ps.get_surface_mesh(PREVIEW).add_color_quantity("print colours", self._colours(), enabled=True)

    def update_planes(self, visible: bool):
        """Translucent cut planes over the form, so cuts can be placed before preparing."""
        m, s = self._mesh(), self.s
        cuts = (s.cut_x, s.cut_y, s.cut_z)
        visible = visible and m is not None and not self.showing and any(f is not None for f in cuts)
        widest = self._widest_fraction()
        key = (visible, m, str(s.up), cuts, widest)
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
        pm = ps.register_surface_mesh(PLANES, V, F, color=helper("plane"), transparency=0.22, smooth_shade=False)
        pm.set_back_face_policy("identical")

    def theme_changed(self):
        self._planes_key = None  # redraw the cut planes in the new theme's colour

    def _widest_fraction(self) -> float:
        """Where the last run put a 'widest section' height cut (0.5 until one ran)."""
        rs, r = self.result_settings, self.result
        if r is not None and rs is not None and rs["cut_z"] is not None and rs["cut_z"] < 0 and rs["up"] == self.s.up:
            return next((f for a, f in r.cuts if a == 2), 0.5)
        return 0.5

    # ------------------------------------------------------------ the job
    def busy(self) -> bool:
        return self.job is not None and not self.job["future"].done()

    def start(self):
        if self.busy():
            return
        if not self.app.baked:
            self.app.bake(wait=True)  # print at full depth
        if not self.app.result:
            return
        m = self.app.result.mesh
        self.s.nozzle_mm = NOZZLES[self.nozzle_idx]
        if self.vessel_mode():  # already a clean closed solid: exported exactly, no voxel remesh
            D = self.app.design.vessel["diameter_mm"]
            self.s.size_mm, self.s.size_axis = D, "width"
            settings = dict(vars(self.s))
            future = self.worker.submit(prepare_exact_arrays, m.V, m.face_ptr, m.face_idx, settings, D / 2.0,
                                        m.vattr.get("wall_mm"))
        else:
            settings = dict(vars(self.s))
            future = self.worker.submit(prepare_arrays, m.V, m.face_ptr, m.face_idx, settings)
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
            result = self.worker.result(job["future"])
        except BrokenProcessPool:
            self.worker.broken()  # the worker died (usually out of memory): start a fresh one next time
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
        if result.voxel_mm > 0:  # voxel runs calibrate the size / time estimates
            res = max(result.dims_mm) / (result.voxel_mm * job["settings"]["detail"])
            self.ref = (len(result.F), res, result.seconds, float(np.prod(result.grid)))
        self.view = "layout"
        self.show_print()
        n = len(result.parts)
        self.app.status = f"print model ready in {result.seconds:.0f}s" + (f": {n} parts" if n > 1 else "")

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
            self.app.status = f"exported {n} file(s) to exports/: verified complete and watertight"

    # ------------------------------------------------------------------ ui
    def ui(self):
        if self.vessel_mode():
            self.vessel_print_ui()
        else:
            ui.explain("Rebuilds the form as closed solids a slicer can read: self-intersections merged, open skins "
                       "thickened, sized in mm. Cutting it into parts lets each print with far less support.")
            self.printer_ui()
            self.size_ui()
            self.orientation_ui()
            self.underside_ui()
            self.cut_ui()
            self.resolution_ui()
            ui.gap(4.0)
            _, self.advanced = ui.check("advanced settings", self.advanced, "pradv",
                                        "Exact voxel size, grid limit, wall, smoothing, mesh detail, solid base.")
            if self.advanced:
                self.advanced_ui()
        ui.gap(10.0)
        if self.busy():
            ui.warn(f"working in a background process … {time.perf_counter() - self.job['started']:.0f}s")
        else:
            w = psim.GetContentRegionAvail()[0]
            if ui.button("prepare print model", "prgo", "Bakes the full depth first, then runs in a separate process "
                         "(the app stays responsive).", width=0.62 * w, kind="primary"):
                self.start()
            psim.SameLine(0.0, 6.0)
            if ui.button("auto settings", "prauto", "Panels lie flat on a solid base; everything else stands up "
                                                    "(model Y up).", width=psim.GetContentRegionAvail()[0]):
                self.defaults_for_shape()
        self.result_ui()

    def vessel_height(self, D: float) -> float:
        """Height of the vessel standing on its rim."""
        op = self.app.design.base.get("opening", 40.0) if self.app.design.base.get("shape") == "sphere_open" else 0.0
        return 0.5 * D * (1 + np.cos(np.radians(op)))

    def set_vessel_size(self, D: float):
        v = self.app.design.vessel
        D = float(min(max(D, 20.0), 400.0))
        if abs(D - v["diameter_mm"]) > 1e-9:
            v["diameter_mm"] = D
            self.app.edited()  # walls are in mm: the geometry is rebuilt for the new size

    def _size_buttons(self, sizes, current, key, setter, fit):
        for mm in sizes:
            if ui.toggle(f"{mm}", abs(current - mm) < 1e-6, f"{key}{mm}", f"{mm} mm"):
                setter(float(mm))
            psim.SameLine(0.0, 4.0)
        if ui.button("fit bed", f"{key}fit", "The largest size that fits the printer bed."):
            fit()

    def vessel_print_ui(self):
        v = self.app.design.vessel
        ui.explain("The vessel is already one clean, closed solid, so it is exported exactly (no voxel remesh) and "
                   "the outside stays a perfect sphere.")
        self.printer_ui()
        ui.subhead("size")
        changed, D = ui.input_float("diameter", v["diameter_mm"], 1.0, 10.0, "%.0f mm",
                                    "Outer diameter of the sphere. Type a value or use - / + (Ctrl: steps of 10). "
                                    "The walls stay as set in mm (0.8 mm windows stay 0.8 mm at any size).", "prvdiam")
        if changed:
            self.set_vessel_size(D)
        ui.row_label("")

        def fit():
            bed = self.bed() - 4.0
            self.set_vessel_size(np.floor(min(bed[0], bed[1], bed[2] / max(self.vessel_height(1.0), 1e-9))))

        self._size_buttons((80, 100, 120, 150, 200), v["diameter_mm"], "prvs", self.set_vessel_size, fit)
        D = v["diameter_mm"]
        H = self.vessel_height(D)
        ok = self.fits(np.array([D, D, H]), any_way=False)
        ui.value("print size", f"{D:.0f} x {D:.0f} x {H:.0f} mm", "hi" if ok else "warn", "W x D x H, on the rim.")
        if not ok:
            ui.warn("Bigger than the bed.")
        self.orientation_ui()
        if self.s.up == "-y":
            ui.note("Upside down: the opening's rim stands on the plate, the dome prints on top.")

    def printer_ui(self):
        ui.subhead("printer", top=0.0)
        changed, self.printer_idx = ui.combo("printer", self.printer_idx, [n for n, _ in PRINTERS],
                                             "Sets the build volume the size and the parts are checked against.",
                                             "prbed")
        if changed:
            self.app.remember("printer", self.printer_idx)
        if PRINTERS[self.printer_idx][1] is not None:
            b = self.bed()
            ui.value("bed", f"{b[0]:.0f} x {b[1]:.0f} x {b[2]:.0f} mm", "dim")
        else:
            changed, v = ui.input_float3("W x D x H", self.custom_bed, "%.0f", "Build volume in mm.", "prcustom")
            if changed:
                self.custom_bed = [max(20.0, float(x)) for x in v]
                self.app.remember("custom_bed", self.custom_bed)
        _, self.nozzle_idx = ui.choice("nozzle", [(i, f"{n}") for i, n in enumerate(NOZZLES)], self.nozzle_idx,
                                       "prnoz", "Nozzle diameter in mm: sets the resolution choices below.")

    def size_ui(self):
        s = self.s
        ui.subhead("size")
        keys = [k for k, _ in SIZE_AXES]
        changed, i = ui.combo("measured", keys.index(s.size_axis) if s.size_axis in keys else 0,
                              [label for _, label in SIZE_AXES], "Which dimension the size below sets.", "prsaxis")
        if changed:
            s.size_axis = keys[i]
        _, v = ui.input_float("size", s.size_mm, 1.0, 10.0, "%.1f mm",
                              "Type a value, or use - / + (hold Ctrl for steps of 10).", "prsize")
        s.size_mm = float(min(max(v, 5.0), 2000.0))
        ui.row_label("")

        def setter(mm):
            s.size_mm = mm

        self._size_buttons((50, 100, 150, 200), s.size_mm, "prs", setter, self.fit_to_bed)
        dims = self.dims_now()
        if dims is None:
            return
        bed = self.bed()
        text = f"{dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm"
        if self.n_parts() == 1:
            ok = self.fits(dims, any_way=False)
            ui.value("print size", text, "hi" if ok else "warn", "W x D x H")
            if not ok:
                ui.warn(f"Bigger than the {bed[0]:.0f} x {bed[1]:.0f} x {bed[2]:.0f} bed: cut it, or fit bed.")
        else:
            ui.value("assembled", text, "fg", "W x D x H")
            part = self.part_dims_estimate(dims)
            ok = self.fits(part, any_way=True)
            ui.value("largest part", f"~{part[0]:.0f} x {part[1]:.0f} x {part[2]:.0f} mm", "hi" if ok else "warn")
            if not ok:
                ui.warn("Too big for the bed: fit bed, or add a cut.")

    def orientation_ui(self):
        s = self.s
        ui.subhead("orientation")
        cur = "y" if s.up == "+y" else str(s.up)
        changed, up = ui.choice("up", [(k, _up_label(k)) for k in UP_KEYS], cur, "prup",
                                "Which model axis points up on the printer. With cuts, this sets the cut directions; "
                                "each part is then turned for printing on its own.")
        if changed:
            s.up = up
        ui.row_label("")
        if ui.button("auto: least support", "prupauto", "Picks the up axis with the least overhang.",
                     enabled=self._mesh() is not None):
            m = self._mesh()
            s.up = best_up(m.V, m.triangles(), float(self.support_deg))
            self.app.status = f"least support with model {_up_label(s.up)} up"

    def underside_ui(self):
        s = self.s
        ui.subhead("undersides")
        changed, on = ui.check("self-supporting undersides", s.undersides > 0, "prunder",
                               "Fills below every overhang with a smooth keel no flatter than the angle, so the print "
                               "needs (almost) no support. Upward-facing surfaces keep all their detail; horizontal "
                               "holes get pointed tops. Set the slicer's threshold angle below this.")
        if changed:
            s.undersides = 45.0 if on else 0.0
        if s.undersides > 0:
            _, s.undersides = ui.slider("keel angle", s.undersides, 30.0, 70.0, "%.0f deg", key="prkeel")
        _, s.foot_mm = ui.slider("flat foot", s.foot_mm, 0.0, 5.0, "%.1f mm",
                                 "Trims the bottom flat by this much, so the print stands on a foot instead of a "
                                 "point.", "prfoot")

    def cut_ui(self):
        s = self.s
        ui.subhead("cut into parts")
        current = tuple(f is not None for f in (s.cut_x, s.cut_y, s.cut_z))
        options = [(i, label) for i, (label, _, _) in enumerate(CUT_PRESETS)]
        cur = next((i for i, (_, cuts, _) in enumerate(CUT_PRESETS) if tuple(f is not None for f in cuts) == current), -1)
        changed, i = ui.segmented(options, cur, "prcut", helps=[tip for _, _, tip in CUT_PRESETS])
        if changed:
            s.cut_x, s.cut_y, s.cut_z = CUT_PRESETS[i][1]
        for a, attr in enumerate(("cut_x", "cut_y", "cut_z")):
            f = getattr(s, attr)
            changed, on = ui.check(AXIS_LABELS[a], f is not None, f"prcut{a}")
            if changed:
                setattr(s, attr, (-1.0 if a == 2 else 0.5) if on else None)
                f = getattr(s, attr)
            if f is None:
                continue
            if a == 2:
                psim.SameLine(0.0, 16.0)
                changed, widest = ui.check("at the widest", f < 0, "prcutw",
                                           "Cut where the form is widest: the biggest flat face for both halves to "
                                           "stand on.")
                if changed:
                    s.cut_z = -1.0 if widest else round(self._widest_fraction(), 2)
                if s.cut_z < 0:
                    continue
            _, pct = ui.slider("position", 100 * getattr(s, attr), 5.0, 95.0, "%.0f %%", key=f"prcutp{a}")
            setattr(s, attr, pct / 100.0)
        if self.n_parts() == 1:
            return
        ui.push_font("small")
        ui.note(f"{self.n_parts()} parts. The planes over the form show the cuts.")
        ui.pop_font()
        _, s.pins = ui.check("alignment pin holes", s.pins, "prpins",
                             "Matching blind holes in both faces of every joint, for gluing the parts in register. "
                             "2.0 mm takes a piece of 1.75 mm filament as the pin.")
        if s.pins:
            _, s.pin_mm = ui.input_float("pin diameter", s.pin_mm, 0.0, 0.0, "%.2f mm", key="prpind")
            _, s.pin_depth_mm = ui.input_float("pin depth", s.pin_depth_mm, 0.0, 0.0, "%.1f mm", key="prpinz")
            s.pin_mm = float(min(max(s.pin_mm, 0.5), 20.0))
            s.pin_depth_mm = float(min(max(s.pin_depth_mm, 1.0), 50.0))
        _, s.part_up = ui.choice("each part", [("auto", "least support"), ("assembly", "as assembled")], s.part_up,
                                 "prpartup", helps=["Turn each part on its own for the least support.",
                                                    "Print every part in its assembled orientation."])

    def resolution_ui(self):
        s = self.s
        ui.subhead("resolution")
        nozzle = NOZZLES[self.nozzle_idx]
        finest = self.effective_voxel() if self._mesh() is None else max(self.finest_voxel(), 0.05)
        options = [(round(k * nozzle, 3), label) for label, k, _ in QUALITY] + [(round(finest, 3), "max")]
        cur = next((v for v, _ in options if abs(s.voxel_mm - v) < 1e-6), None)
        if cur is None and s.voxel_mm < finest - 1e-9:
            cur = round(finest, 3)
        changed, v = ui.segmented(options, cur, "prq", helps=[f"{v:.2f} mm voxels: {tip}." for (v, _), (_, _, tip)
                                                             in zip(options, QUALITY)]
                                  + [f"{finest:.2f} mm: the finest the grid limit allows at this size."])
        if changed:
            s.voxel_mm = v
        eff = self.effective_voxel()
        est = self.estimate()
        ui.value("detail", f"{eff:.2f} mm")
        if est is not None:
            tris, nbytes, ram, secs = est
            ui.value("output", f"~{tris / 1e6:.1f} M triangles · STL ~{nbytes / 1e6:.0f} MB")
            total = _ram_bytes()
            heavy = total is not None and ram > 0.6 * total
            ui.value("needs", f"~{_bytes(ram)} RAM" + (f" · ~{secs / 60:.1f} min" if secs is not None and secs > 90 else
                                                       (f" · ~{secs:.0f} s" if secs is not None else "")),
                     "warn" if heavy else "fg")
            if heavy:
                ui.warn("Close to this computer's memory.")
        if eff > s.voxel_mm + 1e-9:
            ui.warn(f"Limited to {eff:.2f} mm by the grid limit ({s.max_grid} voxels along the longest side). Raise "
                    "it under advanced settings, or print smaller.")

    def advanced_ui(self):
        s = self.s
        _, s.voxel_mm = ui.slider("voxel", s.voxel_mm, 0.05, 1.5, "%.3f mm",
                                  "Exact resolution of the print model (the buttons above set this).", "prvox")
        labels = []
        dims = self.dims_now()
        for g in GRID_LIMITS:
            text = f"{g}"
            if dims is not None:
                v = max(s.voxel_mm, float(dims.max()) / g)
                vox = np.prod(np.ceil(dims / v) + 12)
                text += f"  ({float(dims.max()) / g:.2f} mm, ~{_bytes(vox * BYTES_PER_VOXEL)})"
            labels.append(text)
        idx = GRID_LIMITS.index(s.max_grid) if s.max_grid in GRID_LIMITS else 1
        _, idx = ui.combo("grid limit", idx, labels, "Most voxels along the longest side. Higher = finer detail on big "
                                                     "prints, more memory.", "prgrid")
        s.max_grid = GRID_LIMITS[idx]
        _, s.wall_mm = ui.slider("min wall", s.wall_mm, 0.4, 5.0, "%.2f mm",
                                 "Open or porous forms are printed as a skin of this thickness (>= 2 x nozzle).",
                                 "prwall")
        _, s.smooth = ui.slider_int("smoothing", int(s.smooth), 0, 30, key="prsmooth")
        _, s.soften = ui.slider("soften voxels", s.soften, 0.0, 1.5, "%.2f",
                                "Blurs the voxel solid before the surface is extracted: removes stair-steps (0 = raw "
                                "voxels).", "prsoft")
        _, d = ui.choice("mesh detail", [(1, "full"), (2, "half")], s.detail, "prdetail",
                         helps=["Every voxel cell.", "About 4x smaller files."])
        s.detail = d
        _, s.base = ui.check("solid base (reliefs)", s.base, "prbase",
                             "Fill everything below the form down to a flat plate: turns a relief panel into a tile.")
        if s.base:
            _, s.base_mm = ui.slider("base thickness", s.base_mm, 0.5, 10.0, "%.1f mm", key="prbasemm")

    def result_ui(self):
        r = self.result
        if r is None:
            return
        stale = self.result_key != self._design_key()
        n = len(r.parts)
        ui.subhead("print model")
        ui.begin_card("prresult")
        ui.spaced("STALE: FORM OR SETTINGS CHANGED" if stale else "READY", "warn" if stale else "dim")
        ui.gap(2.0)
        ui.push_font("medium")
        ui.text((f"{n} parts · " if n > 1 else "") + f"{r.dims_mm[0]:.1f} x {r.dims_mm[1]:.1f} x {r.dims_mm[2]:.1f} mm",
                "hi")
        ui.pop_font()
        mass = f"~{r.volume_cm3 * 1.24:.0f} g PLA (all wall)" if r.voxel_mm == 0 else f"~{r.mass_g():.0f} g PLA at 20% infill"
        ui.push_font("small")
        ui.wrap(f"{len(r.F):,} triangles · {r.volume_cm3:.1f} cm3 · {mass} · "
                + (f"detail {r.voxel_mm:.2f} mm" if r.voxel_mm > 0 else "exact mesh")
                + (f" · {r.pins} pin holes" if n > 1 else ""), "dim")
        need, free = stl_bytes(r), self._free_disk()
        ui.wrap(f"STL ~{need / 1e6:.0f} MB · {free / 1e9:.1f} GB free on disk", "warn" if free < need + 64e6 else "dim")
        ui.pop_font()
        ui.end_card()
        for note in r.notes:
            ui.warn(note)
        if r.thin_fraction <= 0.02:
            ui.ok("Thickness check passed.")

        ui.subhead("support check")
        changed, self.support_deg = ui.slider_int("threshold", int(self.support_deg), 5, 85, "%d deg",
                                                  "Same as the slicer's support 'threshold angle' (Bambu / Orca / "
                                                  "Prusa): downward surfaces flatter than this, measured from the "
                                                  "horizontal, get support. 30 = only flat overhangs; 60 = steeper "
                                                  "ones too. A higher number means MORE support, not less.", "prthr")
        if changed and self.colour == "support":
            self.recolour()
        _, per, share = self.overhang()
        total = sum(per)
        ui.value("needs support", f"~{total:.0f} cm2 ({100 * share:.1f}% of the surface)",
                 "hi" if share < 0.03 else "warn", "Shown red on the print model.")
        if n == 1 and share >= 0.03:
            ui.note("Mostly the inside of the dome's top (its ceiling): slicers usually bridge it, or lower the relief "
                    "depth there." if self.vessel_mode() else
                    "Lots of support: cut it into parts (2 halves usually helps most), or try another up direction.")
        bed = self.bed()
        if n > 1:
            for i, p in enumerate(r.parts):
                ok = self.fits(np.array(p.dims_mm), any_way=False)
                ui.wrap(f"{i + 1}. {p.name}: {p.dims_mm[0]:.0f} x {p.dims_mm[1]:.0f} x {p.dims_mm[2]:.0f} mm, prints "
                        f"{_how_printed(p)}, support ~{per[i]:.0f} cm2" + ("" if ok else ", too big for the bed")
                        + (f", {p.shells} loose pieces" if p.shells > 1 else ""), "fg" if ok else "warn")
        elif not self.fits(np.array(r.parts[0].dims_mm), any_way=False):
            ui.warn(f"Bigger than the {bed[0]:.0f} x {bed[1]:.0f} x {bed[2]:.0f} bed.")

        ui.subhead("view")
        if n > 1:
            changed, v = ui.choice("layout", [("layout", "print layout"), ("assembled", "assembled")], self.view,
                                   "prview")
            if changed:
                self.view = v
                self.show_print()
            if self.view == "assembled":
                changed, self.explode = ui.slider("gap", self.explode, 0.0, 60.0, "%.0f mm", key="prexplode")
                if changed and self.showing:
                    self.show_print(reset_camera=False)
        opts = [("support", "support"), ("thin", "too thin")] + ([("parts", "parts")] if n > 1 else [])
        changed, c = ui.choice("colour", opts, self.colour, "prcolour",
                               helps=["Red: surfaces that need support.", "Red: walls thinner than the minimum.",
                                      "One colour per part."][: len(opts)])
        if changed:
            self.colour = c
            self.recolour()
        w = psim.GetContentRegionAvail()[0]
        if self.showing:
            if ui.button("show form", "prform", "Back to the subdivision form.", width=0.4 * w):
                self.show_form()
        elif ui.button("show print model", "prshow", width=0.4 * w):
            self.show_print()
        psim.SameLine(0.0, 6.0)
        if ui.button("export print STL" + ("s" if n > 1 else ""), "prexport",
                     "Saved in exports/ (one STL per part). Load them all into the slicer together.",
                     width=psim.GetContentRegionAvail()[0], kind="primary"):
            self.export()
        for name, info in self.last_export:
            ok = info["complete"] and info["watertight"]
            ui.push_font("small")
            ui.wrap(f"{name}: {info['triangles']:,} triangles, "
                    + ("verified complete and watertight" if ok else f"NOT valid ({info})"), "dim" if ok else "warn")
            ui.pop_font()

    def turntable_ui(self):
        t = self.tt
        _, t["frames"] = ui.slider_int("frames", int(t["frames"]), 12, 240, key="ttframes")
        _, t["seconds"] = ui.slider("seconds", t["seconds"], 1.0, 20.0, "%.1f s", key="ttsec")
        _, t["size"] = ui.slider_int("size", int(t["size"]), 256, 1600, "%d px", key="ttsize")
        _, t["elevation"] = ui.slider("elevation", t["elevation"], -60.0, 80.0, "%.0f deg", key="ttelev")
        _, t["format"] = ui.choice("format", [(0, "GIF"), (1, "MP4")], t["format"], "ttfmt")
        if ui.button("render turntable", "ttgo", "Orbits the camera around whatever is shown (form or print model) "
                                                 "and saves to renders/.", kind="primary"):
            self.render_turntable()

    def render_turntable(self):
        """Orbit whatever is shown: the print model (z-up, mm) or the form (y-up)."""
        if self.showing and self.result is not None:
            V, up = (self.shown_V if self.shown_V is not None else self.result.V), "z"
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
        self.app.status = f"saved renders/{os.path.basename(path)} in {time.perf_counter() - t0:.1f}s"

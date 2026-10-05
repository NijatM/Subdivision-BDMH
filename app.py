"""Interactive Hansmeyer subdivision explorer.

Run:  python app.py            (or double-click run_mac.command / run_windows.bat)
"""

from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from hansmeyer import CC_WEIGHTS, DS_WEIGHTS, MAX_ITERATIONS, SHAPES, Design, IterationSpec, Pipeline, default_spec
from hansmeyer.meshio import export
from hansmeyer.view import apply_scene_theme, apply_ui_theme, set_view, setup_scene, show_mesh, write_png

ROOT = os.path.dirname(os.path.abspath(__file__))
PRESET_DIR = os.path.join(ROOT, "presets")
INPUT_DIR = os.path.join(ROOT, "inputs")
EXPORT_DIR = os.path.join(ROOT, "exports")
RENDER_DIR = os.path.join(ROOT, "renders")
SETTINGS_PATH = os.path.join(ROOT, ".app_settings.json")

AUTO_BAKE_DELAY = 1.5  # seconds of no edits before auto-bake
SHARP = {"w1": -1.0, "w2": -2.0}
EXPORT_FORMATS = ["obj", "stl", "ply"]
WARN = (0.95, 0.55, 0.15, 1.0)
ACTIVE = (0.85, 0.55, 0.20, 1.0)

_OPEN = getattr(psim, "ImGuiTreeNodeFlags_DefaultOpen", 1 << 5)


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    except OSError:
        pass  # a read-only folder just means the theme isn't remembered


def toggle_button(label: str, active: bool) -> bool:
    if active:
        psim.PushStyleColor(psim.ImGuiCol_Button, ACTIVE)
    clicked = psim.Button(label)
    if active:
        psim.PopStyleColor()
    return clicked


class App:
    def __init__(self):
        self.settings = load_settings()
        self.theme = self.settings.get("theme", "dark")
        self.ui_theme_applied = None
        self.pipe = Pipeline(root=ROOT)
        self.presets = self._list_presets()
        self.preset_idx = 0
        self.save_name = "my_form"
        self.tab = 0
        self.auto_bake = True
        self.show_edges = False
        self.export_fmt = 0
        self.last_edit = time.perf_counter()
        self.baked = False
        self.dirty = True
        self.need_view_reset = True
        self.status, self.error = "", ""
        self.result = None
        self.base_mesh = None
        self.obj_files = self._list_inputs()
        default = self._preset_path("fig3_left") or (self.presets[0][1] if self.presets else None)
        self.design = Design.load(default) if default else Design()
        if default:
            self.preset_idx = next((i for i, (_, p) in enumerate(self.presets) if p == default), 0)
            self.save_name = os.path.splitext(os.path.basename(default))[0] + "_edit"
        self.obj_path = self.design.base.get("path", "") if self.design.base.get("shape") == "obj" else ""

    # ------------------------------------------------------------------ theme
    def set_theme(self, name):
        self.theme = name
        apply_scene_theme(name)
        if self.result:
            show_mesh(self.result.mesh, edges=self.show_edges)
        self.settings["theme"] = name
        save_settings(self.settings)

    # ------------------------------------------------------------------ files
    def _list_presets(self):
        paths = sorted(glob.glob(os.path.join(PRESET_DIR, "*.json")))
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]

    def _list_inputs(self):
        return sorted(os.path.relpath(p, ROOT).replace(os.sep, "/") for p in glob.glob(os.path.join(INPUT_DIR, "*.obj")))

    def _preset_path(self, name):
        return next((p for n, p in self.presets if n == name), None)

    def load_preset(self, idx):
        name, path = self.presets[idx]
        try:
            self.design = Design.load(path)
        except (OSError, ValueError, TypeError) as e:
            self.error = f"Could not load {name}: {e}"
            return
        self.preset_idx = idx
        self.save_name = name + "_edit"
        self.obj_path = self.design.base.get("path", "")
        self.tab = 0
        self.need_view_reset = True
        self.edited()
        self.status = f"Loaded {name}"

    def save_preset(self):
        name = "".join(c for c in self.save_name.strip() if c.isalnum() or c in "-_") or "untitled"
        os.makedirs(PRESET_DIR, exist_ok=True)
        path = os.path.join(PRESET_DIR, name + ".json")
        if self.design.name in ("", "Untitled"):
            self.design.name = name
        self.design.save(path)
        self.presets = self._list_presets()
        self.preset_idx = next(i for i, (n, _) in enumerate(self.presets) if n == name)
        self.status = f"Saved presets/{name}.json"

    # ---------------------------------------------------------------- compute
    def edited(self):
        self.dirty = True
        self.baked = False
        self.last_edit = time.perf_counter()

    def compute(self, depth):
        try:
            r = self.pipe.run(self.design, depth)
        except Exception as e:  # bad OBJ, degenerate parameters, ...: keep the last good mesh
            self.error = f"{type(e).__name__}: {e}"
            return False
        self.error = ""
        self.result = r
        self.base_mesh = self.pipe.base(self.design)
        show_mesh(r.mesh, edges=self.show_edges)
        if self.need_view_reset:
            set_view(r.mesh, self.design.view)
            self.need_view_reset = False
        if not np.all(np.isfinite(r.mesh.V)):
            self.error = "Non-finite vertices — reduce the weights"
        return True

    def tick(self):
        if self.dirty:
            self.compute(self.design.preview_depth)
            self.dirty = False
        elif (
            self.auto_bake
            and not self.baked
            and not self.error
            and self.design.full_depth > self.design.preview_depth
            and time.perf_counter() - self.last_edit > AUTO_BAKE_DELAY
            and not psim.IsMouseDown(0)
        ):
            self.bake()

    def bake(self):
        self.baked = self.compute(self.design.full_depth)

    # --------------------------------------------------------------------- ui
    def ui(self):
        # Polyscope re-creates its (dark) style on UI-scale changes and screenshots,
        # so the light style is re-applied every frame; dark is restored once on switch.
        if self.theme == "light" or self.ui_theme_applied != self.theme:
            apply_ui_theme(self.theme)
            self.ui_theme_applied = self.theme
        self.tick()
        psim.PushItemWidth(170)
        self.theme_ui()
        self.preset_ui()
        self.base_ui()
        self.depth_ui()
        if psim.CollapsingHeader("Iteration schedule", _OPEN):
            self.schedule_ui()
        self.view_export_ui()
        self.status_ui()
        psim.PopItemWidth()

    def theme_ui(self):
        psim.Text("Theme")
        psim.SameLine()
        if toggle_button("Dark", self.theme == "dark") and self.theme != "dark":
            self.set_theme("dark")
        psim.SameLine()
        if toggle_button("Light", self.theme == "light") and self.theme != "light":
            self.set_theme("light")
        psim.Separator()

    def preset_ui(self):
        d = self.design
        if not psim.CollapsingHeader("Preset", _OPEN):
            return
        names = [n for n, _ in self.presets] or ["(none)"]
        changed, idx = psim.Combo("preset", self.preset_idx, names)
        if changed and self.presets:
            self.preset_idx = idx
            self.load_preset(idx)
        if d.description:
            psim.TextWrapped(d.description)
        _, self.save_name = psim.InputText("name", self.save_name)
        psim.SameLine()
        if psim.Button("Save"):
            self.save_preset()

    def base_ui(self):
        d = self.design
        if not psim.CollapsingHeader("Base mesh", _OPEN):
            return
        keys = list(SHAPES)
        cur_name = d.base.get("shape", "cube")
        cur = keys.index(cur_name) if cur_name in keys else 0
        changed, idx = psim.Combo("shape", cur, [SHAPES[k].label for k in keys])
        if changed and keys[idx] != cur_name:
            name = keys[idx]
            d.base = default_spec(name)
            if name == "obj":
                d.base["path"] = self.obj_path or (self.obj_files[0] if self.obj_files else "")
                self.obj_path = d.base["path"]
            d.view = SHAPES[name].view
            self.need_view_reset = True
            self.edited()
        shape = SHAPES[d.base.get("shape", "cube")]
        psim.SetItemTooltip(shape.help)

        for p in shape.params:
            val = d.base.get(p.name, p.default)
            if p.integer:
                changed, v = psim.SliderInt(p.label, int(round(val)), int(p.lo), int(p.hi))
            else:
                changed, v = psim.SliderFloat(p.label, float(val), p.lo, p.hi, "%.2f")
            if p.help:
                psim.SetItemTooltip(p.help)
            if changed:
                d.base[p.name] = v
                self.edited()

        if shape.name == "obj":
            self.obj_ui()

        if self.base_mesh is not None:
            m = self.base_mesh
            psim.TextDisabled(f"base: {m.n_verts} verts, {m.n_faces} faces"
                              + (", open" if np.any(m.vert_is_boundary) else ", closed"))
            if np.any(m.vert_is_boundary):
                self.boundary_ui()

    def obj_ui(self):
        d = self.design
        if psim.Button("Rescan inputs/"):
            self.obj_files = self._list_inputs()
        psim.SetItemTooltip("Put .obj files in the inputs/ folder next to app.py.")
        if self.obj_files:
            cur = self.obj_files.index(self.obj_path) if self.obj_path in self.obj_files else -1
            changed, idx = psim.Combo("inputs/", max(cur, 0), self.obj_files)
            if changed or cur < 0 and not self.obj_path:
                self.obj_path = self.obj_files[idx]
        else:
            psim.TextDisabled("no .obj files in inputs/ yet")
        _, self.obj_path = psim.InputText("path", self.obj_path)
        psim.SetItemTooltip("Relative to the project folder, or an absolute path.")
        if psim.Button("Load OBJ"):
            d.base["path"] = self.obj_path.strip()
            self.need_view_reset = True
            self.edited()
        psim.SameLine()
        changed, v = psim.Checkbox("fit to view", bool(d.base.get("normalize", True)))
        if changed:
            d.base["normalize"] = v
            self.edited()
        psim.SameLine()
        changed, v = psim.Checkbox("weld", bool(d.base.get("weld", True)))
        psim.SetItemTooltip("Merge duplicate vertices (needed for meshes exported as separate faces).")
        if changed:
            d.base["weld"] = v
            self.edited()

    def boundary_ui(self):
        d = self.design
        psim.Text("Boundary:")
        psim.SameLine()
        if psim.RadioButton("smooth", d.boundary == "smooth"):
            d.boundary = "smooth"
            self.edited()
        psim.SetItemTooltip("Standard Catmull-Clark boundary rules: the edge relaxes into a B-spline curve.")
        psim.SameLine()
        if psim.RadioButton("locked / tileable", d.boundary == "locked"):
            d.boundary = "locked"
            self.edited()
        psim.SetItemTooltip("Boundary stays fixed and relief fades out near it, so identical tiles meet seamlessly.")
        if d.boundary == "locked":
            changed, v = psim.SliderFloat("fade rows", d.fade_rows, 0.0, 6.0, "%.1f")
            psim.SetItemTooltip("Extrusion fades in over this many base-mesh rows from the boundary.")
            if changed:
                d.fade_rows = v
                self.edited()
            if any(it.scheme == "ds" for it in d.iterations[: d.full_depth]):
                psim.TextColored(WARN, "Doo-Sabin steps shrink open boundaries:\nthe result will not tile.")

    def depth_ui(self):
        d = self.design
        if not psim.CollapsingHeader("Depth & extrusion", _OPEN):
            return
        psim.Text("Extrusion:")
        psim.SameLine()
        if psim.RadioButton("relative", d.extrusion == "relative"):
            d.extrusion = "relative"
            self.edited()
        psim.SetItemTooltip("Displacement = w x local edge length: the same w behaves the same at every depth.")
        psim.SameLine()
        if psim.RadioButton("absolute", d.extrusion == "absolute"):
            d.extrusion = "absolute"
            self.edited()
        psim.SetItemTooltip("Paper-literal: displacement = w in model units (unit normals).")

        changed, v = psim.SliderInt("preview depth", d.preview_depth, 1, MAX_ITERATIONS)
        if changed:
            d.preview_depth = v
            d.full_depth = max(d.full_depth, v)
            self.edited()
        psim.SetItemTooltip("Recomputed live while you drag sliders.")
        changed, v = psim.SliderInt("full depth", d.full_depth, 1, MAX_ITERATIONS)
        if changed:
            d.full_depth = v
            d.preview_depth = min(d.preview_depth, v)
            self.edited()
        psim.SetItemTooltip("Used by Bake / auto-bake and for export. Faces x4 per CC step.")
        if psim.Button("Bake full depth"):
            self.bake()
        psim.SameLine()
        _, self.auto_bake = psim.Checkbox("auto-bake when idle", self.auto_bake)
        changed, v = psim.InputInt("face budget", self.pipe.face_budget, 100_000, 1_000_000)
        if changed:
            self.pipe.face_budget = max(10_000, v)
            self.edited()

    def schedule_ui(self):
        d = self.design
        self.tab = min(self.tab, d.full_depth - 1)
        for k in range(d.full_depth):
            spec = d.iterations[k]
            if k:
                psim.SameLine()
            label = f"{k + 1}{'DS' if spec.scheme == 'ds' else ''}{'*' if k >= d.preview_depth else ''}###it{k}"
            if toggle_button(label, k == self.tab):
                self.tab = k
        psim.TextDisabled("* = only in the baked (full-depth) result")
        psim.Separator()
        self.iteration_ui(self.tab, d.iterations[self.tab])
        if psim.TreeNode("Overview (all iterations)"):
            self.overview_ui()
            psim.TreePop()

    def iteration_ui(self, k, spec: IterationSpec):
        d = self.design
        psim.Text(f"Iteration {k + 1}:")
        psim.SameLine()
        if psim.RadioButton("Catmull-Clark", spec.scheme == "cc"):
            spec.scheme = "cc"
            self.edited()
        psim.SameLine()
        if psim.RadioButton("Doo-Sabin", spec.scheme == "ds"):
            spec.scheme = "ds"
            self.edited()

        defs = CC_WEIGHTS if spec.scheme == "cc" else DS_WEIGHTS
        for wd in defs:
            changed, v = psim.SliderFloat(wd.label, spec.weights[wd.name], wd.lo, wd.hi, "%.2f")
            psim.SetItemTooltip(wd.help + "  (ctrl+click to type a value)")
            if changed:
                spec.weights[wd.name] = float(v)
                self.edited()

        if psim.Button("Reset"):
            spec.weights = {n: 0.0 for n in spec.weights}
            self.edited()
        psim.SetItemTooltip("All weights of this iteration to 0 (standard subdivision).")
        if spec.scheme == "cc":
            psim.SameLine()
            if psim.Button("Sharp"):
                spec.weights.update(SHARP)
                self.edited()
            psim.SetItemTooltip("w1=-1, w2=-2: switches off CC smoothing so extrusions accumulate.")
        psim.SameLine()
        if psim.Button("Copy -> next") and k + 1 < MAX_ITERATIONS:
            d.iterations[k + 1] = IterationSpec(spec.scheme, dict(spec.weights))
            self.edited()
        psim.SameLine()
        if psim.Button("Copy -> all"):
            for j in range(MAX_ITERATIONS):
                if j != k:
                    d.iterations[j] = IterationSpec(spec.scheme, dict(spec.weights))
            self.edited()

    def overview_ui(self):
        """One bar chart per weight: bar i = value at iteration i (zero line in the middle)."""
        d = self.design
        its = d.iterations[: d.full_depth]
        width = psim.GetContentRegionAvail()[0]
        for title, defs in (("Catmull-Clark", CC_WEIGHTS), ("Doo-Sabin", DS_WEIGHTS)):
            vals = {wd.name: np.array([it.weights[wd.name] if (it.scheme == "cc") == (defs is CC_WEIGHTS) else 0.0
                                       for it in its], np.float32) for wd in defs}
            if not any(np.any(v) for v in vals.values()):
                continue
            psim.TextDisabled(title)
            for wd in defs:
                v = vals[wd.name]
                r = max(float(np.abs(v).max()), 0.25)
                psim.PlotHistogram(f"##ov_{wd.name}", v, overlay_text=f"{wd.name}  [{v.min():+.2f} .. {v.max():+.2f}]",
                                   scale_min=-r, scale_max=r, graph_size=(width, 28))

    def view_export_ui(self):
        d = self.design
        if not psim.CollapsingHeader("View & export", _OPEN):
            return
        for i, view in enumerate(("diagonal", "front", "top", "three_quarter")):
            if i:
                psim.SameLine()
            if psim.Button(view):
                d.view = view
                if self.result:
                    set_view(self.result.mesh, view)
        changed, self.show_edges = psim.Checkbox("wireframe", self.show_edges)
        if changed and self.result:
            show_mesh(self.result.mesh, edges=self.show_edges)
        psim.PushItemWidth(80)
        _, self.export_fmt = psim.Combo("##fmt", self.export_fmt, [f.upper() for f in EXPORT_FORMATS])
        psim.PopItemWidth()
        psim.SameLine()
        if psim.Button("Export"):
            self.export_mesh()
        psim.SetItemTooltip("OBJ/PLY keep quads; STL is triangulated. Bakes full depth first. Saved in exports/.")
        psim.SameLine()
        if psim.Button("Screenshot"):
            self.screenshot()

    def status_ui(self):
        psim.Separator()
        if self.result:
            r = self.result
            state = "baked" if self.baked else "preview"
            psim.Text(f"{state}: depth {r.depth_reached}  |  {r.mesh.n_faces:,} faces  |  {r.seconds * 1000:.0f} ms")
            if r.capped_by_budget:
                psim.TextColored(WARN, f"Capped at depth {r.depth_reached} by the face budget")
        if self.error:
            psim.TextColored(WARN, self.error)
        if self.status:
            psim.TextDisabled(self.status)

    # ----------------------------------------------------------------- output
    def _stem(self):
        name = "".join(c for c in self.save_name if c.isalnum() or c in "-_") or "form"
        return f"{name}_d{self.result.depth_reached}"

    def export_mesh(self):
        if not self.baked:
            self.bake()
        if not self.result:
            return
        os.makedirs(EXPORT_DIR, exist_ok=True)
        path = os.path.join(EXPORT_DIR, f"{self._stem()}.{EXPORT_FORMATS[self.export_fmt]}")
        export(path, self.result.mesh)
        self.status = f"Exported exports/{os.path.basename(path)}"

    def screenshot(self):
        if not self.result:
            return
        os.makedirs(RENDER_DIR, exist_ok=True)
        path = os.path.join(RENDER_DIR, self._stem() + ".png")
        write_png(path, ps.screenshot_to_buffer(transparent_bg=False, include_UI=False))
        self.status = f"Saved renders/{os.path.basename(path)}"


def main():
    app = App()
    setup_scene(theme=app.theme)
    ps.set_open_imgui_window_for_user_callback(True)
    ps.set_user_callback(app.ui)
    ps.show()


if __name__ == "__main__":
    main()

"""Interactive Hansmeyer subdivision explorer (Milestone 1).

Run:  python app.py            (or double-click run_mac.command / run_windows.bat)
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from hansmeyer import CC_WEIGHTS, DS_WEIGHTS, MAX_ITERATIONS, SHAPES, Design, IterationSpec, Pipeline
from hansmeyer.view import set_view, setup_scene, show_mesh, write_png

ROOT = os.path.dirname(os.path.abspath(__file__))
PRESET_DIR = os.path.join(ROOT, "presets")
EXPORT_DIR = os.path.join(ROOT, "exports")
RENDER_DIR = os.path.join(ROOT, "renders")

AUTO_BAKE_DELAY = 1.5  # seconds of no edits before auto-bake
SHARP = {"w1": -1.0, "w2": -2.0}

_OPEN = getattr(psim, "ImGuiTreeNodeFlags_DefaultOpen", 1 << 5)


class App:
    def __init__(self):
        self.pipe = Pipeline()
        self.presets = self._list_presets()
        self.preset_idx = 0
        self.save_name = "my_form"
        self.tab = 0
        self.auto_bake = True
        self.show_edges = False
        self.last_edit = time.perf_counter()
        self.baked = False
        self.dirty = True
        self.need_view_reset = True
        self.status = ""
        self.result = None
        default = self._preset_path("fig3_left") or (self.presets[0][1] if self.presets else None)
        self.design = Design.load(default) if default else Design()
        if default:
            self.preset_idx = next((i for i, (_, p) in enumerate(self.presets) if p == default), 0)
            self.save_name = os.path.splitext(os.path.basename(default))[0] + "_edit"

    # ------------------------------------------------------------------ presets
    def _list_presets(self):
        paths = sorted(glob.glob(os.path.join(PRESET_DIR, "*.json")))
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]

    def _preset_path(self, name):
        return next((p for n, p in self.presets if n == name), None)

    def load_preset(self, idx):
        name, path = self.presets[idx]
        self.design = Design.load(path)
        self.save_name = name + "_edit"
        self.tab = 0
        self.need_view_reset = True
        self.edited()
        self.status = f"Loaded {name}"

    def save_preset(self):
        name = "".join(c for c in self.save_name.strip() if c.isalnum() or c in "-_") or "untitled"
        os.makedirs(PRESET_DIR, exist_ok=True)
        path = os.path.join(PRESET_DIR, name + ".json")
        self.design.name = self.design.name if self.design.name != "Untitled" else name
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
        r = self.pipe.run(self.design, depth)
        self.result = r
        show_mesh(r.mesh, edges=self.show_edges)
        if self.need_view_reset:
            set_view(r.mesh, self.design.view)
            self.need_view_reset = False
        if not np.all(np.isfinite(r.mesh.V)):
            self.status = "Warning: non-finite vertices — reduce the weights"

    def tick(self):
        if self.dirty:
            self.compute(self.design.preview_depth)
            self.dirty = False
        elif (
            self.auto_bake
            and not self.baked
            and self.design.full_depth > self.design.preview_depth
            and time.perf_counter() - self.last_edit > AUTO_BAKE_DELAY
            and not psim.IsMouseDown(0)
        ):
            self.bake()

    def bake(self):
        self.compute(self.design.full_depth)
        self.baked = True

    # --------------------------------------------------------------------- ui
    def ui(self):
        self.tick()
        d = self.design
        psim.PushItemWidth(170)

        # ---- presets
        if psim.CollapsingHeader("Preset", _OPEN):
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

        # ---- base + global settings
        if psim.CollapsingHeader("Base & depth", _OPEN):
            shapes = list(SHAPES)
            cur = shapes.index(d.base.get("shape", "cube")) if d.base.get("shape", "cube") in shapes else 0
            changed, idx = psim.Combo("base shape", cur, shapes)
            if changed:
                d.base = {"shape": shapes[idx]}
                self.need_view_reset = True
                self.edited()
            psim.TextDisabled("(more base shapes arrive in milestone 2)")

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

        # ---- schedule
        if psim.CollapsingHeader("Iteration schedule", _OPEN):
            self.schedule_ui()

        # ---- view / export
        if psim.CollapsingHeader("View & export", _OPEN):
            for view in ("diagonal", "front", "top", "three_quarter"):
                if psim.Button(view):
                    d.view = view
                    if self.result:
                        set_view(self.result.mesh, view)
                psim.SameLine()
            psim.NewLine()
            psim.SetItemTooltip("'diagonal' looks down the cube's body diagonal, like Fig. 3.")
            changed, self.show_edges = psim.Checkbox("wireframe", self.show_edges)
            if changed and self.result:
                show_mesh(self.result.mesh, edges=self.show_edges)
            if psim.Button("Export OBJ"):
                self.export_obj()
            psim.SameLine()
            if psim.Button("Screenshot"):
                self.screenshot()

        # ---- status
        psim.Separator()
        if self.result:
            r = self.result
            state = "baked" if self.baked else "preview"
            psim.Text(
                f"{state}: depth {r.depth_reached}  |  {r.mesh.n_faces:,} faces  |  {r.seconds * 1000:.0f} ms"
            )
            if r.capped_by_budget:
                psim.TextColored((0.9, 0.4, 0.1, 1.0), f"Capped at depth {r.depth_reached} by the face budget")
        if self.status:
            psim.TextDisabled(self.status)
        psim.PopItemWidth()

    def schedule_ui(self):
        d = self.design
        self.tab = min(self.tab, d.full_depth - 1)
        for k in range(d.full_depth):
            spec = d.iterations[k]
            if k:
                psim.SameLine()
            active = k == self.tab
            if active:
                psim.PushStyleColor(psim.ImGuiCol_Button, (0.85, 0.55, 0.20, 1.0))
            label = f"{k + 1}{'DS' if spec.scheme == 'ds' else ''}{'*' if k >= d.preview_depth else ''}###it{k}"
            if psim.Button(label):
                self.tab = k
            if active:
                psim.PopStyleColor()
        psim.TextDisabled("* = only in the baked (full-depth) result")
        psim.Separator()
        self.iteration_ui(self.tab, d.iterations[self.tab])

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

    # ----------------------------------------------------------------- output
    def _stem(self):
        r = self.result
        name = "".join(c for c in self.save_name if c.isalnum() or c in "-_") or "form"
        return f"{name}_d{r.depth_reached}"

    def export_obj(self):
        if not self.baked:
            self.bake()
        os.makedirs(EXPORT_DIR, exist_ok=True)
        path = os.path.join(EXPORT_DIR, self._stem() + ".obj")
        self.result.mesh.write_obj(path)
        self.status = f"Exported exports/{os.path.basename(path)}"

    def screenshot(self):
        os.makedirs(RENDER_DIR, exist_ok=True)
        path = os.path.join(RENDER_DIR, self._stem() + ".png")
        write_png(path, ps.screenshot_to_buffer(transparent_bg=False, include_UI=False))
        self.status = f"Saved renders/{os.path.basename(path)}"


def main():
    setup_scene()
    ps.set_open_imgui_window_for_user_callback(True)
    app = App()
    ps.set_user_callback(app.ui)
    ps.show()


if __name__ == "__main__":
    main()

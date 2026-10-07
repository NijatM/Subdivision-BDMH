"""Section cut for app.py: a plane that opens up whatever is shown, so the inside of a form (a vessel's inner
relief, a cage's bars) can be seen while you keep modelling. Lives in the toolbar over the 3D view."""

from __future__ import annotations

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import ui_style as ui
from hansmeyer.view import helper


class SectionTool:
    def __init__(self, app):
        self.app = app
        self.on = False
        self.axis = 2  # 0 x, 1 y, 2 z
        self.pos = 0.5  # across the shown object's bounding box
        self.keep_positive = False  # which half stays
        self.show_plane = False
        self.gizmo = False
        self.plane = None
        self._applied = None

    def bounds(self):
        pp = self.app.print_panel
        V = pp.shown_V if pp.showing and pp.shown_V is not None else (
            self.app.result.mesh.V if self.app.result is not None else None)
        if V is None:
            return None
        return V.min(0), V.max(0)

    def open_cut(self):
        self.on = True

    def theme_changed(self):
        if self.plane is not None:
            self._style_plane()

    def _style_plane(self):
        c = helper("plane")
        self.plane.set_color(c)
        self.plane.set_transparency(0.10)
        self.plane.set_grid_line_color(c)

    def toolbar_ui(self):
        """'cut' toggle plus its settings in a popup, for the toolbar."""
        label = f"cut {'xyz'[self.axis]} {100 * self.pos:.0f}%" if self.on else "cut"
        if ui.toggle(label, self.on, "seccut", "Section cut: hide one side of a plane to see inside (also cuts "
                                                "screenshots and turntables). Click the … box for settings."):
            self.on = not self.on
        psim.SameLine(0.0, -1.0)
        if ui.button("…", "secmore", "Section settings"):
            psim.OpenPopup("##sectionpop")
        psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, (16.0, 14.0))
        psim.SetNextWindowSize((390.0, 0.0))
        if psim.BeginPopup("##sectionpop"):
            ui.spaced("SECTION CUT", "dim")
            ui.gap(4.0)
            self.settings_ui()
            psim.EndPopup()
        psim.PopStyleVar()

    def settings_ui(self):
        _, self.on = ui.check("cut the view open", self.on, "secon")
        ch, a = ui.choice("plane across", [(0, "X"), (1, "Y"), (2, "Z")], self.axis, "secaxis")
        if ch:
            self.axis, self.on = a, True
        _, pct = ui.slider("position", 100 * self.pos, 0.0, 100.0, "%.0f %%", key="secpos")
        self.pos = pct / 100.0
        if ui.button("flip side", "secflip", "Keep the other half."):
            self.keep_positive = not self.keep_positive
        psim.SameLine(0.0, 6.0)
        if ui.button("face the cut", "secface", "Point the camera straight at the cut face."):
            self.face_the_cut()
        _, self.show_plane = ui.check("show plane", self.show_plane, "secplane")
        psim.SameLine(0.0, 16.0)
        _, self.gizmo = ui.check("drag in the view", self.gizmo, "secgizmo",
                                 "Shows a handle on the plane: drag to move it, rotate the rings to tilt it.")

    def _pose(self):
        lo, hi = self.bounds()
        c = 0.5 * (lo + hi)
        c[self.axis] = lo[self.axis] + self.pos * (hi[self.axis] - lo[self.axis])
        n = np.zeros(3)
        n[self.axis] = 1.0 if self.keep_positive else -1.0  # the plane keeps the side its normal points to
        return c, n, float(np.linalg.norm(hi - lo))

    def apply(self):
        b = self.bounds()
        if b is None:
            return
        state = (self.on, self.axis, round(self.pos, 4), self.keep_positive, self.show_plane, self.gizmo,
                 tuple(np.round(b[0], 4)), tuple(np.round(b[1], 4)))
        if state == self._applied:
            return  # (a plane dragged with the handle keeps its pose until a setting changes)
        self._applied = state
        if self.plane is None:
            if not self.on:
                return
            self.plane = ps.add_scene_slice_plane()
            self._style_plane()
        c, n, _ = self._pose()
        self.plane.set_pose(tuple(c), tuple(n))
        self.plane.set_active(self.on)
        self.plane.set_draw_plane(self.on and self.show_plane)
        self.plane.set_draw_widget(self.on and self.gizmo)

    def face_the_cut(self):
        if self.bounds() is None:
            return
        self.on = True
        self.apply()
        c, n, size = self._pose()
        up = np.zeros(3)
        up[1 if self.axis != 1 else 2] = 1.0
        eye = c - n * 1.6 * size + up * 0.25 * size  # from the removed side, looking at the cut face
        ps.look_at(tuple(eye), tuple(c))

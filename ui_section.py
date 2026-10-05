"""Section view for app.py: a cutting plane that opens up whatever is shown, so the inside of a form
(a vessel's inner relief, a cage's bars) can be seen while you keep modelling."""

from __future__ import annotations

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from ui_common import toggle_button


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

    def ui(self):
        if psim.CollapsingHeader("Section view (see inside)"):
            changed, self.on = psim.Checkbox("cut the view open", self.on)
            psim.SetItemTooltip("Hides everything on one side of a plane, so you can see inside while you keep "
                                "editing. Screenshots and turntables are cut too.")
            psim.Text("Plane across:")
            for a, name in enumerate("XYZ"):
                psim.SameLine()
                if toggle_button(f"{name}##sec", self.axis == a):
                    self.axis, self.on = a, True
            psim.SameLine()
            if psim.Button("Flip##sec"):
                self.keep_positive = not self.keep_positive
            psim.SetItemTooltip("Keep the other half.")
            _, pct = psim.SliderFloat("position##sec", 100 * self.pos, 0.0, 100.0, "%.0f %%")
            self.pos = pct / 100.0
            _, self.show_plane = psim.Checkbox("show plane##sec", self.show_plane)
            psim.SameLine()
            _, self.gizmo = psim.Checkbox("drag in the view##sec", self.gizmo)
            psim.SetItemTooltip("Shows a handle on the plane: drag to move it, rotate the rings to tilt it.")
            if psim.Button("Face the cut##sec"):
                self.face_the_cut()
            psim.SetItemTooltip("Point the camera straight at the cut face.")
        self.apply()

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
            self.plane.set_color((0.95, 0.55, 0.15))  # a faint orange sheet instead of the default magenta grid
            self.plane.set_transparency(0.12)
            self.plane.set_grid_line_color((0.95, 0.55, 0.15))
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

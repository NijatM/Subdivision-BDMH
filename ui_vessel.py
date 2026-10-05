"""Vessel panel for app.py: a thin-walled lithophane sphere, smooth outside, the subdivision relief inside."""

from __future__ import annotations

import numpy as np
import polyscope.imgui as psim

from hansmeyer import default_spec
from ui_common import WARN, toggle_button, wrapped_colored

OK = (0.45, 0.85, 0.55, 1.0)


class VesselPanel:
    def __init__(self, app):
        self.app = app

    def use_sphere(self):
        app, d = self.app, self.app.design
        d.base = default_spec("sphere_open")
        d.boundary = "locked"
        d.view = "three_quarter"
        app.need_view_reset = True
        app.print_panel.defaults_for_shape_pending = True
        app.edited()

    def ui(self):
        app, d = self.app, self.app.design
        v = d.vessel
        if not psim.CollapsingHeader("Vessel (lithophane sphere)"):
            return
        psim.TextWrapped("Turns the form into the inside of a thin shell: an exact, smooth sphere outside, the "
                         "subdivision relief inside. Where the relief rises the wall is thin and glows when lit "
                         "from inside; where it sinks the wall is thick and dark.")
        changed, on = psim.Checkbox("vessel on", v["enabled"])
        if changed:
            v["enabled"] = on
            if on and d.base.get("shape") != "sphere_open":
                self.use_sphere()
            app.print_panel.defaults_for_shape_pending = True
            app.edited()
        if d.base.get("shape") != "sphere_open":
            psim.TextColored(WARN, "Made for the 'Sphere with opening' base shape.")
            psim.SameLine()
            if psim.Button("Use it##vessel"):
                self.use_sphere()
        if not v["enabled"]:
            return

        edited = False
        psim.PushItemWidth(120)
        ch, val = psim.InputFloat("diameter (mm)##vessel", v["diameter_mm"], 5.0, 20.0, "%.0f")
        if ch:
            v["diameter_mm"] = float(min(max(val, 20.0), 400.0))
            edited = True
        psim.PopItemWidth()
        ch, val = psim.SliderFloat("thinnest wall (mm)", v["min_wall_mm"], 0.4, 3.0, "%.2f")
        if ch:
            v["min_wall_mm"] = val
            v["max_wall_mm"] = max(v["max_wall_mm"], val + 0.1)
            edited = True
        psim.SetItemTooltip("Brightest spots. Keep it >= 2 nozzle widths (0.8 mm for a 0.4 nozzle).")
        ch, val = psim.SliderFloat("thickest wall (mm)", v["max_wall_mm"], 0.5, 8.0, "%.2f")
        if ch:
            v["max_wall_mm"] = max(val, v["min_wall_mm"] + 0.1)
            edited = True
        psim.SetItemTooltip("Darkest areas. White PLA lithophanes usually span about 0.8 - 3 mm.")
        ch, val = psim.SliderFloat("rim wall (mm)", v["rim_mm"], 0.8, 8.0, "%.1f")
        if ch:
            v["rim_mm"], edited = val, True
        psim.SetItemTooltip("Solid ring around the opening: what the print stands on.")
        ch, val = psim.SliderFloat("contrast", v["contrast"], 0.3, 1.0, "%.2f")
        if ch:
            v["contrast"], edited = val, True
        psim.SetItemTooltip("Share of the relief's height range spread between thinnest and thickest wall.\n"
                            "Lower = more of the wall at the extremes (bolder light pattern).")
        ch, val = psim.SliderInt("relief scale", int(v["smooth"]), 2, 200)
        if ch:
            v["smooth"], edited = val, True
        psim.SetItemTooltip("How large a shape still counts as relief: small = only fine detail shows in the light,\n"
                            "large = big folds and lobes show too.")
        ch, val = psim.Checkbox("invert light", v["invert"])
        if ch:
            v["invert"], edited = val, True
        if edited:
            app.edited()

        info = app.result.mesh.info.get("vessel") if app.result is not None else None
        if info:
            D = info["diameter_mm"]
            lo, hi = info["wall_mm"]
            psim.Text(f"wall {lo:.2f} - {hi:.2f} mm")
            op = info.get("opening_deg")
            if op is not None:
                psim.Text(f"opening {D * np.sin(np.radians(op)):.0f} mm across ({op:.0f} deg from the top)")
                ok = op >= 35
                wrapped_colored(OK if ok else WARN,
                                "Prints upside down on the rim: " + (
                                     f"the outside starts at {op:.0f} deg from flat - no support needed."
                                     if ok else f"the outside starts only {op:.0f} deg from flat near the rim: it "
                                     "may need support there. Widen the opening (Base mesh) to 35 deg or more."))
        if toggle_button("Light preview", app.color_mode == "light"):
            app.color_mode = "none" if app.color_mode == "light" else "light"
            app.refresh_display()
        psim.SetItemTooltip("Colours the sphere by how much light gets through the wall when lit from inside\n"
                            "(bright = thin). Use the Section view to see the relief itself.")

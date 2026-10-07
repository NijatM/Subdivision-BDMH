"""Vessel section for app.py: a thin-walled lithophane sphere, smooth outside, the subdivision relief inside."""

from __future__ import annotations

import numpy as np
import polyscope.imgui as psim

import ui_style as ui
from hansmeyer import default_spec


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
        ui.explain("Turns the form into the inside of a thin shell: an exact, smooth sphere outside, the "
                   "subdivision relief inside, exactly what the schedule's weights build, so edit them as on any "
                   "form. The parts reaching out furthest press against the shell as thin windows that glow when lit "
                   "from inside; deeper relief is thicker and darker.")
        if d.base.get("shape") != "sphere_open":
            ui.empty("Made for the 'Sphere with opening' base shape.")
            if ui.button("use the open sphere", "vsluse", "Switches the base mesh (01) to the open sphere.",
                         kind="primary"):
                self.use_sphere()
            if not v["enabled"]:
                return
        changed, on = ui.check("vessel on", v["enabled"], "vslon")
        if changed:
            v["enabled"] = on
            if on and d.base.get("shape") != "sphere_open":
                self.use_sphere()
            app.print_panel.defaults_for_shape_pending = True
            app.edited()
        if not v["enabled"]:
            return

        edited = False
        ui.subhead("size")
        ch, val = ui.slider("diameter", v["diameter_mm"], 30.0, 300.0, "%.0f mm",
                            "The real, printed size (also in base mesh and print). Ctrl/Cmd+click to type. The view "
                            "always fits the sphere to the screen; walls stay as set in mm.", "vsldiam")
        if ch:
            v["diameter_mm"] = float(min(max(val, 20.0), 400.0))
            edited = True
        ui.subhead("relief")
        ch, val = ui.slider("depth", v["depth"], 0.1, 4.0, "x %.2f",
                            "1 = the form's true proportions; 2 = twice as deep into the sphere.", "vsldepth")
        if ch:
            v["depth"], edited = val, True
        ch, val = ui.slider("glowing share", v["glow"], 1.0, 60.0, "%.0f %%",
                            "How much of the inside (the parts reaching out furthest) presses against the shell as "
                            "thin, glowing windows.", "vslglow")
        if ch:
            v["glow"], edited = val, True
        ch, val = ui.check("turn the relief inside out", v["invert"], "vslinv")
        if ch:
            v["invert"], edited = val, True
        ui.subhead("walls")
        ch, val = ui.slider("window", v["min_wall_mm"], 0.4, 3.0, "%.2f mm",
                            "The thinnest wall: the glowing windows. Keep it >= 2 nozzle widths (0.8 mm for a 0.4 "
                            "nozzle).", "vslmin")
        if ch:
            v["min_wall_mm"] = val
            v["max_wall_mm"] = max(v["max_wall_mm"], val + 0.1)
            edited = True
        ch, val = ui.slider("deepest", v["max_wall_mm"], 1.0, 30.0, "%.1f mm",
                            "A cap on how far the relief may reach into the sphere (it flattens anything deeper).",
                            "vslmax")
        if ch:
            v["max_wall_mm"] = max(val, v["min_wall_mm"] + 0.1)
            edited = True
        ch, val = ui.slider("rim", v["rim_mm"], 0.8, 8.0, "%.1f mm",
                            "Solid ring around the opening: what the print stands on.", "vslrim")
        if ch:
            v["rim_mm"], edited = val, True
        if edited:
            app.edited()

        info = app.result.mesh.info.get("vessel") if app.result is not None else None
        if info:
            ui.subhead("result")
            D = info["diameter_mm"]
            lo, hi = info["wall_mm"]
            ui.value("wall", f"{lo:.2f} - {hi:.2f} mm")
            if hi >= v["max_wall_mm"] - 1e-6:
                ui.warn("The relief reaches the cap: deeper parts are flattened.")
            op = info.get("opening_deg")
            if op is not None:
                ui.value("opening", f"{D * np.sin(np.radians(op)):.0f} mm across ({op:.0f} deg from the top)")
                if op >= 35:
                    ui.ok(f"Prints upside down on the rim: the outside starts at {op:.0f} deg from flat, no support "
                          "needed.")
                else:
                    ui.warn(f"The outside starts only {op:.0f} deg from flat near the rim: it may need support there. "
                            "Widen the opening (base mesh) to 35 deg or more.")
        ui.gap(4.0)
        if ui.toggle("light preview", app.color_mode == "light", "vsllight",
                     "Colours the sphere by how much light gets through the wall when lit from inside (bright = "
                     "thin). Use the cut tool over the view to see the relief itself."):
            app.set_color_mode("none" if app.color_mode == "light" else "light")
        psim.SameLine(0.0, 6.0)
        if ui.button("cut it open", "vslcut", "Turn on the section cut (toolbar) to see inside."):
            app.section_tool.open_cut()

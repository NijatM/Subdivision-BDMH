"""Attractor panel for app.py: add / edit / drag attractors and visualise their influence."""

from __future__ import annotations

import copy
import os

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import ui_style as ui
from hansmeyer import CC_WEIGHTS, DS_WEIGHTS, MAX_ITERATIONS
from hansmeyer.attractors import (
    AXES,
    CURVE_TYPES,
    FALLOFFS,
    SPLINE_KNOTS,
    curve_points,
    face_positions,
    falloff,
    influence,
    load_polyline,
    normalize,
    to_polyline,
)
from hansmeyer.view import helper
from ui_common import rules_editor

HANDLE = "attractor handle"
POINTS = "attractor points"
CURVE_PREFIX = "attractor curve "


def _color(a, selected):
    """Monochrome: weight sets light, modifiers mid grey, switched-off dark; the selected one full white."""
    if not a["enabled"]:
        return helper("off")
    if selected:
        return helper("main")
    return helper("set" if a["payload"] == "set" else "mod")


class AttractorPanel:
    def __init__(self, app):
        self.app = app
        self.sel = 0
        self.iter_tab = 0
        self.ctrl = 0
        self.show = True
        self.import_path = ""
        self.geom_dirty = True
        self.gizmo = False  # the drag handle shows only while the Attractors section is open
        self._curve_names: set[str] = set()
        self._handle = None
        self._handle_pos = None

    # ------------------------------------------------------------ helpers
    @property
    def design(self):
        return self.app.design

    @property
    def atts(self):
        return self.design.attractors

    def selected(self):
        return self.atts[self.sel] if 0 <= self.sel < len(self.atts) else None

    def changed(self, geometry=True):
        self.app.edited()
        if geometry:
            self.geom_dirty = True

    def _scene(self):
        m = self.app.base_mesh
        if m is None:
            return np.zeros(3), 2.0
        lo, hi = m.V.min(0), m.V.max(0)
        return 0.5 * (lo + hi), float(np.linalg.norm(hi - lo))

    def reset(self):
        """Called after loading a preset."""
        self.sel = 0 if self.atts else -1
        self.ctrl = 0
        self.geom_dirty = True

    # --------------------------------------------------------------- anchor
    def _anchor(self, a):
        """The point the gizmo moves: a point attractor, a polyline control point, or a curve's centre."""
        if a["kind"] == "point":
            return np.array(a["position"], float)
        c = a["curve"]
        if c["type"] == "polyline" and c["points"]:
            self.ctrl = min(self.ctrl, len(c["points"]) - 1)
            return np.array(c["points"][self.ctrl], float)
        if c["type"] == "line":
            return 0.5 * (np.array(c["start"], float) + np.array(c["end"], float))
        return np.array(c["center"], float)

    def _move_anchor(self, a, new):
        delta = np.asarray(new, float) - self._anchor(a)
        if a["kind"] == "point":
            a["position"] = list(map(float, new))
            return
        c = a["curve"]
        if c["type"] == "polyline" and c["points"]:
            c["points"][self.ctrl] = list(map(float, new))
        elif c["type"] == "line":
            c["start"] = list(map(float, np.add(c["start"], delta)))
            c["end"] = list(map(float, np.add(c["end"], delta)))
        else:
            c["center"] = list(map(float, new))

    # --------------------------------------------------------- scene sync
    def sync(self, gizmo: bool = True):
        """Per frame: pick up gizmo drags, then redraw attractor geometry if needed."""
        if gizmo != self.gizmo:
            self.gizmo, self.geom_dirty = gizmo, True
        a = self.selected()
        if self._handle is not None and a is not None and self._handle_pos is not None:
            pos = np.array(self._handle.get_transform())[:3, 3]
            if np.linalg.norm(pos - self._handle_pos) > 1e-7:
                self._move_anchor(a, pos)
                self._handle_pos = pos
                self.changed()
        if self.geom_dirty:
            self.draw()
            self.geom_dirty = False

    def draw(self):
        visible = self.show
        pts = [(i, a) for i, a in enumerate(self.atts) if a["kind"] == "point"]
        if pts and visible:
            P = np.array([a["position"] for _, a in pts], float)
            pc = ps.register_point_cloud(POINTS, P, radius=0.012, color=helper("set"))
            pc.add_color_quantity("type", np.array([_color(a, i == self.sel) for i, a in pts]), enabled=True)
        else:
            ps.remove_point_cloud(POINTS, error_if_absent=False)

        wanted = set()
        for i, a in enumerate(self.atts):
            if a["kind"] != "curve" or not visible:
                continue
            name = f"{CURVE_PREFIX}{i}"
            Q = curve_points(a["curve"], 160)
            edges = np.stack([np.arange(len(Q) - 1), np.arange(1, len(Q))], 1)
            ps.register_curve_network(name, Q, edges, radius=0.004 if i != self.sel else 0.007,
                                      color=_color(a, i == self.sel))
            if a["curve"]["type"] == "polyline" and i == self.sel:
                ctrl = np.array(a["curve"]["points"], float).reshape(-1, 3)
                if len(ctrl):
                    ps.register_point_cloud(name + " ctrl", ctrl, radius=0.008, color=helper("main"))
                    wanted.add(name + " ctrl")
            wanted.add(name)
        for name in self._curve_names - wanted:
            if name.endswith(" ctrl"):
                ps.remove_point_cloud(name, error_if_absent=False)
            else:
                ps.remove_curve_network(name, error_if_absent=False)
        self._curve_names = wanted

        a = self.selected()
        if a is not None and visible and self.gizmo:
            if self._handle is None:
                self._handle = ps.register_point_cloud(HANDLE, np.zeros((1, 3)), radius=0.016, color=helper("main"))
                self._handle.set_transform_gizmo_enabled(True)
            self._handle.set_color(helper("main"))
            self._handle.set_enabled(True)
            self._handle.set_transform_gizmo_enabled(True)
            pos = self._anchor(a)
            T = np.eye(4)
            T[:3, 3] = pos
            self._handle.set_transform(T)
            self._handle_pos = pos
        elif self._handle is not None:
            self._handle.set_enabled(False)
            self._handle.set_transform_gizmo_enabled(False)  # (hiding the point alone leaves the gizmo drawn)
            self._handle_pos = None

    def influence_values(self, mesh):
        a = self.selected()
        if a is None:
            return None, None
        P = face_positions(mesh, self.design.attractor_space)
        return influence(a, P), (0.0, max(float(a["strength"]), 1e-6))

    # ------------------------------------------------------------------ ui
    def ui(self):
        d = self.design
        changed, self.show = ui.check("show in view", self.show, "attshow", "Draw the attractors and the drag handle.")
        if changed:
            self.geom_dirty = True
        psim.SameLine(0.0, 16.0)
        changed, on = ui.check("influence map", self.app.color_mode == "influence", "attinfl",
                               "Colour the form by the selected attractor's influence (strength x falloff).")
        if changed:
            self.app.set_color_mode("influence" if on else "none")
        changed, v = ui.choice("measure at", [("current", "current"), ("rest", "original")], d.attractor_space,
                               "attspace", helps=["Distance from each face's current position: growth reacts to "
                                                  "where it has moved.", "Distance from where the face sat on the "
                                                  "input mesh: stable zoning."])
        if changed:
            d.attractor_space = v
            self.changed(False)
        changed, v = ui.slider("background", d.background, 0.0, 3.0, "%.2f",
                               "Influence of the main schedule in the weight-set blend (paper eq. 8-9). 0 = "
                               "paper-pure: only the sets count, and a single set then applies fully everywhere "
                               "inside its radius. Use > 0 for a gradual effect.", "attbg")
        if changed:
            d.background = v
            self.changed(False)
        ui.explain("Weight sets carry a whole schedule of their own and are blended by distance (paper eq. 8-9, "
                   "Fig. 4). Modifiers scale or offset chosen weights near them. Drag the white handle in the view "
                   "to move the selected attractor.")

        ui.subhead("attractors")
        centre, diag = self._scene()
        if ui.button("+ point", "attaddp", "A point attractor: influence falls off with distance from it."):
            self.add({"kind": "point", "position": (centre + [0.35 * diag, 0, 0]).round(3).tolist(),
                      "radius": round(0.35 * diag, 3)})
        psim.SameLine(0.0, 6.0)
        if ui.button("+ curve", "attaddc", "A curve attractor (circle, helix, sine, line, polyline, ...)."):
            self.add({"kind": "curve", "radius": round(0.2 * diag, 3),
                      "curve": {"type": "helix", "center": centre.round(3).tolist(), "radius": round(0.25 * diag, 3),
                                "height": round(0.6 * diag, 3), "length": round(0.6 * diag, 3),
                                "size": round(0.3 * diag, 3)}})
        ui.gap(2.0)
        ui.begin_list()
        for i, a in enumerate(self.atts):
            what = a["kind"] if a["kind"] == "point" else a["curve"]["type"]
            if ui.list_row(f"att{i}", f"{a['name']}  {what} · {'set' if a['payload'] == 'set' else 'modifier'}",
                           "on" if a["enabled"] else "off", selected=i == self.sel, faint=not a["enabled"]):
                self.sel, self.ctrl = i, 0
                self.geom_dirty = True
                self.app.refresh_display()
        ui.end_list()
        if not self.atts:
            ui.empty("No attractors: the weights are the same everywhere in each iteration.")
            return
        a = self.selected()
        if a is None:
            return
        self.editor(a)

    def add(self, spec):
        n = len(self.atts) + 1
        names = {a["name"] for a in self.atts}
        while f"A{n}" in names:
            n += 1
        a = normalize({"name": f"A{n}", **spec})
        if not a["mods"]:
            a["mods"] = [{"weight": "w_f", "op": "scale", "value": 2.0, "from": 1, "to": MAX_ITERATIONS}]
        self.atts.append(a)
        self.sel, self.ctrl = len(self.atts) - 1, 0
        self.changed()

    # -------------------------------------------------------------- editor
    def editor(self, a):
        ui.subhead(f"edit {a['name']}")
        _, a["name"] = ui.input_text("name", a["name"], key="attname")
        changed, a["enabled"] = ui.check("on", a["enabled"], "atton", "Switch it off without deleting it.")
        if changed:
            self.changed()
        psim.SameLine(0.0, 16.0)
        if ui.button("duplicate", "attdup"):
            dup = copy.deepcopy(a)
            dup["name"] = dup["name"] + "'"
            self.atts.insert(self.sel + 1, dup)
            self.sel += 1
            self.changed()
        psim.SameLine(0.0, 6.0)
        if ui.button("delete", "attdel"):
            self.atts.pop(self.sel)
            self.sel = min(self.sel, len(self.atts) - 1)
            self.changed()
            return

        changed, kind = ui.choice("kind", [("point", "point"), ("curve", "curve")], a["kind"], "attkind")
        if changed:
            a["kind"] = kind
            self.changed()
        if a["kind"] == "point":
            changed, v = ui.drag3("position", a["position"], 0.01, "Drag here, or drag the handle in the view.",
                                  "attpos")
            if changed:
                a["position"] = list(v)
                self.changed()
        else:
            self.curve_ui(a)

        changed, payload = ui.choice("payload", [("set", "weight set"), ("modifier", "modifier")], a["payload"],
                                     "attpay", helps=["A full per-iteration weight schedule, blended with the others "
                                                      "by distance (eq. 8-9).",
                                                      "Scales or offsets chosen weights near the attractor."])
        if changed:
            a["payload"] = payload
            self.changed()
        changed, v = ui.slider("strength", a["strength"], 0.0, 5.0, "%.2f", "h in eq. 8.", "attstr")
        if changed:
            a["strength"] = v
            self.changed()
        _, diag = self._scene()
        changed, v = ui.slider("radius", a["radius"], 0.01, max(2.0 * diag, 1.0), "%.2f",
                               "Distance over which the falloff runs from 1 to its end value.", "attrad")
        if changed:
            a["radius"] = v
            self.changed()
        self.falloff_ui(a)
        if a["payload"] == "set":
            self.set_ui(a)
        else:
            self.mods_ui(a)

    def curve_ui(self, a):
        c = a["curve"]
        types = list(CURVE_TYPES)
        changed, idx = ui.combo("curve", types.index(c["type"]), types, key="atttype")
        if changed and types[idx] != c["type"]:
            if types[idx] == "polyline" and not c["points"]:
                c.update(to_polyline(c))
            c["type"] = types[idx]
            self.ctrl = 0
            self.changed()
        t = c["type"]
        if t == "line":
            for key in ("start", "end"):
                changed, v = ui.drag3(key, c[key], 0.01, key=f"att{key}")
                if changed:
                    c[key] = list(v)
                    self.changed()
        elif t == "polyline":
            self.polyline_ui(c)
        else:
            changed, v = ui.drag3("centre", c["center"], 0.01, "Or drag the handle in the view.", "attcentre")
            if changed:
                c["center"] = list(v)
                self.changed()
            if t != "lissajous":
                changed, ax = ui.choice("axis", [(x, x) for x in AXES], c["axis"], "attaxis")
                if changed:
                    c["axis"] = ax
                    self.changed()
            params = {"circle": [("radius", 0.01, 10)], "helix": [("radius", 0.01, 10), ("height", 0.01, 20), ("turns", 0.1, 12)],
                      "sine": [("length", 0.01, 20), ("amplitude", 0.0, 5), ("waves", 0.1, 12)],
                      "lissajous": [("size", 0.01, 10), ("phase", 0.0, 2.0)]}[t]
            for key, lo, hi in params:
                changed, v = ui.slider(key, float(c[key]), lo, hi, "%.2f", key=f"attc{key}")
                if changed:
                    c[key] = v
                    self.changed()
            if t == "lissajous":
                for key in ("a", "b", "c"):
                    changed, v = ui.slider_int(f"freq {key}", int(c[key]), 1, 8, key=f"attf{key}")
                    if changed:
                        c[key] = v
                        self.changed()
            if ui.button("convert to polyline", "attconv", "Turns the curve into control points you can drag one "
                                                           "by one."):
                c.update(to_polyline(c))
                self.ctrl = 0
                self.changed()

    def polyline_ui(self, c):
        pts = c["points"]
        if pts:
            changed, v = ui.slider_int("control point", self.ctrl, 0, len(pts) - 1, key="attctrl")
            if changed:
                self.ctrl = v
                self.geom_dirty = True
            changed, v = ui.drag3("point xyz", pts[self.ctrl], 0.01, "Or drag the handle in the view.", "attpt")
            if changed:
                pts[self.ctrl] = list(v)
                self.changed()
        if ui.button("+ point after", "attptadd"):
            if pts:
                nxt = pts[self.ctrl + 1] if self.ctrl + 1 < len(pts) else np.add(pts[self.ctrl], [0.3, 0, 0])
                pts.insert(self.ctrl + 1, list(map(float, 0.5 * (np.array(pts[self.ctrl]) + np.array(nxt)))))
                self.ctrl += 1
            else:
                pts.append([0.0, 0.0, 0.0])
            self.changed()
        psim.SameLine(0.0, 6.0)
        if ui.button("- remove point", "attptdel", enabled=len(pts) > 2):
            pts.pop(self.ctrl)
            self.ctrl = min(self.ctrl, len(pts) - 1)
            self.changed()
        changed, c["closed"] = ui.check("closed", bool(c["closed"]), "attclosed")
        if changed:
            self.changed()
        psim.SameLine(0.0, 16.0)
        changed, c["smooth"] = ui.check("smooth", bool(c["smooth"]), "attsmooth",
                                        "Catmull-Rom spline through the control points.")
        if changed:
            self.changed()
        _, self.import_path = ui.input_text("import file", self.import_path,
                                            "Polyline from Rhino / Blender: .obj (v + l) or .csv / .txt with x y z "
                                            "per line. Relative paths start in the project folder (inputs/curve.csv).",
                                            "attimp")
        if ui.button("import", "attimpbtn", enabled=bool(self.import_path.strip())):
            path = self.import_path.strip()
            full = path if os.path.isabs(path) else os.path.join(self.app.root, path)
            try:
                P = load_polyline(full)
                c["points"] = P.round(5).tolist()
                self.ctrl = 0
                self.app.status = f"imported {len(P)} points"
                self.changed()
            except (OSError, ValueError) as e:
                self.app.error = f"Import failed: {e}"

    def falloff_ui(self, a):
        f = a["falloff"]
        kinds = list(FALLOFFS)
        changed, idx = ui.combo("falloff", kinds.index(f["type"]), kinds,
                                "power = the paper's (1 - d)^t; spline = draw your own with the 5 sliders.", "attfo")
        if changed:
            f["type"] = kinds[idx]
            self.changed()
        if f["type"] == "power":
            changed, v = ui.slider("tightness t", f["tightness"], 0.1, 8.0, "%.2f", key="atttight")
            if changed:
                f["tightness"] = v
                self.changed()
        if f["type"] == "spline":
            ui.row_label("shape")
            for k in range(5):
                if k:
                    psim.SameLine(0.0, 4.0)
                changed, v = psim.VSliderFloat(f"##sp{k}", (26, 64), f["spline"][k], 0.0, 1.0, "")
                ui.tip(f"influence at d = {SPLINE_KNOTS[k]:.2f} x radius")
                if changed:
                    f["spline"][k] = v
                    self.changed()
        u = np.linspace(0, 1.25, 64)
        ui.row_label("curve", "Influence (up) against distance / radius (across).")
        psim.PlotLines("##falloff", falloff(u, f).astype(np.float32), scale_min=0.0, scale_max=1.0,
                       graph_size=(psim.GetContentRegionAvail()[0], 52))

    def set_ui(self, a):
        d = self.design
        ui.subhead("weight set")
        ui.note("The weights this set pulls toward (the scheme follows the main schedule).")
        self.iter_tab = min(self.iter_tab, d.full_depth - 1)
        options = [(k, f"{k + 1}{'ds' if d.iterations[k].scheme == 'ds' else ''}") for k in range(d.full_depth)]
        _, self.iter_tab = ui.segmented(options, self.iter_tab, "settab")
        k = self.iter_tab
        defs = CC_WEIGHTS if d.iterations[k].scheme == "cc" else DS_WEIGHTS
        w = a["weights"][k]
        for wd in defs:
            changed, v = ui.slider(" ".join(wd.label.split()[:2]), w[wd.name], wd.lo, wd.hi, "%.2f",
                                   f"{' '.join(wd.label.split())}\n{wd.help}", f"set{wd.name}")
            if changed:
                w[wd.name] = float(v)
                self.changed(False)
        if ui.button("from schedule", "setcopy", "Copy the main schedule into this set."):
            for j in range(MAX_ITERATIONS):
                a["weights"][j] = dict(d.iterations[j].weights)
            self.changed(False)
        psim.SameLine(0.0, 6.0)
        if ui.button("sharp", "setsharp", "w1 = -1, w2 = -2 for this iteration.",
                     enabled=d.iterations[k].scheme == "cc"):
            w.update({"w1": -1.0, "w2": -2.0})
            self.changed(False)
        psim.SameLine(0.0, 6.0)
        if ui.button("→ next", "setnext", "Copy this iteration to the next one.", enabled=k + 1 < MAX_ITERATIONS):
            a["weights"][k + 1] = dict(w)
            self.changed(False)
        psim.SameLine(0.0, 6.0)
        if ui.button("→ all", "setall", "Copy this iteration to all others."):
            for j in range(MAX_ITERATIONS):
                a["weights"][j] = dict(w)
            self.changed(False)
        if len([x for x in self.atts if x["payload"] == "set" and x["enabled"]]) == 1 and d.background <= 0:
            ui.warn("A single set with background 0 applies fully inside its radius: raise background for a "
                    "gradient.")

    def mods_ui(self, a):
        ui.subhead("modifier rules")
        ui.note("Applied near this attractor, after the weight-set blend.")
        if rules_editor(a["mods"], f"mod{self.sel}"):
            self.changed(False)

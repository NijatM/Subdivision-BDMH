"""Attractor panel for app.py: add / edit / drag attractors and visualise their influence."""

from __future__ import annotations

import copy
import os

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

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
from hansmeyer.schedule import ALL_WEIGHTS

HANDLE = "attractor handle"
POINTS = "attractor points"
CURVE_PREFIX = "attractor curve "
COL_SET = (0.95, 0.55, 0.15)  # orange: weight-set attractors
COL_MOD = (0.25, 0.75, 0.95)  # cyan: modifiers
COL_OFF = (0.45, 0.45, 0.45)  # disabled
WARN = (0.95, 0.55, 0.15, 1.0)
WEIGHT_LABELS = {w.name: w.label for w in ALL_WEIGHTS}
_OPEN = getattr(psim, "ImGuiTreeNodeFlags_DefaultOpen", 1 << 5)


def _color(a, selected):
    if not a["enabled"]:
        return COL_OFF
    c = np.array(COL_SET if a["payload"] == "set" else COL_MOD)
    return tuple(np.minimum(c * 1.25 + 0.15, 1.0)) if selected else tuple(c)


class AttractorPanel:
    def __init__(self, app):
        self.app = app
        self.sel = 0
        self.iter_tab = 0
        self.ctrl = 0
        self.show = True
        self.import_path = ""
        self.geom_dirty = True
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
    def sync(self):
        """Per frame: pick up gizmo drags, then redraw attractor geometry if needed."""
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
            pc = ps.register_point_cloud(POINTS, P, radius=0.012, color=COL_SET)
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
                    ps.register_point_cloud(name + " ctrl", ctrl, radius=0.008, color=(1, 1, 1))
                    wanted.add(name + " ctrl")
            wanted.add(name)
        for name in self._curve_names - wanted:
            if name.endswith(" ctrl"):
                ps.remove_point_cloud(name, error_if_absent=False)
            else:
                ps.remove_curve_network(name, error_if_absent=False)
        self._curve_names = wanted

        a = self.selected()
        if a is not None and visible:
            if self._handle is None:
                self._handle = ps.register_point_cloud(HANDLE, np.zeros((1, 3)), radius=0.016, color=(1, 1, 1))
                self._handle.set_transform_gizmo_enabled(True)
            self._handle.set_enabled(True)
            pos = self._anchor(a)
            T = np.eye(4)
            T[:3, 3] = pos
            self._handle.set_transform(T)
            self._handle_pos = pos
        elif self._handle is not None:
            self._handle.set_enabled(False)
            self._handle_pos = None

    def influence_values(self, mesh):
        a = self.selected()
        if a is None:
            return None, None
        P = face_positions(mesh, self.design.attractor_space)
        return influence(a, P), (0.0, max(float(a["strength"]), 1e-6))

    # ------------------------------------------------------------------ ui
    def ui(self):
        if not psim.CollapsingHeader("Attractors", _OPEN):
            return
        d = self.design
        changed, self.show = psim.Checkbox("show", self.show)
        if changed:
            self.geom_dirty = True
        psim.SameLine()
        changed, on = psim.Checkbox("influence map", self.app.color_mode == "influence")
        psim.SetItemTooltip("Colour the form by the selected attractor's influence (strength x falloff).")
        if changed:
            self.app.color_mode = "influence" if on else "none"
            self.app.refresh_display()

        psim.Text("Measure at:")
        psim.SameLine()
        if psim.RadioButton("current", d.attractor_space == "current"):
            d.attractor_space = "current"
            self.changed(False)
        psim.SetItemTooltip("Distance from each face's current position: growth reacts to where it has moved.")
        psim.SameLine()
        if psim.RadioButton("original", d.attractor_space == "rest"):
            d.attractor_space = "rest"
            self.changed(False)
        psim.SetItemTooltip("Distance from where the face sat on the input mesh: stable zoning.")
        changed, v = psim.SliderFloat("background", d.background, 0.0, 3.0, "%.2f")
        psim.SetItemTooltip(
            "Influence of the main schedule in the weight-set blend (paper eq. 8-9).\n"
            "0 = paper-pure: only the sets count. A single set then applies fully\n"
            "everywhere inside its radius - use >0 for a gradual effect.")
        if changed:
            d.background = v
            self.changed(False)

        centre, diag = self._scene()
        if psim.Button("+ Point"):
            self.add({"kind": "point", "position": (centre + [0.35 * diag, 0, 0]).round(3).tolist(),
                      "radius": round(0.35 * diag, 3)})
        psim.SameLine()
        if psim.Button("+ Curve"):
            self.add({"kind": "curve", "radius": round(0.2 * diag, 3),
                      "curve": {"type": "helix", "center": centre.round(3).tolist(), "radius": round(0.25 * diag, 3),
                                "height": round(0.6 * diag, 3), "length": round(0.6 * diag, 3),
                                "size": round(0.3 * diag, 3)}})
        psim.SameLine()
        if psim.Button("Duplicate") and self.selected() is not None:
            dup = copy.deepcopy(self.selected())
            dup["name"] = dup["name"] + "'"
            self.atts.insert(self.sel + 1, dup)
            self.sel += 1
            self.changed()
        psim.SameLine()
        if psim.Button("Delete") and self.selected() is not None:
            self.atts.pop(self.sel)
            self.sel = min(self.sel, len(self.atts) - 1)
            self.changed()

        for i, a in enumerate(self.atts):
            tag = f"{a['name']}  -  {a['kind']}{' ' + a['curve']['type'] if a['kind'] == 'curve' else ''}, " \
                  f"{'weight set' if a['payload'] == 'set' else 'modifier'}{'' if a['enabled'] else '  (off)'}"
            if psim.Selectable(f"{tag}##att{i}", i == self.sel):
                self.sel, self.ctrl = i, 0
                self.geom_dirty = True
                self.app.refresh_display()
        if not self.atts:
            psim.TextDisabled("No attractors: weights are uniform per iteration.")
            return
        a = self.selected()
        if a is None:
            return
        psim.Separator()
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
        _, a["name"] = psim.InputText("name", a["name"])
        psim.SameLine()
        changed, a["enabled"] = psim.Checkbox("on", a["enabled"])
        if changed:
            self.changed()

        psim.Text("Kind:")
        psim.SameLine()
        for kind in ("point", "curve"):
            if psim.RadioButton(kind, a["kind"] == kind):
                if a["kind"] != kind:
                    a["kind"] = kind
                    self.changed()
            psim.SameLine()
        psim.NewLine()
        if a["kind"] == "point":
            changed, v = psim.DragFloat3("position", a["position"], 0.01)
            psim.SetItemTooltip("Drag here, or drag the gizmo in the viewport.")
            if changed:
                a["position"] = list(v)
                self.changed()
        else:
            self.curve_ui(a)

        psim.Text("Payload:")
        psim.SameLine()
        if psim.RadioButton("weight set", a["payload"] == "set"):
            if a["payload"] != "set":
                a["payload"] = "set"
                self.changed()
        psim.SetItemTooltip("A full per-iteration weight schedule, blended with the others by distance (eq. 8-9).")
        psim.SameLine()
        if psim.RadioButton("modifier", a["payload"] == "modifier"):
            if a["payload"] != "modifier":
                a["payload"] = "modifier"
                self.changed()
        psim.SetItemTooltip("Scales or offsets chosen weights near the attractor.")

        changed, v = psim.SliderFloat("strength", a["strength"], 0.0, 5.0, "%.2f")
        psim.SetItemTooltip("h in eq. 8.")
        if changed:
            a["strength"] = v
            self.changed()
        _, diag = self._scene()
        changed, v = psim.SliderFloat("radius", a["radius"], 0.01, max(2.0 * diag, 1.0), "%.2f")
        psim.SetItemTooltip("Distance over which the falloff runs from 1 to its end value.")
        if changed:
            a["radius"] = v
            self.changed()
        self.falloff_ui(a)
        psim.Separator()
        if a["payload"] == "set":
            self.set_ui(a)
        else:
            self.mods_ui(a)

    def curve_ui(self, a):
        c = a["curve"]
        types = list(CURVE_TYPES)
        changed, idx = psim.Combo("curve", types.index(c["type"]), types)
        if changed and types[idx] != c["type"]:
            if types[idx] == "polyline" and not c["points"]:
                c.update(to_polyline(c))
            c["type"] = types[idx]
            self.ctrl = 0
            self.changed()
        t = c["type"]
        if t == "line":
            for key in ("start", "end"):
                changed, v = psim.DragFloat3(key, c[key], 0.01)
                if changed:
                    c[key] = list(v)
                    self.changed()
        elif t == "polyline":
            self.polyline_ui(c)
        else:
            if t != "lissajous":
                changed, v = psim.DragFloat3("centre", c["center"], 0.01)
                if changed:
                    c["center"] = list(v)
                    self.changed()
                psim.Text("axis:")
                for ax in AXES:
                    psim.SameLine()
                    if psim.RadioButton(f"{ax}##axis", c["axis"] == ax):
                        c["axis"] = ax
                        self.changed()
            else:
                changed, v = psim.DragFloat3("centre", c["center"], 0.01)
                if changed:
                    c["center"] = list(v)
                    self.changed()
            params = {"circle": [("radius", 0.01, 10)], "helix": [("radius", 0.01, 10), ("height", 0.01, 20), ("turns", 0.1, 12)],
                      "sine": [("length", 0.01, 20), ("amplitude", 0.0, 5), ("waves", 0.1, 12)],
                      "lissajous": [("size", 0.01, 10), ("phase", 0.0, 2.0)]}[t]
            for key, lo, hi in params:
                changed, v = psim.SliderFloat(f"{key}##c", float(c[key]), lo, hi, "%.2f")
                if changed:
                    c[key] = v
                    self.changed()
            if t == "lissajous":
                for key in ("a", "b", "c"):
                    changed, v = psim.SliderInt(f"freq {key}", int(c[key]), 1, 8)
                    if changed:
                        c[key] = v
                        self.changed()
            if psim.Button("Convert to editable polyline"):
                c.update(to_polyline(c))
                self.ctrl = 0
                self.changed()
            psim.SetItemTooltip("Turns the curve into control points you can drag one by one.")

    def polyline_ui(self, c):
        pts = c["points"]
        if pts:
            changed, v = psim.SliderInt("control point", self.ctrl, 0, len(pts) - 1)
            if changed:
                self.ctrl = v
                self.geom_dirty = True
            changed, v = psim.DragFloat3("point xyz", pts[self.ctrl], 0.01)
            psim.SetItemTooltip("Or drag the gizmo in the viewport.")
            if changed:
                pts[self.ctrl] = list(v)
                self.changed()
        if psim.Button("+ point after"):
            if pts:
                nxt = pts[self.ctrl + 1] if self.ctrl + 1 < len(pts) else np.add(pts[self.ctrl], [0.3, 0, 0])
                pts.insert(self.ctrl + 1, list(map(float, 0.5 * (np.array(pts[self.ctrl]) + np.array(nxt)))))
                self.ctrl += 1
            else:
                pts.append([0.0, 0.0, 0.0])
            self.changed()
        psim.SameLine()
        if psim.Button("- remove point") and len(pts) > 2:
            pts.pop(self.ctrl)
            self.ctrl = min(self.ctrl, len(pts) - 1)
            self.changed()
        changed, c["closed"] = psim.Checkbox("closed", bool(c["closed"]))
        if changed:
            self.changed()
        psim.SameLine()
        changed, c["smooth"] = psim.Checkbox("smooth", bool(c["smooth"]))
        psim.SetItemTooltip("Catmull-Rom spline through the control points.")
        if changed:
            self.changed()
        _, self.import_path = psim.InputText("file", self.import_path)
        psim.SetItemTooltip("Polyline from Rhino/Blender: .obj (v + l) or .csv/.txt with x y z per line.\n"
                            "Relative paths start in the project folder (e.g. inputs/curve.csv).")
        psim.SameLine()
        if psim.Button("Import"):
            path = self.import_path.strip()
            full = path if os.path.isabs(path) else os.path.join(self.app.root, path)
            try:
                P = load_polyline(full)
                c["points"] = P.round(5).tolist()
                self.ctrl = 0
                self.app.status = f"Imported {len(P)} points"
                self.changed()
            except (OSError, ValueError) as e:
                self.app.error = f"Import failed: {e}"

    def falloff_ui(self, a):
        f = a["falloff"]
        kinds = list(FALLOFFS)
        changed, idx = psim.Combo("falloff", kinds.index(f["type"]), kinds)
        psim.SetItemTooltip("power = the paper's (1 - d)^t; spline = draw your own with the 5 sliders.")
        if changed:
            f["type"] = kinds[idx]
            self.changed()
        if f["type"] == "power":
            changed, v = psim.SliderFloat("tightness t", f["tightness"], 0.1, 8.0, "%.2f")
            if changed:
                f["tightness"] = v
                self.changed()
        if f["type"] == "spline":
            for k in range(5):
                if k:
                    psim.SameLine()
                changed, v = psim.VSliderFloat(f"##sp{k}", (28, 70), f["spline"][k], 0.0, 1.0, "")
                psim.SetItemTooltip(f"influence at d = {SPLINE_KNOTS[k]:.2f} x radius")
                if changed:
                    f["spline"][k] = v
                    self.changed()
            psim.SameLine()
        u = np.linspace(0, 1.25, 64)
        psim.PlotLines("##falloff", falloff(u, f).astype(np.float32), overlay_text="influence vs distance / radius",
                       scale_min=0.0, scale_max=1.0, graph_size=(0, 70))

    def set_ui(self, a):
        d = self.design
        psim.TextDisabled("Weights this set pulls toward (scheme follows the main schedule):")
        if psim.Button("Copy main schedule into this set"):
            for k in range(MAX_ITERATIONS):
                a["weights"][k] = dict(d.iterations[k].weights)
            self.changed(False)
        self.iter_tab = min(self.iter_tab, d.full_depth - 1)
        for k in range(d.full_depth):
            if k:
                psim.SameLine()
            active = k == self.iter_tab
            if active:
                psim.PushStyleColor(psim.ImGuiCol_Button, (0.85, 0.55, 0.20, 1.0))
            ds = d.iterations[k].scheme == "ds"
            if psim.Button(f"{k + 1}{'DS' if ds else ''}##set{k}"):
                self.iter_tab = k
            if active:
                psim.PopStyleColor()
        k = self.iter_tab
        defs = CC_WEIGHTS if d.iterations[k].scheme == "cc" else DS_WEIGHTS
        w = a["weights"][k]
        for wd in defs:
            changed, v = psim.SliderFloat(f"{wd.label}##set", w[wd.name], wd.lo, wd.hi, "%.2f")
            psim.SetItemTooltip(wd.help)
            if changed:
                w[wd.name] = float(v)
                self.changed(False)
        if psim.Button("Sharp##set") and d.iterations[k].scheme == "cc":
            w.update({"w1": -1.0, "w2": -2.0})
            self.changed(False)
        psim.SameLine()
        if psim.Button("Copy -> next##set") and k + 1 < MAX_ITERATIONS:
            a["weights"][k + 1] = dict(w)
            self.changed(False)
        psim.SameLine()
        if psim.Button("Copy -> all##set"):
            for j in range(MAX_ITERATIONS):
                a["weights"][j] = dict(w)
            self.changed(False)
        if len([x for x in self.atts if x["payload"] == "set" and x["enabled"]]) == 1 and d.background <= 0:
            psim.TextColored(WARN, "Single set with background 0: it applies fully\ninside its radius (raise background for a gradient).")

    def mods_ui(self, a):
        psim.TextDisabled("Rules applied near this attractor (after the weight-set blend):")
        names = [w.name for w in ALL_WEIGHTS]
        remove = None
        for i, m in enumerate(a["mods"]):
            psim.PushItemWidth(110)
            changed, idx = psim.Combo(f"##mw{i}", names.index(m["weight"]) if m["weight"] in names else 0,
                                      [WEIGHT_LABELS[n] for n in names])
            if changed:
                m["weight"] = names[idx]
                self.changed(False)
            psim.SameLine()
            psim.PushItemWidth(70)
            changed, op = psim.Combo(f"##mo{i}", 0 if m["op"] == "scale" else 1, ["scale x", "offset +"])
            psim.PopItemWidth()
            if changed:
                m["op"] = "scale" if op == 0 else "offset"
                m["value"] = 2.0 if m["op"] == "scale" else 0.3
                self.changed(False)
            psim.SameLine()
            lo, hi = (0.0, 5.0) if m["op"] == "scale" else (-1.5, 1.5)
            changed, v = psim.SliderFloat(f"##mv{i}", m["value"], lo, hi, "%.2f")
            if changed:
                m["value"] = v
                self.changed(False)
            psim.SameLine()
            if psim.Button(f"x##mx{i}"):
                remove = i
            changed, rng = psim.SliderInt2(f"iterations##mr{i}", [m["from"], m["to"]], 1, MAX_ITERATIONS)
            if changed:
                m["from"], m["to"] = min(rng), max(rng)
                self.changed(False)
            psim.PopItemWidth()
        if remove is not None:
            a["mods"].pop(remove)
            self.changed(False)
        if psim.Button("+ rule"):
            a["mods"].append({"weight": "w_f", "op": "scale", "value": 2.0, "from": 1, "to": MAX_ITERATIONS})
            self.changed(False)

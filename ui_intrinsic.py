"""Paper-feature sections for app.py: groups (tag & lock), intrinsic rules (motifs, measures), porosity."""

from __future__ import annotations

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import ui_style as ui
from hansmeyer import MAX_ITERATIONS
from hansmeyer import intrinsic as I
from hansmeyer.view import helper
from ui_common import item_header, iteration_range, rules_editor, weight_combo

PICK_MESH = "input mesh (tagging)"
PICK_VERTS = "group vertices"


class IntrinsicPanel:
    def __init__(self, app):
        self.app = app
        self.sel = 0
        self.pick = "off"  # "off" | "faces" | "verts"
        self.normal_dir = 2
        self.normal_angle = 30.0
        self.band_axis = 1
        self.band = [0.0, 0.2]
        self.kth = 2
        self.motif_sel = 0
        self.overlay_dirty = True

    @property
    def design(self):
        return self.app.design

    def selected(self):
        g = self.design.groups
        return g[self.sel] if 0 <= self.sel < len(g) else None

    def reset(self):
        self.sel = 0 if self.design.groups else -1
        self.set_pick("off")

    def changed(self):
        self.app.edited()
        self.overlay_dirty = True

    # ------------------------------------------------------- picking overlay
    def set_pick(self, mode):
        self.pick = mode
        self.overlay_dirty = True
        if mode == "off":
            ps.remove_surface_mesh(PICK_MESH, error_if_absent=False)
            ps.remove_point_cloud(PICK_VERTS, error_if_absent=False)
            if ps.has_surface_mesh("form"):
                ps.get_surface_mesh("form").set_enabled(True)

    def sync(self):
        """Per frame: draw the tagging overlay and handle clicks while picking."""
        if self.pick == "off" or self.app.base_mesh is None:
            return
        m = self.app.base_mesh
        g = self.selected()
        if self.overlay_dirty:
            disp = m.display_faces()
            base, picked = helper("base"), helper("main")
            sm = ps.register_surface_mesh(PICK_MESH, m.V, disp, color=base, edge_width=1.0,
                                          edge_color=helper("mod"), material="flat")
            if g is not None:
                col = np.tile(base, (m.n_faces, 1))
                if g["faces"]:
                    col[[f for f in g["faces"] if f < m.n_faces]] = picked
                if len(disp) != m.n_faces:
                    col = np.repeat(col, m.face_size - 2, axis=0)
                sm.add_color_quantity("group", col, defined_on="faces", enabled=True)
            verts = [v for v in (g["verts"] if g else []) if v < m.n_verts]
            if verts:
                ps.register_point_cloud(PICK_VERTS, m.V[verts], radius=0.012, color=helper("main"))
            else:
                ps.remove_point_cloud(PICK_VERTS, error_if_absent=False)
            if ps.has_surface_mesh("form"):
                ps.get_surface_mesh("form").set_enabled(False)
            self.overlay_dirty = False
        io = psim.GetIO()
        if g is None or io.WantCaptureMouse or not psim.IsMouseClicked(0):
            return
        res = ps.pick(screen_coords=tuple(io.MousePos))
        if not res.is_hit or res.structure_name != PICK_MESH:
            return
        data = res.structure_data
        if data.get("element_type") == "face":
            f = int(data["index"])
            if len(m.display_faces()) != m.n_faces:  # triangulated display -> polygon index
                f = int(np.repeat(np.arange(m.n_faces), m.face_size - 2)[f])
        else:
            f = None
        if self.pick == "faces" and f is not None:
            self._toggle(g["faces"], f)
        elif self.pick == "verts":
            # nearest corner of the clicked face (or the clicked vertex)
            if data.get("element_type") == "vertex":
                v = int(data["index"])
            else:
                face = m.face_idx[m.face_ptr[f]:m.face_ptr[f + 1]] if f is not None else np.arange(m.n_verts)
                v = int(face[np.argmin(np.linalg.norm(m.V[face] - res.position, axis=1))])
            self._toggle(g["verts"], v)
        self.changed()

    @staticmethod
    def _toggle(items, i):
        if i in items:
            items.remove(i)
        else:
            items.append(i)
            items.sort()

    # ---------------------------------------------------------------- groups
    def groups_ui(self):
        d = self.design
        ui.explain("Paper Fig. 9: tag faces or vertices of the input mesh. Locked vertices keep their position for "
                   "a number of iterations (locked edges become creases, locked points spikes); a group's weight "
                   "rules change the weights of its faces.")
        if ui.button("+ group", "grpadd"):
            d.groups.append(I.normalize_group({"name": f"G{len(d.groups) + 1}"}))
            self.sel = len(d.groups) - 1
            self.changed()
        ui.gap(2.0)
        ui.begin_list()
        for i, x in enumerate(d.groups):
            detail = f"{len(x['faces'])}f {len(x['verts'])}v" + (f" · lock {x['lock']}" if x["lock"] else "") \
                + (f" · {len(x['rules'])} rules" if x["rules"] else "")
            if ui.list_row(f"grp{i}", f"{x['name']}  {detail}", "on" if x["enabled"] else "off",
                           selected=i == self.sel, faint=not x["enabled"]):
                self.sel = i
                self.overlay_dirty = True
        ui.end_list()
        if not d.groups:
            ui.empty("No groups. Add one, then click faces or vertices of the input mesh in the view.")
            return
        g = self.selected()
        if g is None:
            return
        ui.subhead(f"edit {g['name']}")
        _, g["name"] = ui.input_text("name", g["name"], key="grpname")
        ch, g["enabled"] = ui.check("on", g["enabled"], "grpon", "Switch it off without deleting it.")
        if ch:
            self.changed()
        psim.SameLine(0.0, 16.0)
        if ui.button("delete", "grpdel"):
            d.groups.pop(self.sel)
            self.sel = min(self.sel, len(d.groups) - 1)
            self.changed()
            return

        ch, mode = ui.choice("click to tag", [("off", "off"), ("faces", "faces"), ("verts", "vertices")], self.pick,
                             "grppick", "Shows the input mesh; click faces / vertices in the view to add or "
                                        "remove them.")
        if ch:
            self.set_pick(mode)
        self.select_tools(g)

        ui.subhead("lock")
        ch, g["lock"] = ui.slider_int("iterations", int(g["lock"]), 0, MAX_ITERATIONS,
                                      help="Group vertices keep their position for this many iterations (paper Fig. "
                                           "9, L_e): locked edges become sharp creases, locked points spikes.",
                                      key="grplock")
        if ch:
            self.changed()
        ch, g["lock_faces"] = ui.check("+ face corners", g["lock_faces"], "grplockf",
                                       "Also lock every corner of the group's faces (keeps whole regions flat).")
        if ch:
            self.changed()
        ui.subhead("weight rules")
        ui.note("For the group's faces.")
        if rules_editor(g["rules"], "grpr"):
            self.changed()

    def select_tools(self, g):
        m = self.app.base_mesh
        if m is None:
            return
        ui.subhead("select by rule")
        target = g["verts"] if self.pick == "verts" else g["faces"]
        what = "vertices" if self.pick == "verts" else "faces"
        ui.push_font("small")
        ui.note(f"Adds to the group's {what} (switch 'click to tag' to choose).")
        ui.pop_font()
        dirs = list(I.AXIS_VECTORS)
        _, self.normal_dir = ui.combo("facing", self.normal_dir, dirs, key="selface")
        _, self.normal_angle = ui.slider("within", self.normal_angle, 1.0, 90.0, "%.0f deg", key="selang")
        if ui.button("add facing", "seladdf"):
            faces = I.select_by_normal(m, dirs[self.normal_dir], self.normal_angle)
            self._add(g, faces, from_faces=True)
        ui.gap(2.0)
        _, self.band_axis = ui.combo("band axis", self.band_axis, ["x", "y", "z"], key="selbax")
        _, band = ui.slider_float2("band (0-1)", self.band, 0.0, 1.0, "%.2f", key="selband")
        self.band = [min(band), max(band)]
        if ui.button("add band", "seladdb"):
            sel = I.select_by_height(m, self.band_axis, *self.band, faces=self.pick != "verts")
            self._add(g, sel, from_faces=self.pick != "verts")
        ui.gap(2.0)
        _, self.kth = ui.slider_int("every k-th", self.kth, 2, 12, key="selk")
        if ui.button("add every k-th", "seladdk"):
            n = m.n_verts if self.pick == "verts" else m.n_faces
            self._add(g, I.select_every_kth(n, self.kth), from_faces=self.pick != "verts")
        labels = list(I.motif_counts(m))
        if labels:
            ui.gap(2.0)
            self.motif_sel = min(self.motif_sel, len(labels) - 1)
            _, self.motif_sel = ui.combo("motif", self.motif_sel, labels, key="selmotif")
            if ui.button("add motif vertices", "seladdm"):
                g["verts"] = sorted(set(g["verts"]) | set(I.select_by_motif(m, labels[self.motif_sel]).tolist()))
                self.changed()
        ui.gap(2.0)
        if ui.button("clear", "selclear", f"Remove all the group's {what}."):
            target.clear()
            self.changed()
        psim.SameLine(0.0, 6.0)
        if ui.button("invert", "selinv", f"Swap the group's {what} for all the others."):
            n = m.n_verts if self.pick == "verts" else m.n_faces
            inv = sorted(set(range(n)) - set(target))
            target.clear()
            target.extend(inv)
            self.changed()

    def _add(self, g, idx, from_faces):
        idx = np.asarray(idx, np.int64)
        if self.pick == "verts" and from_faces:  # faces chosen, vertices wanted: use their corners
            m = self.app.base_mesh
            idx = np.unique(np.concatenate([m.face_idx[m.face_ptr[f]:m.face_ptr[f + 1]] for f in idx])) \
                if len(idx) else idx
            g["verts"] = sorted(set(g["verts"]) | set(idx.tolist()))
        elif self.pick == "verts":
            g["verts"] = sorted(set(g["verts"]) | set(idx.tolist()))
        else:
            g["faces"] = sorted(set(g["faces"]) | set(idx.tolist()))
        self.changed()

    # ------------------------------------------------------------ intrinsic
    def intrinsic_ui(self):
        self.motifs_ui()
        self.measures_ui()

    def motifs_ui(self):
        d = self.design
        m = self.app.base_mesh
        ui.subhead("motifs", top=0.0)
        ui.note("Vertices classed by their incident faces / edges (3F3E, 4F4E, ...). A motif's U attracts (+) or "
                "deflects (-) nearby points through w6 / w7 in the schedule.")
        ui.explain("Paper Fig. 7, eq. 10-11: the same weights act differently on differently connected vertices, "
                   "so a column's capital and base differentiate from the mesh's own topology.")
        counts = I.motif_counts(m) if m is not None else {}
        if self.app.result is not None and self.app.result.depth_reached > 0:
            for k, v in I.motif_counts(self.app.result.mesh).items():
                counts.setdefault(k, 0)
        for label in sorted(set(counts) | set(d.motifs), key=lambda s: (I.parse_label(s) // 100, I.parse_label(s) % 100)):
            n = counts.get(label, 0)
            ch, v = ui.slider(f"U {label}", float(d.motifs.get(label, 0.0)), -1.0, 1.0, "%.2f",
                              f"{n} vertices of motif {label} on the input mesh.", f"motif{label}")
            if ch:
                d.motifs[label] = v
                self.changed()
        if d.motifs and not any(it.weights.get("w6") or it.weights.get("w7") for it in d.iterations[: d.full_depth]):
            ui.warn("Set w6 / w7 in some iteration (schedule) for the motifs to act.")

    def measures_ui(self):
        d = self.design
        ui.subhead("measure rules")
        ui.note("A per-face measure t in [0, 1] sets, scales or adds to a weight, from 'at 0' to 'at 1'.")
        ui.explain("Paper Fig. 8: two sub-values interpolated by distance or curvature. Colour the form by the "
                   "measure (button below each rule) to see where it is high.")
        keys = list(I.MEASURES)
        remove = None
        for i, r in enumerate(d.intrinsic):
            if i:
                ui.gap(4.0)
            toggled, r["enabled"], rm = item_header(f"rule {i + 1}", f"mrule{i}", r["enabled"])
            if rm:
                remove = i
            c1, k = ui.combo("measure", keys.index(r["measure"]), [I.MEASURES[x] for x in keys], key=f"mrm{i}")
            if c1:
                r["measure"] = keys[k]
            c2, r["weight"] = weight_combo("weight", r["weight"], f"mrw{i}")
            c3, op = ui.choice("operation", [(o, o) for o in I.RULE_OPS], r["op"], f"mro{i}")
            if c3:
                r["op"] = op
            c4, r["a"] = ui.slider("at 0", r["a"], -2.0, 2.0, "%.2f", key=f"mra{i}")
            c5, r["b"] = ui.slider("at 1", r["b"], -2.0, 2.0, "%.2f", key=f"mrb{i}")
            c6, r["gamma"] = ui.slider("gamma", r["gamma"], 0.2, 5.0, "%.2f", "Shapes the ramp: t^gamma.", f"mrg{i}")
            c7 = iteration_range("iterations", r, f"mrr{i}")
            mode = f"measure:{r['measure']}"
            if ui.toggle("show measure", self.app.color_mode == mode, f"mrshow{i}", "Colour the form by this measure."):
                self.app.set_color_mode("none" if self.app.color_mode == mode else mode)
            if toggled or c1 or c2 or c3 or c4 or c5 or c6 or c7:
                self.changed()
        if remove is not None:
            d.intrinsic.pop(remove)
            self.changed()
        if not d.intrinsic:
            ui.empty("No measure rules.")
        if ui.button("+ measure rule", "mradd"):
            d.intrinsic.append(I.normalize_rule({}))
            self.changed()

    # ----------------------------------------------------------------- merge
    def merge_ui(self):
        mg = self.design.merge
        ui.explain("Vertices of different parts of the surface that grow into contact are welded. Where a welded "
                   "vertex would exceed the max valence, its faces are dropped: the surface opens into holes and "
                   "handles (watch the Euler characteristic).")
        ch0, mg["enabled"] = ui.check("merging on", mg["enabled"], "mergeon")
        ch1, mg["distance"] = ui.slider("distance", float(mg["distance"]), 0.01, 1.0, "%.3f",
                                        "x local edge length (relative) or model units (absolute).", "mergedist")
        ch2, mg["relative"] = ui.check("relative distance", mg["relative"], "mergerel",
                                       "Measure the distance in local edge lengths instead of model units.")
        ch3, mg["max_valence"] = ui.slider_int("max valence", int(mg["max_valence"]), 0, 12,
                                               help="0 = unlimited (welds only). 4-6 opens holes where sheets meet.",
                                               key="mergeval")
        ch4 = iteration_range("iterations", mg, "mergerange")
        if ch0 or ch1 or ch2 or ch3 or ch4:
            self.changed()
        r = self.app.result
        if r is not None and mg["enabled"]:
            ui.subhead("last step")
            info = r.mesh.info.get("merge")
            if info:
                ui.value("welded", f"{info['merged']} vertices, {info['dropped']} faces dropped")
                if info.get("skipped"):
                    ui.warn("Skipped: it would have destroyed the form.")
            m = r.mesh
            ui.value("topology", f"Euler {m.euler_characteristic()} · {int(m.edge_is_boundary.sum())} hole edges")

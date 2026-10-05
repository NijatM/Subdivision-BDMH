"""Paper-feature panels for app.py (M5): groups (tag & lock), motifs, measure rules, vertex merging."""

from __future__ import annotations

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from hansmeyer import MAX_ITERATIONS
from hansmeyer import intrinsic as I
from ui_common import WARN, iteration_range, list_selector, rules_editor, toggle_button, weight_combo

PICK_MESH = "input mesh (tagging)"
PICK_VERTS = "group vertices"
PICK_COLOR = (0.95, 0.55, 0.15)
BASE_COLOR = (0.55, 0.57, 0.62)


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
            sm = ps.register_surface_mesh(PICK_MESH, m.V, disp, color=BASE_COLOR, edge_width=1.0,
                                          edge_color=(0.1, 0.1, 0.1), material="flat")
            if g is not None:
                col = np.tile(BASE_COLOR, (m.n_faces, 1))
                if g["faces"]:
                    col[[f for f in g["faces"] if f < m.n_faces]] = PICK_COLOR
                if len(disp) != m.n_faces:
                    col = np.repeat(col, m.face_size - 2, axis=0)
                sm.add_color_quantity("group", col, defined_on="faces", enabled=True)
            verts = [v for v in (g["verts"] if g else []) if v < m.n_verts]
            if verts:
                ps.register_point_cloud(PICK_VERTS, m.V[verts], radius=0.012, color=PICK_COLOR)
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

    # ------------------------------------------------------------------ ui
    def ui(self):
        if psim.CollapsingHeader("Groups: tag & lock (paper Fig. 9)"):
            self.groups_ui()
        if psim.CollapsingHeader("Intrinsic: motifs & measures (paper Fig. 6-8)"):
            self.motifs_ui()
            psim.Separator()
            self.measures_ui()
        if psim.CollapsingHeader("Vertex merging / porosity"):
            self.merge_ui()

    # ---------------------------------------------------------------- groups
    def groups_ui(self):
        d = self.design
        if psim.Button("+ Group"):
            d.groups.append(I.normalize_group({"name": f"G{len(d.groups) + 1}"}))
            self.sel = len(d.groups) - 1
            self.changed()
        g = self.selected()
        if g is not None:
            psim.SameLine()
            if psim.Button("Delete##grp"):
                d.groups.pop(self.sel)
                self.sel = min(self.sel, len(d.groups) - 1)
                self.changed()
        rows = [f"{x['name']}  {len(x['faces'])} faces, {len(x['verts'])} verts"
                f"{', lock ' + str(x['lock']) if x['lock'] else ''}{', ' + str(len(x['rules'])) + ' rules' if x['rules'] else ''}"
                f"{'' if x['enabled'] else '  (off)'}" for x in d.groups]
        new = list_selector(rows, self.sel, "grp")
        if new != self.sel:
            self.sel = new
            self.overlay_dirty = True
        if not d.groups:
            psim.TextDisabled("Tag input-mesh faces / vertices, then give them weight rules or lock them.")
            return
        g = self.selected()
        if g is None:
            return
        psim.Separator()
        _, g["name"] = psim.InputText("name##grp", g["name"])
        psim.SameLine()
        ch, g["enabled"] = psim.Checkbox("on##grp", g["enabled"])
        if ch:
            self.changed()

        psim.Text("Click to tag:")
        for mode, label in (("off", "off"), ("faces", "faces"), ("verts", "vertices")):
            psim.SameLine()
            if toggle_button(f"{label}##pick", self.pick == mode):
                self.set_pick(mode)
        psim.SetItemTooltip("Shows the input mesh; click faces / vertices in the viewport to add or remove them.")
        self.select_tools(g)

        ch, g["lock"] = psim.SliderInt("lock iterations##grp", int(g["lock"]), 0, MAX_ITERATIONS)
        psim.SetItemTooltip("Group vertices keep their position for this many iterations (paper Fig. 9, L_e):\n"
                            "locked edges become sharp creases, locked points spikes.")
        if ch:
            self.changed()
        psim.SameLine()
        ch, g["lock_faces"] = psim.Checkbox("+ face corners##grp", g["lock_faces"])
        psim.SetItemTooltip("Also lock every corner of the group's faces (keeps whole regions flat).")
        if ch:
            self.changed()
        psim.TextDisabled("Weight rules for the group's faces:")
        if rules_editor(g["rules"], "grpr"):
            self.changed()

    def select_tools(self, g):
        m = self.app.base_mesh
        if m is None or not psim.TreeNode("Select by rule##grp"):
            return
        target = g["verts"] if self.pick == "verts" else g["faces"]
        what = "vertices" if self.pick == "verts" else "faces"
        psim.TextDisabled(f"Adds to the group's {what} (switch 'Click to tag' to choose).")
        dirs = list(I.AXIS_VECTORS)
        _, self.normal_dir = psim.Combo("facing##sel", self.normal_dir, dirs)
        _, self.normal_angle = psim.SliderFloat("within deg##sel", self.normal_angle, 1.0, 90.0, "%.0f")
        if psim.Button("Add facing##sel"):
            faces = I.select_by_normal(m, dirs[self.normal_dir], self.normal_angle)
            self._add(g, faces, from_faces=True)
        _, self.band_axis = psim.Combo("band axis##sel", self.band_axis, ["x", "y", "z"])
        _, band = psim.SliderFloat2("band (0-1)##sel", self.band, 0.0, 1.0, "%.2f")
        self.band = [min(band), max(band)]
        if psim.Button("Add band##sel"):
            sel = I.select_by_height(m, self.band_axis, *self.band, faces=self.pick != "verts")
            self._add(g, sel, from_faces=self.pick != "verts")
        _, self.kth = psim.SliderInt("every k-th##sel", self.kth, 2, 12)
        psim.SameLine()
        if psim.Button("Add##kth"):
            n = m.n_verts if self.pick == "verts" else m.n_faces
            self._add(g, I.select_every_kth(n, self.kth), from_faces=self.pick != "verts")
        labels = list(I.motif_counts(m))
        if labels:
            self.motif_sel = min(self.motif_sel, len(labels) - 1)
            _, self.motif_sel = psim.Combo("motif##sel", self.motif_sel, labels)
            psim.SameLine()
            if psim.Button("Add verts##motif"):
                g["verts"] = sorted(set(g["verts"]) | set(I.select_by_motif(m, labels[self.motif_sel]).tolist()))
                self.changed()
        if psim.Button("Clear##sel"):
            target.clear()
            self.changed()
        psim.SameLine()
        if psim.Button("Invert##sel"):
            n = m.n_verts if self.pick == "verts" else m.n_faces
            inv = sorted(set(range(n)) - set(target))
            target.clear()
            target.extend(inv)
            self.changed()
        psim.TreePop()

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

    # ---------------------------------------------------------------- motifs
    def motifs_ui(self):
        d = self.design
        m = self.app.base_mesh
        psim.TextWrapped("Motifs: vertices classed by incident faces / edges. Each motif's U attracts (+) or "
                         "deflects (-) nearby face / edge points via w6 / w7 in the iteration schedule (eq. 10-11).")
        counts = I.motif_counts(m) if m is not None else {}
        if self.app.result is not None and self.app.result.depth_reached > 0:
            for k, v in I.motif_counts(self.app.result.mesh).items():
                counts.setdefault(k, 0)
        for label in sorted(set(counts) | set(d.motifs), key=lambda s: (I.parse_label(s) // 100, I.parse_label(s) % 100)):
            n = counts.get(label, 0)
            ch, v = psim.SliderFloat(f"U {label}  ({n} on input)##motif", float(d.motifs.get(label, 0.0)), -1.0, 1.0, "%.2f")
            if ch:
                d.motifs[label] = v
                self.changed()
        if d.motifs and not any(it.weights.get("w6") or it.weights.get("w7") for it in d.iterations[: d.full_depth]):
            psim.TextColored(WARN, "Set w6 / w7 in some iteration for the motifs to act.")

    # -------------------------------------------------------------- measures
    def measures_ui(self):
        d = self.design
        psim.TextWrapped("Measure rules: a per-face measure t in [0, 1] sets / scales / offsets a weight "
                         "from 'at 0' to 'at 1' (paper: two sub-values interpolated by distance or curvature).")
        keys = list(I.MEASURES)
        remove = None
        for i, r in enumerate(d.intrinsic):
            psim.PushID(f"rule{i}")
            ch, r["enabled"] = psim.Checkbox("##on", r["enabled"])
            psim.SameLine()
            psim.PushItemWidth(190)
            c1, k = psim.Combo("##measure", keys.index(r["measure"]), [I.MEASURES[x] for x in keys])
            psim.PopItemWidth()
            if c1:
                r["measure"] = keys[k]
            psim.SameLine()
            if psim.Button("x"):
                remove = i
            psim.PushItemWidth(110)
            c2, r["weight"] = weight_combo("##w", r["weight"])
            psim.SameLine()
            c3, op = psim.Combo("##op", I.RULE_OPS.index(r["op"]), list(I.RULE_OPS))
            psim.PopItemWidth()
            if c3:
                r["op"] = I.RULE_OPS[op]
            c4, r["a"] = psim.SliderFloat("at 0##a", r["a"], -2.0, 2.0, "%.2f")
            c5, r["b"] = psim.SliderFloat("at 1##b", r["b"], -2.0, 2.0, "%.2f")
            c6, r["gamma"] = psim.SliderFloat("gamma##g", r["gamma"], 0.2, 5.0, "%.2f")
            c7 = iteration_range("iterations##r", r)
            if psim.Button("show measure"):
                self.app.color_mode = f"measure:{r['measure']}"
                self.app.refresh_display()
            psim.PopID()
            psim.Separator()
            if ch or c1 or c2 or c3 or c4 or c5 or c6 or c7:
                self.changed()
        if remove is not None:
            d.intrinsic.pop(remove)
            self.changed()
        if psim.Button("+ measure rule"):
            d.intrinsic.append(I.normalize_rule({}))
            self.changed()

    # ----------------------------------------------------------------- merge
    def merge_ui(self):
        mg = self.design.merge
        psim.TextWrapped("Welds vertices of different parts of the surface that grow into contact. Where a "
                         "welded vertex would exceed the max valence, faces are not formed -> porosity.")
        ch0, mg["enabled"] = psim.Checkbox("enabled##merge", mg["enabled"])
        ch1, mg["distance"] = psim.SliderFloat("distance##merge", float(mg["distance"]), 0.01, 1.0, "%.3f")
        psim.SetItemTooltip("x local edge length (relative) or model units (absolute).")
        psim.SameLine()
        ch2, mg["relative"] = psim.Checkbox("relative##merge", mg["relative"])
        ch3, mg["max_valence"] = psim.SliderInt("max valence##merge", int(mg["max_valence"]), 0, 12)
        psim.SetItemTooltip("0 = unlimited (welds only). 4-6 opens holes where sheets meet.")
        ch4 = iteration_range("iterations##merge", mg)
        if ch0 or ch1 or ch2 or ch3 or ch4:
            self.changed()
        r = self.app.result
        if r is not None and mg["enabled"]:
            info = r.mesh.info.get("merge")
            if info:
                note = "  (skipped: would destroy the form)" if info.get("skipped") else ""
                psim.Text(f"last step: {info['merged']} welded, {info['dropped']} faces dropped{note}")
            m = r.mesh
            psim.Text(f"Euler characteristic {m.euler_characteristic()}, {int(m.edge_is_boundary.sum())} hole edges")

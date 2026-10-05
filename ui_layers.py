"""Function layers panel for app.py (M4): fields driving weights or displacement, and folds."""

from __future__ import annotations

import copy
import os

import polyscope.imgui as psim

from hansmeyer import functions as F
from hansmeyer.layers import BLENDS, normalize
from ui_common import WARN, iteration_range, list_selector, params_editor, toggle_button, weight_combo

TARGET_HELP = {
    "weight": "Drives one weight per face before an iteration (pattern -> where folds/spikes grow).",
    "displacement": "Moves vertices along their normals after an iteration (adds relief directly).",
    "fold": "Deforms the whole mesh with a fold map after an iteration (Mandelbox-style space folding).",
}
BLEND_HELP = ("add: w + v    multiply: w x (1 + v)    replace: v    min / max: min(w, v) / max(w, v)\n"
              "where v = amplitude x f(p) + offset, faded by the mask.")


class LayerPanel:
    def __init__(self, app):
        self.app = app
        self.sel = 0
        self.plugin_dir = os.path.join(app.root, "functions")
        self.reload_plugins(initial=True)

    @property
    def layers(self):
        return self.app.design.layers

    def selected(self):
        return self.layers[self.sel] if 0 <= self.sel < len(self.layers) else None

    def reset(self):
        self.sel = 0 if self.layers else -1

    def changed(self):
        self.app.edited()
        if self.app.color_mode == "layer":
            self.app.refresh_display()

    def reload_plugins(self, initial=False):
        errors = F.load_plugins(self.plugin_dir)
        if not initial:
            self.app.pipe.clear()
            self.app.edited()
            self.app.status = f"Plug-ins reloaded ({len([f for f in F.REGISTRY.values() if f.source])} functions)"
        return errors

    # ------------------------------------------------------------------ ui
    def ui(self):
        if not psim.CollapsingHeader("Function layers (M4)"):
            return
        psim.TextDisabled("Our extension: mathematical fields and folds on top of Hansmeyer's process.")
        for target, label in (("weight", "+ Weight field"), ("displacement", "+ Displacement"), ("fold", "+ Fold")):
            if psim.Button(label):
                self.add(target)
            psim.SetItemTooltip(TARGET_HELP[target])
            psim.SameLine()
        psim.NewLine()
        a = self.selected()
        if a is not None:
            if psim.Button("Duplicate##ly"):
                dup = copy.deepcopy(a)
                dup["name"] += "'"
                self.layers.insert(self.sel + 1, dup)
                self.sel += 1
                self.changed()
            psim.SameLine()
            if psim.Button("Delete##ly"):
                self.layers.pop(self.sel)
                self.sel = min(self.sel, len(self.layers) - 1)
                self.changed()
            psim.SameLine()
            if psim.Button("Up##ly") and self.sel > 0:
                self.layers[self.sel - 1], self.layers[self.sel] = self.layers[self.sel], self.layers[self.sel - 1]
                self.sel -= 1
                self.changed()
            psim.SameLine()
            if psim.Button("Down##ly") and self.sel < len(self.layers) - 1:
                self.layers[self.sel + 1], self.layers[self.sel] = self.layers[self.sel], self.layers[self.sel + 1]
                self.sel += 1
                self.changed()
            psim.SameLine()
        if psim.Button("Reload plug-ins"):
            self.reload_plugins()
        psim.SetItemTooltip("Re-imports functions/*.py (see functions/README.md).")
        for err in F.PLUGIN_ERRORS:
            psim.TextColored(WARN, err)

        rows = []
        for ly in self.layers:
            what = ly["weight"] if ly["target"] == "weight" else ly["target"]
            fd = F.REGISTRY.get(ly["function"])
            rows.append(f"{ly['name']}  {fd.label if fd else ly['function'] + ' (missing!)'} -> {what}"
                        f"  [{ly['from']}-{ly['to']}]{'' if ly['enabled'] else '  (off)'}")
        new = list_selector(rows, self.sel, "ly")
        if new != self.sel:
            self.sel = new
            self.app.refresh_display()
        if not self.layers:
            psim.TextDisabled("No layers.")
            return
        a = self.selected()
        if a is not None:
            psim.Separator()
            self.editor(a)

    def add(self, target):
        names = {ly["name"] for ly in self.layers}
        n = len(self.layers) + 1
        while f"L{n}" in names:
            n += 1
        defaults = {"weight": {"function": "gyroid", "amplitude": 0.3},
                    "displacement": {"function": "worley", "amplitude": 0.15, "params": {"mode": 1}},
                    "fold": {"function": "twist" if "twist" in F.REGISTRY else "kaleido", "amplitude": 0.3,
                             "from": 1, "to": 2}}[target]
        self.layers.append(normalize({"name": f"L{n}", "target": target, **defaults}))
        self.sel = len(self.layers) - 1
        self.changed()

    def editor(self, ly):
        _, ly["name"] = psim.InputText("name##ly", ly["name"])
        psim.SameLine()
        ch, ly["enabled"] = psim.Checkbox("on##ly", ly["enabled"])
        if ch:
            self.changed()
        psim.Text(f"Target: {ly['target']}")
        psim.SetItemTooltip(TARGET_HELP[ly["target"]])

        kind = "fold" if ly["target"] == "fold" else "field"
        funcs = sorted((f for f in F.REGISTRY.values() if f.kind == kind), key=lambda f: (f.family, f.label))
        names = [f.name for f in funcs]
        cur = names.index(ly["function"]) if ly["function"] in names else 0
        ch, idx = psim.Combo("function##ly", cur, [f"{f.family}: {f.label}" for f in funcs])
        if ch:
            ly["function"] = names[idx]
            ly["params"] = {}
            self.changed()
        fd = F.REGISTRY.get(ly["function"])
        if fd is None:
            psim.TextColored(WARN, f"Function '{ly['function']}' not found (plug-in missing?)")
            return
        if fd.help:
            psim.TextWrapped(fd.help)
        if params_editor(fd, ly["params"], "lyp"):
            self.changed()

        if ly["target"] == "weight":
            psim.PushItemWidth(150)
            ch, ly["weight"] = weight_combo("weight##ly", ly["weight"])
            psim.PopItemWidth()
            if ch:
                self.changed()
            ch, b = psim.Combo("blend##ly", BLENDS.index(ly["blend"]), list(BLENDS))
            psim.SetItemTooltip(BLEND_HELP)
            if ch:
                ly["blend"] = BLENDS[b]
                self.changed()
        label = "amount" if ly["target"] == "fold" else "amplitude"
        ch, ly["amplitude"] = psim.SliderFloat(f"{label}##ly", ly["amplitude"], -2.0 if label != "amount" else 0.0,
                                               2.0 if label != "amount" else 1.0, "%.3f")
        psim.SetItemTooltip("fold: 0 = no change, 1 = fully folded" if label == "amount"
                            else "v = amplitude x f(p) + offset. Displacement is in local edge lengths "
                                 "(relative extrusion) or model units (absolute).")
        if ch:
            self.changed()
        if ly["target"] != "fold":
            ch, ly["offset"] = psim.SliderFloat("offset##ly", ly["offset"], -2.0, 2.0, "%.3f")
            if ch:
                self.changed()
        if iteration_range("iterations##ly", ly):
            self.changed()

        if ly["target"] != "fold":
            psim.Text("Evaluate at:")
            psim.SameLine()
            for space, label in (("rest", "original"), ("current", "current")):
                if toggle_button(f"{label}##lysp", ly["space"] == space):
                    ly["space"] = space
                    self.changed()
                psim.SameLine()
            psim.NewLine()
            psim.SetItemTooltip("original: the pattern sticks to the input mesh like a texture.\n"
                                "current: the pattern is fixed in space and the growing form moves through it.")
        names = [""] + [a["name"] for a in self.app.design.attractors]
        cur = names.index(ly["mask"]) if ly["mask"] in names else 0
        ch, idx = psim.Combo("mask##ly", cur, ["(everywhere)"] + names[1:])
        psim.SetItemTooltip("Limit the layer to an attractor's reach (its normalised influence).")
        if ch:
            ly["mask"] = names[idx]
            self.changed()
        if ly["target"] != "fold" and psim.TreeNode("Domain fold (fractal patterns)##ly"):
            self.domain_ui(ly)
            psim.TreePop()

    def domain_ui(self, ly):
        dom = ly["domain"]
        folds = [""] + sorted(f.name for f in F.folds())
        cur = folds.index(dom["fold"]) if dom["fold"] in folds else 0
        ch, idx = psim.Combo("fold##dom", cur, ["(none)"] + [F.REGISTRY[n].label for n in folds[1:]])
        psim.SetItemTooltip("Folds the field's input space before evaluating it: p <- scale x fold(p) + c x p0,\n"
                            "repeated. Mandelbox step x 3-5 with c = 1 gives fractal ornament.")
        if ch:
            dom["fold"] = folds[idx] or "none"
            dom["params"] = {}
            self.changed()
        if dom["fold"] in ("none", ""):
            return
        ch1, dom["repeat"] = psim.SliderInt("repeat##dom", int(dom["repeat"]), 1, 8)
        ch2, dom["scale"] = psim.SliderFloat("scale##dom", float(dom["scale"]), -3.0, 3.0, "%.2f")
        ch3, dom["c"] = psim.SliderFloat("c (add p0)##dom", float(dom["c"]), 0.0, 1.0, "%.2f")
        fd = F.REGISTRY.get(dom["fold"])
        ch4 = params_editor(fd, dom["params"], "domp") if fd else False
        if ch1 or ch2 or ch3 or ch4:
            self.changed()

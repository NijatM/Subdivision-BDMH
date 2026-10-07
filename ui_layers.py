"""Function layers panel for app.py: fields driving weights or displacement, and folds."""

from __future__ import annotations

import copy
import os

import polyscope.imgui as psim

import ui_style as ui
from hansmeyer import functions as F
from hansmeyer.layers import BLENDS, FACING, normalize
from ui_common import iteration_range, params_editor, weight_combo

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
            self.app.status = f"plug-ins reloaded ({len([f for f in F.REGISTRY.values() if f.source])} functions)"
        return errors

    # ------------------------------------------------------------------ ui
    def ui(self):
        ui.explain("Our extension of Hansmeyer's process: a field f(p) (gyroid, noise, Worley cells, ...) is "
                   "evaluated on every face and blended into one weight, or moves the surface along its normals "
                   "after a step; a fold deforms the whole mesh. Plug-ins in functions/ add your own.")
        for target, label in (("weight", "+ weight field"), ("displacement", "+ displacement"), ("fold", "+ fold")):
            if ui.button(label, f"lyadd{target}", TARGET_HELP[target]):
                self.add(target)
            psim.SameLine(0.0, 6.0)
        psim.NewLine()

        ui.subhead("stack")
        ui.begin_list()
        for i, ly in enumerate(self.layers):
            what = ly["weight"] if ly["target"] == "weight" else ly["target"]
            fd = F.REGISTRY.get(ly["function"])
            label = f"{ly['name']}  {fd.label if fd else ly['function'] + ' (missing)'} → {what}"
            if ui.list_row(f"ly{i}", label, f"{ly['from']}-{ly['to']}", selected=i == self.sel,
                           faint=not ly["enabled"], help="Layers apply top to bottom."):
                self.sel = i
                self.app.refresh_display()
        ui.end_list()
        if not self.layers:
            ui.empty("No layers: the schedule alone shapes the form.")
        a = self.selected()
        if a is not None:
            if ui.button("↑", "lyup", "Move up (applies earlier).", enabled=self.sel > 0):
                self.layers[self.sel - 1], self.layers[self.sel] = self.layers[self.sel], self.layers[self.sel - 1]
                self.sel -= 1
                self.changed()
            psim.SameLine(0.0, 4.0)
            if ui.button("↓", "lydown", "Move down (applies later).", enabled=self.sel < len(self.layers) - 1):
                self.layers[self.sel + 1], self.layers[self.sel] = self.layers[self.sel], self.layers[self.sel + 1]
                self.sel += 1
                self.changed()
            psim.SameLine(0.0, 4.0)
            if ui.button("duplicate", "lydup"):
                dup = copy.deepcopy(a)
                dup["name"] += "'"
                self.layers.insert(self.sel + 1, dup)
                self.sel += 1
                self.changed()
            psim.SameLine(0.0, 4.0)
            if ui.button("delete", "lydel"):
                self.layers.pop(self.sel)
                self.sel = min(self.sel, len(self.layers) - 1)
                self.changed()
                a = self.selected()
        if a is not None:
            self.editor(a)

        ui.subhead("plug-ins")
        if ui.button("reload functions/", "lyreload", "Re-imports functions/*.py (see functions/README.md)."):
            self.reload_plugins()
        for err in F.PLUGIN_ERRORS:
            ui.warn(err)

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
        ui.subhead(f"edit {ly['name']} · {ly['target']}")
        _, ly["name"] = ui.input_text("name", ly["name"], key="lyname")
        ch, ly["enabled"] = ui.check("on", ly["enabled"], "lyon", "Switch it off without deleting it.")
        if ch:
            self.changed()
        psim.SameLine(0.0, 16.0)
        changed, on = ui.check("show field", self.app.color_mode == "layer", "lyshow",
                               "Colour the form by this layer's field.")
        if changed:
            self.app.set_color_mode("layer" if on else "none")
        ui.explain(TARGET_HELP[ly["target"]])

        kind = "fold" if ly["target"] == "fold" else "field"
        funcs = sorted((f for f in F.REGISTRY.values() if f.kind == kind), key=lambda f: (f.family, f.label))
        names = [f.name for f in funcs]
        cur = names.index(ly["function"]) if ly["function"] in names else 0
        ch, idx = ui.combo("function", cur, [f"{f.family}: {f.label}" for f in funcs], key="lyfunc")
        if ch:
            ly["function"] = names[idx]
            ly["params"] = {}
            self.changed()
        fd = F.REGISTRY.get(ly["function"])
        if fd is None:
            ui.warn(f"Function '{ly['function']}' not found (plug-in missing?)")
            return
        if fd.help:
            ui.push_font("small")
            ui.note(fd.help)
            ui.pop_font()
        if params_editor(fd, ly["params"], "lyp"):
            self.changed()

        if ly["target"] == "weight":
            ch, ly["weight"] = weight_combo("weight", ly["weight"], "lyweight", "The weight this field drives.")
            if ch:
                self.changed()
            ch, b = ui.combo("blend", BLENDS.index(ly["blend"]), list(BLENDS), BLEND_HELP, "lyblend")
            if ch:
                ly["blend"] = BLENDS[b]
                self.changed()
        fold = ly["target"] == "fold"
        ch, ly["amplitude"] = ui.slider("amount" if fold else "amplitude", ly["amplitude"], 0.0 if fold else -2.0,
                                        1.0 if fold else 2.0, "%.3f",
                                        "0 = no change, 1 = fully folded." if fold else
                                        "v = amplitude x f(p) + offset. Displacement is in local edge lengths "
                                        "(relative extrusion) or model units (absolute).", "lyamp")
        if ch:
            self.changed()
        if not fold:
            ch, ly["offset"] = ui.slider("offset", ly["offset"], -2.0, 2.0, "%.3f", key="lyoff")
            if ch:
                self.changed()
        if iteration_range("iterations", ly, "lyrange"):
            self.changed()

        if not fold:
            ch, sp = ui.choice("evaluate at", [("rest", "original"), ("current", "current")], ly["space"], "lysp",
                               helps=["The pattern sticks to the input mesh like a texture.",
                                      "The pattern is fixed in space and the growing form moves through it."])
            if ch:
                ly["space"] = sp
                self.changed()
        names = [""] + [a["name"] for a in self.app.design.attractors] + list(FACING)
        cur = names.index(ly["mask"]) if ly["mask"] in names else 0
        ch, idx = ui.combo("mask", cur, ["everywhere"] + names[1:],
                           "Limit the layer to an attractor's reach (its normalised influence), or to surfaces "
                           "facing a direction: 'facing +y' = detail on top, plain undersides (easier to print).",
                           "lymask")
        if ch:
            ly["mask"] = names[idx]
            self.changed()
        if not fold:
            ui.subhead("domain fold")
            ui.explain("Folds the field's input space before evaluating it: p <- scale x fold(p) + c x p0, repeated. "
                       "A Mandelbox step x 3-5 with c = 1 gives fractal ornament.")
            self.domain_ui(ly)

    def domain_ui(self, ly):
        dom = ly["domain"]
        folds = [""] + sorted(f.name for f in F.folds())
        cur = folds.index(dom["fold"]) if dom["fold"] in folds else 0
        ch, idx = ui.combo("fold", cur, ["none"] + [F.REGISTRY[n].label for n in folds[1:]],
                           "Fractal patterns from simple fields: fold the space the field is evaluated in.", "domfold")
        if ch:
            dom["fold"] = folds[idx] or "none"
            dom["params"] = {}
            self.changed()
        if dom["fold"] in ("none", ""):
            return
        ch1, dom["repeat"] = ui.slider_int("repeat", int(dom["repeat"]), 1, 8, key="domrep")
        ch2, dom["scale"] = ui.slider("scale", float(dom["scale"]), -3.0, 3.0, "%.2f", key="domscale")
        ch3, dom["c"] = ui.slider("c (add p0)", float(dom["c"]), 0.0, 1.0, "%.2f", key="domc")
        fd = F.REGISTRY.get(dom["fold"])
        ch4 = params_editor(fd, dom["params"], "domp") if fd else False
        if ch1 or ch2 or ch3 or ch4:
            self.changed()

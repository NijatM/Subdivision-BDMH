"""SubdivisionEngine_BDMH: an interactive explorer for Hansmeyer-style generative subdivision.

Run:  python app.py            (or double-click run_mac.command / run_windows.bat)

Layout (the look of nijatmahamaliyev.com): the sidebar on the left lists the sections in three groups
(FORM, GROWTH, MAKE); the inspector on the right holds the selected section's controls; the bar over the
3D view has the views and display tools; the bar under it has undo, depth and bake. Tab hides the panels,
H shows the longer help texts, B bakes, Ctrl/Cmd+Z undoes.
"""

from __future__ import annotations

import glob
import json
import os
import time
import traceback
from collections import deque

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import ui_style as ui
from hansmeyer import CC_WEIGHTS, DS_WEIGHTS, MAX_ITERATIONS, SHAPES, Design, IterationSpec, Pipeline, default_spec
from hansmeyer import functions, intrinsic, vessel
from hansmeyer.attractors import face_positions
from hansmeyer.meshio import atomic_write, export
from hansmeyer.view import apply_scene_theme, set_view, setup_scene, show_mesh, write_png
from ui_attractors import AttractorPanel
from ui_intrinsic import IntrinsicPanel
from ui_layers import LayerPanel
from ui_print import PrintPanel
from ui_section import SectionTool
from ui_vessel import VesselPanel

ROOT = os.path.dirname(os.path.abspath(__file__))
PRESET_DIR = os.path.join(ROOT, "presets")
INPUT_DIR = os.path.join(ROOT, "inputs")
EXPORT_DIR = os.path.join(ROOT, "exports")
RENDER_DIR = os.path.join(ROOT, "renders")
SETTINGS_PATH = os.path.join(ROOT, ".app_settings.json")
ERROR_LOG = os.path.join(ROOT, "app_errors.log")
LAST_SESSION = os.path.join(PRESET_DIR, "_last_session.json")

TITLE = "SubdivisionEngine_BDMH"
DEFAULT_PRESET = "cube_six_arms"
AUTO_BAKE_DELAY = 1.5  # seconds of no edits before auto-bake
SHARP = {"w1": -1.0, "w2": -2.0}
EXPORT_FORMATS = ["obj", "stl", "ply"]
NAV_W, INSPECTOR_W, TOOLBAR_H, STATUS_H = 282.0, 410.0, 76.0, 92.0
DEFAULT_WINDOW = (1480, 880)
FLAGS = (psim.ImGuiWindowFlags_NoDecoration | psim.ImGuiWindowFlags_NoMove | psim.ImGuiWindowFlags_NoSavedSettings
         | psim.ImGuiWindowFlags_NoBringToFrontOnFocus | psim.ImGuiWindowFlags_NoFocusOnAppearing)

# (group, name in the sidebar, title in the inspector, one line on what it does)
SECTIONS = [
    ("FORM", "presets", "Presets", "Start from a recipe. Every preset is a JSON file in presets/."),
    ("FORM", "base mesh", "Base mesh", "The coarse input the subdivision starts from."),
    ("FORM", "schedule", "Iteration schedule", "The weights of every subdivision step: where the form extrudes, "
                                                "bulges and creases."),
    ("GROWTH", "attractors", "Attractors", "Points and curves that change the weights near them."),
    ("GROWTH", "layers", "Function layers", "Mathematical fields and folds that drive weights or move the surface."),
    ("GROWTH", "groups", "Groups", "Tag faces or vertices of the input mesh, then lock them or give them rules."),
    ("GROWTH", "intrinsic", "Intrinsic rules", "Weights read from the mesh itself: vertex motifs and measures."),
    ("GROWTH", "porosity", "Porosity", "Weld parts of the surface that grow into contact, opening holes."),
    ("MAKE", "vessel", "Vessel", "A lithophane: smooth sphere outside, the relief inside, glowing where thin."),
    ("MAKE", "print", "Print", "A watertight STL in mm for FDM: size, orientation, cuts into parts, support."),
    ("MAKE", "export", "Export & render", "Mesh files, screenshots and turntable animations."),
]
(PRESETS, BASE, SCHEDULE, ATTRACTORS, LAYERS, GROUPS, INTRINSIC, POROSITY, VESSEL, PRINT, EXPORT) = range(11)

FAMILIES = [("cube", "CUBE"), ("column", "COLUMN"), ("panel", "PANEL"), ("solid", "SOLID"), ("cage", "CAGE"),
            ("vessel", "VESSEL")]
VIEWS = [("diagonal", "diag"), ("front", "front"), ("top", "top"), ("three_quarter", "3/4")]
# short labels for the weight sliders (the full description is in each slider's tooltip)
WEIGHT_SHORT = {"w_f": "w_f  face", "w_e": "w_e  edge", "w_c": "w_c  corner", "w1": "w1   edge bias",
                "w2": "w2   corner bias", "w3": "w3   V/F bias", "w4": "w4   diag bias", "w6": "w6   motif face",
                "w7": "w7   motif edge"}
COLOUR_MODES = [("none", "plain"), ("influence", "attractor influence"), ("layer", "selected layer field"),
                ("tags", "group tags"), ("light", "light through wall")] + \
    [(f"measure:{k}", v) for k, v in intrinsic.MEASURES.items()]


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    except OSError:
        pass  # a read-only folder just means choices aren't remembered


def preset_family(name: str) -> str:
    """'cube_six_arms' -> 'CUBE'; names without a known family prefix are the user's own ('SAVED')."""
    prefix = name.split("_", 1)[0]
    return next((label for key, label in FAMILIES if key == prefix), "SAVED")


class History:
    """Undo / redo over design snapshots (JSON strings)."""

    def __init__(self, limit: int = 120):
        self.stack: list[str] = []
        self.pos = -1
        self.limit = limit

    def push(self, snap: str) -> bool:
        if self.pos >= 0 and self.stack[self.pos] == snap:
            return False
        del self.stack[self.pos + 1:]
        self.stack.append(snap)
        if len(self.stack) > self.limit:
            del self.stack[0]
        self.pos = len(self.stack) - 1
        return True

    def can_undo(self) -> bool:
        return self.pos > 0

    def can_redo(self) -> bool:
        return self.pos < len(self.stack) - 1

    def undo(self) -> str | None:
        if not self.can_undo():
            return None
        self.pos -= 1
        return self.stack[self.pos]

    def redo(self) -> str | None:
        if not self.can_redo():
            return None
        self.pos += 1
        return self.stack[self.pos]


class App:
    def __init__(self):
        self.root = ROOT
        self.settings = load_settings()
        self.theme = self.settings.get("theme", "dark")
        ui.set_palette(self.theme)
        ui.set_help(self.settings.get("help", False))
        self.section = int(self.settings.get("section", PRESETS)) % len(SECTIONS)
        self._scroll_reset = True
        self.panels_hidden = False
        self.polyscope_ui = False
        self.pipe = Pipeline(root=ROOT)
        self.presets = self._list_presets()
        self.auto_bake = True
        self.show_edges = False
        self.export_fmt = 0
        self.last_edit = time.perf_counter()
        self.baked = False
        self.dirty = True
        self.need_view_reset = True
        self.log: deque = deque(maxlen=40)
        self._error = ""
        self.result = None
        self.base_mesh = None
        self.obj_files = self._list_inputs()
        self.schedule_tab = 0
        self.history = History()
        self._snap_due = True
        self._snap_checked = 0.0
        self._window_seen = None
        self.preset_name, self.loaded_snap, self.modified = None, None, False
        self.last_session = self._read_last_session()  # the previous run's final design, read before any autosave
        self.design = Design()
        self.color_mode = "none"  # none | influence | layer | light | measure:<name> | tags
        self.attr_panel = AttractorPanel(self)
        self.layer_panel = LayerPanel(self)  # also loads functions/ plug-ins
        self.intr_panel = IntrinsicPanel(self)
        self.print_panel = PrintPanel(self)
        self.vessel_panel = VesselPanel(self)
        self.section_tool = SectionTool(self)
        if self.last_session is not None:
            self.restore_last_session(startup=True)
        else:
            self.load_preset(self._preset_path(DEFAULT_PRESET) and DEFAULT_PRESET
                             or (self.presets[0][0] if self.presets else None))
        self.save_name = self._default_save_name()

    # ------------------------------------------------------------- messages
    @property
    def status(self) -> str:
        return self.log[-1][1] if self.log else ""

    @status.setter
    def status(self, msg: str):
        if msg:
            self.log.append((time.strftime("%H:%M:%S"), msg))

    @property
    def error(self) -> str:
        return self._error

    @error.setter
    def error(self, msg: str):
        if msg and msg != self._error:
            self.log.append((time.strftime("%H:%M:%S"), "! " + msg))
        self._error = msg

    # ------------------------------------------------------------- settings
    def remember(self, key, value):
        """Persist a user choice (theme, printer, ...) in .app_settings.json."""
        self.settings[key] = value
        save_settings(self.settings)

    def set_theme(self, name):
        self.theme = name
        ui.apply_style(name)
        apply_scene_theme(name)
        self.attr_panel.geom_dirty = True
        self.intr_panel.overlay_dirty = True
        self.print_panel.theme_changed()
        self.section_tool.theme_changed()
        self.refresh_display()
        self.remember("theme", name)

    def set_section(self, i: int):
        i %= len(SECTIONS)
        if i == self.section:
            return
        if self.section == GROUPS and self.intr_panel.pick != "off":
            self.intr_panel.set_pick("off")  # clicking the view only tags while the Groups section is open
        self.section = i
        self._scroll_reset = True
        self.remember("section", i)

    # ---------------------------------------------------------------- files
    def _list_presets(self):
        paths = sorted(glob.glob(os.path.join(PRESET_DIR, "*.json")))
        names = [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]
        return [(n, p) for n, p in names if not n.startswith("_")]

    def _list_inputs(self):
        return sorted(os.path.relpath(p, ROOT).replace(os.sep, "/") for p in glob.glob(os.path.join(INPUT_DIR, "*.obj")))

    def _preset_path(self, name):
        return next((p for n, p in self.presets if n == name), None)

    def _default_save_name(self) -> str:
        name = self.preset_name or "my_form"
        return name if name.endswith("_edit") else name + "_edit"

    def _read_last_session(self):
        try:
            with open(LAST_SESSION, encoding="utf-8") as fh:
                data = json.load(fh)
            return data, os.path.getmtime(LAST_SESSION)
        except (OSError, ValueError):
            return None

    def _install(self, design: Design):
        """Make `design` the current one and bring every panel in line with it."""
        view = self.design.view
        self.design = design
        self.obj_path = design.base.get("path", "") if design.base.get("shape") == "obj" else ""
        self.attr_panel.reset()
        self.layer_panel.reset()
        self.intr_panel.reset()
        self.print_panel.show_form()
        self.print_panel.defaults_for_shape_pending = True
        self.schedule_tab = 0
        self.edited()
        return view

    def load_preset(self, name):
        path = self._preset_path(name) if name else None
        if path is None:
            self._install(Design())
            return
        try:
            design = Design.load(path)
        except (OSError, ValueError, TypeError) as e:
            self.error = f"Could not load {name}: {e}"
            return
        self._install(design)
        self.need_view_reset = True
        self.preset_name = name
        self.loaded_snap = self.snapshot()
        self.save_name = self._default_save_name()
        self.remember("preset", name)
        self.status = f"loaded presets/{name}.json"

    def restore_last_session(self, startup: bool = False):
        data, mtime = self.last_session
        try:
            design = Design.from_dict(data)
        except (TypeError, ValueError) as e:
            self.error = f"Could not restore the last session: {e}"
            return
        self._install(design)
        self.need_view_reset = True
        self.preset_name, self.loaded_snap = data.get("_preset"), None
        snap = self.snapshot()
        for name, path in self.presets:  # which preset is this (or was it edited from)?
            try:
                preset_snap = self.snapshot(Design.load(path))
            except (OSError, ValueError, TypeError):
                continue
            if name == self.preset_name or (self.preset_name is None and preset_snap == snap):
                self.preset_name, self.loaded_snap = name, preset_snap
                break
        else:
            self.preset_name = None
        self.save_name = self._default_save_name()
        when = time.strftime("%b %d, %H:%M", time.localtime(mtime))
        self.status = ("reopened" if startup else "restored") + f" the last session (saved {when})"

    def save_preset(self):
        name = "".join(c for c in self.save_name.strip() if c.isalnum() or c in "-_") or "untitled"
        name = name.lstrip("_") or "untitled"
        path = os.path.join(PRESET_DIR, name + ".json")
        if self.design.name in ("", "Untitled"):
            self.design.name = name
        try:
            os.makedirs(PRESET_DIR, exist_ok=True)
            atomic_write(path, self.design.save)
        except OSError as e:
            self.error = f"Could not save preset: {e}"
            return
        self.presets = self._list_presets()
        self.preset_name = name
        self.loaded_snap = self.snapshot()
        self.modified = False
        self.remember("preset", name)
        self.status = f"saved presets/{name}.json"

    # -------------------------------------------------------------- history
    def snapshot(self, design: Design | None = None) -> str:
        d = (design or self.design).to_dict()
        d.pop("view", None)  # camera choices are not design edits
        return json.dumps(d, sort_keys=True)

    def history_tick(self):
        """Record a step once an edit has settled (no slider held, no gizmo dragged)."""
        now = time.perf_counter()
        if not self._snap_due and now - self._snap_checked < 1.0:
            return  # (the periodic check catches edits that don't call edited(), such as renaming)
        if psim.IsAnyItemActive() or psim.IsMouseDown(0):
            return
        self._snap_due, self._snap_checked = False, now
        snap = self.snapshot()
        self.history.push(snap)
        self.modified = self.loaded_snap is not None and snap != self.loaded_snap

    def undo(self, redo: bool = False):
        snap = self.history.redo() if redo else self.history.undo()
        if snap is None:
            return
        view = self._install(Design.from_dict(json.loads(snap)))
        self.design.view = view
        self.modified = self.loaded_snap is not None and snap != self.loaded_snap
        self.status = "redo" if redo else "undo"

    # -------------------------------------------------------------- compute
    def edited(self):
        self.dirty = True
        self.baked = False
        self.last_edit = time.perf_counter()
        self._snap_due = True

    def compute(self, depth):
        try:
            r = self.pipe.run(self.design, depth)
        except Exception as e:  # bad OBJ, degenerate parameters, ...: keep the last good mesh
            self.error = f"{type(e).__name__}: {e}"
            return False
        self.error = ""
        self.result = r
        base = self.pipe.base(self.design)
        if base is not self.base_mesh:
            self.intr_panel.overlay_dirty = True
        self.base_mesh = base
        self.refresh_display()
        if self.need_view_reset:
            set_view(r.mesh, self.design.view)
            self.need_view_reset = False
        if not np.all(np.isfinite(r.mesh.V)):
            self.error = "Non-finite vertices: reduce the weights"
        return True

    def color_values(self, mesh):
        """Per-face values for the 'colour by' mode (or None)."""
        mode = self.color_mode
        try:
            if mode == "influence":
                return self.attr_panel.influence_values(mesh)
            if mode == "layer":
                ly = self.layer_panel.selected()
                if ly is None or ly["target"] == "fold":
                    return None, None
                P = face_positions(mesh, ly["space"])
                return functions.evaluate(ly["function"], P, ly["params"], ly["domain"]), (-1.0, 1.0)
            if mode == "light":
                return (vessel.light(mesh), (0.0, 1.0)) if "wall_mm" in mesh.vattr else (None, None)
            if mode.startswith("measure:"):
                return intrinsic.measure(mesh, mode.split(":", 1)[1]), (0.0, 1.0)
            if mode == "tags" and "tags" in mesh.fattr:
                g = self.intr_panel.sel
                if g < 0:
                    return (mesh.fattr["tags"] != 0).astype(float), (0.0, 1.0)
                return ((mesh.fattr["tags"] >> g) & 1).astype(float), (0.0, 1.0)
        except Exception as e:  # never let a visualisation break the app
            self.error = f"colour: {e}"
        return None, None

    def set_color_mode(self, mode: str):
        self.color_mode = mode
        self.refresh_display()

    def refresh_display(self):
        """(Re)draw the current result, coloured by the active 'colour by' mode."""
        if not self.result:
            return
        values, vrange = self.color_values(self.result.mesh)
        show_mesh(self.result.mesh, edges=self.show_edges, face_values=values, vrange=vrange,
                  label=self.color_mode.replace("measure:", ""), cmap="inferno" if self.color_mode == "light" else "viridis")
        if self.intr_panel.pick != "off" or self.print_panel.showing:
            ps.get_surface_mesh("form").set_enabled(False)

    def tick(self):
        if self.dirty:
            self.compute(self.design.preview_depth)
            self.dirty = False
        elif (
            self.auto_bake
            and not self.baked
            and not self.error
            and self.design.full_depth > self.design.preview_depth
            and time.perf_counter() - self.last_edit > AUTO_BAKE_DELAY
            and not psim.IsMouseDown(0)
        ):
            self.bake()

    def bake(self):
        self.baked = self.compute(self.design.full_depth)
        if self.baked:
            self.autosave()

    def autosave(self):
        """Keep the current design in presets/_last_session.json, so the next start reopens it."""
        try:
            data = self.design.to_dict()
            data["_preset"] = self.preset_name
            text = json.dumps(data, indent=2)
        except (TypeError, ValueError):
            return
        if text == getattr(self, "_autosaved", None):
            return

        def write(tmp):
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)

        try:
            atomic_write(LAST_SESSION, write)
            self._autosaved = text
        except OSError:
            pass  # a full disk only costs the autosave

    # ----------------------------------------------------------------- frame
    def ui(self):
        io = psim.GetIO()
        W, H = io.DisplaySize
        self.tick()
        self.attr_panel.sync(gizmo=self.section == ATTRACTORS and not self.panels_hidden)
        self.intr_panel.sync()
        self.print_panel.poll()
        self.shortcuts()
        if self.panels_hidden:
            self.hidden_hint(H)
        else:
            if not self.polyscope_ui:  # Polyscope pins its own panel top left, so it takes the sidebar's place
                self.nav_window(H)
            self.inspector_window(W, H)
            self.toolbar_window(W)
            self.status_window(W, H)
        self.print_panel.update_planes(self.section == PRINT and not self.panels_hidden)
        self.section_tool.apply()
        self.history_tick()
        self.track_window(W, H)

    def track_window(self, W, H):
        """Remember the window size between sessions (once it has stopped changing)."""
        now = time.perf_counter()
        size = (int(W), int(H))
        if self._window_seen is None or self._window_seen[0] != size:
            self._window_seen = (size, now)
        elif now - self._window_seen[1] > 1.0 and self.settings.get("window") != list(size) and min(size) > 200:
            self.remember("window", list(size))

    def shortcuts(self):
        io = psim.GetIO()
        if io.WantTextInput:
            return

        def pressed(key):
            k = getattr(psim, "ImGuiKey_" + key, None)
            return k is not None and psim.IsKeyPressed(k, False)

        mod = io.KeyCtrl or io.KeySuper  # Cmd on a Mac
        if mod:
            if pressed("Z"):
                self.undo(redo=io.KeyShift)
            elif pressed("Y"):
                self.undo(redo=True)
            elif pressed("S"):
                self.save_preset()
            return
        if pressed("Tab"):
            self.panels_hidden = not self.panels_hidden
        elif pressed("B"):
            self.bake()
        elif pressed("E"):
            self.export_mesh()
        elif pressed("W"):
            self.show_edges = not self.show_edges
            self.refresh_display()
        elif pressed("H"):
            ui.set_help(not ui.help_on())
            self.remember("help", ui.help_on())
        elif pressed("RightBracket"):
            self.set_section(self.section + 1)
        elif pressed("LeftBracket"):
            self.set_section(self.section - 1)
        for i, (view, _) in enumerate(VIEWS):
            if pressed(str(i + 1)):
                self.set_view(view)

    def set_view(self, view):
        self.design.view = view
        if self.print_panel.showing:
            self.print_panel.show_print()  # the print model has its own (z-up) camera
        elif self.result:
            set_view(self.result.mesh, view)

    # ------------------------------------------------------------ the shell
    def _vertical_rule(self, x, H):
        dl = psim.GetWindowDrawList()
        dl.PushClipRectFullScreen()
        dl.AddLine((x, 0.0), (x, H), ui.u32("line"))
        dl.PopClipRect()

    def nav_window(self, H):
        psim.SetNextWindowPos((0.0, 0.0))
        psim.SetNextWindowSize((NAV_W, H))
        psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, (22.0, 28.0))
        psim.PushStyleVar(psim.ImGuiStyleVar_ItemSpacing, (8.0, 1.0))
        psim.Begin("##nav", FLAGS)
        self._vertical_rule(NAV_W - 1.0, H)
        ui.push_font("name")
        ui.text("Nijat Mahamaliyev", "hi")
        ui.pop_font()
        ui.push_font("small")
        ui.gap(3.0)
        ui.wrap("Subdivision as a generative system", "dim")
        ui.text("HTMAA 2026 · week 03", "dim")
        ui.pop_font()
        group = None
        for i, (grp, name, title, intro) in enumerate(SECTIONS):
            if grp != group:
                group = grp
                ui.gap(24.0)
                ui.spaced(grp)
                ui.gap(6.0)
            state, faint = self.section_state(i)
            if ui.list_row(f"nav{i}", name, state, selected=i == self.section, num=f"{i:02d}", faint=faint,
                           help=f"{title}: {intro}\n[ / ] previous / next section"):
                self.set_section(i)
        self.nav_footer(H)
        psim.End()
        psim.PopStyleVar(2)

    def nav_footer(self, H):
        ui.push_font("small")
        y = max(psim.GetCursorPosY() + 24.0, H - 74.0)
        psim.SetCursorPosY(y)
        x0, y0 = psim.GetCursorScreenPos()
        ui.dashed_line((x0, y0), (x0 + NAV_W - 44.0, y0), "line2")
        ui.gap(10.0)
        ui.text("after Hansmeyer (2010)", "faint")
        ui.text("nijatmahamaliyev.com", "faint")
        ui.pop_font()

    def section_state(self, i: int):
        """(text for the sidebar's right column, faint?) for section i."""
        d = self.design
        if i == PRESETS:
            return (self.preset_name or "unsaved") + ("*" if self.modified else ""), False
        if i == BASE:
            return d.base.get("shape", "cube").replace("_", " "), False
        if i == SCHEDULE:
            return f"d{d.preview_depth}/{d.full_depth}", False
        counts = {ATTRACTORS: sum(a["enabled"] for a in d.attractors), LAYERS: sum(ly["enabled"] for ly in d.layers),
                  GROUPS: sum(g["enabled"] for g in d.groups),
                  INTRINSIC: sum(1 for v in d.motifs.values() if v) + sum(r["enabled"] for r in d.intrinsic)}
        if i in counts:
            return (str(counts[i]) if counts[i] else "—"), counts[i] == 0
        if i == POROSITY:
            return ("on", False) if d.merge["enabled"] else ("off", True)
        if i == VESSEL:
            if vessel.active(d):
                return f"{d.vessel['diameter_mm']:.0f} mm", False
            return ("off" if d.base.get("shape") == "sphere_open" else "—"), True
        if i == PRINT:
            pp = self.print_panel
            if pp.busy():
                return f"{time.perf_counter() - pp.job['started']:.0f}s", False
            if pp.result is not None:
                return ("stale" if pp.result_key != pp._design_key() else "ready"), False
            return "", False
        return "", False

    def inspector_window(self, W, H):
        psim.SetNextWindowPos((W - INSPECTOR_W, 0.0))
        psim.SetNextWindowSize((INSPECTOR_W, H))
        psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, (20.0, 26.0))
        psim.Begin("##inspector", FLAGS)
        self._vertical_rule(W - INSPECTOR_W, H)
        if self._scroll_reset:
            psim.SetScrollY(0.0)
            self._scroll_reset = False
        group, _, title, intro = SECTIONS[self.section]
        ui.spaced(f"{self.section:02d} · {group}", "dim")
        ui.gap(2.0)
        ui.push_font("title")
        ui.text(title, "hi")
        ui.pop_font()
        ui.gap(8.0)
        x0, y0 = psim.GetCursorScreenPos()
        ui.dashed_line((x0, y0), (x0 + psim.GetContentRegionAvail()[0], y0), "line2")
        ui.gap(10.0)
        ui.note(intro)
        ui.gap(4.0)
        [self.presets_ui, self.base_ui, self.schedule_ui, self.attr_panel.ui, self.layer_panel.ui,
         self.intr_panel.groups_ui, self.intr_panel.intrinsic_ui, self.intr_panel.merge_ui, self.vessel_panel.ui,
         self.print_panel.ui, self.export_ui][self.section]()
        ui.gap(30.0)
        psim.End()
        psim.PopStyleVar()

    def toolbar_window(self, W):
        x, w = NAV_W, W - NAV_W - INSPECTOR_W
        psim.SetNextWindowPos((x, 0.0))
        psim.SetNextWindowSize((w, TOOLBAR_H))
        psim.SetNextWindowBgAlpha(0.88)
        psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, (18.0, 12.0))
        psim.Begin("##toolbar", FLAGS | psim.ImGuiWindowFlags_NoScrollbar)
        dl = psim.GetWindowDrawList()
        dl.PushClipRectFullScreen()
        dl.AddLine((x, TOOLBAR_H - 1.0), (x + w, TOOLBAR_H - 1.0), ui.u32("line"))
        dl.PopClipRect()
        # line 1: the prompt, and the app-wide toggles on the right
        ui.push_font("small")
        px, py = psim.GetCursorScreenPos()
        ui.text("root@nijat:~$", "hi")
        psim.SameLine(0.0, 7.0)
        ui.text(f"./{TITLE}", "fg")
        cx = psim.GetItemRectMax()[0] + 3.0
        ui.cursor_block(cx, py + 1.0, psim.GetTextLineHeight() - 2.0)
        ui.pop_font()
        right = [("?", ui.help_on(), "help"), ("light" if self.theme == "dark" else "dark", False, "theme"),
                 ("polyscope", self.polyscope_ui, "ps")]
        pad = psim.GetStyle().FramePadding[0]
        widths = [psim.CalcTextSize(lbl)[0] + 2 * pad for lbl, _, _ in right]
        bx = w - 18.0 - sum(widths) - 6.0 * (len(right) - 1)
        helps = {"help": "Show the longer explanations inline (H).",
                 "theme": "Switch between the dark and the light palette.",
                 "ps": "Polyscope's own panel (structures, camera, materials, ground plane). It takes the "
                       "sidebar's place until you switch it off."}
        for i, (lbl, on, key) in enumerate(right):
            psim.SetCursorPos((bx, 8.0))
            bx += widths[i] + 6.0
            if ui.toggle(lbl, on, f"tb{key}", helps[key]):
                if key == "help":
                    ui.set_help(not ui.help_on())
                    self.remember("help", ui.help_on())
                elif key == "theme":
                    self.set_theme("light" if self.theme == "dark" else "dark")
                else:
                    self.polyscope_ui = not self.polyscope_ui
                    ps.set_build_default_gui_panels(self.polyscope_ui)
        # line 2: views and display
        psim.SetCursorPosY(38.0)
        psim.PushStyleVar(psim.ImGuiStyleVar_ItemSpacing, (-1.0, 4.0))  # joined boxes
        for i, (view, label) in enumerate(VIEWS):
            if i:
                psim.SameLine()
            if ui.toggle(label, self.design.view == view, f"view{view}", f"{view.replace('_', ' ')} view ({i + 1})"):
                self.set_view(view)
        psim.PopStyleVar()
        psim.SameLine(0.0, 16.0)
        changed, self.show_edges = ui.check("wire", self.show_edges, "tbwire", "Show the mesh edges (W).")
        if changed:
            self.refresh_display()
        psim.SameLine(0.0, 16.0)
        psim.AlignTextToFramePadding()
        ui.text("colour", "dim")
        psim.SameLine(0.0, 6.0)
        keys = [k for k, _ in COLOUR_MODES]
        cur = keys.index(self.color_mode) if self.color_mode in keys else 0
        changed, idx = ui.combo_box("##colourby", cur, [v for _, v in COLOUR_MODES], 190.0)
        ui.tip("Colour the form by a field or a measure, to see where things act.")
        if changed:
            self.set_color_mode(keys[idx])
        psim.SameLine(0.0, 16.0)
        self.section_tool.toolbar_ui()
        psim.End()
        psim.PopStyleVar()

    def status_window(self, W, H):
        x, w = NAV_W, W - NAV_W - INSPECTOR_W
        psim.SetNextWindowPos((x, H - STATUS_H))
        psim.SetNextWindowSize((w, STATUS_H))
        psim.SetNextWindowBgAlpha(0.88)
        psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, (18.0, 12.0))
        psim.PushStyleVar(psim.ImGuiStyleVar_ItemSpacing, (8.0, 5.0))
        psim.Begin("##status", FLAGS | psim.ImGuiWindowFlags_NoScrollbar)
        dl = psim.GetWindowDrawList()
        dl.PushClipRectFullScreen()
        dl.AddLine((x, H - STATUS_H), (x + w, H - STATUS_H), ui.u32("line"))
        dl.PopClipRect()
        d = self.design
        if ui.button("‹ undo", "undo", "Undo the last design change (Ctrl/Cmd+Z).", enabled=self.history.can_undo()):
            self.undo()
        psim.SameLine(0.0, 4.0)
        if ui.button("redo ›", "redo", "Redo (Ctrl/Cmd+Shift+Z).", enabled=self.history.can_redo()):
            self.undo(redo=True)
        psim.SameLine(0.0, 22.0)
        changed, v = ui.stepper("preview", d.preview_depth, 1, MAX_ITERATIONS, "pd",
                                "Depth recomputed live while you edit.")
        if changed:
            d.preview_depth = v
            d.full_depth = max(d.full_depth, v)
            self.edited()
        psim.SameLine(0.0, 18.0)
        changed, v = ui.stepper("full", d.full_depth, 1, MAX_ITERATIONS, "fd",
                                "Depth of the baked result, used for export and print. Faces x4 per Catmull-Clark step.")
        if changed:
            d.full_depth = v
            d.preview_depth = min(d.preview_depth, v)
            self.edited()
        psim.SameLine(0.0, 18.0)
        _, self.auto_bake = ui.check("auto-bake", self.auto_bake, "autobake",
                                     f"Bake the full depth after {AUTO_BAKE_DELAY:.1f} s without edits.")
        psim.SameLine(0.0, 12.0)
        if ui.button("bake", "bake", "Compute the full depth now (B).", kind="primary"):
            self.bake()
        # line 2: the state of the result; line 3: the newest message
        ui.push_font("small")
        if self.result:
            r = self.result
            state = "baked" if self.baked else "preview"
            size = f" · {d.vessel['diameter_mm']:.0f} mm sphere" if vessel.active(d) else ""
            ui.text(f"{state} · depth {r.depth_reached} · {r.mesh.n_faces:,} faces · {r.seconds * 1000:.0f} ms{size}",
                    "fg")
            if r.capped_by_budget:
                psim.SameLine(0.0, 12.0)
                ui.text(f"capped at depth {r.depth_reached} by the face budget (schedule section)", "warn")
        else:
            ui.text("computing…", "dim")
        if self.error:
            ui.text("! " + ui.truncate(self.error, w - 50.0), "warn")
            ui.tip(self.error)
        elif self.log:
            stamp, msg = self.log[-1]
            ui.text(f"> {ui.truncate(msg, w - 130.0)}", "dim")
            ui.tip("\n".join(f"{t}  {m}" for t, m in list(self.log)[-12:]))
            psim.SameLine()
            ui.text(stamp, "faint")
        ui.pop_font()
        psim.End()
        psim.PopStyleVar(2)

    def hidden_hint(self, H):
        psim.SetNextWindowPos((14.0, H - 34.0))
        psim.SetNextWindowBgAlpha(0.0)
        psim.Begin("##hint", FLAGS | psim.ImGuiWindowFlags_NoInputs | psim.ImGuiWindowFlags_AlwaysAutoResize)
        ui.push_font("small")
        ui.text("tab · show panels", "faint")
        ui.pop_font()
        psim.End()

    # -------------------------------------------------------- 00 presets
    def presets_ui(self):
        d = self.design
        ui.begin_card("loaded")
        ui.spaced("EDITED" if self.modified else ("LOADED" if self.preset_name else "UNSAVED"), "dim")
        ui.gap(2.0)
        ui.push_font("medium")
        ui.text((self.preset_name or d.name or "untitled") + ("*" if self.modified else ""), "hi")
        ui.pop_font()
        if d.description:
            ui.gap(2.0)
            ui.push_font("small")
            ui.wrap(d.description, "dim")
            ui.pop_font()
        ui.end_card()
        if self.last_session is not None:
            when = time.strftime("%b %d, %H:%M", time.localtime(self.last_session[1]))
            if ui.button(f"restore last session · {when}", "restore", "The design as it was when the app last closed "
                         "(it reopens automatically on start). Undo brings back what you have now."):
                self.restore_last_session()

        ui.subhead("save as")
        exists = self._preset_path("".join(c for c in self.save_name.strip() if c.isalnum() or c in "-_")) is not None
        bw = psim.CalcTextSize("overwrite")[0] + 2 * psim.GetStyle().FramePadding[0]
        psim.SetNextItemWidth(psim.GetContentRegionAvail()[0] - bw - 8.0)
        _, self.save_name = psim.InputText("##savename", self.save_name)
        ui.tip("File name in presets/ (letters, digits, - and _). Ctrl/Cmd+S saves.")
        psim.SameLine(0.0, 8.0)
        if ui.button("overwrite" if exists else "save", "savebtn",
                     "Replaces the preset of that name." if exists else "Saves a new preset.",
                     width=bw, kind="normal" if exists else "primary"):
            self.save_preset()

        order = [label for _, label in FAMILIES] + ["SAVED"]
        for fam in order:
            names = [n for n, _ in self.presets if preset_family(n) == fam]
            if not names:
                continue
            ui.gap(14.0)
            ui.spaced(fam)
            ui.gap(4.0)
            ui.begin_list()
            for name in names:
                short = name.split("_", 1)[1] if fam != "SAVED" and "_" in name else name
                if ui.list_row(f"pre{name}", short, "*" if name == self.preset_name and self.modified else "",
                               selected=name == self.preset_name, help=f"presets/{name}.json"):
                    self.load_preset(name)
            ui.end_list()
        ui.gap(10.0)
        ui.push_font("small")
        ui.note("Loading replaces the design; undo (Ctrl/Cmd+Z) brings it back.")
        ui.pop_font()

    # ------------------------------------------------------ 01 base mesh
    def base_ui(self):
        d = self.design
        keys = list(SHAPES)
        cur_name = d.base.get("shape", "cube")
        cur = keys.index(cur_name) if cur_name in keys else 0
        shape = SHAPES[keys[cur]]
        changed, idx = ui.combo("shape", cur, [SHAPES[k].label for k in keys], shape.help, "shape")
        if changed and keys[idx] != cur_name:
            name = keys[idx]
            d.base = default_spec(name)
            if name == "obj":
                d.base["path"] = self.obj_path or (self.obj_files[0] if self.obj_files else "")
                self.obj_path = d.base["path"]
            d.view = SHAPES[name].view
            self.need_view_reset = True
            self.print_panel.defaults_for_shape_pending = True
            self.edited()
        shape = SHAPES[d.base.get("shape", "cube")]
        ui.explain(shape.help)

        for p in shape.params:
            val = d.base.get(p.name, p.default)
            if p.integer:
                changed, v = ui.slider_int(p.label, val, p.lo, p.hi, help=p.help or None, key=f"bp{p.name}")
            else:
                changed, v = ui.slider(p.label, val, p.lo, p.hi, help=p.help or None, key=f"bp{p.name}")
            if changed:
                d.base[p.name] = v
                self.edited()

        if shape.name == "sphere_open":
            self.vessel_size_ui()
        if shape.name == "obj":
            self.obj_ui()

        if self.base_mesh is not None:
            m = self.base_mesh
            is_open = bool(np.any(m.vert_is_boundary))
            ui.value("input", f"{m.n_verts} verts · {m.n_faces} faces · {'open' if is_open else 'closed'}", "dim")
            if is_open:
                self.boundary_ui()

    def vessel_size_ui(self):
        """The sphere's real size: with the vessel on, its walls are in mm, so the size belongs to the design."""
        d = self.design
        ui.subhead("size")
        if not vessel.active(d):
            ui.note("Turn on the vessel (08) to give the sphere a size in mm.")
            if ui.button("open vessel →", "gotovessel"):
                self.set_section(VESSEL)
            return
        D = d.vessel["diameter_mm"]
        changed, D = ui.slider("diameter", D, 30.0, 300.0, "%.0f mm",
                               "The real, printed size of the sphere (Ctrl/Cmd+click to type). The view always fits "
                               "the sphere to the screen; walls stay as set in mm, so a smaller sphere looks "
                               "thicker-walled.", "basediam")
        if changed:
            d.vessel["diameter_mm"] = float(min(max(D, 20.0), 400.0))
            self.edited()
        H = 0.5 * D * (1 + np.cos(np.radians(d.base.get("opening", 40.0))))
        ui.value("print size", f"{D:.0f} x {D:.0f} x {H:.0f} mm", "hi", "Standing on its rim.")

    def obj_ui(self):
        d = self.design
        ui.subhead("obj file")
        if self.obj_files:
            cur = self.obj_files.index(self.obj_path) if self.obj_path in self.obj_files else -1
            changed, idx = ui.combo("inputs/", max(cur, 0), self.obj_files, "Put .obj files in the inputs/ folder.",
                                    "objfiles")
            if changed or cur < 0 and not self.obj_path:
                self.obj_path = self.obj_files[idx]
        else:
            ui.empty("No .obj files in inputs/ yet. Put one there and rescan, or type a path.")
        _, self.obj_path = ui.input_text("path", self.obj_path, "Relative to the project folder, or absolute.", "objpath")
        if ui.button("load obj", "loadobj", kind="primary"):
            d.base["path"] = self.obj_path.strip()
            self.need_view_reset = True
            self.edited()
        psim.SameLine(0.0, 6.0)
        if ui.button("rescan inputs/", "rescan"):
            self.obj_files = self._list_inputs()
        changed, v = ui.check("fit to view", bool(d.base.get("normalize", True)), "objfit",
                              "Scale and centre the mesh to a unit size.")
        if changed:
            d.base["normalize"] = v
            self.edited()
        psim.SameLine(0.0, 16.0)
        changed, v = ui.check("weld", bool(d.base.get("weld", True)), "objweld",
                              "Merge duplicate vertices (needed for meshes exported as separate faces).")
        if changed:
            d.base["weld"] = v
            self.edited()

    def boundary_ui(self):
        d = self.design
        ui.subhead("open boundary")
        changed, v = ui.choice("boundary", [("smooth", "smooth"), ("locked", "locked / tile")], d.boundary, "bnd",
                               helps=["Standard Catmull-Clark rules: the edge relaxes into a B-spline curve.",
                                      "The boundary stays fixed and relief fades out near it, so identical tiles "
                                      "meet seamlessly."])
        if changed:
            d.boundary = v
            self.edited()
        if d.boundary == "locked":
            changed, v = ui.slider("fade rows", d.fade_rows, 0.0, 6.0, "%.1f",
                                   "Extrusion fades in over this many base-mesh rows from the boundary.", "fade")
            if changed:
                d.fade_rows = v
                self.edited()
            if any(it.scheme == "ds" for it in d.iterations[: d.full_depth]):
                ui.warn("Doo-Sabin steps shrink open boundaries: the result will not tile.")

    # ------------------------------------------------------- 02 schedule
    def schedule_ui(self):
        d = self.design
        changed, v = ui.choice("extrusion", [("relative", "relative"), ("absolute", "absolute")], d.extrusion, "ext",
                               "How far a weight of 1 pushes.",
                               ["Displacement = w x local edge length: the same w behaves the same at every depth.",
                                "Paper-literal: displacement = w in model units (unit normals)."])
        if changed:
            d.extrusion = v
            self.edited()
        ui.subhead("iteration")
        self.schedule_tab = min(self.schedule_tab, d.full_depth - 1)
        options = [(k, f"{k + 1}{'ds' if d.iterations[k].scheme == 'ds' else ''}{'*' if k >= d.preview_depth else ''}")
                   for k in range(d.full_depth)]
        _, self.schedule_tab = ui.segmented(options, self.schedule_tab, "itab")
        ui.push_font("small")
        ui.note("* only in the baked (full-depth) result")
        ui.pop_font()
        self.iteration_ui(self.schedule_tab, d.iterations[self.schedule_tab])

        ui.subhead("overview")
        self.overview_ui()
        ui.subhead("limits")
        changed, v = ui.input_int("face budget", self.pipe.face_budget, 100_000, 1_000_000,
                                  "Stop subdividing before the mesh would pass this many faces.", "budget")
        if changed:
            self.pipe.face_budget = max(10_000, v)
            self.edited()

    def iteration_ui(self, k, spec: IterationSpec):
        d = self.design
        changed, v = ui.choice("scheme", [("cc", "Catmull-Clark"), ("ds", "Doo-Sabin")], spec.scheme, f"sch{k}",
                               helps=["Quads: faces x4 per step (paper eq. 1-4).",
                                      "Corner-cutting: new faces from faces, edges and vertices (eq. 5-6)."])
        if changed:
            spec.scheme = v
            self.edited()
        defs = CC_WEIGHTS if spec.scheme == "cc" else DS_WEIGHTS
        for wd in defs:
            label = WEIGHT_SHORT.get(wd.name, " ".join(wd.label.split()))
            changed, v = ui.slider(label, spec.weights[wd.name], wd.lo, wd.hi, "%.2f",
                                   f"{' '.join(wd.label.split())}\n{wd.help}\nCtrl/Cmd+click to type a value.",
                                   f"w{wd.name}")
            if changed:
                spec.weights[wd.name] = float(v)
                self.edited()
        ui.gap(2.0)
        if ui.button("reset", "itreset", "All weights of this iteration to 0 (standard subdivision)."):
            spec.weights = {n: 0.0 for n in spec.weights}
            self.edited()
        if spec.scheme == "cc":
            psim.SameLine(0.0, 6.0)
            if ui.button("sharp", "itsharp", "w1 = -1, w2 = -2: switches off Catmull-Clark smoothing, so "
                                             "extrusions accumulate."):
                spec.weights.update(SHARP)
                self.edited()
        psim.SameLine(0.0, 6.0)
        if ui.button("copy → next", "itnext", "Copy this iteration to the next one.", enabled=k + 1 < MAX_ITERATIONS):
            d.iterations[k + 1] = IterationSpec(spec.scheme, dict(spec.weights))
            self.edited()
        psim.SameLine(0.0, 6.0)
        if ui.button("copy → all", "itall", "Copy this iteration to every other one."):
            for j in range(MAX_ITERATIONS):
                if j != k:
                    d.iterations[j] = IterationSpec(spec.scheme, dict(spec.weights))
            self.edited()

    def overview_ui(self):
        """One row per weight in use: a bar per iteration, up for +, down for -; the open iteration is bright."""
        d = self.design
        its = d.iterations[: d.full_depth]
        shown = False
        psim.PushStyleVar(psim.ImGuiStyleVar_ItemSpacing, (8.0, 3.0))
        for title, defs in (("Catmull-Clark", CC_WEIGHTS), ("Doo-Sabin", DS_WEIGHTS)):
            for wd in defs:
                v = [it.weights[wd.name] if (it.scheme == "cc") == (defs is CC_WEIGHTS) else 0.0 for it in its]
                if not any(v):
                    continue
                shown = True
                label = WEIGHT_SHORT.get(wd.name, " ".join(wd.label.split())).split()[0] if defs is CC_WEIGHTS \
                    else " ".join(wd.label.split())
                ui.bars(label, v, max(max(abs(x) for x in v), 0.25), self.schedule_tab,
                        f"{title} {wd.name} over iterations 1-{len(its)}:\n"
                        + "  ".join(f"{k + 1}: {x:+.2f}" for k, x in enumerate(v)))
        psim.PopStyleVar()
        if not shown:
            ui.empty("All weights are zero: plain subdivision.")

    # --------------------------------------------------------- 10 export
    def export_ui(self):
        ui.subhead("mesh", top=0.0)
        changed, v = ui.choice("format", [(i, f.upper()) for i, f in enumerate(EXPORT_FORMATS)], self.export_fmt, "fmt",
                               "OBJ and PLY keep quads; STL is triangulated.")
        self.export_fmt = v
        if ui.button("export mesh", "export", "Bakes the full depth first, then saves in exports/ (E). For a "
                     "printable, watertight STL use the print section.", kind="primary"):
            self.export_mesh()
        ui.explain("This is the raw subdivision surface: it may pass through itself. The print section (09) "
                   "rebuilds it as a clean solid in mm.")
        ui.subhead("screenshot")
        if ui.button("save png", "shot", "The 3D view without the panels, saved in renders/."):
            self.screenshot()
        ui.subhead("turntable")
        self.print_panel.turntable_ui()

    # ------------------------------------------------------------ output
    def _stem(self):
        name = "".join(c for c in (self.preset_name or self.save_name) if c.isalnum() or c in "-_") or "form"
        return f"{name}_d{self.result.depth_reached}"

    def export_mesh(self):
        if not self.baked:
            self.bake()
        if not self.result:
            return
        path = os.path.join(EXPORT_DIR, f"{self._stem()}.{EXPORT_FORMATS[self.export_fmt]}")
        try:
            os.makedirs(EXPORT_DIR, exist_ok=True)
            export(path, self.result.mesh)  # checks disk space, writes to a temp file, then renames
        except OSError as e:
            self.error = f"Export failed: {e}"
            return
        note = ""
        if path.endswith(".stl"):
            from hansmeyer.meshio import validate_stl

            info = validate_stl(path)
            note = " (complete; raw surface, use print for a printable STL)" if info["complete"] \
                else " INCOMPLETE, please export again"
        self.status = f"exported exports/{os.path.basename(path)}{note}"

    def screenshot(self):
        if not self.result:
            return
        path = os.path.join(RENDER_DIR, self._stem() + ".png")
        try:
            os.makedirs(RENDER_DIR, exist_ok=True)
            img = ps.screenshot_to_buffer(transparent_bg=False, include_UI=False)
            atomic_write(path, lambda tmp: write_png(tmp, img))
        except OSError as e:
            self.error = f"Screenshot failed: {e}"
            return
        self.status = f"saved renders/{os.path.basename(path)}"

    def safe_ui(self):
        """The Polyscope callback. An exception escaping it would close the app, so anything
        unexpected is logged to app_errors.log and shown in the status bar instead."""
        try:
            self.ui()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e} (details in app_errors.log)"
            try:
                with open(ERROR_LOG, "a", encoding="utf-8") as fh:
                    fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + traceback.format_exc() + "\n")
            except OSError:
                pass


def init_polyscope(settings: dict) -> None:
    """Fonts and style hooks first: Polyscope reads them when it creates its window."""
    ui.set_palette(settings.get("theme", "dark"))
    ps.set_prepare_imgui_fonts_callback(ui.load_fonts)
    ps.set_configure_imgui_style_callback(ui.configure_style)
    ps.set_build_default_gui_panels(False)
    ps.set_open_imgui_window_for_user_callback(False)
    setup_scene(tuple(settings.get("window", DEFAULT_WINDOW)), theme=settings.get("theme", "dark"), title=TITLE)


def main():
    init_polyscope(load_settings())
    app = App()
    ps.set_user_callback(app.safe_ui)
    ps.show()


if __name__ == "__main__":
    main()

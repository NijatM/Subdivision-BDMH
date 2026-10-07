"""The app's look, taken from nijatmahamaliyev.com and the HTMAA site: dark, monochrome, IBM Plex Mono,
1px lines, uppercase letter-spaced labels, dashed borders for what is empty or inactive. Light mode is
the same system inverted. Every panel is built from the small widgets below, so the look lives here.

Layout rule for controls: the label sits in a dim column on the left, the control fills the rest.
"""

from __future__ import annotations

import os

import polyscope.imgui as psim

ROOT = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")


def _hex(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# Tokens from the site's stylesheet (--bg, --bg-2, --line, --line-2, --fg, --fg-hi, --fg-dim, --fg-faint).
# bg-3 is a hover step between bg-2 and line; warn is the one colour used for warnings.
PALETTES = {
    "dark": {"bg": "#0b0b0b", "bg2": "#141414", "bg3": "#1b1b1b", "line": "#232323", "line2": "#3a3a3a",
             "fg": "#d2d2d2", "hi": "#f4f4f4", "dim": "#8a8a8a", "faint": "#4a4a4a", "warn": "#d09a4e"},
    "light": {"bg": "#f4f4f2", "bg2": "#ebebe8", "bg3": "#e2e2df", "line": "#d6d6d2", "line2": "#b4b4b0",
              "fg": "#2b2b2b", "hi": "#0b0b0b", "dim": "#6e6e6c", "faint": "#a6a6a2", "warn": "#a8661a"},
}

# ImGui sizes a font by its line height; Plex Mono's line is 1.3 em, so 17 here = 13 px text on the site.
FONT_SPECS = {"body": ("Regular", 17.0), "small": ("Regular", 14.5), "medium": ("Medium", 17.0),
              "title": ("Medium", 27.0), "name": ("SemiBold", 19.5)}
FONTS: dict = {}
LABEL_W = 118.0  # the label column of the inspector

_state = {"theme": "dark", "help": False}
C: dict = {}  # current palette as RGBA tuples


def set_palette(name: str) -> None:
    _state["theme"] = name
    C.clear()
    C.update({k: (*_hex(v), 1.0) for k, v in PALETTES[name].items()})


set_palette("dark")


def theme() -> str:
    return _state["theme"]


def col(name: str, alpha: float = 1.0) -> tuple:
    c = C[name]
    return (c[0], c[1], c[2], alpha)


def u32(name: str, alpha: float = 1.0) -> int:
    return psim.ColorConvertFloat4ToU32(col(name, alpha))


def help_on() -> bool:
    return _state["help"]


def set_help(on: bool) -> None:
    _state["help"] = bool(on)


# --------------------------------------------------------------------- fonts & style
def load_fonts(atlas):
    """Polyscope's font hook: IBM Plex Mono in the few sizes the UI uses (falls back to ImGui's font)."""
    FONTS.clear()
    for key, (weight, size) in FONT_SPECS.items():
        path = os.path.join(FONT_DIR, f"IBMPlexMono-{weight}.ttf")
        FONTS[key] = atlas.AddFontFromFileTTF(path, size) if os.path.exists(path) else None
    body = FONTS.get("body") or atlas.AddFontDefault()
    for key in FONT_SPECS:
        FONTS[key] = FONTS[key] or body
    return body, body


def apply_style(name: str | None = None) -> None:
    """ImGui sizes and colours for the current theme. Polyscope calls this whenever it rebuilds its style."""
    if name is not None:
        set_palette(name)
    s = psim.GetStyle()
    s.WindowPadding = (20.0, 18.0)
    s.FramePadding = (8.0, 4.0)
    s.ItemSpacing = (8.0, 7.0)
    s.ItemInnerSpacing = (6.0, 4.0)
    s.CellPadding = (6.0, 3.0)
    s.IndentSpacing = 14.0
    s.ScrollbarSize = 8.0
    s.GrabMinSize = 3.0
    for attr in ("WindowRounding", "ChildRounding", "FrameRounding", "PopupRounding", "ScrollbarRounding",
                 "GrabRounding", "TabRounding", "WindowBorderSize", "TabBorderSize"):
        setattr(s, attr, 0.0)
    s.ChildBorderSize = s.PopupBorderSize = s.FrameBorderSize = 1.0
    s.SeparatorTextBorderSize = 1.0
    s.DisabledAlpha = 0.45
    clear = (0.0, 0.0, 0.0, 0.0)
    colours = {
        "Text": col("fg"), "TextDisabled": col("dim"), "WindowBg": col("bg"), "ChildBg": clear,
        "PopupBg": col("bg2"), "Border": col("line"), "BorderShadow": clear,
        "FrameBg": col("bg2"), "FrameBgHovered": col("bg3"), "FrameBgActive": col("bg3"),
        "TitleBg": col("bg"), "TitleBgActive": col("bg2"), "TitleBgCollapsed": col("bg"), "MenuBarBg": col("bg"),
        "ScrollbarBg": clear, "ScrollbarGrab": col("line2"), "ScrollbarGrabHovered": col("dim"),
        "ScrollbarGrabActive": col("fg"), "CheckMark": col("hi"), "SliderGrab": col("dim"),
        "SliderGrabActive": col("hi"), "Button": clear, "ButtonHovered": col("bg3"), "ButtonActive": col("line"),
        "Header": col("bg2"), "HeaderHovered": col("bg2"), "HeaderActive": col("bg3"),
        "Separator": col("line"), "SeparatorHovered": col("line2"), "SeparatorActive": col("dim"),
        "ResizeGrip": clear, "ResizeGripHovered": clear, "ResizeGripActive": clear,
        "Tab": col("bg"), "TabHovered": col("bg3"), "TabSelected": col("bg2"), "TabActive": col("bg2"),
        "TabSelectedOverline": col("hi"), "TabDimmed": col("bg"), "TabDimmedSelected": col("bg2"),
        "PlotLines": col("dim"), "PlotLinesHovered": col("hi"), "PlotHistogram": col("dim"),
        "PlotHistogramHovered": col("hi"), "TableHeaderBg": col("bg2"), "TableBorderStrong": col("line2"),
        "TableBorderLight": col("line"), "TableRowBg": clear, "TableRowBgAlt": col("hi", 0.03),
        "TextSelectedBg": col("dim", 0.45), "TextLink": col("hi"), "DragDropTarget": col("hi"),
        "NavCursor": col("hi"), "NavHighlight": col("hi"), "NavWindowingHighlight": col("hi"),
        "NavWindowingDimBg": (0.0, 0.0, 0.0, 0.5), "ModalWindowDimBg": (0.0, 0.0, 0.0, 0.55),
        "InputTextCursor": col("hi"), "TreeLines": col("line2"),
    }
    colors = s.Colors
    for key, value in colours.items():
        idx = getattr(psim, "ImGuiCol_" + key, None)
        if idx is not None:
            colors[idx] = value


def configure_style():
    """Polyscope's style hook (no arguments)."""
    apply_style()


def push_font(name: str):
    psim.PushFont(FONTS.get(name))


def pop_font():
    psim.PopFont()


# --------------------------------------------------------------------- drawing
def _dl():
    return psim.GetWindowDrawList()


def dashed_line(p0, p1, color: str = "line2", dash: float = 3.0, gap: float = 3.0, dl=None):
    """Horizontal or vertical dashed line (the site's 'border: 1px dashed')."""
    dl = dl or _dl()
    c = u32(color)
    (x0, y0), (x1, y1) = p0, p1
    horizontal = abs(y1 - y0) < abs(x1 - x0)
    a, b = (x0, x1) if horizontal else (y0, y1)
    t = a
    while t < b:
        e = min(t + dash, b)
        dl.AddLine((t, y0) if horizontal else (x0, t), (e, y0) if horizontal else (x0, e), c)
        t = e + gap


def dashed_rect(p0, p1, color: str = "line2", dl=None):
    (x0, y0), (x1, y1) = p0, p1
    for a, b in (((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1 + 1))):
        dashed_line(a, b, color, dl=dl)


def cursor_block(x: float, y: float, h: float, dl=None):
    """The site's blinking block cursor."""
    if int(psim.GetTime() * 2.0) % 2 == 0:
        (dl or _dl()).AddRectFilled((x, y), (x + 0.55 * h, y + h), u32("hi"))


# --------------------------------------------------------------------- text
def text(s: str, color: str = "fg"):
    psim.PushStyleColor(psim.ImGuiCol_Text, col(color))
    psim.TextUnformatted(s)
    psim.PopStyleColor()


def wrap(s: str, color: str = "dim"):
    psim.PushStyleColor(psim.ImGuiCol_Text, col(color))
    psim.TextWrapped(s)
    psim.PopStyleColor()


def note(s: str):
    """A dim line of explanation (always shown)."""
    wrap(s, "dim")


def warn(s: str):
    wrap("! " + s, "warn")


def ok(s: str):
    wrap(s, "hi")


def explain(s: str):
    """Longer help, shown only while the '?' help toggle is on: a quote with a rule on its left."""
    if not help_on() or not s:
        return
    x, y = psim.GetCursorScreenPos()
    psim.Indent(12.0)
    wrap(s, "dim")
    psim.Unindent(12.0)
    y1 = psim.GetItemRectMax()[1]
    _dl().AddLine((x, y + 2), (x, y1 - 1), u32("line2"))


def spaced(s: str, color: str = "faint", font: str = "small", spacing: float = 0.14):
    """Uppercase, letter-spaced label (the site's .label / .kicker), drawn glyph by glyph."""
    push_font(font)
    size = psim.GetFontSize()
    adv = psim.CalcTextSize("M")[0]
    gap = spacing * size / 1.3
    x, y = psim.GetCursorScreenPos()
    dl, f, c = _dl(), psim.GetFont(), u32(color)
    for i, ch in enumerate(s):
        dl.AddText(f, size, (x + i * (adv + gap), y), c, ch)
    psim.Dummy((max(len(s) * (adv + gap) - gap, 1.0), size))
    pop_font()


def subhead(s: str, top: float = 10.0):
    """A small uppercase heading with a dashed rule running to the panel's edge."""
    psim.Dummy((1.0, top))
    spaced(s.upper(), "dim")
    (x0, y0), (x1, y1) = psim.GetItemRectMin(), psim.GetItemRectMax()
    right = x0 + psim.GetContentRegionAvail()[0]
    if x1 + 10 < right:
        y = 0.5 * (y0 + y1) + 1
        dashed_line((x1 + 10, y), (right, y), "line")
    psim.Dummy((1.0, 2.0))


def tip(s: str | None):
    """Hover help for the last item, wrapped to a readable width."""
    if s and psim.BeginItemTooltip():
        psim.PushTextWrapPos(psim.GetFontSize() * 26.0)
        psim.TextUnformatted(s)
        psim.PopTextWrapPos()
        psim.EndTooltip()


def truncate(s: str, width: float) -> str:
    adv = psim.CalcTextSize("M")[0]
    n = int(width // max(adv, 1.0))
    return s if len(s) <= n else s[: max(n - 1, 0)] + "…"


# --------------------------------------------------------------------- rows
def _label(label: str, help: str | None) -> float:
    """Dim label in the left column; the next item fills the rest of the row (returns its width)."""
    x = psim.GetCursorPosX()
    psim.AlignTextToFramePadding()
    text(truncate(label, LABEL_W - 6.0), "dim")
    tip(help)
    psim.SameLine(x + LABEL_W)
    w = max(psim.GetContentRegionAvail()[0], 40.0)
    psim.SetNextItemWidth(w)
    return w


def _clamp01(t: float) -> float:
    return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t


_ACTIVE: dict = {}  # slider id -> active last frame (to keep ImGui's own text while it is being typed into)


def _fmt(fmt: str, v) -> str:
    try:
        return fmt % v
    except (TypeError, ValueError):
        return f"{v}"


def bar_slider(fn, id: str, v, lo, hi, fmt: str, w: float):
    """ImGui's slider drawn as a quiet data bar: filled from zero (or the low end) to the value, a thin end
    marker, a tick at zero for ranges that cross it, and the value in its own column on the right."""
    x, y = psim.GetCursorScreenPos()
    h = psim.GetFrameHeight()
    adv = psim.CalcTextSize("M")[0]
    shown = _fmt(fmt, v)
    vw = max(7, len(shown)) * adv
    bw = max(w - vw - 10.0, 40.0)
    dl = _dl()
    span = float(hi - lo) or 1.0
    z = _clamp01((0.0 - lo) / span) if lo < 0 < hi else 0.0

    def pos(t):
        return x + 1.0 + t * (bw - 2.0)

    t = _clamp01((v - lo) / span)
    dl.AddRectFilled((x, y), (x + bw, y + h), u32("bg2"))
    a, b = sorted((z, t))
    if b - a > 1e-6:
        dl.AddRectFilled((pos(a), y + 1), (pos(b), y + h - 1), u32("line"))
    if lo < 0 < hi:
        dl.AddLine((pos(z), y + 4), (pos(z), y + h - 4), u32("line2"))
    typing = _ACTIVE.get(id, False) and psim.GetIO().WantTextInput
    clear = (0.0, 0.0, 0.0, 0.0)
    psim.PushStyleColor(psim.ImGuiCol_FrameBg, clear)
    psim.PushStyleColor(psim.ImGuiCol_FrameBgHovered, col("hi", 0.035))
    psim.PushStyleColor(psim.ImGuiCol_FrameBgActive, col("hi", 0.05))
    psim.PushStyleColor(psim.ImGuiCol_SliderGrab, clear)
    psim.PushStyleColor(psim.ImGuiCol_SliderGrabActive, clear)
    psim.PushStyleColor(psim.ImGuiCol_Text, col("hi") if typing else clear)
    psim.SetNextItemWidth(bw)
    out = fn(id, v, lo, hi, fmt)
    psim.PopStyleColor(6)
    active, hovered = psim.IsItemActive(), psim.IsItemHovered()
    _ACTIVE[id] = active
    nv = out[1]
    mx = pos(_clamp01((nv - lo) / span))
    dl.AddRectFilled((mx - 1.0, y + 2), (mx + 1.0, y + h - 2), u32("hi" if active or hovered else "fg"))
    shown = _fmt(fmt, nv)
    tw = psim.CalcTextSize(shown)[0]
    dl.AddText((x + w - tw, y + psim.GetStyle().FramePadding[1]), u32("hi" if active or hovered else "fg"), shown)
    return (True, nv) if out[0] else (False, v)  # unchanged: the caller's exact value, not a float32 round trip


def slider(label, v, lo, hi, fmt="%.2f", help=None, key=None):
    w = _label(label, help)
    out = bar_slider(psim.SliderFloat, f"##{key or label}", float(v), lo, hi, fmt, w)
    tip(help)
    return out


def slider_int(label, v, lo, hi, fmt="%d", help=None, key=None):
    w = _label(label, help)
    out = bar_slider(psim.SliderInt, f"##{key or label}", int(v), int(lo), int(hi), fmt, w)
    tip(help)
    return out


def chevron(x: float, y: float, color: str = "dim", size: float = 4.0, dl=None):
    """A small 'v' (the combo's arrow)."""
    dl = dl or _dl()
    c = u32(color)
    dl.AddLine((x - size, y - 0.5 * size), (x, y + 0.5 * size), c, 1.2)
    dl.AddLine((x, y + 0.5 * size), (x + size, y - 0.5 * size), c, 1.2)


def combo_box(id: str, idx: int, items, w: float):
    """A combo with a quiet chevron instead of ImGui's arrow box. Returns (changed, index)."""
    items = list(items)
    x, y = psim.GetCursorScreenPos()
    h = psim.GetFrameHeight()
    preview = truncate(items[idx], w - 34.0) if 0 <= idx < len(items) else ""
    psim.SetNextItemWidth(w)
    changed, out = False, idx
    if psim.BeginCombo(id, preview, psim.ImGuiComboFlags_NoArrowButton):
        for i, item in enumerate(items):
            if psim.Selectable(f"{item}##{i}", i == idx):
                changed, out = i != idx, i
            if i == idx:
                psim.SetItemDefaultFocus()
        psim.EndCombo()
    chevron(x + w - 13.0, y + 0.5 * h, "fg")
    return changed, out


def slider_int2(label, v, lo, hi, help=None, key=None):
    _label(label, help)
    out = psim.SliderInt2(f"##{key or label}", list(v), int(lo), int(hi))
    tip(help)
    return out


def slider_float2(label, v, lo, hi, fmt="%.2f", help=None, key=None):
    _label(label, help)
    out = psim.SliderFloat2(f"##{key or label}", list(v), lo, hi, fmt)
    tip(help)
    return out


def drag3(label, v, speed=0.01, help=None, key=None):
    _label(label, help)
    out = psim.DragFloat3(f"##{key or label}", list(v), speed)
    tip(help)
    return out


def combo(label, idx, items, help=None, key=None):
    w = _label(label, help)
    out = combo_box(f"##{key or label}", int(idx), items, w)
    tip(help)
    return out


def bars(label: str, values, rng: float, highlight: int | None = None, help: str | None = None, h: float = 22.0):
    """One weight over the iterations: a bar per iteration, up for + and down for -, zero in the middle."""
    x0 = psim.GetCursorPosX()
    psim.AlignTextToFramePadding()
    text(truncate(label, LABEL_W - 6.0), "dim")
    psim.SameLine(x0 + LABEL_W)
    x, y = psim.GetCursorScreenPos()
    w = psim.GetContentRegionAvail()[0]
    dl = _dl()
    dl.AddRectFilled((x, y), (x + w, y + h), u32("bg2"))
    mid = y + 0.5 * h
    dl.AddLine((x, mid), (x + w, mid), u32("line2"))
    n = max(len(values), 1)
    bw = w / n
    for i, v in enumerate(values):
        if abs(v) < 1e-9:
            continue
        top = mid - (v / rng) * (0.5 * h - 2.0)
        x1 = x + i * bw + 2.0
        dl.AddRectFilled((x1, min(mid, top)), (x1 + bw - 4.0, max(mid, top)), u32("hi" if i == highlight else "dim"))
    psim.Dummy((w, h))
    tip(help)


def input_text(label, s, help=None, key=None):
    _label(label, help)
    out = psim.InputText(f"##{key or label}", s)
    tip(help)
    return out


def input_float(label, v, step=0.0, step_fast=0.0, fmt="%.2f", help=None, key=None):
    _label(label, help)
    out = psim.InputFloat(f"##{key or label}", float(v), step, step_fast, fmt)
    tip(help)
    return out


def input_float3(label, v, fmt="%.0f", help=None, key=None):
    _label(label, help)
    out = psim.InputFloat3(f"##{key or label}", list(v), fmt)
    tip(help)
    return out


def input_int(label, v, step=1, step_fast=100, help=None, key=None):
    _label(label, help)
    out = psim.InputInt(f"##{key or label}", int(v), step, step_fast)
    tip(help)
    return out


def value(label: str, s: str, color: str = "fg", help: str | None = None):
    """A read-only row: label, then a value."""
    x = psim.GetCursorPosX()
    text(truncate(label, LABEL_W - 6.0), "dim")
    tip(help)
    psim.SameLine(x + LABEL_W)
    psim.PushTextWrapPos(0.0)
    text(s, color)
    psim.PopTextWrapPos()


def row_label(label: str, help: str | None = None):
    """Just the label column, for rows that place their own controls (segmented choices, buttons)."""
    x = psim.GetCursorPosX()
    psim.AlignTextToFramePadding()
    text(truncate(label, LABEL_W - 6.0), "dim")
    tip(help)
    psim.SameLine(x + LABEL_W)


# --------------------------------------------------------------------- buttons
def button(label: str, key: str | None = None, help: str | None = None, width: float = 0.0, kind: str = "normal",
           enabled: bool = True) -> bool:
    """Outlined box. kind: normal | primary (brighter outline) | on (inverted, a chosen toggle)."""
    pushed = 2
    if kind == "on":
        psim.PushStyleColor(psim.ImGuiCol_Button, col("hi"))
        psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, col("fg"))
        psim.PushStyleColor(psim.ImGuiCol_ButtonActive, col("dim"))
        psim.PushStyleColor(psim.ImGuiCol_Text, col("bg"))
        psim.PushStyleColor(psim.ImGuiCol_Border, col("hi"))
        pushed = 5
    else:
        psim.PushStyleColor(psim.ImGuiCol_Border, col("fg" if kind == "primary" else "line2"))
        psim.PushStyleColor(psim.ImGuiCol_Text, col("hi" if kind == "primary" else "fg"))
    if not enabled:
        psim.BeginDisabled()
    clicked = psim.Button(f"{label}##{key or label}", (width, 0.0))
    if not enabled:
        psim.EndDisabled()
    psim.PopStyleColor(pushed)
    tip(help)
    return clicked and enabled


def toggle(label: str, on: bool, key: str | None = None, help: str | None = None, width: float = 0.0) -> bool:
    return button(label, key, help, width, "on" if on else "normal")


def segmented(options, current, key: str, fill: bool = True, helps=None):
    """A row of joined boxes, the chosen one inverted. options: [(value, label)]. Returns (changed, value)."""
    n = len(options)
    avail = psim.GetContentRegionAvail()[0]
    w = (avail + (n - 1)) / n if fill else 0.0
    changed, out = False, current
    psim.PushStyleVar(psim.ImGuiStyleVar_ItemSpacing, (-1.0, psim.GetStyle().ItemSpacing[1]))
    for i, (val, label) in enumerate(options):
        if i:
            psim.SameLine()
        if button(label, f"{key}{i}", helps[i] if helps else None, w if fill else 0.0, "on" if val == current else "normal"):
            changed, out = val != current, val
    psim.PopStyleVar()
    return changed, out


def choice(label: str, options, current, key: str, help: str | None = None, helps=None):
    """Labelled segmented row (replaces radio buttons)."""
    row_label(label, help)
    return segmented(options, current, key, True, helps)


def check(label: str, v: bool, key: str | None = None, help: str | None = None):
    """'[x] label' — a checkbox in the site's terminal manner. Returns (changed, value)."""
    s = f"[{'x' if v else ' '}] {label}"
    size = psim.CalcTextSize(s)
    h = psim.GetFrameHeight()
    x, y = psim.GetCursorScreenPos()
    clicked = psim.InvisibleButton(f"##chk{key or label}", (size[0] + 2.0, h))
    hovered = psim.IsItemHovered()
    _dl().AddText((x, y + 0.5 * (h - size[1])), u32("hi" if hovered or v else "fg"), s)
    tip(help)
    return clicked, (not v) if clicked else v


def stepper(label: str, v: int, lo: int, hi: int, key: str, help: str | None = None):
    """'label ‹ 5 ›' — a compact integer with step buttons. Returns (changed, value)."""
    text(label, "dim")
    tip(help)
    psim.SameLine(0.0, 6.0)
    out = v
    psim.PushStyleVar(psim.ImGuiStyleVar_FramePadding, (5.0, psim.GetStyle().FramePadding[1]))
    if button("‹", f"{key}-", help, enabled=v > lo):
        out = v - 1
    psim.SameLine(0.0, 2.0)
    psim.AlignTextToFramePadding()
    text(f"{v:>2d}", "hi")
    tip(help)
    psim.SameLine(0.0, 2.0)
    if button("›", f"{key}+", help, enabled=v < hi):
        out = v + 1
    psim.PopStyleVar()
    return out != v, out


# --------------------------------------------------------------------- lists & boxes
def begin_list():
    """Rows of a list sit tight, like the site's week list."""
    psim.PushStyleVar(psim.ImGuiStyleVar_ItemSpacing, (psim.GetStyle().ItemSpacing[0], 1.0))


def end_list():
    psim.PopStyleVar()


def list_row(key: str, left: str, right: str = "", selected: bool = False, num: str | None = None,
             faint: bool = False, help: str | None = None, indent: float = 0.0, height: float | None = None) -> bool:
    """A full-width row like the site's week list: [num] title ... right. Current row: bg-2 and bright text."""
    width = psim.GetContentRegionAvail()[0]
    h = height or psim.GetTextLineHeight() + 6.0
    clicked = psim.Selectable(f"##{key}", selected, 0, (width, h))
    hovered = psim.IsItemHovered()
    tip(help)
    (x0, y0), (x1, _) = psim.GetItemRectMin(), psim.GetItemRectMax()
    ty = y0 + 0.5 * (h - psim.GetTextLineHeight())
    dl = _dl()
    main = "faint" if faint and not selected else ("hi" if selected or hovered else "fg")
    x = x0 + 6.0 + indent
    adv = psim.CalcTextSize("M")[0]
    if num is not None:
        dl.AddText((x, ty), u32("hi" if selected else ("faint" if faint else "dim")), num)
        x += (len(num) + 2) * adv
    if right:  # the right column gives way first: the title is what you scan for
        room = int((x1 - 6.0 - x) // adv) - len(left) - 2
        right = right if len(right) <= room else (right[: max(room - 1, 0)] + "…" if room > 1 else "")
    rw = len(right) * adv
    if right:
        dl.AddText((x1 - 6.0 - rw, ty), u32("faint" if faint or not selected else "dim"), right)
    dl.AddText((x, ty), u32(main), truncate(left, x1 - 6.0 - (rw + 2 * adv if right else 0.0) - x))
    return clicked


def empty(s: str, height: float | None = None):
    """A dashed box with faint text: nothing here yet (the site's draft tiles)."""
    width = psim.GetContentRegionAvail()[0]
    x, y = psim.GetCursorScreenPos()
    psim.PushTextWrapPos(psim.GetCursorPosX() + width - 14.0)
    psim.SetCursorScreenPos((x + 12.0, y + 10.0))
    wrap(s, "faint")
    psim.PopTextWrapPos()
    y1 = max(psim.GetItemRectMax()[1] + 10.0, y + (height or 0.0))
    dashed_rect((x, y), (x + width - 1.0, y1), "line2")
    psim.SetCursorScreenPos((x, y1 + 6.0))
    psim.Dummy((1.0, 1.0))


def begin_card(key: str, strong: bool = True) -> None:
    """A bordered box (the site's final-project card). Close with end_card()."""
    psim.PushStyleColor(psim.ImGuiCol_Border, col("line2" if strong else "line"))
    psim.PushStyleVar(psim.ImGuiStyleVar_WindowPadding, (14.0, 12.0))
    flags = (getattr(psim, "ImGuiChildFlags_Borders", 1) | getattr(psim, "ImGuiChildFlags_AutoResizeY", 64)
             | getattr(psim, "ImGuiChildFlags_AlwaysUseWindowPadding", 16))
    psim.BeginChild(key, (0.0, 0.0), flags, psim.ImGuiWindowFlags_NoScrollbar)


def end_card() -> None:
    psim.EndChild()
    psim.PopStyleVar()
    psim.PopStyleColor()


def gap(h: float = 6.0):
    psim.Dummy((1.0, h))

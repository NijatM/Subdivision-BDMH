"""Small UI building blocks shared by the app's panels (built on ui_style's widgets)."""

from __future__ import annotations

import polyscope.imgui as psim

import ui_style as ui
from hansmeyer import MAX_ITERATIONS
from hansmeyer.schedule import ALL_WEIGHTS

WEIGHT_NAMES = [w.name for w in ALL_WEIGHTS]
WEIGHT_LABELS = [" ".join(w.label.split()) for w in ALL_WEIGHTS]
RULE_OPS = [("scale", "scale x"), ("offset", "offset +")]


def weight_combo(label: str, current: str, key: str, help: str | None = None) -> tuple[bool, str]:
    idx = WEIGHT_NAMES.index(current) if current in WEIGHT_NAMES else 0
    changed, idx = ui.combo(label, idx, WEIGHT_LABELS, help, key)
    return changed, WEIGHT_NAMES[idx]


def iteration_range(label: str, rule: dict, key: str, help: str | None = None) -> bool:
    changed, rng = ui.slider_int2(label, [rule["from"], rule["to"]], 1, MAX_ITERATIONS,
                                  help or "First and last iteration this applies to.", key)
    if changed:
        rule["from"], rule["to"] = min(rng), max(rng)
    return changed


def item_header(title: str, key: str, enabled: bool | None = None, removable: bool = True):
    """'TITLE ------ [x] on  ×' above an item in a list of rules. Returns (toggled, enabled, remove)."""
    toggled, remove = False, False
    ui.spaced(title.upper(), "dim")
    right = 0.0
    pad = psim.GetStyle().FramePadding[0]
    if removable:
        right += psim.CalcTextSize("×")[0] + 2 * pad
    if enabled is not None:
        right += psim.CalcTextSize("[x] on")[0] + 12.0
    psim.SameLine()
    x_end = psim.GetCursorPosX() + psim.GetContentRegionAvail()[0]
    psim.SetCursorPosX(x_end - right)
    psim.SetCursorPosY(psim.GetCursorPosY() - 3.0)
    if enabled is not None:
        toggled, enabled = ui.check("on", enabled, f"{key}on", "Switch this off without deleting it.")
        if removable:
            psim.SameLine(0.0, 10.0)
    if removable:
        remove = ui.button("×", f"{key}x", "Delete")
    return toggled, enabled, remove


def rules_editor(rules: list, key: str) -> bool:
    """Weight rules ('scale x' / 'offset +' one weight over a range of iterations). True when anything changed."""
    changed_any, remove = False, None
    for i, m in enumerate(rules):
        if i:
            ui.gap(4.0)
        _, _, rm = item_header(f"rule {i + 1}", f"{key}{i}")
        if rm:
            remove = i
        ch, m["weight"] = weight_combo("weight", m["weight"], f"{key}w{i}")
        changed_any |= ch
        ch, op = ui.choice("operation", RULE_OPS, m["op"], f"{key}o{i}")
        if ch:
            m["op"] = op
            m["value"] = 2.0 if op == "scale" else 0.3
            changed_any = True
        lo, hi = (0.0, 5.0) if m["op"] == "scale" else (-1.5, 1.5)
        ch, m["value"] = ui.slider("value", m["value"], lo, hi, "%.2f", key=f"{key}v{i}")
        changed_any |= ch
        changed_any |= iteration_range("iterations", m, f"{key}r{i}")
    if remove is not None:
        rules.pop(remove)
        changed_any = True
    if not rules:
        ui.empty("No rules yet.")
    if ui.button("+ rule", f"{key}add", "Scale or offset one weight over a range of iterations."):
        rules.append({"weight": "w_f", "op": "scale", "value": 2.0, "from": 1, "to": MAX_ITERATIONS})
        changed_any = True
    return changed_any


def params_editor(fd, params: dict, key: str) -> bool:
    """Auto-generated sliders for a registered function's parameters."""
    changed_any = False
    for p in fd.params:
        val = params.get(p.name, p.default)
        if p.integer:
            ch, v = ui.slider_int(p.name, val, p.lo, p.hi, key=f"{key}{p.name}")
        else:
            ch, v = ui.slider(p.name, val, p.lo, p.hi, "%.3f", key=f"{key}{p.name}")
        if ch:
            params[p.name] = v
            changed_any = True
    return changed_any

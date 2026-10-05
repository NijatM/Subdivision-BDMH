"""Small UI building blocks shared by the app's panels."""

from __future__ import annotations

import polyscope.imgui as psim

from hansmeyer import MAX_ITERATIONS
from hansmeyer.schedule import ALL_WEIGHTS

WARN = (0.95, 0.55, 0.15, 1.0)
ACTIVE = (0.85, 0.55, 0.20, 1.0)
OPEN = getattr(psim, "ImGuiTreeNodeFlags_DefaultOpen", 1 << 5)
WEIGHT_NAMES = [w.name for w in ALL_WEIGHTS]
WEIGHT_LABELS = [w.label for w in ALL_WEIGHTS]


def wrapped_colored(color, text: str) -> None:
    """Coloured text that wraps at the panel edge."""
    psim.PushStyleColor(psim.ImGuiCol_Text, color)
    psim.TextWrapped(text)
    psim.PopStyleColor()


def toggle_button(label: str, active: bool) -> bool:
    if active:
        psim.PushStyleColor(psim.ImGuiCol_Button, ACTIVE)
    clicked = psim.Button(label)
    if active:
        psim.PopStyleColor()
    return clicked


def weight_combo(label: str, current: str) -> tuple[bool, str]:
    idx = WEIGHT_NAMES.index(current) if current in WEIGHT_NAMES else 0
    changed, idx = psim.Combo(label, idx, WEIGHT_LABELS)
    return changed, WEIGHT_NAMES[idx]


def iteration_range(label: str, rule: dict) -> bool:
    changed, rng = psim.SliderInt2(label, [rule["from"], rule["to"]], 1, MAX_ITERATIONS)
    if changed:
        rule["from"], rule["to"] = min(rng), max(rng)
    return changed


def list_selector(items: list[str], selected: int, key: str) -> int:
    for i, text in enumerate(items):
        if psim.Selectable(f"{text}##{key}{i}", i == selected):
            selected = i
    return selected


def rules_editor(rules: list, key: str) -> bool:
    """Rows of 'weight  scale x / offset +  value  [iterations]'. Returns True when anything changed."""
    changed_any, remove = False, None
    for i, m in enumerate(rules):
        psim.PushItemWidth(110)
        ch, m["weight"] = weight_combo(f"##{key}w{i}", m["weight"])
        changed_any |= ch
        psim.SameLine()
        psim.PushItemWidth(70)
        ch, op = psim.Combo(f"##{key}o{i}", 0 if m["op"] == "scale" else 1, ["scale x", "offset +"])
        psim.PopItemWidth()
        if ch:
            m["op"] = "scale" if op == 0 else "offset"
            m["value"] = 2.0 if m["op"] == "scale" else 0.3
            changed_any = True
        psim.SameLine()
        lo, hi = (0.0, 5.0) if m["op"] == "scale" else (-1.5, 1.5)
        ch, m["value"] = psim.SliderFloat(f"##{key}v{i}", m["value"], lo, hi, "%.2f")
        changed_any |= ch
        psim.SameLine()
        if psim.Button(f"x##{key}x{i}"):
            remove = i
        changed_any |= iteration_range(f"iterations##{key}r{i}", m)
        psim.PopItemWidth()
    if remove is not None:
        rules.pop(remove)
        changed_any = True
    if psim.Button(f"+ rule##{key}add"):
        rules.append({"weight": "w_f", "op": "scale", "value": 2.0, "from": 1, "to": MAX_ITERATIONS})
        changed_any = True
    return changed_any


def params_editor(fd, params: dict, key: str) -> bool:
    """Auto-generated sliders for a registered function's parameters."""
    changed_any = False
    for p in fd.params:
        val = params.get(p.name, p.default)
        if p.integer:
            ch, v = psim.SliderInt(f"{p.name}##{key}", int(round(val)), int(p.lo), int(p.hi))
        else:
            ch, v = psim.SliderFloat(f"{p.name}##{key}", float(val), p.lo, p.hi, "%.3f")
        if ch:
            params[p.name] = v
            changed_any = True
    return changed_any

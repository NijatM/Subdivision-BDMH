"""The UI's logic that runs without a window: undo history, preset naming and families, the style tokens."""

import glob
import json
import os

import pytest

from app import FAMILIES, SECTIONS, History, preset_family
from hansmeyer import Design
from ui_style import FONT_DIR, FONT_SPECS, PALETTES, _fmt, _hex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS = sorted(p for p in glob.glob(os.path.join(ROOT, "presets", "*.json"))
                 if not os.path.basename(p).startswith("_"))


def test_history_undo_redo():
    h = History()
    for s in ("a", "b", "c"):
        assert h.push(s)
    assert not h.push("c")  # an unchanged design is not a new step
    assert h.undo() == "b" and h.undo() == "a" and h.undo() is None
    assert h.redo() == "b"
    assert not h.push("b")  # restoring a step must not wipe the redo branch
    assert h.can_redo() and h.redo() == "c"


def test_history_new_edit_drops_redo_and_respects_limit():
    h = History(limit=3)
    for s in "abcd":
        h.push(s)
    assert h.stack == ["b", "c", "d"]
    h.undo()
    h.push("x")
    assert h.stack == ["b", "c", "x"] and not h.can_redo()


def test_preset_family():
    assert preset_family("cube_six_arms") == "CUBE"
    assert preset_family("vessel_lithophane") == "VESSEL"
    assert preset_family("my_form") == "SAVED"
    assert preset_family("cubeish") == "SAVED"


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: os.path.basename(p))
def test_presets_named_by_family(path):
    name = os.path.splitext(os.path.basename(path))[0]
    assert preset_family(name) in {label for _, label in FAMILIES}
    assert not name.startswith("fig")  # descriptive names; the figure goes in the description
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["name"] == name and data.get("description")
    Design.from_dict(data)


def test_preset_names_unique_after_family_prefix():
    shorts = [os.path.splitext(os.path.basename(p))[0] for p in PRESETS]
    assert len(shorts) == len(set(shorts))


def test_sections_and_palettes():
    assert [g for g, *_ in SECTIONS] == ["FORM"] * 3 + ["GROWTH"] * 5 + ["MAKE"] * 3
    assert set(PALETTES["dark"]) == set(PALETTES["light"])
    assert _hex(PALETTES["dark"]["bg"]) == pytest.approx((11 / 255,) * 3)  # the site's --bg
    for weight, _ in FONT_SPECS.values():
        assert os.path.exists(os.path.join(FONT_DIR, f"IBMPlexMono-{weight}.ttf"))


def test_value_format():
    assert _fmt("%.0f %%", 50.0) == "50 %"
    assert _fmt("x %.2f", 1.0) == "x 1.00"
    assert _fmt("%d deg", 30) == "30 deg"

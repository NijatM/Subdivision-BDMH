"""Print prep, part 2: orientation, size along a chosen axis, cutting into parts, pin holes,
support (overhang) check and per-part export."""

import os

import numpy as np
import pytest

from hansmeyer import Design, Pipeline, default_spec, make_base
from hansmeyer.printprep import (UP_KEYS, PrintSettings, best_up, prepare, to_print_coords, up_rotation, up_vector,
                                 write_stl)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = dict(size_mm=40.0, voxel_mm=0.5, smooth=4)


def closed(m) -> bool:
    return bool(np.all(m.he_twin >= 0))


@pytest.fixture(scope="module")
def six_arms():
    d = Design.load(os.path.join(ROOT, "presets", "fig3_left.json"))
    return Pipeline(root=ROOT).run(d, 4).mesh


@pytest.mark.parametrize("key", UP_KEYS)
def test_up_rotation_is_a_proper_rotation(key):
    R = up_rotation(key)
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)  # never a mirror
    assert np.allclose(R @ up_vector(key), [0, 0, 1])


def test_legacy_up_axes_unchanged():
    V = np.random.default_rng(0).normal(size=(20, 3))
    assert np.allclose(to_print_coords(V, "y"), np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1))
    assert np.allclose(to_print_coords(V, "z"), V)


def test_size_along_a_chosen_axis():
    col = make_base(default_spec("column_classic"))  # tall along model Y
    r = prepare(col, PrintSettings(**{**FAST, "size_mm": 30.0, "size_axis": "width"}))
    assert r.dims_mm[0] == pytest.approx(30.0, abs=1.0)
    assert r.dims_mm[2] > r.dims_mm[0]  # still standing: height is the long side
    r = prepare(col, PrintSettings(**{**FAST, "size_axis": "height"}))
    assert r.dims_mm[2] == pytest.approx(40.0, abs=1.0)


def test_best_up_stands_a_column_up():
    col = make_base(default_spec("column_classic"))
    assert best_up(col.V, col.triangles()) in ("y", "-y")


def test_whole_form_is_one_part(six_arms):
    r = prepare(six_arms, PrintSettings(**FAST))
    assert len(r.parts) == 1 and r.parts[0].name == "" and r.pins == 0
    V, F = r.part_arrays(0)
    assert np.allclose(V, r.V) and np.array_equal(F, r.F)


def test_halves_are_closed_flat_and_need_less_support(six_arms):
    whole = prepare(six_arms, PrintSettings(**FAST))
    halves = prepare(six_arms, PrintSettings(**FAST, cut_z=-1.0))
    assert [p.name for p in halves.parts] == ["bottom", "top"]
    for i, p in enumerate(halves.parts):
        m = halves.part_mesh(i)
        assert closed(m) and m.euler_characteristic() == 2 and m.signed_volume() > 0
        V, _ = halves.part_arrays(i)
        assert V[:, 2].min() == pytest.approx(0.0)
        assert (V[:, 2] < 1e-6).sum() > 100  # standing on a flat cut face, not on a point
        assert p.shells == 1
    assert {p.up for p in halves.parts} == {"z", "-z"}  # cut face down: the bottom half turns over
    assert np.allclose(halves.dims_mm, whole.dims_mm, atol=0.6)
    assert halves.volume_cm3 == pytest.approx(whole.volume_cm3, rel=0.05)
    _, _, share_whole = whole.overhang(30)
    _, _, share_halves = halves.overhang(30)
    assert share_halves < 0.6 * share_whole


def test_quarters_meet_without_gaps(six_arms):
    r = prepare(six_arms, PrintSettings(**FAST, cut_x=0.5, cut_y=0.5, pins=False))
    assert len(r.parts) == 4 and r.pins == 0
    V, F, owner = r.assembled(0.0)
    names = [p.name for p in r.parts]
    assert sorted(names) == sorted(["left front", "left back", "right front", "right back"])
    left = np.isin(owner, [i for i, n in enumerate(names) if n.startswith("left")])
    plane = V[left, 0].max()
    assert V[~left, 0].min() == pytest.approx(plane, abs=1e-4)  # the joint faces coincide
    on_left = np.abs(V[left, 0] - plane) < 1e-4
    on_right = np.abs(V[~left, 0] - plane) < 1e-4
    assert on_left.sum() > 50 and on_right.sum() > 50
    # exploding moves the parts apart by the gap along every cut
    Ve, _, _ = r.assembled(10.0)
    assert Ve[~left, 0].min() - Ve[left, 0].max() == pytest.approx(10.0, abs=1e-3)
    # print layout: footprints never overlap
    boxes = [(r.V[p.v0:p.v1, :2].min(0), r.V[p.v0:p.v1, :2].max(0)) for p in r.parts]
    for i in range(4):
        for j in range(i + 1, 4):
            (a0, a1), (b0, b1) = boxes[i], boxes[j]
            assert np.any(a1 < b0) or np.any(b1 < a0)


def test_pin_holes_match_across_the_joint(six_arms):
    s = dict(**{**FAST, "size_mm": 60.0, "voxel_mm": 0.3}, cut_z=-1.0)
    plain = prepare(six_arms, PrintSettings(**s, pins=False))
    pinned = prepare(six_arms, PrintSettings(**s, pins=True, pin_mm=2.0, pin_depth_mm=4.0))
    assert pinned.pins >= 1
    hole = np.pi * 1.0 ** 2 * 8.0 / 1000  # cm3 per hole (both sides)
    assert plain.volume_cm3 - pinned.volume_cm3 == pytest.approx(pinned.pins * hole, rel=0.15)
    for i in range(2):
        m = pinned.part_mesh(i)
        assert closed(m) and m.euler_characteristic() == 2  # blind holes keep each part a single solid


def test_parts_as_assembled_keep_the_up_direction(six_arms):
    r = prepare(six_arms, PrintSettings(**FAST, cut_z=-1.0, part_up="assembly"))
    assert all(p.up == "z" for p in r.parts)


def test_export_one_stl_per_part(six_arms, tmp_path):
    r = prepare(six_arms, PrintSettings(**FAST, cut_x=0.5, cut_y=0.5))
    for i, p in enumerate(r.parts):
        info = write_stl(str(tmp_path / f"p{i}.stl"), r, i)
        assert info["complete"] and info["watertight"] and info["triangles"] == p.f1 - p.f0


def test_overhang_threshold_semantics():
    from hansmeyer.printprep import overhang_mask

    nz = np.array([-1.0, -0.8, -0.3, 0.0, 1.0])  # flat ceiling, 37 deg, 73 deg slope, wall, top
    z = np.full(5, 10.0)
    assert overhang_mask(nz, z, 30).tolist() == [True, False, False, False, False]
    assert overhang_mask(nz, z, 60).tolist() == [True, True, False, False, False]  # higher = more support
    assert overhang_mask(nz, np.zeros(5), 60).tolist() == [False] * 5  # resting on the plate


# ----------------------------------------------- printable undersides, facing masks
def test_self_supporting_fill_obeys_the_angle():
    from scipy import ndimage

    from hansmeyer.printprep import self_supporting

    solid = np.zeros((40, 40, 40), bool)
    solid[18:22, 18:22, 0:30] = True  # a post
    solid[4:36, 10:30, 30:34] = True  # a wide slab on top: a big flat overhang
    out = self_supporting(solid, 45.0)
    assert np.all(out[solid])  # only ever adds material
    assert np.array_equal(out[:, :, 30:], solid[:, :, 30:])  # nothing above the overhang changes
    for z in range(1, 40):  # every layer rests on the one below, grown by 1 voxel (45 deg) ...
        grown = ndimage.binary_dilation(out[:, :, z - 1], structure=np.ones((3, 3), bool))
        hanging = out[:, :, z] & ~grown
        assert not ndimage.binary_erosion(hanging, structure=np.ones((3, 3), bool)).any()  # ... but thin keel edges
    steep = self_supporting(solid, 63.5)  # steeper keels reach further down: more material
    assert steep.sum() > out.sum()


def test_undersides_and_foot_make_the_form_self_supporting(six_arms):
    plain = prepare(six_arms, PrintSettings(**FAST))
    keeled = prepare(six_arms, PrintSettings(**FAST, undersides=45.0, foot_mm=1.0))
    assert closed(keeled.mesh()) and keeled.parts[0].shells == 1
    assert keeled.overhang(30)[2] < 0.25 * plain.overhang(30)[2]
    V, _ = keeled.part_arrays(0)
    assert (V[:, 2] < 1e-6).sum() > 20  # a flat foot instead of a point
    assert keeled.volume_cm3 > plain.volume_cm3  # material was added, never removed above the foot


def test_facing_mask_keeps_undersides_plain():
    from hansmeyer import IterationSpec, functions, layers

    n = np.array([[0, 1, 0], [0, 0, 1], [0, -1, 0]], float)
    assert np.allclose(layers.facing_mask(n, "facing +y"), [1, layers.facing_mask(n[1:2], "facing +y")[0], 0])
    assert np.allclose(functions.evaluate("constant", np.zeros((4, 3)), {}), 1.0)
    its = [IterationSpec("cc", {"w1": -1.0, "w2": -2.0})] * 3
    bump = {"name": "top", "target": "displacement", "function": "constant", "amplitude": 0.0, "offset": 0.4,
            "from": 3, "to": 3, "mask": "facing +y"}
    pipe = Pipeline(root=ROOT)
    a = pipe.run(Design(iterations=its), 3).mesh
    b = pipe.run(Design(iterations=its, layers=[bump]), 3).mesh
    moved = np.linalg.norm(b.V - a.V, axis=1)
    up, down = a.vert_normal[:, 1] > 0.8, a.vert_normal[:, 1] < -0.8
    assert moved[up].min() > 0.01 and moved[down].max() < 1e-9


def test_printable_central_monolith_preset():
    d = Design.load(os.path.join(ROOT, "presets", "printable_central_monolith.json"))
    m = Pipeline(root=ROOT).run(d, 6).mesh
    assert np.all(np.isfinite(m.V))
    r = prepare(m, PrintSettings(**FAST, undersides=45.0, foot_mm=1.0))
    assert closed(r.mesh()) and r.parts[0].shells == 1 and r.overhang(30)[2] < 0.02

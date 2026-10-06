"""Vessel (lithophane sphere): the open-sphere base, the shell, exact print export, light preview."""

import os

import numpy as np
import pytest

from hansmeyer import Design, IterationSpec, Pipeline, default_spec, make_base, vessel
from hansmeyer.printprep import PrintSettings, prepare_exact, write_stl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = {"w1": -1.0, "w2": -2.0}


def closed(m) -> bool:
    return bool(np.all(m.he_twin >= 0))


def design(**vessel_kw):
    its = [IterationSpec("cc", {**W, "w_f": 0.3}), IterationSpec("cc", {**W, "w_e": 0.4}),
           IterationSpec("cc", {**W, "w_f": -0.3}), IterationSpec("cc", W)]
    return Design(base={"shape": "sphere_open", "opening": 40.0}, boundary="locked", iterations=its,
                  vessel={"enabled": True, **vessel_kw})


def test_sphere_open_base():
    m = make_base(default_spec("sphere_open"))
    assert m.euler_characteristic() == 1  # a disc: one opening
    assert int((m.he_twin < 0).sum()) == 16  # one boundary loop of `sides` edges
    assert np.einsum("ij,ij->", m.face_area_vec, m.face_centroid) > 0  # outward
    rim = m.V[m.vert_is_boundary]
    assert np.allclose(rim[:, 1], np.cos(np.radians(40.0)))  # a circle at the requested opening


def test_vessel_is_a_clean_closed_shell():
    m = Pipeline(root=ROOT).run(design(min_wall_mm=0.8, max_wall_mm=3.0, rim_mm=2.5, diameter_mm=120.0), 4).mesh
    assert closed(m) and m.euler_characteristic() == 2 and m.signed_volume() > 0
    n = m.n_verts // 2
    outer, inner = m.V[:n], m.V[n:]
    assert np.allclose(np.linalg.norm(outer, axis=1), 1.0)  # an exact sphere outside
    wall = (1.0 - np.linalg.norm(inner, axis=1)) * 60.0  # mm at 120 mm diameter
    assert wall.min() > 0.79 and wall.max() < 3.01  # within the requested walls (the rim is 2.5 mm)
    rim_y = np.cos(np.radians(40.0))
    top = np.abs(m.V[:, 1] - rim_y) < 1e-9
    assert top.sum() >= 2 * 16  # outer and inner rim lie in one flat plane


def test_vessel_walls_follow_the_relief_and_light():
    m = Pipeline(root=ROOT).run(design(min_wall_mm=1.0, max_wall_mm=2.0), 4).mesh
    w = m.vattr["wall_mm"]
    assert w.min() == pytest.approx(1.0) and w.max() <= 2.5 + 1e-9  # body 1-2 mm, rim 2.5 mm
    light = vessel.light(m)
    assert light.min() >= 0 and light.max() <= 1
    wf = m.face_reduce(w[m.face_idx]) / m.face_size
    assert np.corrcoef(wf, light)[0, 1] < -0.9  # thinner wall = brighter


def test_relief_is_exactly_what_the_weights_build():
    pipe = Pipeline(root=ROOT)
    plain = Design(base={"shape": "sphere_open"}, boundary="locked", iterations=[IterationSpec("cc", {})] * 4,
                   vessel={"enabled": True, "min_wall_mm": 0.8})
    m = pipe.run(plain, 4).mesh
    rest = m.vattr["rest"]
    theta = np.degrees(np.arccos(rest[:, 1] / np.linalg.norm(rest, axis=1)))
    body = theta > 40.0 + 10.0 + 1.0  # past the opening and the band where the wall blends into the rim
    assert np.allclose(m.vattr["wall_mm"][body], 0.8)  # no weights: a uniform thin wall

    def walls(**kw):
        return pipe.run(design(**kw), 4).mesh.vattr["wall_mm"]

    shallow, deep = walls(depth=0.5, max_wall_mm=30.0), walls(depth=2.0, max_wall_mm=30.0)
    assert deep.mean() > shallow.mean() + 1.0  # deeper relief = thicker on average
    for glow in (5.0, 30.0):
        share = (walls(glow=glow) <= 0.8 + 1e-9).mean()  # (outer and inner list the same walls)
        assert share == pytest.approx(glow / 100, abs=0.05)


def test_vessel_round_trips_through_presets(tmp_path):
    d = design(diameter_mm=90.0)
    p = tmp_path / "v.json"
    d.save(str(p))
    assert Design.load(str(p)).vessel["diameter_mm"] == 90.0
    assert "vessel" not in Design().to_dict()  # off by default and not written


def test_exact_print_upside_down_on_a_flat_rim(tmp_path):
    m = Pipeline(root=ROOT).run(design(diameter_mm=100.0), 4).mesh
    r = prepare_exact(m, PrintSettings(up="-y"), 50.0)
    assert r.dims_mm[0] == pytest.approx(100.0, abs=0.2) and r.dims_mm[1] == pytest.approx(100.0, abs=0.2)
    assert r.dims_mm[2] == pytest.approx(50.0 * (1 + np.cos(np.radians(40.0))), abs=0.05)
    assert (r.V[:, 2] < 1e-6).sum() >= 32  # the rim ring stands on the plate
    info = write_stl(str(tmp_path / "v.stl"), r)
    assert info["complete"] and info["watertight"]
    assert r.overhang(30)[2] < 0.05  # 40 deg opening: almost nothing needs support


def test_lithophane_preset():
    d = Design.load(os.path.join(ROOT, "presets", "lithophane_sphere.json"))
    m = Pipeline(root=ROOT).run(d, d.preview_depth).mesh
    assert vessel.active(d) and closed(m) and m.euler_characteristic() == 2

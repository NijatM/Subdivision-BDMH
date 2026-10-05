"""Milestone 3: attractors (paper eq. 8-9, modifiers, falloffs, curves)."""

import numpy as np
import pytest

from hansmeyer import Design, IterationSpec, Pipeline, catmull_clark, default_spec, make_base
from hansmeyer.attractors import (
    blend,
    curve_points,
    distance_to_polyline,
    face_positions,
    falloff,
    influence,
    load_polyline,
    make_provider,
    normalize,
    to_polyline,
)

SHARP = {"w1": -1.0, "w2": -2.0}


def point_set(pos, w_by_iter, strength=1.0, radius=2.0, **kw):
    return {"kind": "point", "position": list(pos), "payload": "set", "strength": strength,
            "radius": radius, "weights": w_by_iter, **kw}


# ------------------------------------------------------------------ falloff
@pytest.mark.parametrize("kind", ["power", "linear", "smoothstep", "gaussian"])
def test_falloff_shape(kind):
    u = np.linspace(0, 1.5, 61)
    f = falloff(u, {"type": kind, "tightness": 2.0})
    assert f[0] == pytest.approx(1.0)
    assert np.all(np.diff(f) <= 1e-12), "monotone non-increasing"
    if kind != "gaussian":
        assert np.all(f[u >= 1] == 0)


def test_power_falloff_is_paper_eq8_numerator():
    u = np.array([0.0, 0.25, 0.5, 0.9])
    assert np.allclose(falloff(u, {"type": "power", "tightness": 3.0}), (1 - u) ** 3)


def test_spline_falloff_interpolates_knots():
    vals = [1.0, 0.2, 0.9, 0.1, 0.4]
    f = falloff(np.linspace(0, 1, 5), {"type": "spline", "spline": vals})
    assert np.allclose(f, vals)
    assert falloff(np.array([3.0]), {"type": "spline", "spline": vals})[0] == pytest.approx(0.4)


# ------------------------------------------------------------------- curves
def test_distance_to_polyline_matches_dense_sampling():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(7, 3))
    P = rng.normal(size=(200, 3)) * 2
    t = np.linspace(0, 1, 4001)[:, None]
    dense = np.concatenate([Q[i] + (Q[i + 1] - Q[i]) * t for i in range(len(Q) - 1)])
    brute = np.sqrt(((P[:, None] - dense[None]) ** 2).sum(-1)).min(1)
    assert np.allclose(distance_to_polyline(P, Q), brute, atol=2e-3)


@pytest.mark.parametrize("kind", ["helix", "sine", "lissajous"])
def test_fast_curve_distance_error_bound(kind):
    from hansmeyer.attractors import _segment_distance

    Q = curve_points(normalize({"kind": "curve", "curve": {"type": kind}})["curve"])
    P = np.random.default_rng(1).uniform(-3, 3, size=(20000, 3))
    A, AB = Q[:-1], Q[1:] - Q[:-1]
    exact = _segment_distance(P, A, AB, (AB * AB).sum(1))
    fast = distance_to_polyline(P, Q)
    half_seg = 0.5 * np.linalg.norm(AB, axis=1).max()
    assert np.all(fast >= exact - 1e-12) and np.all(fast - exact <= half_seg + 1e-12)


def test_circle_and_helix_geometry():
    c = normalize({"kind": "curve", "curve": {"type": "circle", "center": [1, 2, 3], "radius": 0.7, "axis": "x"}})["curve"]
    P = curve_points(c)
    assert np.allclose(np.linalg.norm(P[:, 1:] - [2, 3], axis=1), 0.7)  # circle around the x axis
    assert np.allclose(P[:, 0], 1)
    assert np.allclose(P[0], P[-1])  # closed
    h = normalize({"kind": "curve", "curve": {"type": "helix", "height": 4.0, "radius": 1.0, "axis": "y"}})["curve"]
    H = curve_points(h)
    assert H[:, 1].min() == pytest.approx(-2) and H[:, 1].max() == pytest.approx(2)


def test_to_polyline_keeps_shape():
    c = normalize({"kind": "curve", "curve": {"type": "circle", "radius": 1.0}})["curve"]
    p = to_polyline(c, n=12)
    assert p["type"] == "polyline" and len(p["points"]) == 12 and p["closed"]
    assert np.allclose(np.linalg.norm(np.array(p["points"])[:, [0, 2]], axis=1), 1.0, atol=1e-3)


def test_load_polyline_formats(tmp_path):
    csv = tmp_path / "pts.csv"
    csv.write_text("x,y,z\n0,0,0\n1,0,0\n1,1,0\n")
    assert load_polyline(str(csv)).shape == (3, 3)
    obj = tmp_path / "c.obj"
    obj.write_text("v 0 0 0\nv 2 0 0\nv 1 0 0\nl 1 3 2\n")
    assert np.allclose(load_polyline(str(obj))[:, 0], [0, 1, 2])  # ordered by the `l` element


# ------------------------------------------------------------- eq. 8 / eq. 9
def test_eq8_eq9_blend_is_partition_of_unity():
    m = make_base(default_spec("cube"))
    A = point_set([1.5, 0, 0], [{"w_f": 1.0}], radius=3.0)
    B = point_set([-1.5, 0, 0], [{"w_f": -1.0}], radius=3.0, strength=2.0)
    d = Design(attractors=[A, B], background=0.0)
    W = blend(d, m, 0, IterationSpec().weights)
    P = face_positions(m, "current")
    ra = influence(d.attractors[0], P)
    rb = influence(d.attractors[1], P)
    expected = (ra * 1.0 + rb * -1.0) / (ra + rb)
    assert np.allclose(W["w_f"], expected)
    assert W["w_f"][np.argmax(P[:, 0])] > 0.5 and W["w_f"][np.argmin(P[:, 0])] < -0.5


def test_background_blends_main_schedule():
    m = make_base(default_spec("cube"))
    A = point_set([10, 0, 0], [{"w_f": 1.0}], radius=1.0)  # out of reach of every face
    d = Design(attractors=[A], background=1.0)
    W = blend(d, m, 0, {**IterationSpec().weights, "w_f": 0.3})
    assert np.allclose(W["w_f"], 0.3)


def test_no_attractors_gives_identical_result():
    its = [IterationSpec("cc", {"w_f": 0.3, "w_c": -0.4, **SHARP}) for _ in range(4)]
    a = Pipeline().run(Design(iterations=its), 4).mesh
    plain = Pipeline().run(Design(iterations=its), 4, weight_provider=lambda m, l, s: s.weights).mesh
    assert np.array_equal(a.V, plain.V)
    off = Design(iterations=its, attractors=[point_set([0, 0, 0], [{"w_f": 2}], enabled=False)])
    assert np.array_equal(Pipeline().run(off, 4).mesh.V, a.V)


def test_attractor_breaks_symmetry_locally():
    its = [IterationSpec("cc", dict(SHARP)) for _ in range(4)]
    A = point_set([1.5, 0, 0], [{"w_f": 0.6}] * 4, radius=2.0, strength=3.0)
    m = Pipeline().run(Design(iterations=its, attractors=[A]), 4).mesh
    plain = Pipeline().run(Design(iterations=its), 4).mesh
    assert m.V[:, 0].max() > plain.V[:, 0].max() + 0.2  # +x side grows
    assert np.isclose(m.V[:, 0].min(), plain.V[:, 0].min(), atol=0.02)  # -x side (out of reach) unchanged


def test_single_set_without_background_is_uniform_inside_radius():
    """Paper-pure eq. 8 normalises a lone set to influence 1 wherever it reaches."""
    m = make_base(default_spec("cube"))
    d = Design(attractors=[point_set([1.0, 0, 0], [{"w_f": 0.5}], radius=2.5)], background=0.0)
    W = blend(d, m, 0, IterationSpec().weights)
    reached = influence(d.attractors[0], face_positions(m, "current")) > 0
    assert np.allclose(W["w_f"][reached], 0.5) and np.allclose(W["w_f"][~reached], 0.0)


# ---------------------------------------------------------------- modifiers
def test_modifier_scale_and_offset():
    m = make_base(default_spec("cube"))
    mod = {"kind": "point", "position": [1, 0, 0], "payload": "modifier", "radius": 1.5,
           "mods": [{"weight": "w_f", "op": "scale", "value": 3.0, "from": 1, "to": 1},
                    {"weight": "w_e", "op": "offset", "value": 0.5, "from": 2, "to": 3}]}
    d = Design(attractors=[mod])
    base = {**IterationSpec().weights, "w_f": 0.2, "w_e": 0.1}
    W0 = blend(d, m, 0, base)
    P = face_positions(m, "current")
    near, far = np.argmax(P[:, 0]), np.argmin(P[:, 0])
    assert W0["w_f"][near] == pytest.approx(0.2 * 3.0)  # at the attractor: full effect
    assert W0["w_f"][far] == pytest.approx(0.2)  # out of range
    assert np.allclose(W0["w_e"], 0.1)  # iteration 1 not in w_e's range
    W1 = blend(d, m, 1, base)
    assert W1["w_e"][near] == pytest.approx(0.6) and np.allclose(W1["w_f"], 0.2)


# ------------------------------------------------------------ rest position
def test_rest_positions_are_carried():
    d = Design(iterations=[IterationSpec("cc", {"w_f": 0.5, **SHARP})] * 2)
    m = Pipeline().run(d, 2).mesh
    assert m.vattr["rest"].shape == m.V.shape
    assert np.abs(m.V).max() > np.abs(m.vattr["rest"]).max() + 0.2  # extrusion moved V, not rest


def test_rest_space_changes_result():
    its = [IterationSpec("cc", {"w_f": 0.6, **SHARP})] + [IterationSpec("cc", dict(SHARP))] * 3
    A = point_set([2.4, 0, 0], [{}, {"w_f": 0.8}, {"w_f": 0.8}, {"w_f": 0.8}], radius=1.2)
    cur = Pipeline().run(Design(iterations=its, attractors=[A], attractor_space="current"), 4).mesh
    rest = Pipeline().run(Design(iterations=its, attractors=[A], attractor_space="rest"), 4).mesh
    assert not np.allclose(cur.V, rest.V)


# ------------------------------------------------------------------- caching
def test_cache_only_recomputes_from_edited_iteration():
    its = [IterationSpec("cc", dict(SHARP)) for _ in range(6)]
    A = point_set([1.5, 0, 0], [{"w_f": 0.3}] * 6)
    d = Design(iterations=its, attractors=[A])
    p = Pipeline()
    p.run(d, 6)
    d.attractors[0]["weights"][3]["w_f"] = 0.7  # iteration 4
    assert p.run(d, 6).cache_hits == 3
    d.attractors[0]["position"] = [0, 1.5, 0]  # moving the attractor affects every level
    assert p.run(d, 6).cache_hits == 0


def test_inactive_modifier_does_not_invalidate_early_levels():
    its = [IterationSpec("cc", dict(SHARP)) for _ in range(6)]
    mod = {"kind": "point", "payload": "modifier", "mods": [{"weight": "w_f", "op": "offset", "value": 0.3, "from": 5, "to": 6}]}
    d = Design(iterations=its, attractors=[mod])
    p = Pipeline()
    p.run(d, 6)
    d.attractors[0]["position"] = [0, 2, 0]
    assert p.run(d, 6).cache_hits == 4


# ---------------------------------------------------------------------- json
def test_design_roundtrip_with_attractors(tmp_path):
    curve = {"kind": "curve", "payload": "modifier", "curve": {"type": "sine", "amplitude": 0.3},
             "falloff": {"type": "gaussian"}, "mods": [{"weight": "w_e", "op": "offset", "value": 0.2}]}
    d = Design(attractors=[point_set([0, 1, 0], [{"w_f": 0.4}, {"w_c": -0.2}]), curve],
               attractor_space="rest", background=0.5)
    path = tmp_path / "a.json"
    d.save(str(path))
    d2 = Design.load(str(path))
    assert d2.to_dict() == d.to_dict()
    assert d2.attractors[0]["weights"][1]["w_c"] == -0.2
    assert d2.attractors[1]["curve"]["type"] == "sine"


def test_provider_returns_scalars_without_attractors():
    spec = IterationSpec("cc", {"w_f": 0.1})
    prov = make_provider(Design())
    assert prov(make_base(default_spec("cube")), 0, spec) is spec.weights


def test_curve_attractor_runs_on_column():
    helix = {"kind": "curve", "payload": "modifier", "radius": 0.6, "curve": {"type": "helix", "radius": 0.6, "height": 5.0},
             "mods": [{"weight": "w_f", "op": "offset", "value": 0.5}]}
    d = Design(base=default_spec("column_classic"), iterations=[IterationSpec("cc", dict(SHARP))] * 3, attractors=[helix])
    m = Pipeline().run(d, 3).mesh
    assert np.all(np.isfinite(m.V)) and m.euler_characteristic() == 2

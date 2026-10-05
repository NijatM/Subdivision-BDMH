"""Milestone 4: function library, layer stack, plug-ins."""

import os
import textwrap

import numpy as np
import pytest

from hansmeyer import Design, IterationSpec, Pipeline, default_spec, make_base
from hansmeyer import functions as F
from hansmeyer.layers import apply_weight_layers, post_process
from hansmeyer.noise import perlin, worley

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARP = {"w1": -1.0, "w2": -2.0}
P = np.random.default_rng(0).uniform(-2, 2, size=(4000, 3))


# --------------------------------------------------------------- functions
@pytest.mark.parametrize("name", [f.name for f in F.fields() if not f.source])
def test_fields_are_finite_bounded_and_deterministic(name):
    a = F.evaluate(name, P, {})
    b = F.evaluate(name, P, {})
    assert a.shape == (len(P),) and np.all(np.isfinite(a))
    assert np.array_equal(a, b)
    assert a.min() >= -1.0001 and a.max() <= 1.0001


@pytest.mark.parametrize("name", [f.name for f in F.folds() if not f.source])
def test_folds_return_positions(name):
    out = F.evaluate(name, P, {})
    assert out.shape == P.shape and np.all(np.isfinite(out))


def test_gyroid_matches_formula():
    p = np.array([[0.1, 0.2, 0.3]])
    x, y, z = 2 * np.pi * p[0]
    expected = (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)) / 1.5
    assert F.evaluate("gyroid", p, {"frequency": 1.0})[0] == pytest.approx(expected)


def test_perlin_is_continuous_and_zero_on_lattice():
    q = np.random.default_rng(1).uniform(-5, 5, size=(500, 3))
    assert np.abs(perlin(q) - perlin(q + 1e-4)).max() < 1e-2
    assert np.allclose(perlin(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])), 0.0)


def test_worley_f1_le_f2():
    f1, f2 = worley(P)
    assert np.all(f1 <= f2 + 1e-12) and np.all(f1 >= 0)


def test_box_fold_identity_inside():
    q = np.random.default_rng(2).uniform(-0.9, 0.9, size=(100, 3))
    assert np.allclose(F.evaluate("box_fold", q, {"limit": 1.0}), q)


def test_kaleido_keeps_radius_and_height():
    out = F.evaluate("kaleido", P, {"k": 6})
    assert np.allclose(np.hypot(out[:, 0], out[:, 2]), np.hypot(P[:, 0], P[:, 2]))
    assert np.allclose(out[:, 1], P[:, 1])
    ang = np.arctan2(out[:, 2], out[:, 0])
    assert ang.min() >= -1e-9 and ang.max() <= np.pi / 6 + 1e-9  # one wedge


def test_spherical_harmonic_symmetry():
    v = F.evaluate("spherical_harmonic", P, {"l": 2, "m": 0})
    assert np.allclose(v, F.evaluate("spherical_harmonic", -P, {"l": 2, "m": 0}))  # even l: antipodal symmetric


def test_domain_fold_changes_pattern():
    a = F.evaluate("gyroid", P, {})
    b = F.evaluate("gyroid", P, {}, {"fold": "mandelbox", "repeat": 3, "scale": 1.0, "c": 1.0})
    assert not np.allclose(a, b)


# ------------------------------------------------------------------ layers
def _cube_mesh():
    return make_base(default_spec("cube"))


@pytest.mark.parametrize("blend, expected", [
    ("add", lambda w, v: w + v), ("multiply", lambda w, v: w * (1 + v)), ("replace", lambda w, v: v),
    ("min", np.minimum), ("max", np.maximum)])
def test_weight_layer_blend_modes(blend, expected):
    m = _cube_mesh()
    d = Design(layers=[{"function": "rose", "params": {"k": 3}, "weight": "w_f", "blend": blend,
                        "amplitude": 0.5, "offset": 0.1, "space": "current"}])
    W = apply_weight_layers(d, m, 0, {**IterationSpec().weights, "w_f": 0.2})
    v = 0.5 * F.evaluate("rose", m.face_centroid, {"k": 3}) + 0.1
    assert np.allclose(W["w_f"], expected(np.full(m.n_faces, 0.2), v))


def test_layer_iteration_range_and_disable():
    m = _cube_mesh()
    d = Design(layers=[{"function": "gyroid", "weight": "w_e", "amplitude": 1.0, "from": 2, "to": 3}])
    base = IterationSpec().weights
    assert apply_weight_layers(d, m, 0, base) is base  # iteration 1: outside range
    assert np.any(apply_weight_layers(d, m, 1, base)["w_e"] != 0)
    d.layers[0]["enabled"] = False
    assert apply_weight_layers(d, m, 1, base) is base


def test_layer_mask_by_attractor():
    m = _cube_mesh()
    att = {"name": "spot", "kind": "point", "position": [1.0, 0, 0], "radius": 1.2, "payload": "modifier"}
    d = Design(attractors=[att], layers=[{"function": "schwarz_p", "weight": "w_f", "blend": "replace",
                                          "amplitude": 0.0, "offset": 0.7, "mask": "spot", "space": "current"}])
    W = apply_weight_layers(d, m, 0, IterationSpec().weights)
    x = m.face_centroid[:, 0]
    assert W["w_f"][np.argmax(x)] == pytest.approx(0.7)  # under the attractor: replaced
    assert W["w_f"][np.argmin(x)] == pytest.approx(0.0)  # out of reach: untouched


def test_displacement_layer_moves_along_normals():
    m = _cube_mesh()
    d = Design(layers=[{"target": "displacement", "function": "schwarz_p", "amplitude": 0.0, "offset": 0.25,
                        "space": "current"}])
    out = post_process(d, m, 0, relative=False)
    assert np.allclose(out.V - m.V, 0.25 * m.vert_normal)


def test_fold_layer_amount():
    m = _cube_mesh()
    d = Design(layers=[{"target": "fold", "function": "sphere_fold", "amplitude": 0.0}])
    assert np.allclose(post_process(d, m, 0).V, m.V)
    d.layers[0]["amplitude"] = 1.0
    assert np.allclose(post_process(d, m, 0).V, F.evaluate("sphere_fold", m.V, {}))


def test_layers_in_pipeline_and_cache():
    its = [IterationSpec("cc", dict(SHARP)) for _ in range(5)]
    d = Design(iterations=its, layers=[{"function": "gyroid", "params": {"frequency": 1.5}, "weight": "w_f",
                                        "amplitude": 0.4, "from": 3, "to": 5}])
    p = Pipeline()
    m = p.run(d, 5).mesh
    plain = Pipeline().run(Design(iterations=its), 5).mesh
    assert not np.allclose(m.V, plain.V) and m.euler_characteristic() == 2
    d.layers[0]["amplitude"] = 0.2
    assert p.run(d, 5).cache_hits == 2  # iterations 1-2 unaffected by a layer active from 3


def test_design_roundtrip_with_layers(tmp_path):
    d = Design(layers=[{"function": "worley", "params": {"mode": 1}, "target": "displacement", "amplitude": 0.1,
                        "domain": {"fold": "kaleido", "repeat": 1, "params": {"k": 5}}}])
    path = tmp_path / "l.json"
    d.save(str(path))
    assert Design.load(str(path)).to_dict() == d.to_dict()


# ---------------------------------------------------------------- plug-ins
def test_plugin_loading_and_errors(tmp_path):
    (tmp_path / "good.py").write_text(textwrap.dedent('''
        import numpy as np
        from hansmeyer.functions import field
        @field("stripes_test", family="plug-in", n=(1, 9, 3, "int"))
        def stripes(p, n):
            return np.sign(np.sin(n * p[:, 0]))
    '''))
    (tmp_path / "bad.py").write_text("raise RuntimeError('boom')\n")
    (tmp_path / "_ignored.py").write_text("raise RuntimeError('should not run')\n")
    errors = F.load_plugins(str(tmp_path))
    try:
        assert len(errors) == 1 and "bad.py" in errors[0]
        fd = F.REGISTRY["stripes_test"]
        assert fd.params[0].integer and fd.source.endswith("good.py")
        assert F.evaluate("stripes_test", P, {"n": 4}).shape == (len(P),)
    finally:
        F.load_plugins(os.path.join(ROOT, "functions"))  # restore the project's plug-ins
    assert "stripes_test" not in F.REGISTRY


def test_project_plugins_load():
    errors = F.load_plugins(os.path.join(ROOT, "functions"))
    assert errors == []
    assert F.REGISTRY["ripples"].kind == "field" and F.REGISTRY["twist"].kind == "fold"

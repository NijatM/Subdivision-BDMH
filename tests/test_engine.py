import glob
import os

import numpy as np
import pytest

from hansmeyer import Design, IterationSpec, Pipeline, PolyMesh, catmull_clark, doo_sabin
from hansmeyer.shapes import cube
from tests import reference as ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------- test meshes
def pyramid():
    V = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1.5]]
    F = [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    return PolyMesh.from_faces(V, F)


def grid(n=3):
    """Open n x n quad panel (has boundary edges)."""
    xs = np.linspace(-1, 1, n + 1)
    V = [[x, y, 0.1 * np.sin(3 * x) * np.cos(2 * y)] for y in xs for x in xs]
    F = [[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i] for j in range(n) for i in range(n)]
    return PolyMesh.from_faces(V, F)


def as_lists(m: PolyMesh):
    return [tuple(map(float, p)) for p in m.V], [list(map(int, f)) for f in m.faces_list()]


def same_mesh(m: PolyMesh, verts, faces, tol=1e-9):
    """Same positions (matched by nearest neighbour) and same oriented faces."""
    R = np.array(verts, float)
    if len(R) != m.n_verts or len(faces) != m.n_faces:
        return False
    d2 = ((m.V[:, None, :] - R[None, :, :]) ** 2).sum(-1)
    match = d2.argmin(axis=1)
    if d2[np.arange(m.n_verts), match].max() > tol**2 or len(set(match.tolist())) != m.n_verts:
        return False

    def canon(fs):
        out = set()
        for f in fs:
            f = list(f)
            r = f.index(min(f))
            out.add(tuple(f[r:] + f[:r]))
        return out

    return canon([match[f] for f in m.faces_list()]) == canon(faces)


MESHES = {"cube": cube, "pyramid": pyramid, "grid": grid}


# --------------------------------------------------- zero weights == standard
@pytest.mark.parametrize("name", list(MESHES))
def test_cc_zero_weights_match_reference(name):
    m = MESHES[name]()
    rv, rf = as_lists(m)
    for _ in range(3):
        m = catmull_clark.subdivide(m, {})
        rv, rf = ref.catmull_clark(rv, rf)
        assert same_mesh(m, rv, rf)


@pytest.mark.parametrize("name", ["cube", "pyramid"])
def test_ds_zero_weights_match_reference(name):
    m = MESHES[name]()
    rv, rf = as_lists(m)
    for _ in range(3):
        m = doo_sabin.subdivide(m, {})
        rv, rf = ref.doo_sabin(rv, rf)
        assert same_mesh(m, rv, rf)


def test_mixed_cc_ds_zero_weights_match_reference():
    m = pyramid()
    rv, rf = as_lists(m)
    for scheme in ["cc", "ds", "cc", "ds"]:
        if scheme == "cc":
            m, (rv, rf) = catmull_clark.subdivide(m, {}), ref.catmull_clark(rv, rf)
        else:
            m, (rv, rf) = doo_sabin.subdivide(m, {}), ref.doo_sabin(rv, rf)
        assert same_mesh(m, rv, rf)


def test_cube_corner_textbook_value():
    m = catmull_clark.subdivide(cube(2.0), {})
    corner = m.V[:8][np.argmax(m.V[:8].sum(axis=1))]
    assert np.allclose(corner, 5 / 9)


# ------------------------------------------------------------ counts/topology
def test_face_count_cube_depth_8():
    r = Pipeline().run(Design(), 8)
    assert r.mesh.n_faces == 6 * 4**8 == 393_216
    assert r.depth_reached == 8


def test_ds_face_count():
    m = cube()
    m2 = doo_sabin.subdivide(m, {})
    assert m2.n_faces == m.n_faces + m.n_edges + m.n_verts == 26


def test_topology_with_weights_mixed_schedule():
    rng = np.random.default_rng(1)
    its = []
    for k, s in enumerate(["cc", "ds", "cc", "cc", "ds", "cc"]):
        w = {n: float(rng.uniform(-0.4, 0.4)) for n in ["w_f", "w_e", "w_c", "w1", "w2", "w3", "w4",
                                                         "ds_w1_face", "ds_wf_face", "ds_w1_edge",
                                                         "ds_wf_edge", "ds_w1_vert", "ds_wf_vert"]}
        its.append(IterationSpec(s, w))
    r = Pipeline().run(Design(iterations=its), 6)
    m = r.mesh
    assert m.euler_characteristic() == 2  # genus 0 preserved
    _ = m.he_twin  # raises if non-manifold or inconsistently oriented
    assert np.all(m.he_twin >= 0)  # closed


# ------------------------------------------------------- paper equation checks
def test_eq2_w1_one_gives_face_point_average():
    m = cube()
    out = catmull_clark.subdivide(m, {"w1": 1.0})
    Fp = m.face_centroid
    ef = m.edge_faces
    E_pts = out.V[m.n_verts:m.n_verts + m.n_edges]
    assert np.allclose(E_pts, 0.5 * (Fp[ef[:, 0]] + Fp[ef[:, 1]]))


def test_eq3_reduces_to_textbook_formula_for_w2_zero():
    m = pyramid()
    out = catmull_clark.subdivide(m, {"w2": 0.0})
    rv, _ = ref.catmull_clark(*as_lists(m))
    assert np.allclose(out.V[:m.n_verts], np.array(rv[:m.n_verts]))


def test_eq4_face_point_after_cc():
    m1 = catmull_clark.subdivide(cube(), {})
    w3, w4 = 0.7, -0.3
    Fp = catmull_clark.face_points(m1, {"w3": w3, "w4": w4})
    f = 0
    vs = m1.faces_list()[f]  # [corner, edge, face, edge]
    Pv, Pe1, Pf, Pe2 = m1.V[vs]
    expected = ((Pv * (1 + w3) + Pf * (1 - w3)) * (1 + w4) + (Pe1 + Pe2) * (1 - w4)) / 4
    assert np.allclose(Fp[f], expected)


def test_eq5_doo_sabin_quad():
    V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    m = PolyMesh.from_faces(V, [[0, 1, 2, 3]])
    w1 = 0.2
    out = doo_sabin.subdivide(m, {"ds_w1_face": w1})
    P = np.array(V, float)
    expected = (P[0] * (2.25 + 2 * w1) + (P[1] + P[3]) * (0.75 - w1) + 0.25 * P[2]) / 4
    assert np.allclose(out.V[0], expected)


def test_eq6_doo_sabin_triangle():
    V = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    m = PolyMesh.from_faces(V, [[0, 1, 2]])
    w1 = -0.4
    out = doo_sabin.subdivide(m, {"ds_w1_face": w1})
    P = np.array(V, float)
    expected = 2 / 3 * P[0] * (1 + w1 / 2) + 1 / 6 * (P[1] + P[2]) * (1 - w1)
    assert np.allclose(out.V[0], expected)


def test_bias_weights_are_affine_invariant():
    w = {"w1": 0.6, "w2": -0.8, "w3": 0.3, "w4": 0.4}
    a = catmull_clark.subdivide(catmull_clark.subdivide(pyramid(), w), w)
    p = pyramid()
    p.V = p.V + np.array([3.0, -2.0, 5.0])
    b = catmull_clark.subdivide(catmull_clark.subdivide(p, w), w)
    assert np.allclose(a.V + np.array([3.0, -2.0, 5.0]), b.V)


def test_relative_extrusion_is_scale_invariant():
    w = {"w_f": 0.5, "w_e": -0.3, "w_c": 0.2}
    a = catmull_clark.subdivide(cube(2.0), w, relative=True)
    b = catmull_clark.subdivide(cube(6.0), w, relative=True)
    assert np.allclose(a.V * 3.0, b.V)
    c = catmull_clark.subdivide(cube(6.0), w, relative=False)
    assert not np.allclose(a.V * 3.0, c.V)


def test_face_extrusion_direction():
    m = cube()
    out = catmull_clark.subdivide(m, {"w_f": 0.5})
    Fp = out.V[m.n_verts + m.n_edges:]
    # face points pushed outward by 0.5 x edge length (2.0) = 1.0 beyond the face plane at 1.0
    assert np.allclose(np.abs(Fp).max(axis=1), 2.0)


# ------------------------------------------------------------------ pipeline
def test_cache_reuses_unchanged_levels():
    p = Pipeline()
    d = Design()
    p.run(d, 6)
    d.iterations[5].weights["w_f"] = 0.3
    r = p.run(d, 6)
    assert r.cache_hits == 5


def test_face_budget_caps_depth():
    p = Pipeline(face_budget=10_000)
    r = p.run(Design(), 8)
    assert r.capped_by_budget and r.depth_reached == 5 and r.mesh.n_faces == 6144


def test_design_json_roundtrip(tmp_path):
    d = Design(name="t", iterations=[IterationSpec("ds", {"ds_wf_face": 0.25}), IterationSpec("cc", {"w_f": -0.1})])
    path = tmp_path / "d.json"
    d.save(str(path))
    d2 = Design.load(str(path))
    assert d2.to_dict() == d.to_dict()


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "presets", "*.json"))))
def test_presets_load_and_run(path):
    d = Design.load(path)
    r = Pipeline().run(d, min(d.preview_depth, 5))
    assert r.mesh.n_faces > 0
    assert np.all(np.isfinite(r.mesh.V))

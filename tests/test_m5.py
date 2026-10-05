"""Milestone 5: groups & locking, motifs (eq. 10-11), intrinsic measures, vertex merging."""

import numpy as np
import pytest

from hansmeyer import Design, IterationSpec, Pipeline, PolyMesh, catmull_clark, default_spec, doo_sabin, make_base
from hansmeyer import intrinsic as I
from hansmeyer.merge import close_pairs, make_manifold, merge_vertices

SHARP = {"w1": -1.0, "w2": -2.0}


def decorated(name="cube", design=None, **spec):
    m = make_base({**default_spec(name), **spec})
    m.vattr["rest"] = m.V.copy()
    I.decorate_base(design or Design(), m)
    return m


# ------------------------------------------------------------------- motifs
def test_motif_labels_cube_and_after_cc():
    m = decorated()
    assert I.motif_counts(m) == {"3F3E": 8}
    m1 = catmull_clark.subdivide(m, {})
    assert I.motif_counts(m1) == {"3F3E": 8, "4F4E": 18}  # the two motifs CC adds (paper p. 290)


def test_motif_label_roundtrip():
    assert I.parse_label("12F6E") == 1206 and I.motif_label(1206) == "12F6E"


def test_eq10_face_attraction():
    m = decorated()
    U = np.zeros(m.n_verts)
    U[0] = 1.0  # one cube corner attracts
    w6 = 0.4
    base = catmull_clark.subdivide(m, {})
    att = catmull_clark.subdivide(m, {"w6": w6}, vU=U)
    F0 = base.V[m.n_verts + m.n_edges:]
    F1 = att.V[m.n_verts + m.n_edges:]
    for f, face in enumerate(m.faces_list()):
        expected = F0[f] + w6 * sum((m.V[v] - F0[f]) * U[v] for v in face)
        assert np.allclose(F1[f], expected)


def test_eq11_edge_attraction():
    m = decorated()
    U = np.linspace(-1, 1, m.n_verts)
    w7 = 0.3
    base = catmull_clark.subdivide(m, {})
    att = catmull_clark.subdivide(m, {"w7": w7}, vU=U)
    E0 = base.V[m.n_verts:m.n_verts + m.n_edges]
    E1 = att.V[m.n_verts:m.n_verts + m.n_edges]
    a, b = m.edge_verts[:, 0], m.edge_verts[:, 1]
    expected = E0 + w7 * ((m.V[a] - E0) * U[a][:, None] + (m.V[b] - E0) * U[b][:, None])
    assert np.allclose(E1, expected)


def test_motif_table_in_pipeline():
    its = [IterationSpec("cc", {**SHARP, "w6": 0.5, "w7": 0.5})] * 3
    plain = Pipeline().run(Design(iterations=its), 3).mesh
    pulled = Pipeline().run(Design(iterations=its, motifs={"3F3E": 1.0}), 3).mesh
    assert not np.allclose(plain.V, pulled.V)
    assert np.allclose(Pipeline().run(Design(iterations=its, motifs={"9F9E": 1.0}), 3).mesh.V, plain.V)


# ----------------------------------------------------------------- measures
def test_topological_distances_fig8():
    m1 = catmull_clark.subdivide(decorated(), {})
    dv = I.vertex_distance(m1, "dist_vertex")
    de = I.vertex_distance(m1, "dist_edge")
    N, E = 8, 12
    assert np.all(dv[:N] == 0) and np.all(dv[N:N + E] == 1) and np.all(dv[N + E:] == 2)
    assert np.all(de[:N + E] == 0) and np.all(de[N + E:] == 1)


def test_original_edge_marker_survives_cc_and_clears_after_ds():
    m = catmull_clark.subdivide(catmull_clark.subdivide(decorated(), {}), {})
    on_edge = m.vattr["te"] > 0.5
    rest = m.vattr["rest"]
    # every marked vertex lies on a cube edge: two coordinates at +-1
    assert np.all((np.isclose(np.abs(rest[on_edge]), 1.0)).sum(1) >= 2)
    assert np.all(doo_sabin.subdivide(m, {}).vattr["te"] == 0)


def test_planarity_and_bend():
    flat = decorated("panel")
    assert np.allclose(flat.face_planarity, 0) and np.allclose(flat.face_bend, 0)
    cube = decorated()
    assert np.allclose(cube.face_bend, 1.0)  # every neighbour at 90 degrees
    twisted = PolyMesh.from_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0.5], [0, 1, 0]], [[0, 1, 2, 3]])
    assert twisted.face_planarity[0] > 0.1


def test_intrinsic_rule_set_and_scale():
    m1 = catmull_clark.subdivide(decorated(), {})
    d = Design(intrinsic=[{"measure": "dist_vertex", "weight": "w_f", "op": "set", "a": 0.0, "b": 1.0}])
    W = I.apply_rules(d, m1, 1, IterationSpec().weights)
    t = I.measure(m1, "dist_vertex")
    assert np.allclose(W["w_f"], t) and t.min() >= 0 and t.max() <= 1
    d.intrinsic[0].update(op="scale", a=2.0, b=2.0)
    W = I.apply_rules(d, m1, 1, {**IterationSpec().weights, "w_f": 0.1})
    assert np.allclose(W["w_f"], 0.2)


# ------------------------------------------------------------------- groups
def test_tags_follow_children():
    d = Design(groups=[{"name": "top", "faces": [3]}])
    m = decorated(design=d)
    c = catmull_clark.subdivide(m, {})
    assert np.sum(c.fattr["tags"] & 1) == 4  # the tagged quad became four tagged quads
    s = doo_sabin.subdivide(c, {})
    tagged = s.fattr["tags"] & 1
    assert tagged.sum() >= 4 and len(tagged) == s.n_faces


def test_group_rules_only_touch_tagged_faces():
    d = Design(groups=[{"name": "g", "faces": [0, 1], "rules": [{"weight": "w_f", "op": "offset", "value": 0.5}]}])
    m = decorated(design=d)
    W = I.apply_group_rules(d, m, 0, {**IterationSpec().weights, "w_f": 0.1})
    assert np.allclose(W["w_f"][[0, 1]], 0.6) and np.allclose(W["w_f"][2:], 0.1)


def test_rule_based_selection():
    cube = decorated()
    up = I.select_by_normal(cube, "+y", 10)
    assert len(up) == 1 and np.allclose(cube.face_normal[up[0]], [0, 1, 0])
    assert len(I.select_by_height(cube, 1, 0.9, 1.0, faces=False)) == 4
    assert list(I.select_every_kth(10, 3)) == [0, 3, 6, 9]
    assert len(I.select_by_motif(cube, "3F3E")) == 8


def test_locked_edge_stays_a_sharp_crease_fig9():
    cube = decorated()
    edge_verts = cube.edge_verts[0]
    d = Design(groups=[{"name": "edge", "verts": edge_verts.tolist(), "lock": 4}],
               iterations=[IterationSpec("cc", {**SHARP, "w_f": 0.3})] * 5)
    m4 = Pipeline().run(d, 4).mesh
    a, b = cube.V[edge_verts[0]], cube.V[edge_verts[1]]
    assert np.allclose(m4.V[edge_verts], cube.V[edge_verts])  # endpoints never moved
    on = m4.vattr["lock"] > 0
    assert on.sum() == 2 ** 4 + 1  # the locked edge was split 4 times
    P = m4.V[on]
    t = (P - a) @ (b - a) / np.dot(b - a, b - a)
    assert np.allclose(P, a + t[:, None] * (b - a))  # all still on the straight crease
    m5 = Pipeline().run(d, 5).mesh  # lock = 4: released in iteration 5
    assert not np.allclose(m5.V[edge_verts], cube.V[edge_verts])


def test_groups_change_base_key():
    d = Design()
    k = d.base_key()
    d.groups = [I.normalize_group({"faces": [1], "lock": 2, "lock_faces": True})]
    assert d.base_key() != k


# ------------------------------------------------------------------ merging
def two_quads_sharing_an_edge_position():
    V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [1, 0, 0], [2, 0, 0], [2, 1, 0], [1.0001, 1, 0]]  # second quad's left edge duplicates x = 1
    return PolyMesh.from_faces(V, [[0, 1, 2, 3], [4, 5, 6, 7]])


def test_close_pairs():
    pairs = close_pairs(two_quads_sharing_an_edge_position().V, 0.01)
    assert {tuple(p) for p in pairs.tolist()} == {(1, 4), (2, 7)}


def test_merge_is_pairwise():
    V = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], float)  # a chain of three close points
    m = PolyMesh.from_faces(np.concatenate([V, V + [0, 1, 0], V + [0, 0, 1]]),
                            [[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    _, stats = merge_vertices(m, 0.015)
    assert stats["merged"] <= 3  # pairs only: no transitive chains


def test_merge_welds_into_manifold_strip():
    m, stats = merge_vertices(two_quads_sharing_an_edge_position(), 0.01)
    assert stats["merged"] == 2 and m.n_verts == 6 and m.n_faces == 2
    assert np.sum(m.he_twin >= 0) == 2  # the welded edge is now shared


def test_merge_max_valence_makes_holes():
    m, stats = merge_vertices(two_quads_sharing_an_edge_position(), 0.01, max_valence=2)
    assert stats["dropped"] >= 1 and m.n_faces == 1


def test_make_manifold_drops_third_face():
    faces = [np.array([0, 1, 2]), np.array([1, 0, 3]), np.array([0, 1, 4])]  # 0->1 used twice
    keep = make_manifold(faces, np.ones(3, bool))
    assert keep.tolist() == [True, True, False]


def test_merge_ignores_topological_neighbours():
    from hansmeyer.merge import _share_a_face

    m = catmull_clark.subdivide(make_base(default_spec("cube")), {})
    allp = np.array([(i, j) for i in range(m.n_verts) for j in range(i + 1, m.n_verts)])
    d = np.linalg.norm(m.V[allp[:, 0]] - m.V[allp[:, 1]], axis=1)
    share = _share_a_face(m, allp)
    gap = d[~share].min()  # nearest pair that is NOT topologically adjacent
    assert d[share].min() < gap  # neighbours are closer than that...
    out, stats = merge_vertices(m, 0.99 * gap)
    assert stats["merged"] == 0 and out.n_faces == 24  # ...but are never welded


def test_merge_in_pipeline_keeps_mesh_valid():
    # Fig. 3-left recipe: the cube corners are pulled into the centre, where sheets collide
    its = [IterationSpec("cc", {**SHARP, "w_c": -1.2, "w_f": 0.3}), IterationSpec("cc", {**SHARP, "w_c": -0.4})] + \
          [IterationSpec("cc", {**SHARP, "w_f": 0.4})] * 3
    d = Design(iterations=its, merge={"enabled": True, "distance": 0.1, "relative": True, "max_valence": 6,
                                      "from": 2, "to": 5})
    m = Pipeline().run(d, 5).mesh
    _ = m.he_twin  # raises if non-manifold / inconsistently oriented
    assert np.all(np.isfinite(m.V)) and m.n_faces > 1000
    assert m.info["merge"]["merged"] > 0
    assert m.euler_characteristic() != 2 or m.edge_is_boundary.any()  # topology changed: porosity
    # the result can still be subdivided further
    from hansmeyer import catmull_clark as cc
    assert cc.subdivide(m, {}).n_faces == m.n_halfedges

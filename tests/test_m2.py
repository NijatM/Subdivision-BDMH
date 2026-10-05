"""Milestone 2: base shapes, boundaries, attributes, mesh I/O."""

import os
import struct

import numpy as np
import pytest

from hansmeyer import Design, IterationSpec, Pipeline, PolyMesh, catmull_clark, default_spec, doo_sabin, make_base
from hansmeyer.meshio import MeshImportError, clean_mesh, load_obj, orient_faces, read_obj, write_obj, write_ply, write_stl
from hansmeyer.shapes import SHAPES, UNIT_RADIUS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOSED = [n for n in SHAPES if n not in ("panel", "obj", "sphere_open")]  # open shapes are tested elsewhere


def rand_weights(rng, scale=0.3):
    names = ["w_f", "w_e", "w_c", "w1", "w2", "w3", "w4"]
    return {n: float(rng.uniform(-scale, scale)) for n in names}


# -------------------------------------------------------------------- shapes
@pytest.mark.parametrize("name", CLOSED)
def test_closed_shapes_are_valid_solids(name):
    m = make_base(default_spec(name))
    chi = -8 if name == "cage" else 2  # the cage is a frame with six openings (genus 5)
    assert np.all(m.he_twin >= 0), "closed"
    assert m.euler_characteristic() == chi
    assert m.signed_volume() > 0, "outward oriented"
    m2 = catmull_clark.subdivide(m, {})
    assert m2.euler_characteristic() == chi and m2.signed_volume() > 0


@pytest.mark.parametrize(
    "name, n_faces, size",
    [("tetrahedron", 4, 3), ("cube", 6, 4), ("octahedron", 8, 3), ("dodecahedron", 12, 5), ("icosahedron", 20, 3)],
)
def test_platonic_solids(name, n_faces, size):
    m = make_base(default_spec(name))
    assert m.n_faces == n_faces and np.all(m.face_size == size)
    r = np.linalg.norm(m.V, axis=1)
    assert np.allclose(r, r[0]), "all vertices on the circumsphere"
    assert np.allclose(m.edge_length, m.edge_length[0]), "regular"
    if name != "cube":
        assert np.isclose(r[0], UNIT_RADIUS)


def test_column_box_matches_fig4_description():
    m = make_base(default_spec("column_box"))
    assert m.n_faces == 14 and np.all(m.face_size == 4)  # elongated cube on a base
    m = make_base({**default_spec("column_box"), "shaft_segments": 4})
    assert m.n_faces == 14 + 4 * 3


def test_column_classic_has_profile():
    m = make_base(default_spec("column_classic"))
    radius_by_height = {}
    for p in m.V:
        radius_by_height.setdefault(round(p[1], 6), []).append(np.hypot(p[0], p[2]))
    radii = [np.mean(v) for _, v in sorted(radius_by_height.items())]
    shaft = radii[2:-3]
    assert max(radii) > max(shaft), "base/capital wider than the shaft"


def test_panel_is_open_and_faces_plus_z():
    m = make_base(default_spec("panel"))
    assert np.any(m.vert_is_boundary)
    assert m.euler_characteristic() == 1
    assert np.allclose(m.face_normal, [0, 0, 1])


def test_integer_params_are_rounded():
    m = make_base({"shape": "panel", "nx": 3.4, "ny": 2.6})
    assert m.n_faces == 3 * 3


# ----------------------------------------------------------------- boundaries
def test_boundary_distance():
    m = make_base({"shape": "panel", "nx": 4, "ny": 4})
    d = m.boundary_distance()
    assert d.min() == 0 and d.max() == 2  # centre vertex of a 4x4 grid is 2 hops in
    assert np.all(np.isinf(make_base(default_spec("cube")).boundary_distance()))


def _run(design, depth):
    return Pipeline(root=ROOT).run(design, depth).mesh


def test_locked_boundary_is_tileable():
    rng = np.random.default_rng(3)
    its = [IterationSpec("cc", rand_weights(rng, 0.5)) for _ in range(4)]
    d = Design(base={"shape": "panel", "nx": 3, "ny": 3}, boundary="locked", fade_rows=1.0, iterations=its)
    m = _run(d, 4)
    b = m.vert_is_boundary
    P = m.V[b]
    # every boundary vertex stays on the original square outline, in the panel plane
    assert np.allclose(P[:, 2], 0)
    assert np.all(np.isclose(np.abs(P[:, 0]), 1) | np.isclose(np.abs(P[:, 1]), 1))
    # opposite edges carry identical vertex distributions -> copies tile seamlessly
    left = np.sort(P[np.isclose(P[:, 0], -1), 1])
    right = np.sort(P[np.isclose(P[:, 0], 1), 1])
    assert np.allclose(left, right)
    # relief exists in the interior
    assert np.abs(m.V[:, 2]).max() > 0.05


def test_smooth_boundary_moves():
    rng = np.random.default_rng(3)
    its = [IterationSpec("cc", rand_weights(rng, 0.5)) for _ in range(3)]
    d = Design(base={"shape": "panel", "nx": 3, "ny": 3, "bend": 0.5}, boundary="smooth", iterations=its)
    m = _run(d, 3)
    P = m.V[m.vert_is_boundary]
    assert not np.all(np.isclose(np.abs(P[:, 0]), 1) | np.isclose(np.abs(P[:, 1]), 1))


def test_fade_suppresses_extrusion_near_boundary():
    base = make_base({"shape": "panel", "nx": 6, "ny": 6})
    bd = base.boundary_distance()
    from hansmeyer.pipeline import boundary_fade

    out = catmull_clark.subdivide(base, {"w_f": 0.5}, lock_boundary=True, vmask=boundary_fade(bd, 2.0))
    Fp = out.V[base.n_verts + base.n_edges:]
    face_bd = base.face_reduce(bd[base.face_idx]) / base.face_size
    assert np.allclose(Fp[face_bd < 0.6, 2], 0, atol=0.05)  # ring touching the boundary: (almost) flat
    full = 0.5 * (2.0 / 6)  # w_f x edge length of a 6x6 panel of width 2
    assert np.allclose(Fp[face_bd >= 2, 2], full)  # deep interior: full extrusion


def test_locked_mode_does_not_change_closed_meshes():
    its = [IterationSpec("cc", {"w_f": 0.3, "w_c": -0.4}) for _ in range(3)]
    a = _run(Design(iterations=its, boundary="smooth"), 3)
    b = _run(Design(iterations=its, boundary="locked"), 3)
    assert np.allclose(a.V, b.V)


# ----------------------------------------------------------------- attributes
def test_attributes_follow_subdivision():
    m = make_base(default_spec("cube"))
    m.vattr["x"] = m.V[:, 0].copy()
    m.vattr["p"] = m.V.copy()
    cc = catmull_clark.subdivide(m, {})
    assert cc.vattr["x"].shape == (cc.n_verts,) and cc.vattr["p"].shape == (cc.n_verts, 3)
    ds = doo_sabin.subdivide(m, {})
    assert np.allclose(ds.vattr["p"], ds.V)  # DS masks are affine, so a linear attribute == position


# --------------------------------------------------------------------- mesh io
CUBE_OBJ = """# cube with mixed index syntax
v -1 -1 -1
v -1 -1 1
v -1 1 -1
v -1 1 1
v 1 -1 -1
v 1 -1 1
v 1 1 -1
v 1 1 1
vt 0 0
vn 0 0 1
f 1/1/1 2/1/1 4/1/1 3/1/1
f 5//1 7//1 8//1 6//1
f 1 5 6 2
f 3 4 8 7
f -8 -6 -2 -4
f 2 6 8 4
"""


def test_read_obj_syntax(tmp_path):
    p = tmp_path / "c.obj"
    p.write_text(CUBE_OBJ)
    V, F = read_obj(str(p))
    assert V.shape == (8, 3) and len(F) == 6 and F[4] == [0, 2, 6, 4]


def test_load_obj_repairs_orientation(tmp_path):
    p = tmp_path / "c.obj"
    p.write_text(CUBE_OBJ)
    m = load_obj(str(p), normalize=False)
    assert m.signed_volume() == pytest.approx(8.0)
    assert np.all(m.he_twin >= 0)


def test_load_sample_input_welds():
    m = load_obj(os.path.join(ROOT, "inputs", "l_block.obj"))
    assert m.n_verts == 16 and m.n_faces == 14 and m.euler_characteristic() == 2
    assert m.signed_volume() > 0
    assert np.isclose(np.linalg.norm(m.V, axis=1).max(), UNIT_RADIUS)


def test_whole_mesh_inverted_gets_flipped():
    m = make_base(default_spec("cube"))
    faces = [list(f)[::-1] for f in m.faces_list()]
    fixed = orient_faces(m.V, faces)
    assert PolyMesh.from_faces(m.V, fixed).signed_volume() > 0


def test_non_manifold_is_rejected():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], float)
    faces = [[0, 1, 2], [1, 0, 3], [0, 1, 4]]  # three faces on edge 0-1
    with pytest.raises(MeshImportError, match="non-manifold"):
        orient_faces(V, faces)


def test_clean_mesh_drops_degenerate_and_duplicates():
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], float)
    faces = [[0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 2], [4, 1, 2, 3]]
    V2, F2 = clean_mesh(V, faces)
    assert len(V2) == 4 and len(F2) == 1
    assert np.allclose(V2[F2[0]], V[[0, 1, 2, 3]])  # same quad, same winding (indices renumbered)


def test_obj_roundtrip(tmp_path):
    m = make_base(default_spec("column_classic"))  # quads + octagon caps
    p = tmp_path / "rt.obj"
    write_obj(str(p), m)
    m2 = load_obj(str(p), normalize=False)
    assert m2.n_faces == m.n_faces and np.isclose(m2.signed_volume(), m.signed_volume(), rtol=1e-5)


def test_stl_and_ply_writers(tmp_path):
    m = make_base(default_spec("column_classic"))
    stl = tmp_path / "m.stl"
    write_stl(str(stl), m)
    n_tri = len(m.triangles())
    data = stl.read_bytes()
    assert struct.unpack("<I", data[80:84])[0] == n_tri and len(data) == 84 + 50 * n_tri
    ply = tmp_path / "m.ply"
    write_ply(str(ply), m)
    head = ply.read_bytes().split(b"end_header\n")[0].decode()
    assert f"element vertex {m.n_verts}" in head and f"element face {m.n_faces}" in head


def test_obj_base_reimports_when_file_changes(tmp_path):
    p = tmp_path / "c.obj"
    p.write_text(CUBE_OBJ)
    d = Design(base={"shape": "obj", "path": str(p), "normalize": False})
    k1 = d.base_key()
    os.utime(p, (1, 1))
    assert d.base_key() != k1


def test_design_roundtrip_with_new_fields(tmp_path):
    d = Design(base=default_spec("panel"), boundary="locked", fade_rows=3.0)
    path = tmp_path / "d.json"
    d.save(str(path))
    d2 = Design.load(str(path))
    assert d2.boundary == "locked" and d2.fade_rows == 3.0 and d2.base == d.base

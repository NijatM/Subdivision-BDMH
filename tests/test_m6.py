"""Milestone 6: watertight print preparation and turntable output."""

import os

import numpy as np
import pytest

from hansmeyer import Design, IterationSpec, Pipeline, PolyMesh, default_spec, make_base
from hansmeyer.printprep import PrintSettings, _barycentric, grid_for, prepare, taubin, to_print_coords, write_stl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARP = {"w1": -1.0, "w2": -2.0}


def closed(m: PolyMesh) -> bool:
    return bool(np.all(m.he_twin >= 0))  # also raises if non-manifold / inconsistently oriented


def run(preset_or_design, depth=4):
    d = Design.load(os.path.join(ROOT, "presets", preset_or_design + ".json")) if isinstance(preset_or_design, str) \
        else preset_or_design
    return Pipeline(root=ROOT).run(d, depth).mesh


FAST = dict(size_mm=40.0, voxel_mm=0.5, smooth=4)


def test_smooth_cube_prints_watertight_and_true_to_size():
    m = run(Design(iterations=[IterationSpec("cc")] * 3), 3)
    r = prepare(m, PrintSettings(**FAST))
    pm = r.mesh()
    assert closed(pm) and pm.euler_characteristic() == 2  # one closed genus-0 shell
    assert max(r.dims_mm) == pytest.approx(40.0, abs=2 * 0.5)
    assert min(r.V[:, 2]) == pytest.approx(0.0)  # standing on the build plate
    expected = m.signed_volume() * (40.0 / np.ptp(m.V, axis=0).max()) ** 3 / 1000
    assert r.volume_cm3 == pytest.approx(expected, rel=0.12)


def test_self_intersecting_form_becomes_one_closed_solid():
    m = run("fig3_left", 4)
    r = prepare(m, PrintSettings(**FAST))
    pm = r.mesh()
    assert closed(pm) and r.volume_cm3 > 0
    assert pm.signed_volume() > 0  # outward oriented


def test_open_panel_skin_and_base():
    m = run("panel_tile", 4)
    skin = prepare(m, PrintSettings(**{**FAST, "up": "z", "wall_mm": 1.5}))
    assert closed(skin.mesh())
    assert any("no enclosed volume" in n for n in skin.notes)
    assert skin.dims_mm[2] < 10  # a thin skin, not a block
    tile = prepare(m, PrintSettings(**{**FAST, "up": "z", "base": True, "base_mm": 3.0}))
    assert closed(tile.mesh())
    assert tile.volume_cm3 > 2 * skin.volume_cm3  # filled down to a solid plate
    bottom = tile.V[tile.V[:, 2] < 0.3]
    assert len(bottom) > 50  # a flat face on the plate


def test_up_axis_rotation():
    col = make_base(default_spec("column_classic"))
    P = to_print_coords(col.V, "y")
    assert np.ptp(P[:, 2]) == pytest.approx(np.ptp(col.V[:, 1]))  # model height -> printer Z
    assert np.allclose(to_print_coords(col.V, "z"), col.V)


def test_thin_features_are_flagged():
    sheet = make_base({"shape": "panel", "nx": 4, "ny": 4})  # zero-thickness sheet printed as a 0.6 mm skin
    r = prepare(sheet, PrintSettings(size_mm=30, voxel_mm=0.15, wall_mm=0.4, up="z", smooth=0, nozzle_mm=0.4))
    assert r.thin_fraction > 0.5 and any("thinner" in n for n in r.notes)
    cube = make_base(default_spec("cube"))
    r2 = prepare(cube, PrintSettings(size_mm=20, voxel_mm=0.4, smooth=0))
    assert r2.thin_fraction < 0.05


def test_voxel_cap():
    s = PrintSettings(size_mm=100, voxel_mm=0.01, max_grid=200)
    _, voxel, _, shape = grid_for(make_base(default_spec("cube")), s)
    assert voxel == pytest.approx(0.5) and max(shape) <= 200 + 20


def test_barycentric_grid_and_taubin():
    B = _barycentric(4)
    assert len(B) == 15 and np.allclose(B.sum(1), 1) and B.min() >= 0
    from hansmeyer import catmull_clark

    m = make_base(default_spec("icosahedron"))
    for _ in range(3):
        m = catmull_clark.subdivide(m, {})
    r0 = np.linalg.norm(m.V, axis=1).mean()
    V = taubin(m.V, m.triangles(), 20)
    assert np.linalg.norm(V, axis=1).mean() == pytest.approx(r0, rel=0.02)  # dense mesh: no shrinking


def test_stl_export(tmp_path):
    r = prepare(make_base(default_spec("cube")), PrintSettings(size_mm=20, voxel_mm=0.5, smooth=2))
    path = tmp_path / "p.stl"
    write_stl(str(path), r)
    assert path.stat().st_size == 84 + 50 * len(r.F)


def test_turntable_writes_gif(tmp_path):
    ps = pytest.importorskip("polyscope")
    try:
        from hansmeyer.view import setup_scene, show_mesh, turntable

        setup_scene((200, 200))
    except Exception as e:  # no display / OpenGL available
        pytest.skip(f"no OpenGL context: {e}")
    m = make_base(default_spec("cube"))
    show_mesh(m)
    path = turntable(str(tmp_path / "t.gif"), np.zeros(3), 1.7, frames=6, seconds=1, size=64)
    import imageio.v3 as iio

    frames = iio.imread(path, index=None)
    assert frames.shape[:3] == (6, 64, 64)
    assert not np.array_equal(frames[0], frames[1])
    ps.remove_all_structures()


# ------------------------------------------------------------- safe output
def test_validate_stl_detects_truncation_and_holes(tmp_path):
    from hansmeyer.meshio import validate_stl, write_stl as raw_stl

    r = prepare(make_base(default_spec("cube")), PrintSettings(size_mm=20, voxel_mm=0.5, smooth=2))
    good = tmp_path / "good.stl"
    info = write_stl(str(good), r)
    assert info == {"triangles": len(r.F), "complete": True, "watertight": True, "bad_edges": 0}
    cut = tmp_path / "cut.stl"
    cut.write_bytes(good.read_bytes()[:84])  # what a full disk left behind
    assert validate_stl(str(cut))["complete"] is False
    holed = tmp_path / "holed.stl"
    m = r.mesh()
    raw_stl(str(holed), PolyMesh(m.V, m.face_ptr[:-1], m.face_idx[:-3]))  # drop one triangle
    v = validate_stl(str(holed))
    assert v["complete"] and not v["watertight"] and v["bad_edges"] == 3


def test_export_refuses_when_disk_is_full(tmp_path, monkeypatch):
    import shutil

    from hansmeyer import meshio

    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda p: type(real(p))(10**12, 10**12 - 10**5, 10**5))
    path = tmp_path / "x.stl"
    with pytest.raises(meshio.DiskSpaceError, match="not enough disk space"):
        meshio.export(str(path), make_base(default_spec("cube")))
    assert not path.exists() and not (tmp_path / "x.stl.part").exists()


def test_atomic_write_leaves_no_partial_file(tmp_path):
    from hansmeyer.meshio import atomic_write

    target = tmp_path / "f.bin"
    target.write_bytes(b"old")

    def failing(tmp):
        open(tmp, "wb").write(b"half")
        raise OSError("disk full")

    with pytest.raises(OSError):
        atomic_write(str(target), failing)
    assert target.read_bytes() == b"old" and not (tmp_path / "f.bin.part").exists()


def test_print_prep_runs_in_a_worker_process():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from hansmeyer.printprep import prepare_arrays

    m = make_base(default_spec("cube"))
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        r = ex.submit(prepare_arrays, m.V, m.face_ptr, m.face_idx, {"size_mm": 20, "voxel_mm": 0.5}).result(timeout=120)
    assert closed(r.mesh()) and max(r.dims_mm) == pytest.approx(20, abs=1)


def test_half_detail_is_smaller_and_still_watertight():
    m = make_base(default_spec("cube"))
    full = prepare(m, PrintSettings(size_mm=30, voxel_mm=0.4))
    half = prepare(m, PrintSettings(size_mm=30, voxel_mm=0.4, detail=2))
    assert len(half.F) < 0.4 * len(full.F) and closed(half.mesh())


@pytest.mark.parametrize("detail", [1, 2])
def test_porous_form_prints_watertight_at_every_detail(detail, tmp_path):
    """Regression: a marching-cubes step of 2 used to leave non-manifold edges on porous forms."""
    r = prepare(run("porous_grotto", 5), PrintSettings(size_mm=60, voxel_mm=0.3, detail=detail))
    info = write_stl(str(tmp_path / "p.stl"), r)
    assert info["complete"] and info["watertight"] and closed(r.mesh())


def test_print_keeps_requested_size():
    m = run(Design(iterations=[IterationSpec("cc")] * 3), 3)
    for detail, tol in ((1, 0.35), (2, 0.7)):
        r = prepare(m, PrintSettings(size_mm=40, voxel_mm=0.3, smooth=4, detail=detail))
        assert max(r.dims_mm) == pytest.approx(40, abs=tol)

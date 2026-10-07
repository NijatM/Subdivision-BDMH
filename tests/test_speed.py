"""The speed-ups must not change any result: shared connectivity tables, the one-sort edge table, the Worley
cell table, the memory-capped level cache, and the shared-memory hand-over to the background process."""

import os
import pickle

import numpy as np
import pytest

from hansmeyer import Design, Pipeline, PolyMesh, background, catmull_clark, doo_sabin, noise
from hansmeyer.mesh import scatter_add
from tests.test_engine import ROOT, grid, pyramid


def _fresh(m: PolyMesh) -> PolyMesh:
    """The same mesh with nothing cached or shared."""
    return PolyMesh(m.V.copy(), m.face_ptr.copy(), m.face_idx.copy(), vtype=m.vtype, fclass=m.fclass)


@pytest.mark.parametrize("make", [pyramid, grid])
def test_edge_table_matches_definition(make):
    m = make()
    for _ in range(2):
        m = catmull_clark.subdivide(m, {"w_f": 0.2})
    # every half-edge's twin runs the other way along the same edge; edges numbered by sorted vertex pair
    he, tw = m.he_edge, m.he_twin
    has = tw >= 0
    assert np.array_equal(m.he_from[tw[has]], m.he_to[has]) and np.array_equal(he[tw[has]], he[has])
    pairs = np.sort(np.stack([m.he_from, m.he_to], 1), axis=1)
    keys = pairs[:, 0] * m.n_verts + pairs[:, 1]
    assert np.array_equal(np.unique(keys, return_inverse=True)[1].ravel(), he)
    lowest = np.full(m.n_edges, len(he))
    np.minimum.at(lowest, he, np.arange(len(he)))
    assert np.array_equal(m.edge_he, lowest)


def test_subdivision_shares_connectivity_and_matches_fresh_meshes():
    m = grid(4)
    a = catmull_clark.subdivide(m, {"w_f": 0.3})
    b = catmull_clark.subdivide(m, {"w_f": -0.1})
    assert a.topo is b.topo  # same input connectivity: the child tables are built once
    for mesh in (a, b):
        f = _fresh(mesh)
        for name in ("he_twin", "edge_verts", "edge_faces", "valence", "vert_is_boundary"):
            assert np.array_equal(getattr(mesh, name), getattr(f, name)), name
    ds = doo_sabin.subdivide(a, {"ds_wf_face": 0.1})
    assert np.array_equal(ds.face_idx, doo_sabin.subdivide(_fresh(a), {"ds_wf_face": 0.1}).face_idx)


def test_base_meshes_with_the_same_faces_share_tables():
    pipe = Pipeline()
    d = Design()
    a = pipe.base(d)
    d.base = {"shape": "cube", "size": 1.7}
    b = pipe.base(d)
    assert a is not b and a.topo is b.topo


def test_scatter_add_matches_add_at():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 50, 1000)
    vals = rng.normal(size=(1000, 3))
    ref = np.zeros((50, 3))
    np.add.at(ref, idx, vals)
    assert np.array_equal(scatter_add(idx, vals, 50), ref)


@pytest.mark.parametrize("scale", [0.5, 6.0, 300.0])
def test_worley_table_matches_direct(scale):
    p = np.random.default_rng(3).uniform(-1, 1, (4000, 3)) * scale
    f1, f2 = noise.worley(p, 5, 0.9)
    g1, g2 = noise._worley_direct(p, np.floor(p).astype(np.int64), 5, 0.9)
    assert np.array_equal(f1, g1) and np.array_equal(f2, g2)


def test_level_cache_stays_under_its_memory_cap():
    pipe = Pipeline(max_bytes=3_000_000)
    d = Design()
    for k in range(4):
        d.iterations[0].weights["w_f"] = 0.1 * k
        pipe.run(d, 6)
    assert pipe._bytes <= 3_000_000 or len(pipe._cache) == 1
    assert pipe._bytes == sum(pipe._sizes.values())


def test_mesh_pickles_without_its_tables():
    m = catmull_clark.subdivide(grid(3), {"w_f": 0.2})
    _ = m.he_twin, m.vert_normal  # fill the caches
    data = pickle.dumps(m)
    assert len(data) < 4 * (m.V.nbytes + m.face_idx.nbytes)
    back = pickle.loads(data)
    assert np.array_equal(back.V, m.V) and np.array_equal(back.he_twin, m.he_twin)


def test_shared_memory_round_trip():
    big = {"V": np.arange(300_000, dtype=float).reshape(-1, 3), "F": np.arange(400_000).reshape(-1, 4), "note": "x"}
    msg = background.share(big)
    assert msg[0] == "shm" and len(pickle.dumps(msg)) < 10_000  # only a small message goes through the pipe
    back = background.unshare(msg)
    background.release()
    assert np.array_equal(back["V"], big["V"]) and np.array_equal(back["F"], big["F"]) and back["note"] == "x"
    back["V"][0, 0] = -1.0  # a private, writable copy
    small = background.unshare(background.share([1, 2, 3]))
    assert small == [1, 2, 3]


def test_background_bake_matches_a_local_run(tmp_path):
    d = Design.load(os.path.join(ROOT, "presets", "cube_six_arms.json"))
    w = background.Worker()
    try:
        r = w.result(w.submit(background.bake, d.to_dict(), 4, 2_000_000, str(tmp_path)))
    finally:
        w.stop()
    local = Pipeline(root=str(tmp_path)).run(d, 4)
    assert np.array_equal(r.mesh.V, local.mesh.V) and np.array_equal(r.mesh.face_idx, local.mesh.face_idx)

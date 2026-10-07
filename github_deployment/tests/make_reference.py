"""Reference results from the desktop (Python) engine, for the web engine's parity test.

    python github_deployment/tests/make_reference.py OUT_DIR

writes, for every preset in presets/ (and a few extra designs), the mesh the Python pipeline computes at the
preset's preview depth: OUT_DIR/<name>.bin (float64 positions, then int32 face_ptr and face_idx) and a
manifest.json. Run tests/parity.mjs with the JavaScript shell afterwards (see tests/README.md).
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from hansmeyer import Design, Pipeline  # noqa: E402
from hansmeyer import functions  # noqa: E402

EXTRA = {
    # every scheme / feature combination the presets don't reach on their own
    "x_ds_only": {"base": {"shape": "dodecahedron"}, "iterations": [
        {"scheme": "ds", "weights": {"ds_w1_face": 0.4, "ds_wf_face": 0.2}},
        {"scheme": "ds", "weights": {"ds_w1_edge": -0.5, "ds_wf_vert": 0.3, "ds_wf_edge": -0.2}},
        {"scheme": "cc", "weights": {"w_f": 0.2, "w3": 1.0, "w4": -1.0}},
        {"scheme": "ds", "weights": {"ds_w1_vert": 0.7, "ds_wf_face": 0.1}}], "preview_depth": 4},
    "x_panel_smooth_ds": {"base": {"shape": "panel", "nx": 5, "ny": 3, "bend": 0.4}, "iterations": [
        {"scheme": "cc", "weights": {"w_f": 0.3, "w_e": 0.2, "w_c": -0.3}},
        {"scheme": "ds", "weights": {"ds_wf_face": 0.2}},
        {"scheme": "cc", "weights": {"w1": -1, "w2": -2, "w_f": 0.2}}], "preview_depth": 3},
    "x_obj_lblock": {"base": {"shape": "obj", "path": "inputs/l_block.obj"}, "iterations": [
        {"scheme": "cc", "weights": {"w_f": 0.3, "w_c": -0.5}}, {"scheme": "cc", "weights": {"w_e": 0.2}},
        {"scheme": "cc", "weights": {}}], "preview_depth": 3},
    "x_layers_all": {"base": {"shape": "icosahedron"}, "preview_depth": 4, "iterations": [
        {"scheme": "cc", "weights": {"w_f": 0.2}}, {"scheme": "cc", "weights": {"w_f": 0.1}},
        {"scheme": "cc", "weights": {}}, {"scheme": "cc", "weights": {}}],
        "attractors": [{"name": "A1", "kind": "curve", "payload": "set", "radius": 1.2,
                        "curve": {"type": "lissajous", "size": 1.2},
                        "falloff": {"type": "spline", "spline": [1, 0.9, 0.4, 0.1, 0]},
                        "weights": [{"w_f": 0.5}, {"w_e": 0.3}, {"w_c": -0.4}]}],
        "background": 0.5,
        "layers": [
            {"name": "L1", "target": "weight", "function": "schwarz_d", "weight": "w_e", "blend": "multiply", "amplitude": 0.5},
            {"name": "L2", "target": "weight", "function": "neovius", "weight": "w_c", "blend": "min", "amplitude": 0.4,
             "mask": "facing +y"},
            {"name": "L3", "target": "weight", "function": "ridged", "weight": "w_f", "blend": "max", "amplitude": 0.3,
             "space": "current", "params": {"octaves": 3, "seed": 5}},
            {"name": "L4", "target": "weight", "function": "spherical_harmonic", "weight": "w_f", "blend": "replace",
             "amplitude": 0.2, "mask": "A1", "params": {"l": 5, "m": -2}},
            {"name": "L5", "target": "displacement", "function": "superformula", "amplitude": 0.05, "from": 2, "to": 3},
            {"name": "L6", "target": "displacement", "function": "warped", "amplitude": 0.04, "from": 3, "to": 4,
             "params": {"seed": 3}},
            {"name": "L7", "target": "fold", "function": "sphere_fold", "amplitude": 0.2, "from": 2, "to": 2},
            {"name": "L8", "target": "weight", "function": "superquadric", "weight": "w_f", "amplitude": 0.2,
             "domain": {"fold": "mandelbox", "repeat": 3, "c": 1.0}},
            {"name": "L9", "target": "weight", "function": "perlin", "weight": "w_e", "amplitude": 0.2,
             "domain": {"fold": "kaleido", "repeat": 1, "params": {"k": 5}}},
            {"name": "L10", "target": "fold", "function": "box_fold", "amplitude": 0.1, "from": 4, "to": 4},
            {"name": "L11", "target": "weight", "function": "worley", "weight": "w_f", "amplitude": 0.2,
             "params": {"mode": 1, "jitter": 0.7, "seed": 11, "frequency": 3.0}}]},
    "x_groups_locks": {"base": {"shape": "cube"}, "preview_depth": 4,
                       "iterations": [{"scheme": "cc", "weights": {"w_f": 0.4, "w1": -1, "w2": -2}},
                                      {"scheme": "ds", "weights": {"ds_wf_face": 0.2}},
                                      {"scheme": "cc", "weights": {"w_f": 0.3}}, {"scheme": "cc", "weights": {"w_e": 0.2}}],
                       "groups": [{"name": "G1", "faces": [0, 2], "verts": [1, 6], "lock": 2, "lock_faces": True,
                                   "rules": [{"weight": "w_f", "op": "scale", "value": 3.0, "from": 1, "to": 4}]},
                                  {"name": "G2", "faces": [3, 5], "rules": [{"weight": "w_e", "op": "offset", "value": 0.4}]}],
                       "motifs": {"3F3E": 0.6, "4F4E": -0.3},
                       "intrinsic": [{"measure": "dist_vertex", "weight": "w_f", "op": "scale", "a": 0.5, "b": 2.0, "gamma": 1.5},
                                     {"measure": "bend", "weight": "w_e", "op": "add", "a": 0.0, "b": 0.3},
                                     {"measure": "planarity", "weight": "w_c", "op": "set", "a": -0.1, "b": 0.2}]},
    "x_merge": {"base": {"shape": "cube"}, "preview_depth": 4,
                "iterations": [{"scheme": "cc", "weights": {"w_f": 0.9, "w_c": -1.4, "w1": -1, "w2": -2}},
                               {"scheme": "cc", "weights": {"w_f": 0.8, "w1": -1, "w2": -2}},
                               {"scheme": "cc", "weights": {"w_f": 0.9, "w_e": 0.3, "w1": -1, "w2": -2}},
                               {"scheme": "cc", "weights": {"w_f": 0.5}}],
                "merge": {"enabled": True, "distance": 0.4, "relative": True, "max_valence": 5, "from": 2, "to": 4}},
}


def main(out):
    os.makedirs(out, exist_ok=True)
    functions.load_plugins(os.path.join(ROOT, "functions"))
    manifest = []
    items = [(os.path.splitext(os.path.basename(p))[0], json.load(open(p))) for p in sorted(glob.glob(os.path.join(ROOT, "presets", "*.json")))]
    items = [(n, d) for n, d in items if not n.startswith("_")] + list(EXTRA.items())
    for name, data in items:
        d = Design.from_dict(data)
        depth = d.preview_depth
        r = Pipeline(root=ROOT).run(d, depth)
        m = r.mesh
        with open(os.path.join(out, name + ".bin"), "wb") as fh:
            fh.write(np.ascontiguousarray(m.V, "<f8").tobytes())
            fh.write(np.ascontiguousarray(m.face_ptr, "<i4").tobytes())
            fh.write(np.ascontiguousarray(m.face_idx, "<i4").tobytes())
        manifest.append({"name": name, "design": data, "depth": depth, "reached": r.depth_reached,
                         "n_verts": int(m.n_verts), "n_faces": int(m.n_faces), "n_he": int(m.n_halfedges),
                         "info": {k: v for k, v in m.info.items()}})
        print(f"{name:28s} depth {r.depth_reached} {m.n_faces:8d} faces {r.seconds:.2f}s")
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, default=float)


if __name__ == "__main__":
    main(sys.argv[1])

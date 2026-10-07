"""Render presets to PNG without the interactive UI.

    python render.py presets/cube_six_arms.json             -> renders/cube_six_arms.png
    python render.py presets/*.json --depth 6 --montage all.png
    python render.py presets/cube_petal_flower.json --turntable renders/cube_petal_flower.gif   (or .mp4)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import polyscope as ps

from hansmeyer import Design, Pipeline, functions
from hansmeyer.view import THEMES, setup_scene, set_view, show_mesh, turntable, write_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("presets", nargs="+")
    ap.add_argument("--depth", type=int, default=None, help="default: the preset's full depth")
    ap.add_argument("--view", default=None, help="diagonal | front | top | three_quarter")
    ap.add_argument("--size", type=int, default=700)
    ap.add_argument("--out", default="renders")
    ap.add_argument("--montage", default=None, help="also combine all renders into one image")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--theme", default="dark", choices=["dark", "light"])
    ap.add_argument("--turntable", default=None, help="write an orbit animation (.gif or .mp4) of the first preset")
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    for err in functions.load_plugins(os.path.join(root, "functions")):
        print("plug-in error:", err)
    setup_scene((args.size, args.size), theme=args.theme, ssaa=2)  # stills: always supersampled
    ps.set_ground_plane_mode("none")
    os.makedirs(args.out, exist_ok=True)
    pipe = Pipeline(root=os.path.dirname(os.path.abspath(__file__)))
    tiles = []
    for path in args.presets:
        d = Design.load(path)
        depth = args.depth or d.full_depth
        r = pipe.run(d, depth)
        show_mesh(r.mesh)
        set_view(r.mesh, args.view or d.view)
        img = ps.screenshot_to_buffer(transparent_bg=False, include_UI=False)
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out, f"{name}.png")
        write_png(out, img)
        tiles.append(img)
        print(f"{out}: depth {r.depth_reached}, {r.mesh.n_faces:,} faces, {r.seconds:.2f}s")
        if args.turntable and path == args.presets[0]:
            lo, hi = r.mesh.V.min(0), r.mesh.V.max(0)
            turntable(args.turntable, 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo)), args.frames,
                      args.seconds, args.size)
            print("turntable:", args.turntable)

    if args.montage and tiles:
        h, w = tiles[0].shape[:2]
        cols = min(args.cols, len(tiles))
        rows = -(-len(tiles) // cols)
        bg = np.array(THEMES[args.theme]["screen"] + (255,))[: tiles[0].shape[2]]
        canvas = np.empty((rows * h, cols * w, tiles[0].shape[2]), np.uint8)
        canvas[:] = bg.astype(np.uint8)
        for i, t in enumerate(tiles):
            r_, c_ = divmod(i, cols)
            canvas[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = t[:h, :w]
        write_png(args.montage, canvas)
        print("montage:", args.montage)


if __name__ == "__main__":
    main()

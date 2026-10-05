"""Render presets to PNG without the interactive UI.

    python render.py presets/fig3_left.json                 -> renders/fig3_left.png
    python render.py presets/*.json --depth 6 --montage all.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import polyscope as ps

from hansmeyer import Design, Pipeline
from hansmeyer.view import setup_scene, set_view, show_mesh, write_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("presets", nargs="+")
    ap.add_argument("--depth", type=int, default=None, help="default: the preset's full depth")
    ap.add_argument("--view", default=None, help="diagonal | front | top | three_quarter")
    ap.add_argument("--size", type=int, default=700)
    ap.add_argument("--out", default="renders")
    ap.add_argument("--montage", default=None, help="also combine all renders into one image")
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    setup_scene((args.size, args.size))
    ps.set_ground_plane_mode("none")
    os.makedirs(args.out, exist_ok=True)
    pipe = Pipeline()
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

    if args.montage and tiles:
        h, w = tiles[0].shape[:2]
        cols = min(args.cols, len(tiles))
        rows = -(-len(tiles) // cols)
        canvas = np.full((rows * h, cols * w, tiles[0].shape[2]), 255, np.uint8)
        for i, t in enumerate(tiles):
            r_, c_ = divmod(i, cols)
            canvas[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = t[:h, :w]
        write_png(args.montage, canvas)
        print("montage:", args.montage)


if __name__ == "__main__":
    main()

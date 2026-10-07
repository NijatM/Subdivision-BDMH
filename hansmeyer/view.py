"""Shared Polyscope look (used by the app and the render tool)."""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np
import polyscope as ps

from .mesh import PolyMesh

MESH_NAME = "form"
MESH_COLOR = (0.84, 0.84, 0.83)
BACK_COLOR = (0.40, 0.40, 0.40)

# Polyscope tone-maps the background, so these are the values that land on screen as the site's #0b0b0b
# (dark) and its inverted paper white #f4f4f2 (light); measured from screenshots.
THEMES = {
    "dark": {"background": (0.00099, 0.00099, 0.00099), "ground": "shadow_only", "edge": (0.10, 0.10, 0.10),
             "screen": (11, 11, 11)},
    "light": {"background": (0.825, 0.825, 0.808), "ground": "shadow_only", "edge": (0.30, 0.30, 0.30),
              "screen": (244, 244, 242)},
}
# Monochrome scene helpers (attractors, cut planes, tagging overlays) that read on either background.
HELPERS = {
    "dark": {"main": (1.0, 1.0, 1.0), "set": (0.86, 0.86, 0.86), "mod": (0.56, 0.56, 0.56),
             "off": (0.28, 0.28, 0.28), "plane": (0.92, 0.92, 0.92), "base": (0.32, 0.32, 0.32)},
    "light": {"main": (0.0, 0.0, 0.0), "set": (0.14, 0.14, 0.14), "mod": (0.45, 0.45, 0.45),
              "off": (0.72, 0.72, 0.72), "plane": (0.10, 0.10, 0.10), "base": (0.70, 0.70, 0.70)},
}
_theme = {"name": "dark"}


def apply_scene_theme(name: str) -> None:
    """Background / ground plane. Safe to call outside an ImGui frame."""
    t = THEMES[name]
    _theme["name"] = name
    ps.set_background_color(t["background"])
    ps.set_ground_plane_mode(t["ground"])


def helper(kind: str) -> tuple:
    """Colour for a scene helper in the current theme: main | set | mod | off | plane | base."""
    return HELPERS[_theme["name"]][kind]


VIEWS = {
    # Fig. 3 looks down the cube's body diagonal (three-fold symmetry)
    "diagonal": (np.array([1.0, 1.0, 1.0]), "y_up"),
    "front": (np.array([0.0, 0.0, 1.0]), "y_up"),
    "top": (np.array([0.0, 1.0, 0.0001]), "z_up"),
    "three_quarter": (np.array([1.0, 0.6, 1.4]), "y_up"),
}


def setup_scene(window=(1280, 900), theme: str = "dark", title: str = "SubdivisionEngine_BDMH", ssaa: int | None = None):
    """ssaa: supersampling factor; by default 2 on ordinary screens and 1 on Retina screens, which already draw
    2 x 2 pixels per point (supersampling those as well renders 16 pixels per point, ~3x the GPU time per frame)."""
    ps.set_program_name(title)
    ps.set_window_size(*window)
    ps.set_up_dir("y_up")
    ps.init()
    if ssaa is None:
        (w, _), (bw, _) = ps.get_window_size(), ps.get_buffer_size()
        ssaa = 1 if bw >= 1.5 * max(w, 1) else 2
    ps.set_SSAA_factor(int(ssaa))
    apply_scene_theme(theme)


_shown = {"faces": None, "label": None}  # what the "form" structure was registered with


def show_mesh(mesh: PolyMesh, edges: bool | None = None, face_values: np.ndarray | None = None,
              vrange: tuple | None = None, label: str = "influence", cmap: str = "viridis"):
    """Register the form; optional per-face scalar (e.g. attractor influence) shown as a colour map.
    When only the positions changed (a slider step), the vertices are updated in place: much cheaper than
    registering the mesh again."""
    edges = mesh.n_faces <= 30_000 if edges is None else edges
    disp = mesh.display_faces()
    prev = _shown["faces"]  # (face_ptr, face_idx, vertex count)
    same = (ps.has_surface_mesh(MESH_NAME) and prev is not None and prev[2] == len(mesh.V)
            and (prev[1] is mesh.face_idx or (len(prev[1]) == len(mesh.face_idx) and np.array_equal(prev[0], mesh.face_ptr)
                                              and np.array_equal(prev[1], mesh.face_idx))))
    if same:
        s = ps.get_surface_mesh(MESH_NAME)
        s.update_vertex_positions(mesh.V)
        s.set_edge_width(0.6 if edges else 0.0)
        s.set_edge_color(THEMES[_theme["name"]]["edge"])
        if face_values is None or label != _shown["label"]:
            s.remove_all_quantities()
    else:
        s = ps.register_surface_mesh(
            MESH_NAME,
            mesh.V,
            disp,
            color=MESH_COLOR,
            material="clay",
            smooth_shade=False,
            edge_width=0.6 if edges else 0.0,
            edge_color=THEMES[_theme["name"]]["edge"],
            back_face_policy="custom",
            back_face_color=BACK_COLOR,
        )
        _shown["faces"] = (mesh.face_ptr, mesh.face_idx, len(mesh.V))
    _shown["label"] = label if face_values is not None else None
    if face_values is not None:
        vals = np.asarray(face_values, float)
        if len(disp) != mesh.n_faces:  # display was triangulated: one value per triangle
            vals = np.repeat(vals, mesh.face_size - 2)
        lo, hi = vrange if vrange else (float(vals.min()), float(max(vals.max(), vals.min() + 1e-9)))
        s.add_scalar_quantity(label, vals, defined_on="faces", enabled=True, cmap=cmap, vminmax=(lo, hi))
    return s


def look(eye, target) -> None:
    """Point the camera at `target` and make it the orbit / zoom centre. (Polyscope's look_at alone keeps
    the old centre's distance along the new view ray, so the form would orbit around an empty point and
    zoom in steps sized by that stale distance.)"""
    ps.look_at(tuple(map(float, eye)), tuple(map(float, target)))
    ps.set_view_center_raw(tuple(map(float, target)))


def set_view(mesh: PolyMesh, view: str = "diagonal", distance: float = 3.2):
    direction, up = VIEWS.get(view, VIEWS["diagonal"])
    ps.set_up_dir(up)
    lo, hi = mesh.V.min(axis=0), mesh.V.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * np.linalg.norm(hi - lo)
    look(center + direction / np.linalg.norm(direction) * radius * distance, center)


# Scroll zoom. Polyscope moves the camera by a share of the distance to the orbit centre times the raw
# wheel delta, which on macOS trackpads and accelerated wheels varies wildly from event to event. Ours:
# every notch covers the same share of the remaining distance, one frame's delta is capped, and the
# distance stays within sensible bounds of the object's size, so the form can't be overshot or lost.
ZOOM_RATE = 0.12  # per wheel notch: 1 - exp(-0.12) = 11 % of the remaining distance
ZOOM_MAX_STEP = 2.5  # wheel units per frame (trackpad momentum spikes)
ZOOM_NEAR, ZOOM_FAR = 0.01, 15.0  # closest / furthest distance to the centre, x the object's size


def zoomed_view(view_mat, center, wheel: float, size: float) -> np.ndarray:
    """The camera view matrix after `wheel` notches of zoom toward `center` (positive = closer)."""
    M = np.array(view_mat, float)
    R = M[:3, :3]
    pos = -R.T @ M[:3, 3]
    forward = -R[2]  # cameras look down -z
    d = float(np.dot(np.asarray(center, float) - pos, forward))
    if d <= 1e-9:  # centre behind the camera: fall back to the straight distance
        d = float(np.linalg.norm(np.asarray(center, float) - pos)) or size
    step = float(np.clip(wheel, -ZOOM_MAX_STEP, ZOOM_MAX_STEP))
    new_d = float(np.clip(d * np.exp(-ZOOM_RATE * step), ZOOM_NEAR * size, ZOOM_FAR * size))
    out = M.copy()
    out[:3, 3] = -R @ (pos + forward * (d - new_d))
    return out


def write_png(path: str, rgba: np.ndarray) -> None:
    """Minimal PNG writer (avoids an imaging dependency)."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, c = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    color_type = 6 if c == 4 else 2
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def turntable(path: str, center, radius: float, frames: int = 72, seconds: float = 6.0, size: int = 640,
              elevation_deg: float = 20.0, distance: float = 3.0, up: str = "y", progress=None) -> str:
    """Orbit the camera around the up axis ("y" for forms, "z" for print models), capture
    frames and write a GIF or MP4. Renders whatever is currently shown. Returns the path."""
    import imageio.v2 as iio
    from PIL import Image

    if path.lower().endswith(".mp4"):
        size = max(16, int(round(size / 16)) * 16)  # video codecs want multiples of 16
    saved_view, saved_up = ps.get_view_as_json(), ps.get_up_dir()
    ps.set_up_dir(f"{up}_up")
    center = np.asarray(center, float)
    el = np.radians(elevation_deg)
    images = []
    try:
        for k in range(int(frames)):
            az = 2 * np.pi * k / frames
            ring, height = np.cos(el) * np.array([np.cos(az), np.sin(az)]), np.sin(el)
            d = np.array([ring[0], height, ring[1]]) if up == "y" else np.array([ring[0], ring[1], height])
            ps.look_at(tuple(center + radius * distance * d), tuple(center))
            buf = ps.screenshot_to_buffer(transparent_bg=False, include_UI=False)[:, :, :3]
            h, w = buf.shape[:2]
            side = min(h, w)
            crop = buf[(h - side) // 2:(h - side) // 2 + side, (w - side) // 2:(w - side) // 2 + side]
            images.append(np.asarray(Image.fromarray(crop).resize((size, size), Image.LANCZOS)))
            if progress:
                progress(k + 1, frames)
    finally:
        ps.set_up_dir(saved_up)
        ps.set_view_from_json(saved_view)
    fps = frames / max(seconds, 0.1)
    root, ext = os.path.splitext(path)
    tmp = root + ".part" + ext  # complete files only: write aside, then rename
    try:
        if ext.lower() == ".mp4":
            with iio.get_writer(tmp, fps=fps, codec="libx264", quality=8, macro_block_size=16) as wr:
                for im in images:
                    wr.append_data(im)
        else:
            Image.fromarray(images[0]).save(tmp, format="GIF", save_all=True, loop=0, optimize=True,
                                            append_images=[Image.fromarray(i) for i in images[1:]],
                                            duration=int(round(1000 / fps)))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path

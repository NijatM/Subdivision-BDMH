"""Shared Polyscope look (used by the app and the render tool)."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import polyscope as ps

from .mesh import PolyMesh

MESH_NAME = "form"
MESH_COLOR = (0.86, 0.85, 0.82)

# Polyscope tone-maps the background (gamma ~2.2), so colours are given in linear space:
# 0.0065 linear ~ 0.10 on screen (a deep charcoal).
THEMES = {
    "dark": {"background": (0.0060, 0.0065, 0.0080), "ground": "shadow_only", "edge": (0.18, 0.18, 0.20)},
    "light": {"background": (1.0, 1.0, 1.0), "ground": "shadow_only", "edge": (0.25, 0.25, 0.25)},
}
_theme = {"name": "dark", "polyscope_style": None}


def apply_scene_theme(name: str) -> None:
    """Background / ground plane. Safe to call outside an ImGui frame."""
    t = THEMES[name]
    _theme["name"] = name
    ps.set_background_color(t["background"])
    ps.set_ground_plane_mode(t["ground"])


def apply_ui_theme(name: str) -> None:
    """ImGui colours — must be called inside a frame (e.g. from the user callback)."""
    import polyscope.imgui as psim

    colors = psim.GetStyle().Colors
    if _theme["polyscope_style"] is None:  # remember Polyscope's own dark style once
        _theme["polyscope_style"] = [tuple(colors[i]) for i in range(psim.ImGuiCol_COUNT)]
    if name == "dark":
        for i, c in enumerate(_theme["polyscope_style"]):
            colors[i] = c
    else:
        psim.StyleColorsLight()
        colors = psim.GetStyle().Colors
        accent = (0.36, 0.62, 0.50, 1.0)
        for key, c in [("Header", accent), ("HeaderHovered", (0.42, 0.70, 0.57, 1.0)),
                       ("Button", (0.36, 0.62, 0.50, 0.55)), ("ButtonHovered", (0.42, 0.70, 0.57, 0.9)),
                       ("TitleBgActive", accent), ("TitleBg", accent), ("SliderGrab", accent),
                       ("CheckMark", (0.20, 0.45, 0.35, 1.0))]:
            colors[getattr(psim, "ImGuiCol_" + key)] = c


VIEWS = {
    # Fig. 3 looks down the cube's body diagonal (three-fold symmetry)
    "diagonal": (np.array([1.0, 1.0, 1.0]), "y_up"),
    "front": (np.array([0.0, 0.0, 1.0]), "y_up"),
    "top": (np.array([0.0, 1.0, 0.0001]), "z_up"),
    "three_quarter": (np.array([1.0, 0.6, 1.4]), "y_up"),
}


def setup_scene(window=(1280, 900), theme: str = "dark"):
    ps.set_program_name("Hansmeyer Subdivision")
    ps.set_window_size(*window)
    ps.set_up_dir("y_up")
    ps.set_SSAA_factor(2)
    ps.init()
    apply_scene_theme(theme)


def show_mesh(mesh: PolyMesh, edges: bool | None = None):
    edges = mesh.n_faces <= 30_000 if edges is None else edges
    s = ps.register_surface_mesh(
        MESH_NAME,
        mesh.V,
        mesh.display_faces(),
        color=MESH_COLOR,
        material="clay",
        smooth_shade=False,
        edge_width=0.6 if edges else 0.0,
        edge_color=THEMES[_theme["name"]]["edge"],
        back_face_policy="custom",
        back_face_color=(0.55, 0.42, 0.40),
    )
    return s


def set_view(mesh: PolyMesh, view: str = "diagonal", distance: float = 3.2):
    direction, up = VIEWS.get(view, VIEWS["diagonal"])
    ps.set_up_dir(up)
    lo, hi = mesh.V.min(axis=0), mesh.V.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * np.linalg.norm(hi - lo)
    eye = center + direction / np.linalg.norm(direction) * radius * distance
    ps.look_at(tuple(eye), tuple(center))


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

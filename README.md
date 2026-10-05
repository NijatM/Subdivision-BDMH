# Subdivision as a Generative System

An interactive engine for **Michael Hansmeyer's extended Catmull-Clark / Doo-Sabin subdivision**
(*From Mesh to Ornament*, eCAADe 2010), the method behind his Subdivided Columns and
Digital Grotesque (with Benjamin Dillenburger).

- Background research: [`info.md`](info.md)
- Agreed plan and milestones: [`PLAN.md`](PLAN.md)

![Fig. 3 reproductions](renders/fig3_repro.png)
*Our Fig. 3 presets (cube, 8 iterations, non-stationary weights). Left: six arms. Centre: corner cup with a spike field. Right: Catmull-Clark + Doo-Sabin flower.*

![Milestone 2 presets](renders/m2_presets.png)
*Milestone 2 presets: Fig. 4 box column, Fig. 7 classic column, grotto panel, dodecahedron coral, icosahedron crystal, Fig. 3 right.*

## Run it

| | First run (installs everything into a local `.venv`) | Afterwards |
|---|---|---|
| **Mac** | double-click `run_mac.command` | same |
| **Windows** | double-click `run_windows.bat` | same |

Requirements: **Python 3.10+** ([python.org](https://www.python.org/downloads/); on Windows tick *"Add python.exe to PATH"*).

- macOS may block the file the first time: right-click → *Open*, or run `chmod +x run_mac.command` in a terminal.
- Manual alternative (both OSes): create a venv, `pip install -r requirements.txt`, `python app.py`.

## Using the app

- **Theme** (top of the panel): *Dark* is the default; *Light* is available. Your choice is remembered in `.app_settings.json`.
- **Preset**: load any design from `presets/`. Type a name and press *Save* to store your own.
- **Base mesh**: pick a shape; its parameters appear as sliders.
  - Cube and Platonic solids (tetrahedron, octahedron, dodecahedron, icosahedron).
  - **Column: box on base** (paper Fig. 4) and **Column: classic** (Fig. 7 style: base, shaft with entasis and taper, echinus, abacus).
  - **Panel**: an open relief tile with optional bend or saddle.
  - **Import OBJ**: drop `.obj` files into `inputs/` (a sample `l_block.obj` is included), pick one and press *Load OBJ*. Duplicate vertices are welded, winding is repaired, and the mesh is scaled to fit. Non-manifold meshes are rejected with an explanation.
- **Boundary** (open meshes only):
  - *smooth* uses the standard Catmull-Clark boundary rules.
  - *locked / tileable* keeps the outline fixed and fades the relief out over *fade rows*, so identical tiles meet seamlessly (see below). Doo-Sabin steps shrink open boundaries, and the app warns when they'd break tiling.
- **Depth & extrusion**: *preview depth* recomputes live while you drag. After 1.5 s idle (or *Bake full depth*), the **full depth** is computed. The face budget caps runaway depths.
- **Iteration schedule**: pick an iteration, choose Catmull-Clark or Doo-Sabin, and set its weights. *Sharp* sets `w1=-1, w2=-2`; *Copy → next / all* duplicates a step. *Overview* shows every weight as bars across iterations.
- **View & export**: *diagonal* looks down the cube's body diagonal like the paper. Export **OBJ / PLY** (quads kept) or **STL** (triangulated) to `exports/`; *Screenshot* writes to `renders/`.
- Ctrl+click any slider to type an exact value. Hover a control for its explanation.
- Pinkish surfaces are **back faces**: the surface has folded through itself.

![Tiling](renders/panel_tiling.png)
*Four copies of `panel_tile` side by side: the locked boundary makes the seams line up.*

### Weight cheat-sheet (relative extrusion: 1.0 = one local edge length)

| Weight | Equation | Effect |
|---|---|---|
| `w_f` | 1 | push face centres out (+) / in (−) → bumps, spikes, pits |
| `w_e` | 2 | push edge points along the edge normal → ridges, ribs, rims |
| `w_c` | 3 | push corner points along the vertex normal. **Strongly negative pulls corners in → arms / cups** |
| `w1` | 2 | edge point bias: −1 = plain edge midpoint (no smoothing) |
| `w2` | 3 | corner bias: negative = anti-smoothing. **`w1=-1, w2=-2` keeps forms crisp** |
| `w3`, `w4` | 4 | from iteration 2: face-point bias toward the old corner/face vertex (`w3`) and diagonal/edge vertices (`w4`). `w3=-1, w4=1` concentrates growth on the previous step's tips |
| DS `w1`, `w_f` | 5–6 | Doo-Sabin corner pull and extrusion, set separately for face-, edge- and vertex-derived faces |

**Lesson from tuning:** with standard smoothing (`w1 = w2 = 0`), whatever the extrusions build gets averaged away in later steps. Switch smoothing off (*Sharp*) and the extrusions accumulate.

## Render presets from the command line

```bash
python render.py presets/fig3_left.json --depth 8                 # → renders/fig3_left.png
python render.py presets/*.json --depth 6 --montage renders/all.png
python render.py presets/panel_tile.json --theme light            # white background
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite checks:
- Zero weights reproduce textbook Catmull-Clark and Doo-Sabin, compared against an independent loop-based reference (`tests/reference.py`). This covers closed and open meshes and mixed CC/DS.
- The paper's eq. 2, 3, 4, 5 and 6 numerically.
- Face counts, Euler characteristic and manifold orientation.
- That every base shape is a valid, outward-facing solid, and that the Platonic solids are regular.
- **Locked panels are tileable**: the outline stays fixed and opposite edges match. The boundary fade is checked too.
- OBJ parsing (all index syntaxes), welding, orientation repair, non-manifold rejection, and OBJ/STL/PLY writers.
- Affine and scale invariance, caching, the face budget, JSON round-trips, and that every preset loads.

## Layout

```
app.py              interactive Polyscope app
render.py           headless preset → PNG
hansmeyer/
  mesh.py           polygon mesh, vectorised half-edges, normals, per-vertex attributes
  catmull_clark.py  extended CC (eq. 1–4), smooth / locked boundaries
  doo_sabin.py      extended DS (eq. 5–6)
  schedule.py       weight definitions, per-iteration schedule, Design (= preset JSON)
  pipeline.py       runs a Design: caching, face budget, boundary fade
  shapes.py         base mesh registry (Platonic solids, columns, panel, OBJ)
  meshio.py         OBJ import + cleanup; OBJ / STL / PLY export
  view.py           shared rendering style and dark/light themes
presets/            JSON designs
inputs/             drop .obj files here to import them
tests/              pytest suite + independent reference implementation
```

## Faithfulness to the paper

- **From the paper:** eq. 1–6, non-stationary weights, the CC + DS combination, provenance (corner/edge/face points; face/edge/vertex-derived faces), and the Fig. 4 / Fig. 7 column inputs.
- **Our extensions (labelled in code):**
  - Eq. 1/3 applied to n-gons and any valence.
  - A Doo-Sabin `w1` generalisation for n ≠ 3, 4.
  - Relative extrusion (the paper doesn't say whether normals are unit length; the *absolute* mode is the literal reading).
  - Boundary rules and the locked/tileable mode for open meshes.
- **Fig. 3:** the paper publishes no weight values, so the presets reproduce the *character* of the figures, not exact geometry. The left figure's arms are there, but its fluted fans at the arm tips aren't yet.
- **Fig. 4 zoning** (four weight sets along y) needs attractors, which come in M3. The M2 column presets use uniform weights.

## Roadmap

1. ✅ Core engine, tests, minimal viewer, Fig. 3 presets
2. ✅ Dark/light theme, Platonic solids, columns (Fig. 4/7), open panel with smooth/locked boundaries, OBJ import, OBJ/STL/PLY export, schedule overview
3. Attractors (points and space curves, falloff curves, weight sets and modifiers)
4. Function layer stack (TPMS, noise, analytic, folds) and a `functions/` plug-in folder
5. Tagging/locking, intrinsic motifs, topological distance and curvature, vertex merging (porosity)
6. Watertight voxel remesh for FDM printing, turntable GIF/MP4

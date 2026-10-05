# Subdivision as a Generative System

An interactive engine for **Michael Hansmeyer's extended Catmull-Clark / Doo-Sabin subdivision**
(*From Mesh to Ornament*, eCAADe 2010), the method behind his Subdivided Columns and
Digital Grotesque (with Benjamin Dillenburger).

- Background research: [`info.md`](info.md)
- Agreed plan and milestones: [`PLAN.md`](PLAN.md)

![Fig. 3 reproductions](renders/fig3_repro.png)
*Our Fig. 3 presets (cube, 8 iterations, non-stationary weights). Left: six arms. Centre: corner cup with a spike field. Right: Catmull-Clark + Doo-Sabin flower.*

## Run it

| | First run (installs everything into a local `.venv`) | Afterwards |
|---|---|---|
| **Mac** | double-click `run_mac.command` | same |
| **Windows** | double-click `run_windows.bat` | same |

Requirements: **Python 3.10+** ([python.org](https://www.python.org/downloads/); on Windows tick *"Add python.exe to PATH"*).

- macOS may block the file the first time: right-click → *Open*, or run `chmod +x run_mac.command` in a terminal.
- Manual alternative (both OSes): create a venv, `pip install -r requirements.txt`, `python app.py`.

## Using the app

- **Preset**: load `fig3_left`, `fig3_centre`, `fig3_right`, `sharp_starter` or `smooth`. Type a name and press *Save* to store your own in `presets/`.
- **Preview depth** is recomputed live while you drag. After 1.5 s idle (or *Bake full depth*), the **full depth** is computed. Iterations marked `*` only show in the baked result.
- **Iteration schedule**: pick an iteration, choose Catmull-Clark or Doo-Sabin, and set its weights. *Sharp* sets `w1=-1, w2=-2`; *Copy → next / all* duplicates a step.
- **View & export**: *diagonal* looks down the cube's body diagonal like the paper. *Export OBJ* writes to `exports/`; *Screenshot* writes to `renders/`.
- Ctrl+click any slider to type an exact value. Hover a slider for its equation.
- Pinkish surfaces are **back faces**: the surface has folded through itself. That's useful as a warning, and sometimes it's the look you want.

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
python render.py presets/fig3_left.json --depth 8               # → renders/fig3_left.png
python render.py presets/*.json --depth 6 --montage renders/all.png
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

What the tests check:
- Zero weights reproduce textbook Catmull-Clark and Doo-Sabin, compared against an independent loop-based reference (`tests/reference.py`). This covers closed and open meshes and mixed CC/DS.
- The paper's eq. 2, 3, 4, 5 and 6 numerically.
- Face counts (cube, depth 8 = 6·4⁸ = 393,216) and Euler characteristic / manifold orientation.
- Affine and scale invariance, caching, the face budget, JSON round-trips, and that every preset loads.

## Layout

```
app.py              interactive Polyscope app
render.py           headless preset → PNG
hansmeyer/
  mesh.py           polygon mesh, vectorised half-edges, normals, OBJ
  catmull_clark.py  extended CC (eq. 1–4)
  doo_sabin.py      extended DS (eq. 5–6)
  schedule.py       weight definitions, per-iteration schedule, Design (= preset JSON)
  pipeline.py       runs a Design with per-iteration caching + face budget
  shapes.py         base meshes
  view.py           shared rendering style
presets/            JSON designs
tests/              pytest suite + independent reference implementation
```

## Faithfulness to the paper

- **From the paper:** eq. 1–6, non-stationary weights, the CC + DS combination, and provenance (corner/edge/face points; face/edge/vertex-derived faces).
- **Our extensions (labelled in code):**
  - Eq. 1/3 applied to n-gons and any valence.
  - A Doo-Sabin `w1` generalisation for n ≠ 3, 4.
  - Relative extrusion (the paper doesn't say whether normals are unit length; the *absolute* mode is the literal reading).
  - Standard boundary rules for open meshes.
- **Fig. 3:** the paper publishes no weight values, so the presets reproduce the *character* of the figures, not exact geometry. The left figure's arms are there, but its fluted fans at the arm tips aren't yet.

## Roadmap

1. ✅ Core engine, tests, minimal viewer, Fig. 3 presets
2. Schedule overview plot, Platonic solids, columns (Fig. 4/7), open panel with smooth/locked boundaries, OBJ import, STL/PLY export
3. Attractors (points and space curves, falloff curves, weight sets and modifiers)
4. Function layer stack (TPMS, noise, analytic, folds) and a `functions/` plug-in folder
5. Tagging/locking, intrinsic motifs, topological distance and curvature, vertex merging (porosity)
6. Watertight voxel remesh for FDM printing, turntable GIF/MP4

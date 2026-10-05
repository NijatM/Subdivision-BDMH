# Hansmeyer Subdivision Engine — Agreed Plan (v1)

Aligned via a grill-me session on 2026-10-04. Background research: `info.md`.

## Runtime & setup
- Python 3.10+ · numpy · Polyscope (ImGui UI); scipy / scikit-image / imageio added in later milestones.
- One-click launchers: `run_mac.command`, `run_windows.bat` — first run creates `.venv` and installs `requirements.txt`.
- Layout: `app.py`, `hansmeyer/` (engine), `functions/` (plug-ins), `presets/`, `tests/`, `info.md`, `README.md`.
- Git is handled by the user only (no init/commit/push by Claude; commands are suggested as text).

## Engine
- General polygon mesh. Each iteration of the schedule picks **extended Catmull-Clark** (eq. 1–4) or **extended Doo-Sabin** (eq. 5–6).
  - Paper equations are used where they apply (quads, triangles); n-gon generalisations are labelled as extensions.
- Provenance tracking: every vertex knows whether it was a corner / edge / face point (eq. 4); every face knows whether it came from a face / edge / vertex (Doo-Sabin per-class weights).
- Extrusion is **relative** by default (displacement = w × local edge length), with a global toggle to **absolute** (paper-literal).
- Live **preview depth** (~5) + **Bake** at full depth (8+), per-iteration caching, face budget (default 2M).

## Inputs
- Cube + Platonic solids; column inputs (Fig. 4 elongated box on base, parametric Fig. 7-style column); open quad panel; OBJ import.
- Boundary mode per mesh: **Smooth** (standard B-spline boundary) or **Locked/tileable** (extrusion fades to 0 over N rows).

## Controls
- Iteration tabs (sliders + CC/DS choice), Copy → next / all, per-weight overview plot.
- **Attractors**: points (gizmo or xyz) and space curves (line, circle, helix, sine, Lissajous; editable control points; Rhino polyline import).
  - Payload: full **weight set** (paper eq. 8–9 blending) or **modifier** (scale/offset chosen weights).
  - Falloff: `(1−d)^t`, linear, smoothstep, gaussian, editable spline.
  - Influence measured at the face's current position by default; toggle to original (rest) position.
- **Function layer stack** (our extension, not Hansmeyer's method):
  - Families: TPMS (gyroid, Schwarz P/D, Neovius), noise & Worley, analytic (superformula, spherical harmonics, superquadric, rose), minimal folds (box, sphere, kaleidoscopic mirror).
  - Targets: weight fields, displacement fields, fold maps. Per layer: blend mode, amplitude, iteration range, attractor mask.
  - `functions/` plug-in folder with auto-generated sliders.
- **Paper advanced features**: tagging + locking (Fig. 9) via click-pick and rule-based selection; intrinsic motifs with U values (eq. 10–11); topological distance + curvature weighting; vertex merging / porosity (paper-style face dropping + manifold cleanup).

## Outputs
- OBJ / STL / PLY, JSON presets, screenshots.
- Watertight voxel remesh for FDM (0.4 mm nozzle, 80–150 mm): ~0.2 mm voxels, warning for features < 0.8 mm.
- Turntable GIF / MP4.

## Success criteria
- Tests: zero weights reproduce standard Catmull-Clark / Doo-Sabin (vs. an independent reference implementation); face counts (6·4⁸ = 393,216); Euler characteristic; orientation.
- Three tuned **Fig. 3 presets** (judged by eye) + body-diagonal camera button.
- The paper publishes no weight values, so Fig. 3 is reproduced in character, not exactly.

## Milestones (check-in after each)
1. ✅ Core engine + tests + minimal viewer + Fig. 3 presets
2. ✅ Schedule UI polish, all base shapes, boundaries, OBJ import/export, presets (+ dark/light theme, user request)
3. ✅ Attractors (points, curves, falloffs, weight sets & modifiers)
4. ✅ Function layer stack + plug-ins
5. ✅ Advanced paper features (tags/locking, motifs, topo distance/curvature, merging)
6. **Watertight print prep + turntable** ← next

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

![Milestone 3 presets](renders/m3_presets.png)
*Milestone 3, attractors. Left: paper Fig. 4, four weight sets along y give a base and three sections. Middle: a helix curve wraps the column in a spiral of ornament. Right: a sine-curve "river" across a tileable panel.*

![Milestone 4 presets](renders/m4_presets.png)
*Milestone 4, function layers: gyroid coral column, twisted rose flutes, Worley cell tile, Mandelbox-folded gyroid.*

![Milestone 5 presets](renders/m5_presets.png)
*Milestone 5, the paper's intrinsic features. Fig. 9 locked edges (sharp creases), Fig. 7 motif-driven column, Fig. 8 topological distance, porosity by vertex merging.*

## Run it

| | First run (installs everything into a local `.venv`) | Afterwards |
|---|---|---|
| **Mac** | double-click `run_mac.command` | same |
| **Windows** | double-click `run_windows.bat` | same |

Requirements: **Python 3.10+** ([python.org](https://www.python.org/downloads/); on Windows tick *"Add python.exe to PATH"*).

- macOS may block the file the first time: right-click → *Open*, or run `chmod +x run_mac.command` in a terminal.
- Manual alternative (both OSes): create a venv, `pip install -r requirements.txt`, `python app.py`.

## Using the app

The app (window title *SubdivisionEngine_BDMH*) shares the look of [nijatmahamaliyev.com](https://nijatmahamaliyev.com) and the HTMAA site: dark, monochrome, IBM Plex Mono (bundled in `assets/fonts/`, OFL licence).

```
+-------------------+----------------------------------------+---------------------+
| Nijat Mahamaliyev | root@nijat:~$ ./SubdivisionEngine_BDMH | 02 · FORM           |
|                   | diag front top 3/4  wire  colour  cut  | Iteration schedule  |
| FORM              |                                        |                     |
| 00 presets        |                                        | the selected        |
| 01 base mesh      |               3D view                  | section's controls  |
| 02 schedule       |                                        |                     |
| GROWTH  03-07     |                                        |                     |
| MAKE    08-10     | undo redo  preview 5  full 8  bake     |                     |
|                   | baked · depth 8 · 98,304 faces         |                     |
+-------------------+----------------------------------------+---------------------+
```

- **Sidebar (left)**: the sections in workflow order. **FORM**: 00 presets, 01 base mesh, 02 schedule. **GROWTH**: 03 attractors, 04 layers, 05 groups, 06 intrinsic, 07 porosity. **MAKE**: 08 vessel, 09 print, 10 export.
  - The right column shows each section's state: the preset (with `*` when edited), the shape, the depths, how many attractors / layers / groups / rules are on, whether the print model is ready. Sections with nothing set are faint.
- **Inspector (right)**: the selected section's controls. Each section opens with one line on what it does; hover any control for its full explanation, or press **?** (or H) to show the longer explanations inline.
- **Toolbar (over the view)**: views (*diag* looks down the cube's body diagonal like the paper), *wire*, *colour by* (attractor influence, the selected layer's field, group tags, light through a vessel wall, any measure), the section *cut* (… for its settings), help, the dark / light palette, and Polyscope's own panel (it takes the sidebar's place while on).
- **Bottom bar (under the view)**: undo / redo, *preview* depth (recomputed live while you edit), *full* depth (baked after 1.5 s idle, or *bake*), auto-bake, then the state of the result and the latest message (hover it for the recent log).
- **The window never waits for the full depth**: bakes run in a background process, and the form is replaced when the result is ready (the bottom bar shows *baking … s*). Heavy forms, those slower than ~60 ms per preview step, also update their preview there while you drag, so sliders stay smooth and the form follows a moment later.
- **Undo / redo** covers every design edit, including loading a preset: a preset loads with one click and one undo brings your design back.
- **The last session reopens on start** (it is kept in `presets/_last_session.json` after every bake). *restore last session* in presets brings it back later.
- **3D view**: drag to orbit, right-drag (or Shift+drag) to pan, scroll or two-finger swipe to zoom. Every scroll notch covers the same share (~11 %) of the distance to the orbit centre, trackpad spikes are capped, and the camera never passes the centre or drifts further than 15x the form's size. **Double-click** a point of the form to orbit and zoom around it; **F** frames the whole form again.
- Ctrl/Cmd+click any slider to type an exact value. Your choices (palette, section, printer, window size) are remembered in `.app_settings.json`.
- Grey surfaces are **back faces**: the surface has folded through itself.

| Key | Action |
|---|---|
| Ctrl/Cmd+Z, Ctrl/Cmd+Shift+Z | undo, redo |
| Ctrl/Cmd+S | save the preset (name in *presets → save as*) |
| B | bake the full depth |
| E | export the mesh (format in *export*) |
| 1 2 3 4 | diag, front, top, 3/4 view |
| F | frame the form again (if you lose it) |
| W | wireframe |
| [ ] | previous / next section |
| H | longer help texts on / off |
| Tab | hide / show all panels (for screenshots and presenting) |

**Presets** are named *family_form* and grouped by base shape: `cube_*`, `column_*`, `panel_*`, `solid_*`, `cage_*`, `vessel_*`. Presets after a figure of the paper say so in their description (*ref: Hansmeyer 2010, Fig. 3 (left)*). Names without a family prefix are listed under SAVED.

**Base mesh** shapes:
- Cube and Platonic solids (tetrahedron, octahedron, dodecahedron, icosahedron).
- **Column: box on base** (paper Fig. 4) and **Column: classic** (Fig. 7 style: base, shaft with entasis and taper, echinus, abacus).
- **Panel**: an open relief tile with optional bend or saddle.
- **Cube cage (frame)**: the 12 edges of a cube as bars, all six faces open.
- **Sphere with opening (vessel)**: a sphere with a circular opening on top. *opening* is its angular radius. With the vessel on, *diameter* sets its real size.
- **Import OBJ**: drop `.obj` files into `inputs/` (a sample `l_block.obj` is included), pick one and press *load obj*. Duplicate vertices are welded, winding is repaired, and the mesh is scaled to fit. Non-manifold meshes are rejected with an explanation.
- **Open boundary** (open meshes only): *smooth* uses the standard Catmull-Clark boundary rules; *locked / tile* keeps the outline fixed and fades the relief out over *fade rows*, so identical tiles meet seamlessly (see below). Doo-Sabin steps shrink open boundaries, and the app warns when they'd break tiling.

**Schedule**: pick an iteration (`*` = only in the baked result), choose Catmull-Clark or Doo-Sabin, and set its weights. Each slider is a bar filled from zero to the value. *sharp* sets `w1=-1, w2=-2`; *copy → next / all* duplicates a step. The *overview* shows every weight in use as bars across the iterations (the open one bright). *face budget* caps runaway depths.

**Export**: **OBJ / PLY** (quads kept) or **STL** (triangulated) to `exports/`; *save png* writes the view (without the panels) to `renders/`; turntables below.

**Section cut** (toolbar *cut*, settings under …): hides everything on one side of a plane, so you can keep editing while looking inside. Pick the plane (*X / Y / Z*), slide its *position*, *flip side*, *drag in the view* to move or tilt it, *face the cut* to point the camera at it. It works on the form and on the print model; screenshots and turntables are cut too.

### Attractors (non-uniform weights)

Attractors make weights vary **in space**, the paper's *extrinsic specification of parameters* (eq. 8–9). Section **03 attractors**:

- **+ point / + curve** adds one near the form. Click a row to select it. **Drag the gizmo** in the view (shown while this section is open), or type coordinates.
- **Kind**:
  - *point*.
  - *curve*: line, circle, helix, sine, Lissajous, or an editable **polyline**. *convert to polyline* turns any curve into draggable control points. *Import* reads `.csv/.txt` (x y z per line) or `.obj` polylines from Rhino or Blender; try `inputs/sample_curve.csv`.
- **Payload**:
  - *weight set*: a full per-iteration schedule of its own. Each face blends all sets by distance: `c_a = f(d_a)·h_a / Σ f(d_i)·h_i`, `w = Σ w_i·c_i` (eq. 8–9).
  - *modifier*: rules such as "scale `w_f` ×3 in iterations 1–4" or "add +0.3 to `w_e`", applied near the attractor.
- **strength** (`h`), **radius**, and **falloff**: `power` is the paper's `(1−d)^t`; also linear, smoothstep, gaussian, or a hand-drawn **spline** (5 sliders). The plot shows the curve.
- **background**: how much the main schedule takes part in the blend. **0 is paper-pure.** With 0, a *single* set applies fully everywhere it reaches, so use background > 0 for a gradient.
- **Measure at**:
  - *current* uses distance from where a face is now, so growth reacts to its own movement.
  - *original* uses where the face sat on the input mesh, which gives stable zoning.
- **influence map** colours the form by the selected attractor's influence.

### Function layers (our extension)

Mathematical fields and folds layered on top of Hansmeyer's process, in section **04 layers**:

- **+ weight field**: a field drives one weight per face (`w_f`, `w_e`, …) before an iteration. The blend can be add, multiply, replace, min or max.
- **+ displacement**: moves vertices along their normals after an iteration (adds relief directly).
- **+ fold**: deforms the whole mesh with a fold map after an iteration.
- **Built-in families:**
  - **TPMS**: gyroid, Schwarz P, Schwarz D, Neovius.
  - **Noise**: Perlin, fBm, ridged, domain-warped, Worley cells or borders.
  - **Analytic**: constant (with a mask, a plain weight or relief boost just where the mask is), spherical harmonics, superformula, superquadric, k-fold rose with twist.
  - **Folds**: box, sphere, kaleidoscopic mirror, Mandelbox step, clamp to box (flat, cut-like outer faces).
- **Per layer:**
  - amplitude and offset.
  - iteration range.
  - **evaluate at** *original* positions (the pattern sticks to the form like a texture) or *current* positions (the form grows through a fixed pattern).
  - **mask** by an attractor's reach, or by **facing direction** (*facing +y* etc.): the layer acts on surfaces facing that way and fades out on surfaces facing away. *facing +y* keeps detail on top and leaves the undersides plain, which is easier to 3D print.
  - **domain fold**: repeat a fold on the field's input, e.g. Mandelbox × 3 with c = 1, for fractal ornament.
- **Plug-ins**: drop a `.py` file into `functions/` with an `@field` or `@fold` function, then press *reload functions/*. Sliders are generated from its parameters. See [`functions/README.md`](functions/README.md) and the examples `ripples.py` and `twist.py`.

### The paper's intrinsic features

- **Groups (05, Fig. 9)**: *+ group*, then *click to tag* faces or vertices on the input mesh, or *select by rule* (facing direction, height band, every k-th, motif).
  - Each group gets **weight rules**.
  - **Lock iterations** hold its vertices in place, so locked edges stay sharp creases and locked points become spikes. Tags pass to every child face.
- **Motifs (06 intrinsic, Fig. 6–7, eq. 10–11)**: every motif found on the input (e.g. `3F3E`, `4F4E`) gets an attractor/deflector value **U**. The schedule's **w6** and **w7** pull face and edge points toward (+) or away from (−) those vertices.
- **Measure rules (06 intrinsic, Fig. 8 + curvature)**: distance to original vertices, distance to original edges, planarity, or bend gives a per-face *t* in [0, 1]. A rule sets, scales or offsets a weight from *at 0* to *at 1*.
- **Vertex merging (07 porosity)**: welds vertices of *different* parts of the surface that grow into contact (pairwise, within a fraction of the local edge length). Where a welded vertex would exceed the **max valence**, faces aren't formed, which opens holes and handles. The Euler characteristic is shown.
- **Colour by** (toolbar): attractor influence, the selected layer's field, group tags, or any measure. It shows where things act.

### Printing (FDM) and turntables

Section **09 print** turns any form into a watertight, printable solid:
- **What it handles:**
  - Self-intersections are merged into one solid.
  - Open or porous skins are thickened to the *min wall*.
  - Reliefs can get a *solid base*.
- **What you get:** the model is placed on the build plate in **millimetres, Z up**. Features thinner than about 2 nozzle widths are shown in **red**.
- **Accuracy:** within a fraction of a voxel of the requested size; volume within about 2%.
- **Runs in a separate process,** so the app stays responsive. Big models can't crash it, and an out-of-memory failure shows up as a message.
- **Exports are safe:**
  - Free disk space is checked first.
  - Data goes to a temporary file that only replaces the target once complete.
  - The written STL is **re-read and verified** to be complete and watertight before you see "verified".
- ***Mesh detail: half*** gives about 4× smaller files (still watertight).
- **Typical times:** 100 mm at 0.3 mm voxels takes about 2–20 s; 200 mm about 1.5 min and 2 GB of RAM.

The section is laid out top to bottom:
- **Printer:** pick your bed (Bambu A1/P1S/X1C, A1 mini, Prusa MK4, CORE One, MINI, or custom). The choice is remembered.
- **Size:** choose what the number measures (longest side, height, width or depth), type it in mm or click 50/100/150/200. **Fit bed** finds the largest size that fits. The live W × D × H line says whether it fits.
- **Orientation:** which model axis points up (±X, ±Y, ±Z). **auto: least support** picks the one with the least overhang.
- **Undersides:**
  - **Self-supporting undersides** fills below every overhang with a smooth keel no flatter than the angle (default 45°). The print then needs almost no support.
    - Upward-facing surfaces keep all their detail, and horizontal holes get pointed (teardrop) tops.
    - It adds material, sometimes doubling the volume of forms with large overhangs.
    - Set the slicer's threshold angle below the keel angle.
  - **Flat foot** trims the bottom flat, so the print stands on a face instead of a point.
- **Cut into parts:** *Whole*, *2 halves* (a horizontal cut at the widest section), *4 quarters* (two vertical cuts: four pillars, like the corners of a cage) or *8 pieces*. You can also tick each cut and move it.
  - Planes over the form show where the cuts go.
  - Every part is its own closed solid, and its cut faces are exactly flat, so mating parts meet without a gap.
  - *Alignment pin holes* (default Ø 2.0 mm × 5 mm, for 1.75 mm filament pins) are drilled into both faces of every joint.
  - Each part is turned on its own to need the least support (usually cut face down), or printed as assembled.
- **Resolution:** *Draft / Standard / Fine* (voxel = 1 / 0.75 / 0.5 × nozzle) or *Max* (the finest the grid limit allows). The line below shows the real detail size, triangles, STL size, RAM and time. If the grid limit coarsens your choice, it says so.
- **Advanced settings:** exact voxel size, grid limit (400–1000 voxels along the longest side, with its RAM cost), min wall, smoothing, voxel softening, mesh detail and solid base.
- **Support check** (after preparing): set the slicer's support *threshold angle*. Surfaces that would get support turn red, and the area per part is listed.
  - The angle works like Bambu Studio, Orca and PrusaSlicer: downward surfaces flatter than it (measured from the horizontal) get support, so **higher = more support**.
- **View:** *Print layout* (parts side by side on the plate) or *Assembled*, with an explode gap. Colour by *needs support*, *too thin* or *parts*.
- **Export print STLs** writes one verified STL per part (`…_part1of4_left-front.stl`, …). Load them all into the slicer together.

**Cube cage** (`presets/cage_cube.json`): a cube frame with flat outer faces and organic openings, printed in 4 quarters.
- It starts from the **Cube cage (frame)** base shape: 12 bars, all six faces open. *bar width* and *segments* are in base mesh.
- Subdivision makes the bars swell, ridge and bead.
  - fBm noise and three *lens* attractors make some bars bulge into the openings while others stay slim.
  - Worley cells texture the inside.
- A final **Clamp to box** fold flattens everything beyond the cube onto its faces, so the outer faces are flat like a cut block.
- To print it: *cut into parts → 4 quarters*. Each quarter lies flat on an outer face, needs almost no support, and has pin holes at the joints, so the cage can be assembled around something placed inside.
  - At 150 mm it is about 620 cm³ (≈370 g PLA at 20% infill). A smaller *bar width* or size makes it lighter.

**Why a whole six-arms form (Fig. 3) gets support everywhere:** its arms stick out sideways, so their undersides are near-horizontal overhangs at any threshold angle. The form also stands on a single arm tip.
- The mesh is fine: shrink-wrapping it changes nothing.
- Cutting it into 2 halves at the widest section cuts the support area by about 4–7×, and each half stands on a large flat face.

**Turntable** (in 10 export) orbits the camera around the form, or around the print model, and saves a GIF or MP4 to `renders/`. From the command line:

```bash
python render.py presets/cube_petal_flower.json --turntable renders/cube_petal_flower.gif
```

If anything unexpected goes wrong, the app logs it to `app_errors.log` and shows it in the bottom bar instead of closing.

![Tiling](renders/panel_tiling.png)
*Four copies of `panel_tileable` side by side: the locked boundary makes the seams line up.*

### Vessel: lithophane sphere

Section **08 vessel** (lithophane sphere) turns the form into the inside of a thin shell (our extension):
- **Outside**: an exact, smooth sphere of the chosen *diameter (mm)*.
- **Inside**: the subdivision relief, exactly what the schedule's weights build. It is measured against the same schedule run with zero weights, so you edit it like any other form.
  - The parts reaching out furthest (*glowing share*, e.g. 15%) press against the shell at the *window wall* (0.8 mm). They glow when lit from inside.
  - Everything else is deeper and thicker, at the form's true proportions times *relief depth*, up to the *deepest wall* cap.
  - *Turn the relief inside out* swaps what is near the shell and what is deep.
- **Opening**: an exact, flat circle with a solid *rim wall*. The vessel prints upside down standing on it, so the dome comes out smooth.
- **light preview** (or *colour → light through wall*): bright = thin. *cut it open* (or the toolbar's *cut*) shows the relief itself.
- **Size**: the *diameter* slider sits in **base mesh**, right under the sphere's settings, with the print size shown below it. The same value is in vessel and in print's **size**. The view always fits the sphere to the screen, so the sphere doesn't grow on screen. Watch the *print size* line instead. Type it, use the 80–200 buttons, or *Fit bed*. The walls stay as set in mm at any size.
- **Printing**: print exports the vessel **exactly**, with no voxel remesh, so the sphere stays perfect. It is set upside down (*-Y up*).
  - An opening of 35° or more needs no support outside.
- **Preset**: `presets/vessel_lithophane.json` (64 mm). A structured, deep relief from pure weights, with no noise:
  - Faces push out and corners pull in, then the edges sink into ribs.
  - The panels swell, then alternate faces and edges into nested coffers.
  - Fine ribbing finishes it.
  - Lit from inside, it reads like a rose window: a lattice of glowing windows framed by dark ribs.
  - At 120 mm the walls are 0.8–7.8 mm: about 136 g of PLA and a 48 MB STL.

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
python render.py presets/cube_six_arms.json --depth 8             # → renders/cube_six_arms.png
python render.py presets/*.json --depth 6 --montage renders/all.png
python render.py presets/panel_tileable.json --theme light        # paper-white background
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
- **Attractors**:
  - eq. 8–9 blending as a partition of unity, plus background blending and modifiers by iteration range.
  - Falloff shapes, curve geometry, curve distance (exact and fast, with a tested error bound) and polyline import.
  - Rest vs current positions.
  - Per-iteration cache invalidation: editing iteration k recomputes only from k.
- **Function layers**:
  - Every built-in field and fold is bounded, finite and deterministic. The gyroid formula, Perlin continuity, Worley F1 ≤ F2, kaleidoscope wedges and SH symmetry are checked.
  - The five blend modes, iteration ranges, attractor masks, displacement and fold amounts, and per-level caching.
  - Plug-in loading, errors and reloading.
- **Intrinsic features**:
  - Motif labels (CC adds exactly `4F4E`).
  - eq. 10 and 11 numerically.
  - Topological distances (Fig. 8), planarity and bend.
  - Rules, tag inheritance through CC and DS, rule-based selection.
  - **Locked edges stay straight creases** (Fig. 9) and release after L iterations.
- **Merging**: pairwise only; topological neighbours are never welded; the max valence opens holes; the result stays manifold and can be subdivided further.
- **Printing**:
  - Closed, watertight output for self-intersecting, open, porous and relief forms, at full and half detail.
  - Requested size and wall thickness are kept.
  - Thin features are flagged, the up-axis rotation is correct, and the voxel cap applies.
  - The STL validator catches truncated and holed files.
  - Up rotations are proper (never mirrored), size can be set along any axis, and *Auto* stands a column up.
  - Halves are closed and stand on a flat cut face. They need much less support than the whole.
  - Quarter joints meet exactly, explode correctly, and never overlap in the print layout.
  - Pin holes have the requested volume and keep each part a single solid. Per-part STLs verify.
  - The support threshold angle follows slicer conventions (higher = more support).
  - Self-supporting undersides keep every layer within the slicer's angle of the one below and only ever add material. The flat foot is flat.
  - *facing +y* masks leave downward-facing vertices untouched.
  - The cage base is a closed genus-5 frame, and *clamp to box* flattens the outside.
  - The cube cage preset prints as 4 single-piece quarters, each lying on a flat face.
  - **Vessel**:
    - The open sphere base has one circular opening.
    - The vessel is a closed, outward-oriented shell: exact sphere outside, walls within the requested band, a flat rim.
    - Thinner walls light up brighter, and presets round-trip.
    - The exact print stands on its rim, verifies watertight, and needs almost no support.
  - A full disk is refused cleanly (no partial file), writes are atomic, and the worker process runs.
- **Turntable**: GIF frames orbit (skipped without OpenGL).
- Affine and scale invariance, caching, the face budget, JSON round-trips, and that every preset loads.
- **Speed-ups change nothing**: connectivity tables shared between meshes with the same faces, the one-sort edge table, the Worley cell table and the background bake all give results identical to building everything from scratch. The level cache stays under its memory cap, and the shared-memory hand-over to the background process round-trips.

## Layout

```
app.py              interactive Polyscope app: sidebar, inspector, toolbar, bottom bar, undo, shortcuts,
                    and the presets / base mesh / schedule / export sections
ui_style.py         the look (site palette, IBM Plex Mono, dark / light) and the widgets every section uses
ui_attractors.py    03 attractors (gizmo, curves, falloff, sets, modifiers)
ui_layers.py        04 function layers
ui_intrinsic.py     05 groups / 06 intrinsic (motifs, measures) / 07 porosity (click-to-tag picking)
ui_vessel.py        08 vessel (lithophane sphere)
ui_print.py         09 print (FDM) and the turntable in 10 export
ui_section.py       section cut (toolbar)
ui_common.py        shared rule editors
assets/fonts/       IBM Plex Mono (SIL Open Font Licence)
render.py           headless preset → PNG
hansmeyer/
  mesh.py           polygon mesh, vectorised half-edges (shared between meshes with the same faces), normals,
                    per-vertex attributes
  catmull_clark.py  extended CC (eq. 1–4), smooth / locked boundaries
  doo_sabin.py      extended DS (eq. 5–6)
  schedule.py       weight definitions, per-iteration schedule, Design (= preset JSON)
  pipeline.py       runs a Design: caching, face budget, boundary fade
  shapes.py         base mesh registry (Platonic solids, columns, panel, cage, open sphere, OBJ)
  attractors.py     eq. 8-9 weight sets, modifiers, falloffs, curves, distances
  functions.py      field / fold registry, built-ins, plug-in loader (@field, @fold)
  noise.py          vectorised Perlin, fBm, ridged, warped, Worley
  layers.py         function layer stack (weights, displacement, folds)
  intrinsic.py      groups & locks, motifs (eq. 10-11), measures & rules
  merge.py          vertex merging / porosity
  printprep.py      watertight voxel remesh for FDM printing (+ thin-feature check), exact export for vessels
  vessel.py         thin-walled lithophane sphere: exact outer sphere, relief inside, flat rim
  meshio.py         OBJ import + cleanup; OBJ / STL / PLY export
  view.py           shared rendering style and dark/light themes, scroll zoom
  background.py     the worker process for bakes and print prep (data passed through shared memory)
presets/            JSON designs
functions/          your plug-in functions (examples included)
inputs/             drop .obj meshes / curve files here (samples included)
tests/              pytest suite + independent reference implementation
```

## Faithfulness to the paper

- **From the paper:**
  - eq. 1–6, non-stationary weights, the CC + DS combination, and provenance (corner/edge/face points; face/edge/vertex-derived faces).
  - The Fig. 4 / Fig. 7 column inputs.
  - **eq. 8–9 extrinsic weight sets** (Fig. 4 zoning, Fig. 5 planar or spatial sets).
  - **Intrinsic parameters:** topological motifs with attractor/deflector values (**eq. 10–11**), topological distance (**Fig. 8**), planarity, tagging and **locking** (**Fig. 9**).
  - **Vertex merging with a maximum valence** (porosity).
- **Our extensions (labelled in code):**
  - Eq. 1/3 applied to n-gons and any valence.
  - A Doo-Sabin `w1` generalisation for n ≠ 3, 4.
  - Relative extrusion (the paper doesn't say whether normals are unit length; the *absolute* mode is the literal reading).
  - Boundary rules and the locked/tileable mode for open meshes.
  - Attractors:
    - A radius and choice of falloff (the paper normalises distance by the largest possible distance).
    - Curve attractors.
    - Modifier attractors.
    - The background set.
    - The current/original measuring position.
  - The whole function layer stack (TPMS, noise, analytic fields, folds, plug-ins).
  - The bend measure.
  - Merging details. The paper only says proximate vertices are joined and faces can't form beyond the max valence. We weld *pairs* of vertices that don't share a face, within a fraction of the local edge length, and drop the faces that would break manifoldness.
- **Fig. 3:** the paper publishes no weight values, so the presets reproduce the *character* of the figures, not exact geometry. The left figure's arms are there, but its fluted fans at the arm tips aren't yet.
- **Fig. 4:** reproduced in character by `column_box_zoned`: four weight sets along y, a base and three sections with gradients. Again, the paper gives no weight values.

## Roadmap

1. ✅ Core engine, tests, minimal viewer, Fig. 3 presets
2. ✅ Dark/light theme, Platonic solids, columns (Fig. 4/7), open panel with smooth/locked boundaries, OBJ import, OBJ/STL/PLY export, schedule overview
3. ✅ Attractors (points and space curves, falloff curves, weight sets and modifiers, gizmo dragging, influence map)
4. ✅ Function layer stack (TPMS, noise, analytic, folds, domain folds) and a `functions/` plug-in folder
5. ✅ Tagging/locking (Fig. 9), motifs (eq. 10–11, Fig. 7), topological distance (Fig. 8) and curvature, vertex merging (porosity), colour-by maps
6. ✅ Watertight voxel remesh for FDM printing (verified STL export, thin-feature check), turntable GIF/MP4
7. ✅ UI redesign in the look of nijatmahamaliyev.com: sidebar + inspector, sections by workflow, undo / redo, shortcuts, presets renamed by family

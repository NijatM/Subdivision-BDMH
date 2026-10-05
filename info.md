# Subdivision as a Generative System — Research Notes

Inspired by Michael Hansmeyer & Benjamin Dillenburger.
Primary reference: Hansmeyer, M. (2010). *From Mesh to Ornament: Subdivision as a Generative System*, eCAADe 28, ETH Zurich — `Ref/ecaade2010_192.content.pdf`

---

# Part 1 — Short explanation

**Subdivision** is a standard computer-graphics technique: take a coarse polygon mesh, split every face into smaller faces, and place the new vertices at weighted averages of their neighbours. Repeat a few times and a blocky cube becomes a smooth blob. Catmull-Clark and Doo-Sabin are the classic recipes, designed only for **smoothing**.

**Hansmeyer's insight:** keep the splitting (topology) rule but change the averaging (weighting) rule. He adds a handful of extra parameters ("weights") that bias where new points go and push them in or out along surface normals. With weights = 0 you get ordinary smoothing. With other values, each split adds folds, ridges, branches, holes and self-similar detail, and the effect compounds every iteration. After 6–8 iterations a 6-face cube becomes hundreds of thousands of faces of ornament.

**You don't design the object; you design the process.** You choose an input mesh plus a set of weights per iteration and per region, and the algorithm produces detail at every scale, from overall proportion down to micro-texture. The line runs from the 2010 paper to subdivided columns, two 3D-printed sandstone rooms (*Digital Grotesque I & II*) and a 30 m 3D-printed concrete tower (*Tor Alva*, 2025).

---

# Part 2 — Full summary

## 1. The core paper (eCAADe 2010)

### 1.1 Two parts of every subdivision scheme
- **Topological rules** — how connectivity is refined (1 quad → 4 quads; new face, edge and corner points).
- **Weighting rules** — where the new vertices are placed (weighted averages of old vertices).

Hansmeyer leaves the topology rules alone and adds parameters to the weighting rules. That one change turns a smoothing algorithm into a form generator.

### 1.2 Extended Catmull-Clark (quad-mesh input)

| New point | Equation | Role of the new weight |
|---|---|---|
| **Face point** (1) | `F' = (P1+P2+P3+P4)/4 + n_f · w_f` | `w_f` pushes the face centre out/in along the face normal → bumps and pits |
| **Edge point** (2) | `E' = ((F1+F2)(1+w1) + (P1+P4)(1−w1))/4 + n_e · w_e` | `w1` biases toward the face points vs. the edge endpoints; `w_e` extrudes along the edge normal (average of adjacent face normals) |
| **Corner point** (3) | `C' = (F(1+w2) + 2E(1−w2/2) + (i−3)C)/i + n_c · w_c` | `w2` biases between the average face point `F` and the average edge midpoint `E`; `w_c` extrudes along the previous vertex normal (`i` = valence) |
| **Face point, later iterations** (4) | `F' = ((V'(1+w3) + F'(1−w3))(1+w4) + (E'1+E'2)(1−w4))/4 + n_f · w_f` | After iteration 1, every vertex is known to be a former face, edge or corner point, so each type gets its own weight (`w3`, `w4`) |

**With all weights at 0 the equations reduce exactly to standard Catmull-Clark**, so the smooth result is one special case of the parameter space.

Stencils (paper Fig. 1): face point = 1,1,1,1 · edge point = (1+w1) on face points, (6−2w1) on endpoints · corner point = (1+w2) face, 6 edge, (15−3w2) centre.

### 1.3 Extended Doo-Sabin
- Quad (5): `P1' = (P1(2.25+2w1) + (P2+P4)(0.75−w1) + 0.25·P3)/4 + n_f · w_f`  (w1 = 0 gives the classic 9/16, 3/16, 3/16, 1/16 stencil)
- Triangle (6): `P1' = (2/3)·P1(1+w1/2) + (1/6)(P2+P3)(1−w1) + n_f · w_f`
- After one iteration, new faces are classed as coming from an original face, an edge or a vertex, and each class can get its own `w1`/`w_f`.
- **The two schemes can be mixed**: the output of either is valid input for the other, so they can alternate between iterations (paper Fig. 3, right).

### 1.4 What the weights produce
Concavity and convexity, branching, porosity, fractalization — globally or locally, and combinable into ornament.

### 1.5 Two key control ideas
1. **Non-stationary weights** — different values at each iteration. Iteration 1 might set large-scale massing, 2–4 mid-scale folds, 5–8 fine texture. Some features come from a single iteration; others only emerge from the interplay of several.
2. **Non-uniform weights** — different values in different places. Without them, a cube's output keeps the cube's symmetry; with them you get local variation (e.g. base / shaft / capital on a column).

### 1.6 How to assign non-uniform weights

**A. Extrinsic (from the environment; suits simple uniform input meshes)**
- Place several **weight sets** (complete parameter bundles) at points in space, like attractors.
- Each face blends the sets by distance:
  - (8) `c_a = (1 − dist(pos_face, pos_set_a))^t · h_a  /  Σ_i (1 − dist(pos_face, pos_set_i))^t · h_i`
    - `h` = strength of a set, `t > 1` = "tightness" exponent, max distance normalised to 1
  - (9) `w_x = Σ_i w_{x,i} · c_i`
- Example: four sets stacked along the Y axis produce a column with a base and three zones that blend smoothly (Figs. 4, 10). Moving, rotating or scaling the mesh relative to the sets changes the output.
- Sets can lie on an axis, in a plane (Fig. 5) or anywhere in 3D space.

**B. Intrinsic (from the mesh itself; suits differentiated input meshes)**
1. **Topological motifs** — classify vertices by valence and incident faces (labels like `2F3E`, `4F4E`, `5F5E`; Fig. 6). Assign a weight set per motif, or an attractor/deflector value `U` used after equations (1) and (2):
   - (10) `F'_attracted = F' + w6 · Σ_{i=1..n} (P_i − F') · u_{P_i}`
   - (11) `E'_attracted = E' + w7 · ((P1 − E') · u_{P1} + (P2 − E') · u_{P2})`
2. **Topological distance** — number of edges from a vertex to an original edge or vertex. Interpolate between a minimum and maximum value of each weight according to that distance (Fig. 8).
3. **Topographical curvature** — measure face planarity and use it to amplify/attenuate weights, or to interpolate between sub-values.

**C. Tagging and locking (both mesh types)**
- Assign vertices/faces to groups, each with its own weight set.
- **Lock** vertices for `L` iterations → creases, hard edges, spikes, curvature control (Fig. 9). Same idea as Pixar's *Geri's Game* (DeRose, Kass, Truong 1998).

### 1.7 Vertex merging (the non-linear trick)
If new vertices that land within a minimum distance are **welded**, the topology changes and new motifs appear that weren't in the input. Capping the maximum valence of a welded vertex means some faces can't form → **porosity** (holes). This is the one place where a small parameter change can make the output jump abruptly.

### 1.8 Uniform vs. differentiated input mesh

| | Uniform input (e.g. a cube) | Differentiated input (e.g. a 64-face rough column) |
|---|---|---|
| Freedom | Highest — ornament as "an object in itself" | Constrained by the input geometry |
| Control | Harder; needs many non-uniform, non-stationary weights | Easier, more predictable, good for families of forms |
| Risk | Can look arbitrary | Can look cellular / repetitive / made of pieces |
| Iterations | More (~8 for the Fig. 4 columns) | Fewer (6 iterations ≈ 250k faces) |

**Face-count check:** each Catmull-Clark iteration multiplies the quad count by 4. 64 × 4⁶ = 262,144 (paper: "~250,000 faces"). A cube after 8 iterations: 6 × 4⁸ = 393,216 faces.

### 1.9 Paper figures worth reproducing
- **Fig. 3** — three cubes after 8 iterations with non-stationary weights (left & centre: extended Catmull-Clark; right: Catmull-Clark + Doo-Sabin combined).
- **Fig. 4 / Fig. 10** — elongated cube + base, four non-stationary weight sets placed along Y → columns.
- **Fig. 7 / Fig. 11** — 64-face column input, weights from 3 intrinsic motifs (`5F5E`, `4F4E`, `2F3E`), 6th generation ≈ 250k faces.
- **Fig. 9** — 7th-generation cube with one edge locked for `Le` iterations.

### 1.10 Conclusions of the paper
- The results are **non-reductionist**: you can't infer the input or the parameters from the output.
- Yet the process is **deterministic, reproducible and mostly linear**: small weight changes give gradual, traceable changes (vertex merging is the exception).
- Future work: catalogue which parameters produce which attributes; add geometric constraints on intermediate meshes, so that subdivision can "transcend [its] role as smoothing algorithms" and become a generative system.

---

## 2. Project lineage

| Year | Project | What's new | Fabrication |
|---|---|---|---|
| 2008 | **Platonic Solids** | First experiments: the five Platonic solids run through modified Catmull-Clark / Doo-Sabin with normal extrusion; only the division parameters change | Digital renders |
| 2008 | **Cabinet of Cupolas** | Dome studies; early mesh-grammar thinking | Digital |
| 2010 | **eCAADe paper** + **Column Prototype I** | Abstracted Doric column as input (shaft, base, capital, fluting, entasis encoded in the mesh), tagged regions; ~6 M faces | **2,700 laser-cut 1 mm greyboard sheets** on a wooden core; 2.7 m; 20 km of cutting path |
| 2011 | **The Sixth Order** | A family of columns — "a new column order" | 10,800 sheets of 1 mm ABS |
| 2012 | **TED Global: "Building Unimaginable Shapes"** | Popularised the work; framed as cell division / morphogenesis | — |
| 2013 | **Mesh Grammars** (with Dillenburger) | Subdivision + **shape grammars**: rules read a mesh's topological/topographical properties and act locally; objects are always treated as embedded in a network (the mesh) | — |
| 2013 | **Digital Grotesque I** | First human-scale room made entirely of 3D-printed sandstone; starts from a simple cube; 260 M surfaces, 42 bn voxels, 78 GB of data | Voxeljet sand printer, 0.3 mm layers; mortar-free "smart bricks" with hollow, force-flow-optimised interiors; 3.2 m; 1 year design / 1 month print / 1 day assembly |
| 2017 | **Digital Grotesque II** (Centre Pompidou, *Imprimer le monde*; permanent collection) | **Topology-changing subdivision** (moves between genera — handles, holes) → porous, multi-layered branching depth. **The system evaluates its own output**: viewer viewpoints simulated; 16,386 variants scored on soft criteria (experience-ability, compressibility, depth-complexity); computed on ETH's Euler cluster | 1.35 bn surfaces; 7 t sandstone; 3.45 m; 2 years of design |
| 2017 | **Astana Columns** (Expo 2017) | Column series at larger scale | 20,000 sheets of 0.6 mm greyboard |
| 2019 | **Muqarna Mutation** (Mori Art Museum, Tokyo) | **Selective recursive subdivision**: each tile decides from its own properties whether to split further | Robot-milled EPS + ~15,000 aluminium tubes; 2.4 m |
| 2025 | **Tor Alva / White Tower** (Mulegns, CH) | Building scale: 32 columns over 4 storeys, thinner and more branched toward the top | Robotically 3D-printed hollow concrete with integrated reinforcement; 30 m — tallest 3D-printed building |

**Benjamin Dillenburger** now leads Digital Building Technologies at ETH Zurich; the same sand-printing research fed structural applications such as the **Smart Slab** at DFAB House (80 m² concrete ceiling cast on 3D-printed sand formwork).

Note: Hansmeyer's site lists Digital Grotesque I at 5.8 t; some press reports say 11 t.

---

## 3. Underlying ideas
1. **Design the process, not the object.** The designer writes rules and explores the space of outcomes.
2. **One operation, all scales.** A single algorithm sets overall proportion, mid-scale folds and micro-texture — no repeated components.
3. **Ornament without an alphabet.** Classical ornament assembles predefined motifs; this approach is operational, with no "a priori idealized components."
4. **Intention vs. emergence.** The designer steers; results can "astonish even their creators."
5. **Designer as curator** (DG I) → **designer + evaluating machine** (DG II).
6. **Fabrication makes complexity free.** In sand printing a billion facets cost the same as a flat block; "highly individualized elements [become] as affordable as a standardized series." Slicing into laser-cut layers achieves the same with cheap machines.

---

## 4. Implementation notes (for our engine)
1. **Catmull-Clark (and optionally Doo-Sabin) with the weights built into the stencils.** Off-the-shelf tools (Weaverbird, Blender Subsurf, OpenSubdiv) don't expose the stencils, so we write our own.
2. **Per-iteration weight schedule** (non-stationary).
3. **Spatial variation**: attractor weight sets (eqs. 8–9), vertex/face tags, later valence motifs and curvature.
4. **Normals** for faces, edges (average of the two adjacent faces) and vertices.
5. **Fabrication**: extrusion causes self-intersections, so meshes for FDM/resin printing need voxel remeshing or boolean cleanup to become watertight. Alternatively use Hansmeyer's column method — slice into layers and laser-cut — which handles self-intersections automatically.

---

## Sources
- Hansmeyer, M. (2010). *From Mesh to Ornament*, eCAADe 28 — `Ref/ecaade2010_192.content.pdf`
- [Digital Grotesque I](https://michael-hansmeyer.com/digital-grotesque-I)
- [Digital Grotesque II](https://michael-hansmeyer.com/digital-grotesque-II) · [ETH DBT](https://center.dbt.arch.ethz.ch/?p=35) · [Designboom: 1.35 billion surfaces](https://www.designboom.com/architecture/digital-grotesque-grotto-2-3d-printed-michael-hansmeyer-benjamin-dillenburger-07-14-2017/)
- [Subdivided Columns](https://michael-hansmeyer.com/subdivided-columns) · [Platonic Solids](https://michael-hansmeyer.com/platonic-solids) · [Cabinet of Cupolas](https://michael-hansmeyer.com/cupolas) · [Muqarna Mutation](https://michael-hansmeyer.com/muqarnas) · [Project index](https://michael-hansmeyer.com/llms.txt)
- [TED: Building Unimaginable Shapes](https://www.ted.com/talks/michael_hansmeyer_building_unimaginable_shapes)
- [Mesh Grammars — CreativeApplications](https://www.creativeapplications.net/project/mesh-grammars-studies-for-a-dome-by-dillenburger-hansmeyer) · [Evolo](https://www.evolo.us/?p=18866)
- [Dezeen: DG I prototype](https://www.dezeen.com/2013/06/26/digital-grotesque-the-worlds-first-3d-printed-room/amp/) · [Architect Magazine on DG II](https://www.architectmagazine.com/technology/michael-hansmeyers-dizzying-piece-for-the-centre-pompidou_o)
- [Dezeen: Tor Alva](https://www.dezeen.com/2025/05/21/worlds-tallest-3d-printed-tower-tor-alva/) · [3DPrinting.com: Tor Alva](https://3dprinting.com/news/30-meter-3d-printed-tor-alva-tower-unveiled-in-swiss-alps/)
- [Dezeen: ETH Smart Slab](https://www.dezeen.com/2018/08/03/eth-zurich-makes-light-concrete-ceiling-using-3d-sand-printing/)
- [Wikipedia: Michael Hansmeyer](https://en.wikipedia.org/wiki/Michael_Hansmeyer)

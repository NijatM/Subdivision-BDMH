// Base (input) meshes; port of hansmeyer/shapes.py. Every shape is registered with its parameters so the
// UI can build sliders automatically. A base spec is an object: {shape: name, ...params}.

import { PolyMesh } from "./mesh.js";
import { vertexFans } from "./dooSabin.js";
import { loadObj } from "./meshio.js";
import { linspace, pyRound } from "./util.js";

const PHI = (1 + 5 ** 0.5) / 2;
export const UNIT_RADIUS = 3 ** 0.5; // circumradius of the default cube, used for all solids

const P = (name, label, lo, hi, def, integer = false, help = "") => ({ name, label, lo, hi, default: def, integer, help });

// ------------------------------------------------------------------ helpers
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Wind faces of a convex solid centred at the origin so normals point outward. */
function orientConvex(V, faces) {
  return faces.map((f) => {
    const p = f.map((i) => V[i]);
    const n = [0, 0, 0];
    for (let i = 0; i < p.length; i++) {
      const c = cross(p[i], p[(i + 1) % p.length]);
      n[0] += c[0]; n[1] += c[1]; n[2] += c[2];
    }
    const m = [0, 0, 0];
    for (const q of p) { m[0] += q[0] / p.length; m[1] += q[1] / p.length; m[2] += q[2] / p.length; }
    return n[0] * m[0] + n[1] * m[1] + n[2] * m[2] > 0 ? [...f] : [...f].reverse();
  });
}

/** All vertex triples whose three sides equal the shortest vertex distance (regular deltahedra). */
function trianglesByEdgeLength(V) {
  const n = V.length;
  const D = [];
  let L = Infinity;
  for (let i = 0; i < n; i++) {
    D.push([]);
    for (let j = 0; j < n; j++) {
      const d = Math.sqrt((V[i][0] - V[j][0]) ** 2 + (V[i][1] - V[j][1]) ** 2 + (V[i][2] - V[j][2]) ** 2);
      D[i].push(d);
      if (d > 1e-9 && d < L) L = d;
    }
  }
  const adj = (i, j) => Math.abs(D[i][j] - L) < 1e-6 * L;
  const out = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
    if (adj(i, j) && adj(j, k) && adj(i, k)) out.push([i, j, k]);
  }
  return out;
}

function scaleToRadius(V, radius) {
  let m = 0;
  for (const p of V) m = Math.max(m, Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]));
  return V.map((p) => p.map((x) => x * (radius / m)));
}

function rows(Vflat) {
  const out = [];
  for (let i = 0; i < Vflat.length; i += 3) out.push([Vflat[i], Vflat[i + 1], Vflat[i + 2]]);
  return out;
}

/** Dual mesh of a closed manifold: face centroids become vertices, vertex fans become faces. */
export function dual(m) {
  const faces = [];
  const hf = m.heFace;
  for (const { k, fans } of vertexFans(m)) {
    for (let i = 0; i < fans.length; i += k) {
      const f = [];
      for (let j = 0; j < k; j++) f.push(hf[fans[i + j]]);
      faces.push(f);
    }
  }
  return PolyMesh.fromFaces(Float64Array.from(m.faceCentroid), faces);
}

// ------------------------------------------------------------ platonic solids
export function tetrahedron(radius = UNIT_RADIUS) {
  const V = scaleToRadius([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], radius);
  return PolyMesh.fromFaces(V, orientConvex(V, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]));
}

export function cube(size = 2.0) {
  const h = size / 2, V = [];
  for (const x of [-h, h]) for (const y of [-h, h]) for (const z of [-h, h]) V.push([x, y, z]);
  const F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
  return PolyMesh.fromFaces(V, F);
}

/** Closed, outward-oriented quad surface of a set of grid cells (filled[x][y][z]), centred on the origin. */
export function polycube(filled, dims, cell = 1.0) {
  const [nx, ny, nz] = dims;
  const at = (x, y, z) => x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && filled[(x * ny + y) * nz + z];
  const quads = [];
  for (let a = 0; a < 3; a++) {
    const b = (a + 1) % 3, c = (a + 2) % 3;
    for (const sign of [1, -1]) {
      const corners = sign > 0 ? [[0, 0], [1, 0], [1, 1], [0, 1]] : [[0, 0], [0, 1], [1, 1], [1, 0]];
      for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
        if (!filled[(x * ny + y) * nz + z]) continue;
        const nb = [x, y, z];
        nb[a] += sign;
        if (at(nb[0], nb[1], nb[2])) continue;
        const q = [];
        for (const [db, dc] of corners) {
          const pt = [x, y, z];
          pt[a] += sign > 0 ? 1 : 0;
          pt[b] += db;
          pt[c] += dc;
          q.push(pt);
        }
        quads.push(q);
      }
    }
  }
  // np.unique(axis=0): points sorted lexicographically, faces through the inverse
  const K = (p) => (p[0] * (ny + 1) + p[1]) * (nz + 1) + p[2];
  const keys = new Set();
  for (const q of quads) for (const p of q) keys.add(K(p));
  const sorted = [...keys].sort((u, v) => u - v);
  const index = new Map(sorted.map((k, i) => [k, i]));
  const V = sorted.map((k) => {
    const z = k % (nz + 1), r = (k - z) / (nz + 1), y = r % (ny + 1), x = (r - y) / (ny + 1);
    return [(x - 0.5 * nx) * cell, (y - 0.5 * ny) * cell, (z - 0.5 * nz) * cell];
  });
  return PolyMesh.fromFaces(V, quads.map((q) => q.map((p) => index.get(K(p)))));
}

/** A cube frame: the 12 edges of a cube as square bars, all six faces open. */
export function cage(size = 2.0, bar = 0.2, segments = 5) {
  const n = Math.max(Math.trunc(segments), 3);
  const band = Math.trunc(Math.min(Math.max(pyRound(bar * n), 1), Math.floor((n - 1) / 2)));
  const edge = (i) => i < band || i >= n - band;
  const filled = new Uint8Array(n * n * n);
  for (let x = 0; x < n; x++) for (let y = 0; y < n; y++) for (let z = 0; z < n; z++) {
    filled[(x * n + y) * n + z] = edge(x) + edge(y) + edge(z) >= 2 ? 1 : 0;
  }
  return polycube(filled, [n, n, n], size / n);
}

/** A unit sphere with a circular opening on top (+y): rings of quads from the opening down to a triangle
 * fan at the bottom pole. `opening` is the opening's angular radius from the top, in degrees. */
export function sphereOpen(sides = 16, rings = 10, opening = 40.0) {
  const n = Math.max(Math.trunc(sides), 3), k = Math.max(Math.trunc(rings), 2);
  const t0 = (Math.min(Math.max(Number(opening), 5.0), 150.0) * Math.PI) / 180;
  const theta = linspace(t0, Math.PI, k + 1).slice(0, k);
  const V = [];
  for (let r = 0; r < k; r++) for (let i = 0; i < n; i++) {
    const T = theta[r], Pp = (2 * Math.PI * i) / n;
    V.push([Math.sin(T) * Math.cos(Pp), Math.cos(T), Math.sin(T) * Math.sin(Pp)]);
  }
  V.push([0.0, -1.0, 0.0]);
  const pole = V.length - 1;
  const faces = [];
  for (let r = 0; r < k - 1; r++) for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    faces.push([r * n + i, (r + 1) * n + i, (r + 1) * n + j, r * n + j]);
  }
  for (let i = 0; i < n; i++) faces.push([(k - 1) * n + i, pole, (k - 1) * n + ((i + 1) % n)]);
  let m = PolyMesh.fromFaces(V, faces);
  const A = m.faceAreaVec, C = m.faceCentroid;
  let s = 0;
  for (let i = 0; i < A.length; i++) s += A[i] * C[i];
  if (s < 0) m = PolyMesh.fromFaces(V, faces.map((f) => [...f].reverse()));
  return m;
}

export function octahedron(radius = UNIT_RADIUS) {
  const V = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].map((p) => p.map((x) => x * radius));
  return PolyMesh.fromFaces(V, orientConvex(V, trianglesByEdgeLength(V)));
}

export function icosahedron(radius = UNIT_RADIUS) {
  let V = [];
  for (const a of [-1, 1]) for (const b of [-1, 1]) V.push([0, a, b * PHI], [a, b * PHI, 0], [b * PHI, 0, a]);
  V = scaleToRadius(V, radius);
  return PolyMesh.fromFaces(V, orientConvex(V, trianglesByEdgeLength(V)));
}

export function dodecahedron(radius = UNIT_RADIUS) {
  const d = dual(icosahedron());
  return PolyMesh.fromFaces(scaleToRadius(rows(d.V), radius), d.facesList());
}

// --------------------------------------------------------------- lathe solids
/** Revolve a profile [[apothem r, height y], ...] (bottom to top) into a closed prism-like solid. */
export function lathe(profile, sides, twist = 0.0) {
  sides = Math.trunc(sides);
  const m = profile.length;
  const corner = 1.0 / Math.cos(Math.PI / sides);
  const V = [];
  const y0 = profile[0][1], span = Math.max(profile[m - 1][1] - y0, 1e-9);
  for (const [r, y] of profile) {
    const t = (twist * (y - y0)) / span;
    for (let j = 0; j < sides; j++) {
      const th = Math.PI / sides + (2 * Math.PI * j) / sides + t;
      V.push([r * corner * Math.cos(th), y, r * corner * Math.sin(th)]);
    }
  }
  const idx = (i, j) => i * sides + (((j % sides) + sides) % sides);
  const F = [];
  for (let i = 0; i < m - 1; i++) for (let j = 0; j < sides; j++) F.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
  const bottom = [], top = [];
  for (let j = 0; j < sides; j++) bottom.push(idx(0, j));
  for (let j = sides - 1; j >= 0; j--) top.push(idx(m - 1, j));
  F.push(bottom, top);
  return PolyMesh.fromFaces(V, F);
}

function centreY(m) {
  const V = Float64Array.from(m.V);
  let lo = Infinity, hi = -Infinity;
  for (let i = 1; i < V.length; i += 3) { lo = Math.min(lo, V[i]); hi = Math.max(hi, V[i]); }
  const c = 0.5 * (lo + hi);
  for (let i = 1; i < V.length; i += 3) V[i] -= c;
  return PolyMesh.fromFaces(V, m.facesList());
}

/** Paper Fig. 4: an elongated cube on a base (square cross-section). */
export function columnBox(height = 6.0, shaftWidth = 0.6, baseWidth = 1.0, baseHeight = 0.5, shaftSegments = 1) {
  const prof = [[baseWidth, 0.0], [baseWidth, baseHeight]];
  for (const t of linspace(0, 1, Math.trunc(shaftSegments) + 1)) prof.push([shaftWidth, baseHeight + t * (height - baseHeight)]);
  return centreY(lathe(prof, 4));
}

/** Paper Fig. 7 style: base, shaft (with entasis and taper), echinus and abacus. */
export function columnClassic(sides = 8, height = 6.0, baseWidth = 0.95, baseHeight = 0.45, shaftWidth = 0.55,
  entasis = 0.08, taper = 0.12, capitalWidth = 1.0, capitalHeight = 0.8, shaftRings = 4) {
  const topShaft = height - capitalHeight;
  const prof = [[baseWidth, 0.0], [baseWidth, baseHeight]];
  for (const t of linspace(0, 1, Math.trunc(shaftRings) + 1)) {
    const r = shaftWidth * (1 + entasis * Math.sin(Math.PI * t)) * (1 - taper * t);
    prof.push([r, baseHeight + t * (topShaft - baseHeight)]);
  }
  const rTop = prof[prof.length - 1][0];
  prof.push([(rTop + capitalWidth) / 2, topShaft + 0.45 * capitalHeight]); // echinus
  prof.push([capitalWidth, topShaft + 0.7 * capitalHeight]); // abacus
  prof.push([capitalWidth, height]);
  return centreY(lathe(prof, Math.trunc(sides)));
}

/** Open quad grid in the x/y plane facing +z: a relief tile. */
export function panel(nx = 4, ny = 4, width = 2.0, height = 2.0, bend = 0.0, saddle = 0.0) {
  nx = Math.trunc(nx);
  ny = Math.trunc(ny);
  const xs = linspace(-width / 2, width / 2, nx + 1), ys = linspace(-height / 2, height / 2, ny + 1);
  const V = [];
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
    const X = xs[i], Y = ys[j], u = (2 * X) / width, v = (2 * Y) / height;
    const Z = 0.5 * bend * (1 - u ** 2) * width / 2 + 0.5 * saddle * (u ** 2 - v ** 2) * width / 2;
    V.push([X, Y, Z]);
  }
  const F = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const a = j * (nx + 1) + i;
    F.push([a, a + 1, a + nx + 2, a + nx + 1]);
  }
  return PolyMesh.fromFaces(V, F);
}

// ----------------------------------------------------------------------- obj
export function obj(path = "", normalize = true, weld = true, ctx = {}) {
  if (!path) throw new Error("choose an .obj file to import");
  const text = ctx.objText ? ctx.objText(path) : undefined;
  if (text === undefined || text === null) throw new Error(`${path}: file not found (import it again)`);
  return loadObj(text, path, { normalize: !!normalize, weld: !!weld });
}

// ------------------------------------------------------------------ registry
const R = P("radius", "radius", 0.5, 4.0, UNIT_RADIUS);
export const SHAPES = {};
const reg = (name, label, build, params, view, help, extra = {}) => {
  SHAPES[name] = { name, label, build, params, view, help, extraDefaults: extra };
};
reg("cube", "Cube", (k) => cube(k.size), [P("size", "size", 0.5, 4.0, 2.0)], "diagonal", "Paper Fig. 3 input.");
reg("cage", "Cube cage (frame)", (k) => cage(k.size, k.bar, k.segments), [
  P("size", "size", 0.5, 4.0, 2.0),
  P("bar", "bar width", 0.05, 0.45, 0.2, false, "Bar width as a share of the cube's side."),
  P("segments", "segments", 3, 15, 5, true, "Quads along each side: more = finer input mesh."),
], "diagonal", "The 12 edges of a cube as bars, all six faces open. Add a 'clamp to box' fold layer for flat outer faces.");
reg("sphere_open", "Sphere with opening (vessel)", (k) => sphereOpen(k.sides, k.rings, k.opening), [
  P("sides", "sides", 4, 48, 16, true, "Quads around."),
  P("rings", "rings", 2, 32, 10, true, "Quads from the opening to the bottom pole."),
  P("opening", "opening (deg)", 10.0, 80.0, 40.0, false,
    "Angular radius of the top opening. The opening's diameter = sphere diameter x sin(angle)."),
], "three_quarter", "A sphere with a circular opening on top. With Vessel on, it becomes a thin-walled lithophane sphere: smooth outside, the subdivision relief inside.");
reg("tetrahedron", "Tetrahedron", (k) => tetrahedron(k.radius), [R], "three_quarter", "Platonic solid, 4 triangles.");
reg("octahedron", "Octahedron", (k) => octahedron(k.radius), [R], "diagonal", "Platonic solid, 8 triangles.");
reg("dodecahedron", "Dodecahedron", (k) => dodecahedron(k.radius), [R], "diagonal", "Platonic solid, 12 pentagons.");
reg("icosahedron", "Icosahedron", (k) => icosahedron(k.radius), [R], "diagonal", "Platonic solid, 20 triangles.");
reg("column_box", "Column: box on base (Fig. 4)", (k) => columnBox(k.height, k.shaft_width, k.base_width, k.base_height, k.shaft_segments), [
  P("height", "height", 2.0, 12.0, 6.0),
  P("shaft_width", "shaft half-width", 0.2, 2.0, 0.6),
  P("base_width", "base half-width", 0.2, 3.0, 1.0),
  P("base_height", "base height", 0.1, 3.0, 0.5),
  P("shaft_segments", "shaft segments", 1, 12, 1, true, "Rings along the shaft (paper: 1)."),
], "three_quarter", "Uniform input mesh of paper Fig. 4: an elongated cube and a base.");
reg("column_classic", "Column: classic (Fig. 7)", (k) => columnClassic(k.sides, k.height, k.base_width, k.base_height,
  k.shaft_width, k.entasis, k.taper, k.capital_width, k.capital_height, k.shaft_rings), [
  P("sides", "sides", 3, 24, 8, true),
  P("height", "height", 2.0, 12.0, 6.0),
  P("base_width", "base half-width", 0.2, 3.0, 0.95),
  P("base_height", "base height", 0.1, 2.0, 0.45),
  P("shaft_width", "shaft half-width", 0.2, 2.0, 0.55),
  P("entasis", "entasis", 0.0, 0.4, 0.08, false, "Swelling of the shaft (Doric entasis)."),
  P("taper", "taper", 0.0, 0.6, 0.12),
  P("capital_width", "capital half-width", 0.2, 3.0, 1.0),
  P("capital_height", "capital height", 0.2, 3.0, 0.8),
  P("shaft_rings", "shaft rings", 1, 16, 4, true),
], "three_quarter", "Differentiated input like paper Fig. 7: base, shaft with entasis, echinus, abacus.");
reg("panel", "Panel (open relief tile)", (k) => panel(k.nx, k.ny, k.width, k.height, k.bend, k.saddle), [
  P("nx", "cells x", 1, 24, 4, true),
  P("ny", "cells y", 1, 24, 4, true),
  P("width", "width", 0.5, 6.0, 2.0),
  P("height", "height", 0.5, 6.0, 2.0),
  P("bend", "bend", -1.0, 1.0, 0.0, false, "Cylindrical curvature."),
  P("saddle", "saddle", -1.0, 1.0, 0.0, false, "Hyperbolic (saddle) curvature."),
], "three_quarter", "Open quad grid facing +z. Use Locked boundary to keep it tileable.");
reg("obj", "Import OBJ", (k, ctx) => obj(k.path, k.normalize, k.weld, ctx), [], "three_quarter",
  "Any polygon mesh (quads recommended). Welded, re-oriented and scaled to fit.", { path: "", normalize: true, weld: true });

export function defaultSpec(name) {
  const s = SHAPES[name];
  const spec = { shape: name };
  for (const p of s.params) spec[p.name] = p.default;
  Object.assign(spec, JSON.parse(JSON.stringify(s.extraDefaults)));
  return spec;
}

export function makeBase(spec, ctx = {}) {
  const name = spec.shape === undefined ? "cube" : spec.shape;
  const s = SHAPES[name];
  if (!s) throw new Error(`unknown base shape '${name}'; available: ${Object.keys(SHAPES).join(", ")}`);
  const kw = {};
  for (const p of s.params) kw[p.name] = p.default;
  Object.assign(kw, s.extraDefaults);
  for (const [k, v] of Object.entries(spec)) if (k in kw) kw[k] = v;
  for (const p of s.params) {
    kw[p.name] = Number(kw[p.name]);
    if (p.integer) kw[p.name] = pyRound(kw[p.name]);
  }
  return s.build(kw, ctx);
}

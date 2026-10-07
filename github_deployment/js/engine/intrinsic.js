// Intrinsic weight specification (Hansmeyer 2010); port of hansmeyer/intrinsic.py.
//   * Groups (tagging & locking, Fig. 9): faces / vertices of the input mesh are tagged; tags pass to every
//     child face, each group has its own weight rules, and group vertices can be locked for L iterations.
//   * Motifs (Fig. 6/7, eq. 10-11): vertices are classified by incident faces / edges ("4F4E"); each motif
//     gets an attractor/deflector value U (pulls face / edge points through w6 / w7).
//   * Measures (Fig. 8 + curvature): topological distance to the original vertices or edges, planarity or
//     bend, normalised to [0, 1] per iteration, drive weight rules between a value at 0 and one at 1.
//
// Group tags are bit fields: groups 0-30 in fattr.tags, groups 31-61 in fattr.tags2 (Int32 words).

import { MAX_ITERATIONS } from "./weights.js";
import { deepCopy, percentile } from "./util.js";

export const MEASURES = {
  dist_vertex: "distance to original vertices",
  dist_edge: "distance to original edges",
  planarity: "non-planarity of the face",
  bend: "bend (dihedral angle to neighbours)",
};
export const RULE_OPS = ["set", "scale", "add"];
export const MAX_GROUPS = 62;

// ------------------------------------------------------------------- motifs
export const motifLabel = (code) => `${Math.floor(code / 100)}F${code % 100}E`;

export function parseLabel(label) {
  const s = String(label).toUpperCase().replace(/E+$/, "");
  const parts = s.split("F");
  if (parts.length !== 2 || parts[0] === "" || parts[1] === "" || Number.isNaN(+parts[0]) || Number.isNaN(+parts[1])) {
    throw new Error(`bad motif label '${label}'`);
  }
  return parseInt(parts[0], 10) * 100 + parseInt(parts[1], 10);
}

export function motifCounts(mesh) {
  const codes = mesh.motifCodes();
  const counts = new Map();
  for (const c of codes) counts.set(c, (counts.get(c) || 0) + 1);
  const out = {};
  for (const c of [...counts.keys()].sort((a, b) => a - b)) out[motifLabel(c)] = counts.get(c);
  return out;
}

/** Per-vertex attractor/deflector value U from the design's motif table (eq. 10-11), or null. */
export function motifU(design, mesh) {
  const table = Object.entries(design.motifs || {}).filter(([, v]) => Number(v) !== 0.0);
  if (!table.length) return null;
  const codes = mesh.motifCodes();
  const U = new Float64Array(mesh.nVerts);
  for (const [label, u] of table) {
    let code;
    try { code = parseLabel(label); } catch { continue; }
    for (let i = 0; i < codes.length; i++) if (codes[i] === code) U[i] = Number(u);
  }
  return U;
}

// ----------------------------------------------------------------- measures
function normalise(x, pct = 95.0) {
  const finite = [];
  for (const v of x) if (Number.isFinite(v)) finite.push(v);
  const out = new Float64Array(x.length);
  if (!finite.length) return out.fill(1);
  let hi;
  if (pct < 100) hi = percentile(finite, pct);
  else hi = finite.reduce((a, b) => (b > a ? b : a), -Infinity);
  hi = hi > 1e-12 ? hi : 1.0;
  for (let i = 0; i < x.length; i++) {
    const v = Number.isFinite(x[i]) ? x[i] / hi : 1.0;
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

/** Edge hops from each vertex to the original vertices ("tv") or original edges ("te"). */
export function vertexDistance(mesh, which) {
  const key = which === "dist_vertex" ? "tv" : "te";
  const a = mesh.vattr[key];
  const seeds = new Uint8Array(mesh.nVerts);
  if (a) for (let i = 0; i < seeds.length; i++) seeds[i] = a[i] > 0.5 ? 1 : 0;
  return mesh.hopDistance(seeds);
}

/** Per-face measure in [0, 1]. */
export function measure(mesh, name) {
  if (name === "dist_vertex" || name === "dist_edge") {
    const d = normalise(vertexDistance(mesh, name), 100.0);
    return mesh.faceMeanOfVerts(d);
  }
  if (name === "planarity") return normalise(mesh.facePlanarity);
  if (name === "bend") return normalise(mesh.faceBend);
  throw new Error(`unknown measure '${name}'`);
}

// -------------------------------------------------------------------- rules
export const DEFAULT_RULE = { enabled: true, measure: "dist_edge", weight: "w_f", op: "set", a: 0.0, b: 0.3, gamma: 1.0, from: 1, to: MAX_ITERATIONS };

export function normalizeRule(r) {
  const out = deepCopy(DEFAULT_RULE);
  for (const [k, v] of Object.entries(r)) if (k in DEFAULT_RULE) out[k] = v;
  if (!(out.measure in MEASURES) || !RULE_OPS.includes(out.op)) throw new Error("unknown intrinsic measure or op");
  return out;
}

const activeRules = (design, level) => design.intrinsic.filter((r) => r.enabled && r.from <= level + 1 && level + 1 <= r.to);

function asArrays(W, F) {
  const out = {};
  for (const [n, v] of Object.entries(W)) out[n] = typeof v === "number" ? new Float64Array(F).fill(v) : Float64Array.from(v);
  return out;
}

export function applyRules(design, mesh, level, W) {
  const rules = activeRules(design, level);
  if (!rules.length) return W;
  const F = mesh.nFaces;
  W = asArrays(W, F);
  const cache = {};
  for (const r of rules) {
    if (!(r.weight in W)) continue;
    if (!(r.measure in cache)) cache[r.measure] = measure(mesh, r.measure);
    const t = cache[r.measure], g = Math.max(Number(r.gamma), 1e-6), w = W[r.weight], out = new Float64Array(F);
    for (let f = 0; f < F; f++) {
      const v = r.a + (r.b - r.a) * t[f] ** g;
      out[f] = r.op === "set" ? v : r.op === "scale" ? w[f] * v : w[f] + v;
    }
    W[r.weight] = out;
  }
  return W;
}

// ------------------------------------------------------------------- groups
export const DEFAULT_GROUP = { name: "G", enabled: true, faces: [], verts: [], lock: 0, lock_faces: false, rules: [] };

export function normalizeGroup(g) {
  const out = deepCopy(DEFAULT_GROUP);
  for (const [k, v] of Object.entries(g)) if (k in DEFAULT_GROUP) out[k] = deepCopy(v);
  out.faces = [...new Set(out.faces.map((i) => Math.trunc(i)))].sort((a, b) => a - b);
  out.verts = [...new Set(out.verts.map((i) => Math.trunc(i)))].sort((a, b) => a - b);
  out.rules = out.rules.map((m) => ({
    weight: m.weight ?? "w_f", op: m.op ?? "scale", value: Number(m.value ?? 1.0),
    from: Math.trunc(m.from ?? 1), to: Math.trunc(m.to ?? MAX_ITERATIONS),
  }));
  return out;
}

/** Whether face f carries group bit `bit`. */
export function hasTag(fattr, f, bit) {
  const word = bit < 31 ? fattr.tags : fattr.tags2;
  if (!word) return false;
  return ((word[f] >>> (bit < 31 ? bit : bit - 31)) & 1) === 1;
}

export function anyTag(fattr, f) {
  return (fattr.tags && fattr.tags[f] !== 0) || (fattr.tags2 && fattr.tags2[f] !== 0);
}

/** Attach the attributes that intrinsic features carry through subdivision. */
export function decorateBase(design, mesh) {
  const n = mesh.nVerts, F = mesh.nFaces;
  mesh.vattr.tv = new Float64Array(n).fill(1); // original vertices
  mesh.vattr.te = new Float64Array(n).fill(1); // vertices lying on original edges
  const tags = new Int32Array(F), tags2 = new Int32Array(F);
  const lock = new Float64Array(n);
  let anyLock = false, anyHigh = false;
  design.groups.slice(0, MAX_GROUPS).forEach((g, bit) => {
    if (!g.enabled) return;
    const faces = g.faces.filter((f) => f >= 0 && f < F);
    for (const f of faces) {
      if (bit < 31) tags[f] |= 1 << bit;
      else { tags2[f] |= 1 << (bit - 31); anyHigh = true; }
    }
    if (g.lock > 0) {
      const verts = new Set(g.verts.filter((v) => v >= 0 && v < n));
      if (g.lock_faces) for (const f of faces) for (const v of mesh.faceVerts(f)) verts.add(v);
      for (const v of verts) { lock[v] = Math.max(lock[v], g.lock); anyLock = true; }
    }
  });
  mesh.fattr.tags = tags;
  if (anyHigh) mesh.fattr.tags2 = tags2;
  if (anyLock) mesh.vattr.lock = lock;
}

export function applyGroupRules(design, mesh, level, W) {
  const groups = [];
  design.groups.slice(0, MAX_GROUPS).forEach((g, bit) => {
    if (g.enabled && g.rules.some((r) => r.from <= level + 1 && level + 1 <= r.to)) groups.push([bit, g]);
  });
  if (!groups.length || !mesh.fattr.tags) return W;
  const F = mesh.nFaces;
  W = asArrays(W, F);
  for (const [bit, g] of groups) {
    const sel = [];
    for (let f = 0; f < F; f++) if (hasTag(mesh.fattr, f, bit)) sel.push(f);
    if (!sel.length) continue;
    for (const r of g.rules) {
      if (!(r.from <= level + 1 && level + 1 <= r.to) || !(r.weight in W)) continue;
      const w = W[r.weight];
      for (const f of sel) w[f] = r.op === "scale" ? w[f] * r.value : w[f] + r.value;
    }
  }
  return W;
}

export function groupSignature(design, level) {
  const out = [];
  for (const g of design.groups) if (g.enabled) out.push(g.rules.filter((r) => r.from <= level + 1 && level + 1 <= r.to));
  return out.some((x) => x.length) ? out : null;
}

/** What groups contribute to the decorated input mesh (tags and locks). */
export function baseSignature(design) {
  const out = design.groups.map((g) => [g.faces, g.verts, g.lock, g.lock_faces, g.enabled]);
  return out.length ? out : null;
}

export function signature(design, level) {
  const rules = activeRules(design, level);
  const motifs = Object.fromEntries(Object.entries(design.motifs || {}).filter(([, v]) => Number(v) !== 0.0));
  return rules.length || Object.keys(motifs).length ? { rules, motifs } : null;
}

// --------------------------------------------------------- selection helpers
export const AXIS_VECTORS = { "+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1] };

export function selectByNormal(mesh, direction, maxAngleDeg = 30.0) {
  const d = AXIS_VECTORS[direction], n = mesh.faceNormal, c = Math.cos((maxAngleDeg * Math.PI) / 180), out = [];
  for (let f = 0; f < mesh.nFaces; f++) if (n[3 * f] * d[0] + n[3 * f + 1] * d[1] + n[3 * f + 2] * d[2] >= c) out.push(f);
  return out;
}

/** Faces (or vertices) whose centroid lies in the [lo, hi] band (fractions of the bounding box). */
export function selectByHeight(mesh, axis, lo, hi, faces = true) {
  const P = faces ? mesh.faceCentroid : mesh.V;
  let vmin = Infinity, vmax = -Infinity;
  for (let i = axis; i < mesh.V.length; i += 3) { vmin = Math.min(vmin, mesh.V[i]); vmax = Math.max(vmax, mesh.V[i]); }
  const out = [];
  for (let i = 0; i < P.length / 3; i++) {
    const t = (P[3 * i + axis] - vmin) / Math.max(vmax - vmin, 1e-12);
    if (t >= lo - 1e-9 && t <= hi + 1e-9) out.push(i);
  }
  return out;
}

export function selectEveryKth(n, k, start = 0) {
  const kk = Math.max(k, 1), out = [];
  for (let i = start % kk; i < n; i += kk) out.push(i);
  return out;
}

export function selectByMotif(mesh, label) {
  const code = parseLabel(label), codes = mesh.motifCodes(), out = [];
  for (let i = 0; i < codes.length; i++) if (codes[i] === code) out.push(i);
  return out;
}

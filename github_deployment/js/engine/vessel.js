// Vessel: a thin-walled lithophane sphere, smooth outside, the subdivision relief inside (our extension);
// port of hansmeyer/vessel.py.
//
// The subdivided form becomes the inner face of a shell: the outer face is an exact sphere of the given
// diameter through every vertex's rest direction; the form's relief (measured against the same schedule run
// with zero weights) becomes the wall: the outermost `glow` share is pressed against the shell at
// `min_wall_mm` (windows that glow when lit from inside), everything else is thicker at true proportion
// times `depth`, up to `max_wall_mm`. Every boundary loop gets a rim joining the two faces; the top opening's
// rim is an exact, flat circle of `rim_mm` wall, so the vessel prints upside down standing on it.

import { PolyMesh } from "./mesh.js";
import { normalizeIteration } from "./weights.js";
import { deepCopy, percentile } from "./util.js";

export const DEFAULT_VESSEL = {
  enabled: false, diameter_mm: 120.0, min_wall_mm: 0.8, max_wall_mm: 12.0, depth: 1.0, glow: 12.0, rim_mm: 2.5,
  rim_band: 10.0, invert: false,
};

export function normalize(v) {
  const out = deepCopy(DEFAULT_VESSEL);
  for (const [k, val] of Object.entries(v || {})) if (k in DEFAULT_VESSEL) out[k] = val;
  out.min_wall_mm = Math.max(Number(out.min_wall_mm), 0.1);
  out.max_wall_mm = Math.max(Number(out.max_wall_mm), out.min_wall_mm);
  out.diameter_mm = Math.max(Number(out.diameter_mm), 1.0);
  return out;
}

export function active(design) {
  return !!(design.vessel && design.vessel.enabled);
}

/** Boundary half-edges grouped into closed loops (in face orientation). */
export function boundaryLoops(mesh) {
  const tw = mesh.heTwin, from = mesh.faceIdx, to = mesh.heTo;
  const hb = [];
  for (let h = 0; h < tw.length; h++) if (tw[h] < 0) hb.push(h);
  if (!hb.length) return [];
  const start = new Map();
  for (const h of hb) start.set(from[h], h);
  const seen = new Set(), loops = [];
  for (const h0 of hb) {
    if (seen.has(h0)) continue;
    const loop = [];
    let h = h0;
    while (h !== undefined && !seen.has(h)) {
      seen.add(h);
      loop.push(h);
      h = start.get(to[h]); // undefined: open chain (pinched boundary), stop here
    }
    loops.push(loop);
  }
  return loops;
}

function smooth(values, mesh, iterations) {
  if (iterations <= 0) return values;
  const a = mesh.faceIdx, b = mesh.heTo, n = mesh.nVerts;
  const deg = new Float64Array(n);
  for (let h = 0; h < a.length; h++) { deg[a[h]] += 1; deg[b[h]] += 1; }
  for (let v = 0; v < n; v++) deg[v] = Math.max(deg[v], 1);
  let x = values;
  for (let it = 0; it < iterations; it++) {
    const acc = new Float64Array(n);
    for (let h = 0; h < a.length; h++) { acc[a[h]] += x[b[h]]; acc[b[h]] += x[a[h]]; }
    const nx = new Float64Array(n);
    for (let v = 0; v < n; v++) nx[v] = 0.5 * x[v] + (0.5 * acc[v]) / deg[v];
    x = nx;
  }
  return x;
}

/** The same input and schedule with all weights zero and no fields: plain subdivision. Its vertices
 * match the form's one to one (merging aside), so the difference is exactly what the weights did. */
export function plainDesign(design, normalizeDesign) {
  return normalizeDesign({
    base: deepCopy(design.base), extrusion: design.extrusion, boundary: design.boundary, fade_rows: design.fade_rows,
    iterations: design.iterations.map((it) => normalizeIteration({ scheme: it.scheme, weights: {} })),
    groups: deepCopy(design.groups),
  });
}

function restDirections(relief) {
  const rest = relief.vattr.rest || relief.V, n = relief.nVerts, d = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const x = rest[3 * i], y = rest[3 * i + 1], z = rest[3 * i + 2];
    const r = Math.max(Math.sqrt(x * x + y * y + z * z), 1e-12);
    d[3 * i] = x / r; d[3 * i + 1] = y / r; d[3 * i + 2] = z / r;
  }
  return d;
}

/** How far (model units, + outward) each vertex sits from the plain subdivision, along its rest direction. */
export function reliefHeight(relief, plain = null) {
  const d = restDirections(relief), V = relief.V, n = relief.nVerts, h = new Float64Array(n);
  for (let i = 0; i < n; i++) h[i] = V[3 * i] * d[3 * i] + V[3 * i + 1] * d[3 * i + 1] + V[3 * i + 2] * d[3 * i + 2];
  if (plain && plain.nVerts === n) {
    const Q = plain.V;
    for (let i = 0; i < n; i++) h[i] -= Q[3 * i] * d[3 * i] + Q[3 * i + 1] * d[3 * i + 1] + Q[3 * i + 2] * d[3 * i + 2];
    return h;
  }
  const s = smooth(h, relief, 40); // topology changed (vertex merging): use the smoothed form instead
  for (let i = 0; i < n; i++) h[i] -= s[i];
  return h;
}

/** Wall thickness (mm) at every vertex of the relief, before the rim blend. */
export function wallThickness(design, relief, plain = null) {
  const v = normalize(design.vessel);
  const h = reliefHeight(relief, plain);
  if (v.invert) for (let i = 0; i < h.length; i++) h[i] = -h[i];
  const top = percentile(h, 100.0 - Math.min(Math.max(Number(v.glow), 0.5), 90.0)); // everything above is a window
  const t = new Float64Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = v.min_wall_mm + Number(v.depth) * (top - h[i]) * 0.5 * v.diameter_mm;
    t[i] = Math.min(Math.max(x, v.min_wall_mm), v.max_wall_mm);
  }
  return t;
}

export function build(design, relief, plain = null) {
  const v = normalize(design.vessel);
  const unit = 2.0 / v.diameter_mm; // model units per mm: the outer sphere has radius 1
  const rest = relief.vattr.rest || relief.V;
  const d = restDirections(relief);
  let t = wallThickness(design, relief, plain);
  const n = relief.nVerts;

  const loops = boundaryLoops(relief);
  const from = relief.faceIdx, to = relief.heTo;
  let top = null, thetaRim = 0, rimV = null;
  if (loops.length) {
    let best = -Infinity;
    for (const lp of loops) {
      let s = 0;
      for (const h of lp) s += d[3 * from[h] + 1];
      s /= lp.length;
      if (s > best) { best = s; top = lp; }
    }
    rimV = top.map((h) => from[h]);
    let mean = 0;
    for (const vi of rimV) mean += Math.acos(Math.min(Math.max(d[3 * vi + 1], -1), 1));
    thetaRim = mean / rimV.length;
    if (design.base.shape === "sphere_open") { // the exact opening of the input sphere
      thetaRim = (Math.min(Math.max(Number(design.base.opening ?? 40.0), 5.0), 150.0) * Math.PI) / 180;
    }
    const band = (Math.max(v.rim_band, 0.1) * Math.PI) / 180;
    const nt = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const theta = Math.acos(Math.min(Math.max(d[3 * i + 1], -1), 1));
      let b = Math.min(Math.max((theta - thetaRim) / band, 0.0), 1.0);
      b = b * b * (3 - 2 * b);
      nt[i] = v.rim_mm + b * (t[i] - v.rim_mm);
    }
    t = nt;
  }
  for (let i = 0; i < n; i++) t[i] = Math.min(t[i], 0.95 / unit); // never through the centre

  const V = new Float64Array(6 * n);
  for (let i = 0; i < n; i++) {
    const k = 1.0 - t[i] * unit;
    for (let c = 0; c < 3; c++) {
      V[3 * i + c] = d[3 * i + c];
      V[3 * (n + i) + c] = d[3 * i + c] * k;
    }
  }
  if (top) { // an exact, flat, circular rim to stand on
    const s = Math.sin(thetaRim), c = Math.cos(thetaRim);
    const ri = Math.max(s - v.rim_mm * unit, 1e-3);
    for (const vi of rimV) {
      const ang = Math.atan2(d[3 * vi + 2], d[3 * vi]);
      const cx = Math.cos(ang), cz = Math.sin(ang);
      V[3 * vi] = cx * s; V[3 * vi + 1] = c; V[3 * vi + 2] = cz * s;
      V[3 * (n + vi)] = cx * ri; V[3 * (n + vi) + 1] = c; V[3 * (n + vi) + 2] = cz * ri;
    }
  }

  // faces: outer as the form, inner reversed, and a quad strip joining them along every boundary loop
  const ptr = relief.facePtr, idx = relief.faceIdx, F = relief.nFaces, H = idx.length;
  let nRim = 0;
  for (const lp of loops) nRim += lp.length;
  const faceIdx = new Int32Array(2 * H + 4 * nRim);
  const facePtr = new Int32Array(2 * F + nRim + 1);
  faceIdx.set(idx, 0);
  for (let f = 0; f < F; f++) {
    const a = ptr[f], k = ptr[f + 1] - a;
    for (let j = 0; j < k; j++) faceIdx[H + a + j] = idx[a + k - 1 - j] + n;
  }
  facePtr.set(ptr, 0);
  for (let f = 1; f <= F; f++) facePtr[F + f] = H + ptr[f];
  let o = 2 * H, fi = 2 * F;
  for (const lp of loops) {
    for (const h of lp) {
      const a = from[h], b = to[h];
      faceIdx[o++] = b; faceIdx[o++] = a; faceIdx[o++] = a + n; faceIdx[o++] = b + n;
      facePtr[++fi] = o;
    }
  }
  const m = new PolyMesh(V, facePtr, faceIdx);
  const rest2 = new Float64Array(6 * n);
  rest2.set(rest, 0);
  rest2.set(rest, 3 * n);
  m.vattr.rest = rest2;
  const wall = new Float64Array(2 * n);
  wall.set(t, 0);
  wall.set(t, n);
  m.vattr.wall_mm = wall;
  const outer = new Float64Array(2 * n);
  outer.fill(1, 0, n);
  m.vattr.vessel_outer = outer;
  let tmin = Infinity, tmax = -Infinity;
  for (const x of t) { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); }
  m.info.vessel = {
    diameter_mm: v.diameter_mm, opening_deg: top ? (thetaRim * 180) / Math.PI : null, wall_mm: [tmin, tmax], loops: loops.length,
  };
  return m;
}

/** Per-face brightness (0..1) of the lit vessel seen from outside: light through PLA falls off roughly as
 * exp(-absorb * thickness_mm), normalised between the thickest and thinnest wall. */
export function light(mesh, absorb = 1.0) {
  const w = mesh.vattr.wall_mm;
  const tf = mesh.faceMeanOfVerts(w);
  let lo = Infinity, hi = -Infinity;
  for (const x of w) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
  const a = Math.exp(-absorb * hi), b = Math.exp(-absorb * lo), out = new Float64Array(tf.length);
  for (let i = 0; i < tf.length; i++) out[i] = Math.min(Math.max((Math.exp(-absorb * tf[i]) - a) / Math.max(b - a, 1e-12), 0), 1);
  return out;
}

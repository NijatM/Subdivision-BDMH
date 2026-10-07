// Attractors: spatially varying (non-uniform) weights; port of hansmeyer/attractors.py.
//
// Hansmeyer 2010, "Uniform input mesh and specification of extrinsic parameters": sets of weights are
// placed in the mesh's environment; for each face the distances to the sets give their level of influence
// (eq. 8) and the face's weights are the influence-weighted blend of the sets (eq. 9):
//     c_a = f(d_a) h_a / sum_i f(d_i) h_i          w_x = sum_i w_{x,i} c_i
// with f(d) = (1 - d)^t in the paper. Here an attractor is a point or a space curve, f is a choice of
// falloff curves over d / radius, the main schedule takes part in the blend as a "background" set, and
// "modifier" attractors additionally scale or offset chosen weights near them.

import { MAX_ITERATIONS, WEIGHT_NAMES } from "./weights.js";
import { clamp, deepCopy, linspace, npRound, stableStringify } from "./util.js";

export const FALLOFFS = ["power", "linear", "smoothstep", "gaussian", "spline"];
export const CURVE_TYPES = ["line", "circle", "helix", "sine", "lissajous", "polyline"];
export const AXES = ["x", "y", "z"];
export const SPLINE_KNOTS = [0, 0.25, 0.5, 0.75, 1];
export const CURVE_SAMPLES = 256;

export const DEFAULT_CURVE = {
  type: "helix", center: [0.0, 0.0, 0.0], axis: "y", radius: 1.0, height: 4.0, turns: 2.0, length: 4.0,
  amplitude: 0.6, waves: 2.0, size: 1.5, a: 3, b: 2, c: 1, phase: 0.5, start: [-2.0, 0.0, 0.0],
  end: [2.0, 0.0, 0.0], points: [], closed: false, smooth: true,
};

export const DEFAULT_ATTRACTOR = {
  name: "A", enabled: true, kind: "point", position: [0.0, 0.0, 0.0], curve: DEFAULT_CURVE, payload: "modifier",
  strength: 1.0, radius: 2.0, falloff: { type: "power", tightness: 2.0, spline: [1.0, 0.8, 0.5, 0.2, 0.0] },
  weights: [], mods: [],
};

// ------------------------------------------------------------------ defaults
/** Fill in defaults so every attractor has every field (presets stay short). */
export function normalize(att) {
  const a = deepCopy(DEFAULT_ATTRACTOR);
  for (const [k, v] of Object.entries(att)) if (k in DEFAULT_ATTRACTOR) a[k] = deepCopy(v);
  a.curve = { ...deepCopy(DEFAULT_CURVE), ...deepCopy(att.curve || {}) };
  a.falloff = { ...deepCopy(DEFAULT_ATTRACTOR.falloff), ...deepCopy(att.falloff || {}) };
  const weights = (att.weights || []).slice(0, MAX_ITERATIONS).map((w) => ({ ...w }));
  while (weights.length < MAX_ITERATIONS) weights.push({});
  a.weights = weights.map((w) => {
    const o = {};
    for (const n of WEIGHT_NAMES) o[n] = Number(w[n] === undefined ? 0.0 : w[n]);
    return o;
  });
  a.mods = (att.mods || []).map((m) => ({
    weight: m.weight === undefined ? "w_f" : m.weight, op: m.op === undefined ? "scale" : m.op,
    value: Number(m.value === undefined ? 1.0 : m.value), from: Math.trunc(m.from === undefined ? 1 : m.from),
    to: Math.trunc(m.to === undefined ? MAX_ITERATIONS : m.to),
  }));
  if (!["point", "curve"].includes(a.kind)) throw new Error("attractor kind must be one of point, curve");
  if (!["set", "modifier"].includes(a.payload)) throw new Error("attractor payload must be one of set, modifier");
  if (!FALLOFFS.includes(a.falloff.type) || !CURVE_TYPES.includes(a.curve.type)) throw new Error("unknown falloff or curve type");
  return a;
}

/** Inverse of normalize for saving: drop zero weights and unused curve data. */
export function compact(att) {
  const a = deepCopy(att);
  if (a.payload === "set") {
    a.weights = a.weights.map((w) => Object.fromEntries(Object.entries(w).filter(([, v]) => v !== 0.0)));
    while (a.weights.length && Object.keys(a.weights[a.weights.length - 1]).length === 0) a.weights.pop();
  } else {
    delete a.weights;
  }
  if (a.payload !== "modifier") delete a.mods;
  if (a.kind === "point") {
    delete a.curve;
  } else {
    delete a.position;
    const used = {
      line: ["start", "end"], circle: ["center", "axis", "radius"], helix: ["center", "axis", "radius", "height", "turns"],
      sine: ["center", "axis", "length", "amplitude", "waves"], lissajous: ["center", "size", "a", "b", "c", "phase"],
      polyline: ["points", "closed", "smooth"],
    }[a.curve.type];
    const c = { type: a.curve.type };
    for (const k of used) c[k] = a.curve[k];
    a.curve = c;
  }
  return a;
}

// ------------------------------------------------------------------ falloff
/** Influence in [0, 1] as a function of normalised distance u = d / radius (scalar). */
export function falloff1(u, spec) {
  u = Math.max(u, 0.0);
  const kind = spec.type || "power";
  if (kind === "power") return clamp(1.0 - u, 0.0, 1.0) ** Math.max(Number(spec.tightness ?? 2.0), 1e-6);
  if (kind === "linear") return clamp(1.0 - u, 0.0, 1.0);
  if (kind === "smoothstep") {
    const t = clamp(u, 0.0, 1.0);
    return 1.0 - t * t * (3 - 2 * t);
  }
  if (kind === "gaussian") return Math.exp(-4.5 * u * u);
  if (kind === "spline") {
    const vals = (spec.spline || [1, 0.8, 0.5, 0.2, 0]).map(Number);
    if (u >= 1) return vals[vals.length - 1];
    let i = 0; // np.interp: the knot interval holding u
    while (i < 3 && u >= SPLINE_KNOTS[i + 1]) i++;
    const slope = (vals[i + 1] - vals[i]) / (SPLINE_KNOTS[i + 1] - SPLINE_KNOTS[i]);
    return slope * (u - SPLINE_KNOTS[i]) + vals[i];
  }
  throw new Error(`unknown falloff '${kind}'`);
}

export function falloff(u, spec) {
  const out = new Float64Array(u.length);
  for (let i = 0; i < u.length; i++) out[i] = falloff1(u[i], spec);
  return out;
}

// ------------------------------------------------------------------- curves
/** Rotation taking local y (the curve's main axis) to the chosen world axis (row-major 3x3). */
function frame(axis) {
  if (axis === "y") return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  if (axis === "x") return [0, -1, 0, 1, 0, 0, 0, 0, 1]; // [[0,1,0],[-1,0,0],[0,0,1]].T
  return [1, 0, 0, 0, 0, -1, 0, 1, 0]; // [[1,0,0],[0,0,1],[0,-1,0]].T
}

function catmullRom(P, closed, samples) {
  const n = P.length;
  if (n < 3) {
    const out = [];
    const t = linspace(0, 1, samples);
    for (let i = 0; i < samples; i++) {
      if (n === 2) out.push([0, 1, 2].map((k) => P[0][k] * (1 - t[i]) + P[1][k] * t[i]));
      else out.push([...P[0]]);
    }
    return out;
  }
  let ext, segs;
  if (closed) {
    ext = [P[n - 1], ...P, P[0], P[1]];
    segs = n;
  } else {
    ext = [P[0].map((x, k) => 2 * x - P[1][k]), ...P, P[n - 1].map((x, k) => 2 * x - P[n - 2][k])];
    segs = n - 1;
  }
  const out = [];
  const per = Math.max(Math.floor(samples / segs), 2);
  const ts = linspace(0, 1, per, false);
  for (let i = 0; i < segs; i++) {
    const p0 = ext[i], p1 = ext[i + 1], p2 = ext[i + 2], p3 = ext[i + 3];
    for (const t of ts) {
      out.push([0, 1, 2].map((k) => 0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t ** 2
        + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t ** 3)));
    }
  }
  out.push(closed ? [...ext[1]] : [...ext[segs + 1]]);
  return out;
}

/** Sample a curve as an array of [x,y,z] (closed curves repeat their first point). */
export function curvePoints(curve, samples = CURVE_SAMPLES) {
  const c = curve;
  const kind = c.type;
  const t = linspace(0.0, 1.0, samples);
  if (kind === "line") {
    const a = c.start.map(Number), b = c.end.map(Number);
    return Array.from(t, (s) => [0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * s));
  }
  if (kind === "polyline") {
    const P = (c.points || []).map((p) => p.map(Number));
    if (P.length === 0) return [[0, 0, 0]];
    if ((c.smooth === undefined || c.smooth) && P.length >= 3) return catmullRom(P, !!c.closed, samples);
    return c.closed && P.length > 2 ? [...P, P[0]] : P;
  }
  let local;
  if (kind === "circle") {
    local = Array.from(t, (s) => {
      const th = 2 * Math.PI * s;
      return [c.radius * Math.cos(th), 0, c.radius * Math.sin(th)];
    });
  } else if (kind === "helix") {
    local = Array.from(t, (s) => {
      const th = 2 * Math.PI * c.turns * s;
      return [c.radius * Math.cos(th), (s - 0.5) * c.height, c.radius * Math.sin(th)];
    });
  } else if (kind === "sine") {
    local = Array.from(t, (s) => [c.amplitude * Math.sin(2 * Math.PI * c.waves * s), (s - 0.5) * c.length, 0]);
  } else if (kind === "lissajous") {
    local = Array.from(t, (s) => {
      const th = 2 * Math.PI * s, sz = c.size;
      return [sz * Math.sin(c.a * th + c.phase * Math.PI), sz * Math.sin(c.b * th), sz * Math.sin(c.c * th + 0.5 * c.phase * Math.PI)];
    });
  } else {
    throw new Error(`unknown curve type '${kind}'`);
  }
  const R = kind !== "lissajous" ? frame(c.axis || "y") : [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const cen = c.center.map(Number);
  // local @ R.T + center
  return local.map((p) => [0, 1, 2].map((r) => p[0] * R[3 * r] + p[1] * R[3 * r + 1] + p[2] * R[3 * r + 2] + cen[r]));
}

/** Convert any curve into an editable polyline with n control points. */
export function toPolyline(curve, n = 8) {
  const closed = curve.type === "circle" || curve.type === "lissajous";
  const P = curvePoints(curve, 512);
  const m = n + (closed ? 1 : 0);
  const idx = Array.from(linspace(0, P.length - 1, m), (x) => npRound(x));
  const pick = closed ? idx.slice(0, -1) : idx;
  const out = deepCopy(curve);
  Object.assign(out, { type: "polyline", points: pick.map((i) => P[i].map((x) => npRound(x, 4))), closed, smooth: true });
  return out;
}

/** Points from .csv/.txt (x y z per line, any separator) or .obj (v, ordered by `l` if present). */
export function loadPolyline(text, name = "curve.csv") {
  let P;
  if (name.toLowerCase().endsWith(".obj")) {
    const V = [], order = [];
    for (const line of text.split(/\r?\n/)) {
      const p = line.trim().split(/\s+/);
      if (p[0] === "v") V.push(p.slice(1, 4).map(Number));
      else if (p[0] === "l") {
        for (const tok of p.slice(1)) {
          let i = parseInt(tok.split("/")[0], 10);
          i = i > 0 ? i - 1 : V.length + i;
          if (!order.length || order[order.length - 1] !== i) order.push(i);
        }
      }
    }
    P = order.length ? order.map((i) => V[i]) : V;
  } else {
    P = [];
    for (const line of text.split(/\r?\n/)) {
      const vals = line.replace(/[,;\t]/g, " ").trim().split(/\s+/).filter(Boolean);
      const nums = vals.map(Number);
      if (!vals.length || nums.some((x) => Number.isNaN(x))) continue; // header lines
      if (nums.length >= 3) P.push(nums.slice(0, 3));
    }
  }
  if (P.length < 2) throw new Error(`${name}: need at least two points`);
  return P;
}

// ----------------------------------------------------------------- distance
/** Distance from packed points P (n*3) to the polyline Q ([[x,y,z], ...]).
 * Short polylines are solved exactly against every segment. Long (densely sampled) curves first find the
 * nearest segment midpoint, then solve exactly on the `window` segments either side of it: never
 * underestimates, and overestimates by at most half a segment length. */
export function distanceToPolyline(P, Q, window = 2) {
  const n = P.length / 3, out = new Float64Array(n);
  if (Q.length === 1) {
    const q = Q[0];
    for (let i = 0; i < n; i++) {
      const dx = P[3 * i] - q[0], dy = P[3 * i + 1] - q[1], dz = P[3 * i + 2] - q[2];
      out[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return out;
  }
  const S = Q.length - 1;
  const A = new Float64Array(3 * S), AB = new Float64Array(3 * S), L2 = new Float64Array(S);
  for (let s = 0; s < S; s++) {
    for (let k = 0; k < 3; k++) {
      A[3 * s + k] = Q[s][k];
      AB[3 * s + k] = Q[s + 1][k] - Q[s][k];
    }
    L2[s] = Math.max(AB[3 * s] ** 2 + AB[3 * s + 1] ** 2 + AB[3 * s + 2] ** 2, 1e-30);
  }
  const segDist2 = (px, py, pz, s) => {
    const ax = A[3 * s], ay = A[3 * s + 1], az = A[3 * s + 2], bx = AB[3 * s], by = AB[3 * s + 1], bz = AB[3 * s + 2];
    let t = ((px - ax) * bx + (py - ay) * by + (pz - az) * bz) / L2[s];
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const dx = ax + t * bx - px, dy = ay + t * by - py, dz = az + t * bz - pz;
    return dx * dx + dy * dy + dz * dz;
  };
  if (S <= 4 * window + 2) {
    for (let i = 0; i < n; i++) {
      const px = P[3 * i], py = P[3 * i + 1], pz = P[3 * i + 2];
      let best = Infinity;
      for (let s = 0; s < S; s++) {
        const d = segDist2(px, py, pz, s);
        if (d < best) best = d;
      }
      out[i] = Math.sqrt(best);
    }
    return out;
  }
  const M = new Float64Array(3 * S), M2 = new Float64Array(S);
  for (let s = 0; s < S; s++) {
    for (let k = 0; k < 3; k++) M[3 * s + k] = A[3 * s + k] + 0.5 * AB[3 * s + k];
    M2[s] = M[3 * s] ** 2 + M[3 * s + 1] ** 2 + M[3 * s + 2] ** 2;
  }
  const grid = n * S > 2_000_000 ? midpointGrid(M, S) : null;
  const cand = grid ? new Int32Array(S) : null;
  for (let i = 0; i < n; i++) {
    const px = P[3 * i], py = P[3 * i + 1], pz = P[3 * i + 2];
    let nearest = 0, bestM = Infinity;
    if (grid) {
      // the same argmin as below, over the midpoints a grid search proves can be nearest
      const nc = grid.candidates(px, py, pz, cand);
      for (let c = 0; c < nc; c++) {
        const s = cand[c];
        const v = M2[s] - 2.0 * (px * M[3 * s] + py * M[3 * s + 1] + pz * M[3 * s + 2]);
        if (v < bestM || (v === bestM && s < nearest)) { bestM = v; nearest = s; }
      }
    } else {
      for (let s = 0; s < S; s++) {
        const v = M2[s] - 2.0 * (px * M[3 * s] + py * M[3 * s + 1] + pz * M[3 * s + 2]);
        if (v < bestM) { bestM = v; nearest = s; }
      }
    }
    let best = Infinity;
    for (let o = -window; o <= window; o++) {
      const s = Math.min(Math.max(nearest + o, 0), S - 1);
      const d = segDist2(px, py, pz, s);
      if (d < best) best = d;
    }
    out[i] = Math.sqrt(best);
  }
  return out;
}

/** A static k-d tree over segment midpoints: candidates(p) lists every midpoint that can be the nearest to p
 * (all midpoints within the nearest one's distance, plus a margin for rounding). */
function midpointGrid(M, S) {
  const order = new Int32Array(S);
  for (let s = 0; s < S; s++) order[s] = s;
  const axisOf = new Int8Array(S);
  const build = (lo, hi, depth) => {
    if (hi - lo <= 1) return;
    // split on the widest axis of this block
    let ax = 0, w = -1;
    for (let a = 0; a < 3; a++) {
      let mn = Infinity, mx = -Infinity;
      for (let i = lo; i < hi; i++) { const v = M[3 * order[i] + a]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mx - mn > w) { w = mx - mn; ax = a; }
    }
    const part = Array.from(order.subarray(lo, hi)).sort((x, y) => M[3 * x + ax] - M[3 * y + ax] || x - y);
    order.set(part, lo);
    const mid = (lo + hi) >> 1;
    axisOf[mid] = ax;
    build(lo, mid, depth + 1);
    build(mid + 1, hi, depth + 1);
  };
  build(0, S, 0);
  // flattened tree: node = mid of [lo, hi); coordinates per node for speed
  const nodeX = new Float64Array(S), nodeY = new Float64Array(S), nodeZ = new Float64Array(S);
  for (let i = 0; i < S; i++) { const s = order[i]; nodeX[i] = M[3 * s]; nodeY[i] = M[3 * s + 1]; nodeZ[i] = M[3 * s + 2]; }
  const stackLo = new Int32Array(128), stackHi = new Int32Array(128);
  const search = (px, py, pz, radius, buf) => {
    // radius < 0: nearest search (returns the squared distance); else collect all within radius into buf
    let best = radius < 0 ? Infinity : radius, n = 0, top = 0;
    stackLo[0] = 0; stackHi[0] = S; top = 1;
    while (top > 0) {
      top--;
      const lo = stackLo[top], hi = stackHi[top];
      if (lo >= hi) continue;
      const mid = (lo + hi) >> 1;
      const dx = nodeX[mid] - px, dy = nodeY[mid] - py, dz = nodeZ[mid] - pz;
      const d = dx * dx + dy * dy + dz * dz;
      if (radius < 0) { if (d < best) best = d; } else if (d <= radius) buf[n++] = order[mid];
      if (hi - lo === 1) continue;
      const ax = axisOf[mid];
      const diff = ax === 0 ? -dx : ax === 1 ? -dy : -dz; // query - node along the split axis
      // push the far side first (popped last), and only if it can hold something closer
      if (diff < 0) {
        if (diff * diff <= best) { stackLo[top] = mid + 1; stackHi[top] = hi; top++; }
        stackLo[top] = lo; stackHi[top] = mid; top++;
      } else {
        if (diff * diff <= best) { stackLo[top] = lo; stackHi[top] = mid; top++; }
        stackLo[top] = mid + 1; stackHi[top] = hi; top++;
      }
    }
    return radius < 0 ? best : n;
  };
  return {
    candidates(px, py, pz, buf) {
      const best = search(px, py, pz, -1, null);
      const r = Math.sqrt(best) * (1 + 1e-9) + 1e-12;
      return search(px, py, pz, r * r, buf);
    },
  };
}

/** The attractor as a polyline (a single point for point attractors). */
export function geometry(att) {
  if (att.kind === "point") return [att.position.map(Number)];
  return curvePoints(att.curve);
}

/** h * f(d / radius) at packed positions P. */
export function influence(att, P, geom = null) {
  geom = geom || geometry(att);
  const d = distanceToPolyline(P, geom);
  const h = Number(att.strength), r = Math.max(Number(att.radius), 1e-9);
  const out = new Float64Array(d.length);
  for (let i = 0; i < d.length; i++) out[i] = h * falloff1(d[i] / r, att.falloff);
  return out;
}

// ------------------------------------------------------------------ weights
export function active(design) {
  return design.attractors.filter((a) => a.enabled !== false);
}

export function facePositions(mesh, space) {
  if (space === "rest" && mesh.vattr.rest) return mesh.faceMeanOfVerts(mesh.vattr.rest, 3);
  return mesh.faceCentroid;
}

/** Per-face weights for one iteration (eq. 8-9 plus modifiers). */
export function blend(design, mesh, level, baseWeights) {
  const atts = active(design);
  if (!atts.length) return baseWeights;
  const P = facePositions(mesh, design.attractor_space);
  const F = P.length / 3;
  const sets = atts.filter((a) => a.payload === "set");
  const mods = atts.filter((a) => a.payload === "modifier" && a.mods.some((m) => m.from <= level + 1 && level + 1 <= m.to));
  const W = {};
  for (const [n, v] of Object.entries(baseWeights)) W[n] = typeof v === "number" ? new Float64Array(F).fill(v) : Float64Array.from(v);

  if (sets.length) {
    const raw = sets.map((a) => influence(a, P));
    const bg = Math.max(Number(design.background), 0.0);
    const total = new Float64Array(F).fill(0);
    for (let f = 0; f < F; f++) {
      let s = 0;
      for (const r of raw) s += r[f];
      total[f] = bg + s;
    }
    for (const n of Object.keys(W)) {
      const w = W[n], out = new Float64Array(F);
      for (let f = 0; f < F; f++) {
        let acc = bg * w[f];
        for (let i = 0; i < sets.length; i++) acc = acc + raw[i][f] * (sets[i].weights[level][n] ?? 0.0);
        out[f] = total[f] > 1e-12 ? acc / total[f] : w[f];
      }
      W[n] = out;
    }
  }

  for (const a of mods) {
    const m = influence(a, P);
    for (const mod of a.mods) {
      if (!(mod.from <= level + 1 && level + 1 <= mod.to) || !(mod.weight in W)) continue;
      const w = W[mod.weight], out = new Float64Array(F);
      if (mod.op === "scale") for (let f = 0; f < F; f++) out[f] = w[f] * (1.0 + (mod.value - 1.0) * m[f]);
      else for (let f = 0; f < F; f++) out[f] = w[f] + mod.value * m[f];
      W[mod.weight] = out;
    }
  }
  return W;
}

/** Weight provider for the pipeline: (mesh, level, spec) -> weights (scalars or per-face arrays). */
export function makeProvider(design) {
  if (!active(design).length) return (mesh, level, spec) => spec.weights;
  return (mesh, level, spec) => blend(design, mesh, level, spec.weights);
}

/** What the attractors contribute to iteration `level` (for per-level cache keys). */
export function levelSignature(design, level) {
  const atts = active(design);
  if (!atts.length) return null;
  const common = [];
  for (const a of atts) {
    const g = { kind: a.kind, payload: a.payload, strength: a.strength, radius: a.radius, falloff: a.falloff };
    g.geom = a.kind === "point" ? a.position : a.curve;
    if (a.payload === "set") {
      g.w = a.weights[level];
    } else {
      g.mods = a.mods.filter((m) => m.from <= level + 1 && level + 1 <= m.to);
      if (!g.mods.length) continue; // inactive at this iteration
    }
    common.push(g);
  }
  if (!common.length) return null;
  return stableStringify({ space: design.attractor_space, bg: design.background, a: common });
}

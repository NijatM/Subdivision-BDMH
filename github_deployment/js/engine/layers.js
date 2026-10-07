// Function layer stack (our extension, not Hansmeyer's method); port of hansmeyer/layers.py.
//
// Each layer evaluates a function from functions.js and applies it in one of three ways:
//   weight        per face, before an iteration:  v = amplitude * f(p) + offset, blended into one weight
//                   add: w + m v   multiply: w (1 + m v)   replace: lerp(w, v, m)   min/max: lerp(w, min/max(w, v), m)
//   displacement  per vertex, after an iteration: move along the vertex normal by m (amplitude f + offset),
//                 in local edge lengths (relative extrusion) or model units (absolute)
//   fold          per vertex, after an iteration: V <- V + m * amplitude * (fold(V) - V)
// m is the layer's mask: 1, the normalised influence of a named attractor, or a facing mask ("facing +y").
// Layers run in list order, only in their iteration range [from, to] (1-based).

import * as functions from "./functions.js";
import { MAX_ITERATIONS } from "./weights.js";
import { deepCopy } from "./util.js";
import { facePositions, influence } from "./attractors.js";

export const TARGETS = ["weight", "displacement", "fold"];
export const BLENDS = ["add", "multiply", "replace", "min", "max"];

export const DEFAULT_LAYER = {
  name: "L", enabled: true, target: "weight", function: "gyroid", params: {}, weight: "w_f", blend: "add",
  amplitude: 0.3, offset: 0.0, from: 1, to: MAX_ITERATIONS, space: "rest", mask: "",
  domain: { fold: "none", repeat: 1, scale: 1.0, c: 0.0, params: {} },
};

export function normalize(layer) {
  const out = deepCopy(DEFAULT_LAYER);
  for (const [k, v] of Object.entries(layer)) if (k in DEFAULT_LAYER) out[k] = deepCopy(v);
  out.domain = { ...deepCopy(DEFAULT_LAYER.domain), ...deepCopy(layer.domain || {}) };
  if (!TARGETS.includes(out.target) || !BLENDS.includes(out.blend)) throw new Error("unknown layer target or blend mode");
  out.from = Math.trunc(out.from);
  out.to = Math.trunc(out.to);
  return out;
}

export function compact(layer) {
  const out = deepCopy(layer);
  if (out.target !== "weight") {
    delete out.weight;
    delete out.blend;
  }
  if ((out.domain.fold ?? "none") === "none") delete out.domain;
  if (!out.mask) delete out.mask;
  return out;
}

export function active(design, level, targets) {
  return design.layers.filter((ly) => ly.enabled && targets.includes(ly.target) && ly.from <= level + 1 && level + 1 <= ly.to);
}

export const FACING = {};
for (const [a, i] of [["x", 0], ["y", 1], ["z", 2]]) {
  for (const [s, sign] of [["+", 1], ["-", -1]]) {
    const v = [0, 0, 0];
    v[i] = sign;
    FACING[`facing ${s}${a}`] = v;
  }
}
const FACING_RANGE = [-0.35, 0.45]; // n . direction: 0 at or below the first (undersides), 1 at or above the second

/** 1 where the surface faces the named direction, smoothly 0 where it faces away. */
export function facingMask(normals, name) {
  const [lo, hi] = FACING_RANGE, d = FACING[name], n = normals.length / 3, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dot = normals[3 * i] * d[0] + normals[3 * i + 1] * d[1] + normals[3 * i + 2] * d[2];
    let t = (dot - lo) / (hi - lo);
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    out[i] = t * t * (3 - 2 * t);
  }
  return out;
}

/** The layer's mask: a number (1 = everywhere) or a per-element array. normals: an array, or a function
 * returning one (only called by facing masks). */
export function mask(design, layer, P, normals = null) {
  if (!layer.mask) return 1.0;
  if (layer.mask in FACING) {
    const n = typeof normals === "function" ? normals() : normals;
    return n === null || n === undefined ? 1.0 : facingMask(n, layer.mask);
  }
  for (const a of design.attractors) {
    if (a.name === layer.mask) {
      if (!a.enabled) return 0.0;
      const inf = influence(a, P), s = Math.max(Number(a.strength), 1e-9);
      for (let i = 0; i < inf.length; i++) inf[i] = Math.min(Math.max(inf[i] / s, 0.0), 1.0);
      return inf;
    }
  }
  return 1.0; // mask attractor not found: no masking
}

export function field(layer, P) {
  return functions.evaluate(layer.function, P, layer.params, layer.domain);
}

const at = (m, i) => (typeof m === "number" ? m : m[i]);

// ------------------------------------------------------------------ weights
export function applyWeightLayers(design, mesh, level, W) {
  const layers = active(design, level, ["weight"]);
  if (!layers.length) return W;
  const F = mesh.nFaces;
  const out = {};
  for (const [n, v] of Object.entries(W)) out[n] = typeof v === "number" ? new Float64Array(F).fill(v) : Float64Array.from(v);
  const cache = {};
  for (const ly of layers) {
    if (!(ly.weight in out)) continue;
    if (!(ly.space in cache)) cache[ly.space] = facePositions(mesh, ly.space);
    const P = cache[ly.space];
    const f = field(ly, P);
    const m = mask(design, ly, P, () => mesh.faceNormal);
    const w = out[ly.weight], b = ly.blend, res = new Float64Array(F);
    for (let i = 0; i < F; i++) {
      const v = ly.amplitude * f[i] + ly.offset, mi = at(m, i), wi = w[i];
      if (b === "add") res[i] = wi + mi * v;
      else if (b === "multiply") res[i] = wi * (1.0 + mi * v);
      else if (b === "replace") res[i] = wi + mi * (v - wi);
      else if (b === "min") res[i] = wi + mi * (Math.min(wi, v) - wi);
      else res[i] = wi + mi * (Math.max(wi, v) - wi);
    }
    out[ly.weight] = res;
  }
  return out;
}

// ------------------------------------------------------------- post steps
/** Displacement and fold layers, applied after iteration `level` (0-based). */
export function postProcess(design, mesh, level, relative = true, vmask = null) {
  const layers = active(design, level, ["displacement", "fold"]);
  if (!layers.length) return mesh;
  let V = Float64Array.from(mesh.V);
  const N = mesh.nVerts;
  const keep = new Float64Array(N).fill(1);
  if (mesh.vattr.lock) for (let i = 0; i < N; i++) keep[i] *= mesh.vattr.lock[i] <= level ? 1 : 0;
  if (vmask && vmask.length === N) for (let i = 0; i < N; i++) keep[i] *= vmask[i];
  for (const ly of layers) {
    const cur = mesh.withPositions(V);
    if (ly.target === "displacement") {
      const P = ly.space === "rest" && cur.vattr.rest ? cur.vattr.rest : V;
      const m = mask(design, ly, P, cur.vertNormal);
      const f = field(ly, P);
      const amount = new Float64Array(N);
      for (let i = 0; i < N; i++) amount[i] = (ly.amplitude * f[i] + ly.offset) * at(m, i) * keep[i];
      if (relative) {
        const s = cur.vertMean(cur.faceScale);
        for (let i = 0; i < N; i++) amount[i] = amount[i] * s[i];
      }
      const n = cur.vertNormal, nV = new Float64Array(V.length);
      for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) nV[3 * i + k] = V[3 * i + k] + n[3 * i + k] * amount[i];
      V = nV;
    } else {
      const target = functions.evaluate(ly.function, V, ly.params);
      const m = mask(design, ly, V, () => cur.vertNormal);
      const nV = new Float64Array(V.length);
      for (let i = 0; i < N; i++) {
        const s = ly.amplitude * at(m, i) * keep[i];
        for (let k = 0; k < 3; k++) nV[3 * i + k] = V[3 * i + k] + s * (target[3 * i + k] - V[3 * i + k]);
      }
      V = nV;
    }
  }
  return mesh.withPositions(V);
}

export function signature(design, level) {
  const out = [];
  for (const ly of design.layers) {
    if (!ly.enabled || !(ly.from <= level + 1 && level + 1 <= ly.to)) continue;
    const sig = {};
    for (const k of ["target", "function", "params", "weight", "blend", "amplitude", "offset", "space", "mask", "domain"]) sig[k] = ly[k];
    sig.fn = functions.signature(ly.function);
    const dom = ly.domain.fold ?? "none";
    sig.dfn = dom !== "none" ? functions.signature(dom) : null;
    out.push(sig);
  }
  return out.length ? out : null;
}

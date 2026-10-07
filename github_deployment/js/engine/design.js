// The full design state, saved as JSON presets (port of the Design class in hansmeyer/schedule.py).
// A design is a plain object; normalizeDesign fills in every default (presets stay short), toDict compacts it
// for saving in the same format as the desktop app, so presets move freely between the two.

import * as attractors from "./attractors.js";
import * as intrinsic from "./intrinsic.js";
import * as layers from "./layers.js";
import * as merge from "./merge.js";
import * as vessel from "./vessel.js";
import { MAX_ITERATIONS, normalizeIteration } from "./weights.js";
import { deepCopy, digest } from "./util.js";

export const DESIGN_FIELDS = ["name", "description", "base", "extrusion", "boundary", "fade_rows", "preview_depth",
  "full_depth", "iterations", "view", "attractors", "attractor_space", "background", "layers", "groups", "motifs",
  "intrinsic", "merge", "vessel"];

export function defaultDesign() {
  return {
    name: "Untitled", description: "", base: { shape: "cube" }, extrusion: "relative", boundary: "smooth", fade_rows: 2.0,
    preview_depth: 5, full_depth: 8, iterations: [], view: "diagonal", attractors: [], attractor_space: "current",
    background: 1.0, layers: [], groups: [], motifs: {}, intrinsic: [], merge: {}, vessel: {},
  };
}

/** Every field present and valid (throws like the desktop app on unknown values). */
export function normalizeDesign(d) {
  const out = defaultDesign();
  for (const k of DESIGN_FIELDS) if (d[k] !== undefined) out[k] = deepCopy(d[k]);
  const its = (out.iterations || []).map((i) => normalizeIteration(i));
  while (its.length < MAX_ITERATIONS) its.push(normalizeIteration({}));
  out.iterations = its.slice(0, MAX_ITERATIONS);
  if (!["relative", "absolute"].includes(out.extrusion)) throw new Error("extrusion must be 'relative' or 'absolute'");
  if (!["smooth", "locked"].includes(out.boundary)) throw new Error("boundary must be 'smooth' or 'locked'");
  if (!["current", "rest"].includes(out.attractor_space)) throw new Error("attractor_space must be 'current' or 'rest'");
  out.preview_depth = Math.trunc(out.preview_depth);
  out.full_depth = Math.trunc(out.full_depth);
  out.fade_rows = Number(out.fade_rows);
  out.background = Number(out.background);
  out.attractors = out.attractors.map(attractors.normalize);
  out.layers = out.layers.map(layers.normalize);
  out.groups = out.groups.map(intrinsic.normalizeGroup);
  out.intrinsic = out.intrinsic.map(intrinsic.normalizeRule);
  out.motifs = Object.fromEntries(Object.entries(out.motifs || {}).map(([k, v]) => [String(k), Number(v)]));
  out.merge = merge.normalize(out.merge);
  out.vessel = vessel.normalize(out.vessel);
  if (typeof out.base !== "object" || out.base === null) out.base = { shape: "cube" };
  return out;
}

/** Compact JSON-ready object (drops all-zero weights, unused attractor data, empty lists). */
export function toDict(design) {
  const d = {};
  for (const k of DESIGN_FIELDS) d[k] = deepCopy(design[k]);
  d.iterations = d.iterations.map((it) => ({
    scheme: it.scheme, weights: Object.fromEntries(Object.entries(it.weights).filter(([, v]) => v !== 0.0)),
  }));
  d.attractors = design.attractors.map(attractors.compact);
  d.layers = design.layers.map(layers.compact);
  d.motifs = Object.fromEntries(Object.entries(design.motifs).filter(([, v]) => v !== 0.0));
  for (const key of ["layers", "groups", "intrinsic", "attractors"]) if (!d[key].length) delete d[key];
  if (!Object.keys(d.motifs).length) delete d.motifs;
  if (!d.merge.enabled) delete d.merge;
  if (!d.vessel.enabled) delete d.vessel;
  return d;
}

export function fromDict(d) {
  return normalizeDesign(d);
}

export function copyDesign(d) {
  return normalizeDesign(toDict(d));
}

/** JSON text of a design in the desktop app's preset format (2-space indent, floats keep a decimal point). */
export function presetText(design) {
  return pyJson(toDict(design), 0) + "\n";
}

function pyNum(x) {
  if (Number.isInteger(x)) return x.toFixed(1);
  return String(x);
}

/** json.dumps(indent=2) in Python's style, keeping floats as floats (1.0, not 1) where the desktop app
 * writes them. Integer-valued fields stay integers. */
function pyJson(v, depth, key = "") {
  const pad = "  ".repeat(depth + 1), end = "  ".repeat(depth);
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "null";
    return INT_KEYS.has(key) && Number.isInteger(v) ? String(v) : pyNum(v);
  }
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) {
    if (!v.length) return "[]";
    const isIntList = key === "faces" || key === "verts";
    return "[\n" + v.map((x) => pad + (isIntList ? String(x) : pyJson(x, depth + 1, key === "points" ? "" : key))).join(",\n") + "\n" + end + "]";
  }
  const keys = Object.keys(v);
  if (!keys.length) return "{}";
  return "{\n" + keys.map((k) => pad + JSON.stringify(k) + ": " + pyJson(v[k], depth + 1, k)).join(",\n") + "\n" + end + "}";
}

// keys whose numbers the desktop app stores as integers
const INT_KEYS = new Set(["preview_depth", "full_depth", "from", "to", "lock", "max_valence", "sides", "rings", "segments",
  "nx", "ny", "shaft_segments", "shaft_rings", "a", "b", "c", "repeat", "octaves", "seed", "mode", "k", "l", "m"]);

// ------------------------------------------------------------- caching keys
export function baseKey(design, objHash = null) {
  const key = {
    base: design.base, extrusion: design.extrusion, boundary: design.boundary, fade: design.fade_rows,
    groups: intrinsic.baseSignature(design),
  };
  if (design.base.shape === "obj") key.obj = objHash; // re-import when the file's content changes
  return digest(key);
}

/** Everything besides the iteration's own weights that affects iteration `level`. */
export function levelSignature(design, level) {
  const sig = {
    att: attractors.levelSignature(design, level), layers: layers.signature(design, level),
    groups: intrinsic.groupSignature(design, level), intr: intrinsic.signature(design, level),
    merge: merge.signature(design, level),
  };
  const out = {};
  for (const [k, v] of Object.entries(sig)) if (v !== null && v !== undefined) out[k] = v;
  return Object.keys(out).length ? out : null;
}

/** Key for the mesh after each iteration: changing iteration k only invalidates levels >= k. */
export function levelKeys(design, depth, objHash = null) {
  const keys = [];
  let prev = baseKey(design, objHash);
  for (let level = 0; level < Math.min(depth, design.iterations.length); level++) {
    prev = digest({ prev, it: design.iterations[level], field: levelSignature(design, level) });
    keys.push(prev);
  }
  return keys;
}

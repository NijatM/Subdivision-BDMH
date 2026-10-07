// Weight definitions and the per-iteration schedule (from hansmeyer/schedule.py). No imports: every
// other engine module may use these constants at load time.

export const MAX_ITERATIONS = 10;

const W = (name, label, lo, hi, help) => ({ name, label, lo, hi, help });

// Extended Catmull-Clark (Hansmeyer 2010, eq. 1-4). All zero == standard Catmull-Clark.
export const CC_WEIGHTS = [
  W("w_f", "w_f  face extrude", -2.0, 2.0, "Eq.1: push face points along the face normal."),
  W("w_e", "w_e  edge extrude", -2.0, 2.0, "Eq.2: push edge points along the edge normal."),
  W("w_c", "w_c  corner extrude", -2.0, 2.0, "Eq.3: push corner points along the vertex normal."),
  W("w1", "w1   edge bias", -2.0, 2.0, "Eq.2: + pulls edge points toward face points, - toward the edge."),
  W("w2", "w2   corner bias", -2.0, 2.0, "Eq.3: + pulls corners toward face points, - toward edge midpoints."),
  W("w3", "w3   V/F bias", -2.0, 2.0, "Eq.4 (iter 2+): face point bias toward the old-corner vs old-face vertex."),
  W("w4", "w4   diag/edge bias", -2.0, 2.0, "Eq.4 (iter 2+): face point bias toward the diagonal vs the edge vertices."),
  W("w6", "w6   motif face pull", -2.0, 2.0, "Eq.10: pull face points toward vertices by their motif value U (Intrinsic panel)."),
  W("w7", "w7   motif edge pull", -2.0, 2.0, "Eq.11: pull edge points toward vertices by their motif value U (Intrinsic panel)."),
];

// Extended Doo-Sabin (eq. 5-6), with separate weights per face class (face-, edge-, vertex-derived).
export const DS_WEIGHTS = [
  W("ds_w1_face", "w1   face-faces", -2.0, 2.0, "Eq.5/6: corner pull toward its own vertex (faces from old faces)."),
  W("ds_wf_face", "w_f  face-faces", -2.0, 2.0, "Extrude new corners along the face normal (faces from old faces)."),
  W("ds_w1_edge", "w1   edge-faces", -2.0, 2.0, "As above, for faces created from old edges (after a DS step)."),
  W("ds_wf_edge", "w_f  edge-faces", -2.0, 2.0, "As above, for faces created from old edges (after a DS step)."),
  W("ds_w1_vert", "w1   vertex-faces", -2.0, 2.0, "As above, for faces created from old vertices (after a DS step)."),
  W("ds_wf_vert", "w_f  vertex-faces", -2.0, 2.0, "As above, for faces created from old vertices (after a DS step)."),
];

export const ALL_WEIGHTS = [...CC_WEIGHTS, ...DS_WEIGHTS];
export const WEIGHT_NAMES = ALL_WEIGHTS.map((w) => w.name);
export const SCHEMES = ["cc", "ds"];

export function zeroWeights() {
  const out = {};
  for (const n of WEIGHT_NAMES) out[n] = 0.0;
  return out;
}

export function normalizeIteration(it = {}) {
  const scheme = it.scheme === undefined ? "cc" : it.scheme;
  if (!SCHEMES.includes(scheme)) throw new Error(`unknown scheme '${scheme}'`);
  const weights = zeroWeights();
  for (const [k, v] of Object.entries(it.weights || {})) if (k in weights) weights[k] = Number(v);
  return { scheme, weights };
}

// Function library for the layer stack (our extension, not part of Hansmeyer's method); port of
// hansmeyer/functions.py.
//
//   fields  f(P, params) -> Float64Array(n) values, roughly in [-1, 1]  (drive weights, displace the surface)
//   folds   g(P, params) -> Float64Array(3n) positions                 (deform the mesh, fold a field's domain)
//
// Plug-ins are small JavaScript files (see functions/README.md) that register per-point functions:
//   field("ripples", { label, family, help, params: { frequency: [0.2, 12, 3] } }, (x, y, z, p) => ...)
//   fold("twist", { params: { turns: [-1, 1, 0.15] } }, (x, y, z, p) => [x2, y2, z2])
// A parameter is [lo, hi, default] or [lo, hi, default, "int"].

import * as noise from "./noise.js";
import { hashString, pyRound } from "./util.js";

export const REGISTRY = {};
export const PLUGIN_ERRORS = [];
const TAU = 2 * Math.PI;

function parseParams(params = {}) {
  return Object.entries(params).map(([name, spec]) => ({
    name, lo: Number(spec[0]), hi: Number(spec[1]), default: Number(spec[2]), integer: spec.length > 3 && spec[3] === "int",
  }));
}

function register(kind, name, opts, fn, source = "", version = "") {
  const key = name.toLowerCase().replace(/ /g, "_");
  REGISTRY[key] = {
    name: key, label: opts.label || name, kind, family: opts.family || "custom", fn,
    params: parseParams(opts.params), help: opts.help || "", source, version,
  };
}

export function defaults(fd) {
  const out = {};
  for (const p of fd.params) out[p.name] = p.default;
  return out;
}

export function call(fd, P, params = {}) {
  const kw = defaults(fd);
  for (const [k, v] of Object.entries(params || {})) if (k in kw) kw[k] = Number(v);
  for (const p of fd.params) if (p.integer) kw[p.name] = pyRound(kw[p.name]);
  return fd.fn(P, kw);
}

export const fields = () => Object.values(REGISTRY).filter((f) => f.kind === "field");
export const folds = () => Object.values(REGISTRY).filter((f) => f.kind === "fold");

const clip1 = (x) => (x < -1 ? -1 : x > 1 ? 1 : x);

function mapField(P, f) {
  const n = P.length / 3, out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = f(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
  return out;
}

function mapFold(P, f) {
  const out = new Float64Array(P.length);
  for (let i = 0; i < P.length; i += 3) {
    const q = f(P[i], P[i + 1], P[i + 2]);
    out[i] = q[0]; out[i + 1] = q[1]; out[i + 2] = q[2];
  }
  return out;
}

function scaled(P, k) {
  const out = new Float64Array(P.length);
  for (let i = 0; i < P.length; i++) out[i] = P[i] * k;
  return out;
}

// =====================================================================  TPMS
const tpms = (fn) => (P, k) => mapField(P, (x, y, z) => {
  const qx = TAU * (k.frequency * x + k.phase), qy = TAU * (k.frequency * y + k.phase), qz = TAU * (k.frequency * z + k.phase);
  return fn(qx, qy, qz);
});
const FREQ = { frequency: [0.05, 6.0, 1.0], phase: [0.0, 1.0, 0.0] };
register("field", "gyroid", { label: "Gyroid", family: "TPMS", help: "Triply periodic minimal surface: labyrinthine, coral-like channels.", params: FREQ },
  tpms((x, y, z) => (Math.sin(x) * Math.cos(y) + Math.sin(y) * Math.cos(z) + Math.sin(z) * Math.cos(x)) / 1.5));
register("field", "schwarz_p", { label: "Schwarz P", family: "TPMS", help: "Primitive surface: a lattice of rounded cells.", params: FREQ },
  tpms((x, y, z) => (Math.cos(x) + Math.cos(y) + Math.cos(z)) / 3.0));
register("field", "schwarz_d", { label: "Schwarz D (diamond)", family: "TPMS", help: "Diamond surface: tetrahedral, crystalline channels.", params: FREQ },
  tpms((x, y, z) => {
    const s = Math.sin, c = Math.cos;
    return (s(x) * s(y) * s(z) + s(x) * c(y) * c(z) + c(x) * s(y) * c(z) + c(x) * c(y) * s(z)) / 1.42;
  }));
register("field", "neovius", { label: "Neovius", family: "TPMS", help: "Neovius surface: cubic cells with tubular necks.", params: FREQ },
  tpms((x, y, z) => (3 * (Math.cos(x) + Math.cos(y) + Math.cos(z)) + 4 * Math.cos(x) * Math.cos(y) * Math.cos(z)) / 13.0));

// ==================================================================== noise
register("field", "perlin", { label: "Perlin noise", family: "noise", help: "Smooth random variation.",
  params: { frequency: [0.05, 12.0, 1.5], seed: [0, 99, 0, "int"] } },
(P, k) => noise.perlin(scaled(P, k.frequency), k.seed).map((v) => clip1(v * 1.4)));
register("field", "fbm", { label: "fBm (fractal noise)", family: "noise", help: "Several octaves of noise: detail at every scale.",
  params: { frequency: [0.05, 12.0, 1.5], octaves: [1, 8, 4, "int"], gain: [0.1, 0.9, 0.5], seed: [0, 99, 0, "int"] } },
(P, k) => noise.fbm(scaled(P, k.frequency), k.octaves, 2.0, k.gain, k.seed).map((v) => clip1(v * 1.6)));
register("field", "ridged", { label: "Ridged noise", family: "noise", help: "Sharp crests and valleys, mountain-like.",
  params: { frequency: [0.05, 12.0, 1.5], octaves: [1, 8, 4, "int"], gain: [0.1, 0.9, 0.5], seed: [0, 99, 0, "int"] } },
(P, k) => noise.ridged(scaled(P, k.frequency), k.octaves, 2.0, k.gain, k.seed));
register("field", "warped", { label: "Domain-warped noise", family: "noise", help: "Noise fed by noise: marbled, flowing patterns.",
  params: { frequency: [0.05, 12.0, 1.0], warp: [0.0, 4.0, 1.5], seed: [0, 99, 0, "int"] } },
(P, k) => noise.warped(scaled(P, k.frequency), k.warp, 4, k.seed).map((v) => clip1(v * 1.8)));
register("field", "worley", { label: "Worley / Voronoi cells", family: "noise",
  help: "Cellular pattern. mode 0: cell centres high; mode 1: cell borders high (F2 - F1).",
  params: { frequency: [0.1, 12.0, 2.0], mode: [0, 1, 0, "int"], jitter: [0.0, 1.0, 1.0], seed: [0, 99, 0, "int"] } },
(P, k) => {
  const [f1, f2] = noise.worley(scaled(P, k.frequency), k.seed, k.jitter);
  const out = new Float64Array(f1.length);
  if (k.mode === 0) for (let i = 0; i < out.length; i++) out[i] = clip1(1.0 - 2.2 * f1[i]);
  else for (let i = 0; i < out.length; i++) out[i] = clip1(1.0 - 6.0 * (f2[i] - f1[i]));
  return out;
});

// ================================================================= analytic
function spherical(x, y, z) {
  const r = Math.sqrt(x * x + y * y + z * z);
  const rr = Math.max(r, 1e-12);
  const theta = Math.acos(Math.min(Math.max(y / rr, -1), 1)); // polar angle from +y
  const phi = Math.atan2(z, x); // azimuth around y
  return [r, theta, phi];
}

function assocLegendre(l, m, x) {
  let pmm = 1.0;
  if (m > 0) {
    const somx2 = Math.sqrt(Math.max(1 - x * x, 0));
    let fact = 1.0;
    for (let i = 0; i < m; i++) {
      pmm = -pmm * fact * somx2;
      fact += 2.0;
    }
  }
  if (l === m) return pmm;
  let pmmp1 = x * (2 * m + 1) * pmm;
  if (l === m + 1) return pmmp1;
  for (let ll = m + 2; ll <= l; ll++) {
    const pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pmmp1;
}

function realSH(l, m, theta, phi) {
  const P = assocLegendre(l, Math.abs(m), Math.cos(theta));
  if (m > 0) return P * Math.cos(m * phi);
  if (m < 0) return P * Math.sin(-m * phi);
  return P;
}

const SH_SCALE = new Map();
function shScale(l, m) {
  const key = `${l},${m}`;
  if (SH_SCALE.has(key)) return SH_SCALE.get(key);
  let mx = 0;
  for (let i = 0; i < 160; i++) {
    const ph = -Math.PI + (2 * Math.PI * i) / 159;
    for (let j = 0; j < 160; j++) {
      const th = Math.acos(1 - (2 * j) / 159);
      mx = Math.max(mx, Math.abs(realSH(l, m, th, ph)));
    }
  }
  const s = mx || 1.0;
  SH_SCALE.set(key, s);
  return s;
}

register("field", "spherical_harmonic", { label: "Spherical harmonic", family: "analytic",
  help: "Real spherical harmonic Y(l, m) around the y axis: lobed, symmetric patterns.",
  params: { l: [0, 10, 4, "int"], m: [-10, 10, 3, "int"] } },
(P, k) => {
  const l = k.l, m = Math.max(-l, Math.min(l, k.m)), s = shScale(l, m);
  return mapField(P, (x, y, z) => {
    const [, th, ph] = spherical(x, y, z);
    return realSH(l, m, th, ph) / s;
  });
});

function sf(angle, m, n1, n2, n3) {
  const t = (m * angle) / 4;
  return (Math.abs(Math.cos(t)) ** n2 + Math.abs(Math.sin(t)) ** n3) ** (-1.0 / Math.max(n1, 1e-6));
}

register("field", "constant", { label: "Constant", family: "analytic",
  help: "1 everywhere: with a mask, applies amplitude + offset only where the mask is (e.g. a weight boost on upward-facing surfaces)." },
(P) => new Float64Array(P.length / 3).fill(1));
register("field", "superformula", { label: "Superformula (Gielis)", family: "analytic",
  help: "Positive inside a Gielis superformula shape, negative outside: displace to morph toward it.",
  params: { m: [0, 16, 6, "int"], n1: [0.1, 20.0, 3.0], n2: [0.1, 20.0, 6.0], n3: [0.1, 20.0, 6.0], size: [0.2, 6.0, 1.8] } },
(P, k) => mapField(P, (x, y, z) => {
  const [r, th, ph] = spherical(x, y, z);
  const rho = Math.min(sf(ph, k.m, k.n1, k.n2, k.n3) * sf(th - Math.PI / 2, k.m, k.n1, k.n2, k.n3), 10.0);
  return clip1((k.size * rho - r) / k.size);
}));
register("field", "superquadric", { label: "Superquadric", family: "analytic",
  help: "Positive inside a superquadric (exponent 2 = sphere, high = cube, <1 = star), negative outside.",
  params: { exponent: [0.3, 12.0, 4.0], size: [0.2, 6.0, 1.6] } },
(P, k) => mapField(P, (x, y, z) => {
  const r = Math.sqrt(x * x + y * y + z * z), rr = Math.max(r, 1e-12), e = k.exponent;
  const s = Math.abs(x / rr) ** e + Math.abs(y / rr) ** e + Math.abs(z / rr) ** e;
  const rho = k.size / Math.max(s, 1e-12) ** (1.0 / e);
  return clip1((rho - r) / k.size);
}));
register("field", "rose", { label: "Rose (k-fold radial)", family: "analytic",
  help: "cos(k * angle) around the y axis: flutes, petals and ribs; twist turns them into spirals.",
  params: { k: [1, 32, 8, "int"], twist: [-6.0, 6.0, 0.0], axial: [0.0, 6.0, 0.0] } },
(P, k) => mapField(P, (x, y, z) => {
  const ang = Math.atan2(z, x) + k.twist * y;
  let out = Math.cos(k.k * ang);
  if (k.axial > 0) out = out * Math.cos((TAU * k.axial * y) / 2);
  return out;
}));

// ==================================================================== folds
const boxFold = (P, k) => mapFold(P, (x, y, z) => [x, y, z].map((v) => Math.min(Math.max(v, -k.limit), k.limit) * 2 - v));
function sphereFoldPt(x, y, z, minR, fixedR) {
  const r2 = x * x + y * y + z * z, mr2 = minR ** 2, fr2 = Math.max(fixedR ** 2, minR ** 2 + 1e-9);
  const k = r2 < mr2 ? fr2 / mr2 : r2 < fr2 ? fr2 / Math.max(r2, 1e-12) : 1.0;
  return [x * k, y * k, z * k];
}
register("fold", "box_fold", { label: "Box fold", family: "fold", help: "Mandelbox box fold: reflects points beyond +/- limit back inside.",
  params: { limit: [0.1, 4.0, 1.0] } }, boxFold);
register("fold", "clamp_box", { label: "Clamp to box", family: "fold",
  help: "Flattens everything outside a box onto its faces: flat, cut-like outer faces (e.g. a cube cage).",
  params: { size: [0.1, 4.0, 1.0] } },
(P, k) => mapFold(P, (x, y, z) => [x, y, z].map((v) => Math.min(Math.max(v, -k.size), k.size))));
register("fold", "sphere_fold", { label: "Sphere fold", family: "fold",
  help: "Mandelbox sphere fold: inverts points inside the fixed radius (scales up the core).",
  params: { min_radius: [0.05, 2.0, 0.5], fixed_radius: [0.1, 4.0, 1.0] } },
(P, k) => mapFold(P, (x, y, z) => sphereFoldPt(x, y, z, k.min_radius, k.fixed_radius)));
register("fold", "kaleido", { label: "Kaleidoscopic mirror", family: "fold", help: "Mirrors space into k wedges around the y axis (k-fold symmetry).",
  params: { k: [2, 24, 6, "int"], rotation: [0.0, 1.0, 0.0] } },
(P, k) => mapFold(P, (x, y, z) => {
  const r = Math.hypot(x, z);
  const ang = Math.atan2(z, x) - (k.rotation * TAU) / k.k;
  const sector = TAU / k.k;
  let a = ang - sector * Math.floor(ang / sector); // np.mod
  a = (a > sector / 2 ? sector - a : a) + (k.rotation * TAU) / k.k;
  return [r * Math.cos(a), y, r * Math.sin(a)];
}));
register("fold", "mandelbox", { label: "Mandelbox step", family: "fold",
  help: "One Mandelbox iteration: box fold, sphere fold, then scale (use 'repeat' in a field's domain).",
  params: { scale: [-3.0, 3.0, 2.0], limit: [0.1, 4.0, 1.0], min_radius: [0.05, 2.0, 0.5], fixed_radius: [0.1, 4.0, 1.0] } },
(P, k) => mapFold(P, (x, y, z) => {
  const b = [x, y, z].map((v) => Math.min(Math.max(v, -k.limit), k.limit) * 2 - v);
  const s = sphereFoldPt(b[0], b[1], b[2], k.min_radius, k.fixed_radius);
  return [k.scale * s[0], k.scale * s[1], k.scale * s[2]];
}));

// ============================================================== evaluation
/** Repeatedly fold the input of a field: p <- scale * fold(p) + c * p0. */
export function foldDomain(P, domain) {
  if (!domain || ["none", "", null, undefined].includes(domain.fold)) return P;
  const fd = REGISTRY[domain.fold];
  if (!fd || fd.kind !== "fold") return P;
  let q = Float64Array.from(P);
  const scale = Number(domain.scale ?? 1.0), c = Number(domain.c ?? 0.0);
  for (let r = 0; r < Math.trunc(domain.repeat ?? 1); r++) {
    const g = call(fd, q, domain.params || {});
    const nq = new Float64Array(q.length);
    for (let i = 0; i < q.length; i++) {
      const v = scale * g[i] + c * P[i];
      nq[i] = v < -1e6 ? -1e6 : v > 1e6 ? 1e6 : v;
    }
    q = nq;
  }
  return q;
}

export function evaluate(name, P, params, domain = null) {
  const fd = REGISTRY[name];
  if (!fd) throw new Error(`unknown function '${name}' (missing plug-in?)`);
  const q = fd.kind === "field" ? foldDomain(P, domain) : P;
  const out = call(fd, q, params);
  const expected = fd.kind === "field" ? P.length / 3 : P.length;
  if (!out || out.length !== expected) throw new Error(`function '${name}' returned ${out ? out.length : 0} values, expected ${expected}`);
  const res = out instanceof Float64Array ? out : Float64Array.from(out);
  for (let i = 0; i < res.length; i++) if (!Number.isFinite(res[i])) res[i] = 0.0;
  return res;
}

// ================================================================== plug-ins
/** Register every plug-in (list of {name, source}); previous plug-in registrations are dropped first.
 * Returns the error messages. */
export function loadPlugins(list) {
  PLUGIN_ERRORS.length = 0;
  for (const [k, f] of Object.entries(REGISTRY)) if (f.source) delete REGISTRY[k];
  for (const { name, source } of list || []) {
    const version = hashString(source);
    const api = {
      field: (key, opts, fn) => register("field", key, opts || {}, (P, k) => mapField(P, (x, y, z) => fn(x, y, z, k)), name, version),
      fold: (key, opts, fn) => register("fold", key, opts || {}, (P, k) => mapFold(P, (x, y, z) => fn(x, y, z, k)), name, version),
    };
    try {
      // eslint-disable-next-line no-new-func
      new Function("field", "fold", `"use strict";\n${source}`)(api.field, api.fold);
    } catch (e) {
      PLUGIN_ERRORS.push(`${name}: ${e && e.message ? e.message : e}`);
    }
  }
  return [...PLUGIN_ERRORS];
}

export function signature(name) {
  const fd = REGISTRY[name];
  return fd ? [fd.name, fd.version] : null;
}

/** Plain description of the registry (for the page: labels, families, parameters). */
export function describe() {
  return Object.values(REGISTRY).map((f) => ({
    name: f.name, label: f.label, kind: f.kind, family: f.family, params: f.params, help: f.help, source: f.source,
  }));
}

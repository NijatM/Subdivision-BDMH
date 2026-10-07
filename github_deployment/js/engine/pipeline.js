// Runs a design's schedule with per-iteration caching and a face budget (port of hansmeyer/pipeline.py).

import * as attractors from "./attractors.js";
import * as cc from "./catmullClark.js";
import * as ds from "./dooSabin.js";
import * as intrinsic from "./intrinsic.js";
import * as layers from "./layers.js";
import * as merge from "./merge.js";
import * as vessel from "./vessel.js";
import { baseKey, levelKeys, normalizeDesign } from "./design.js";
import { makeBase } from "./shapes.js";
import { now } from "./util.js";

export const DEFAULT_FACE_BUDGET = 2_000_000;
const SCHEMES = { cc, ds };

/** 0 on the boundary, smoothly rising to 1 at `rows` base-mesh rows inside. */
export function boundaryFade(dist, rows) {
  const out = new Float64Array(dist.length);
  if (rows <= 0) {
    for (let i = 0; i < dist.length; i++) out[i] = dist[i] > 0 ? 1 : 0;
    return out;
  }
  for (let i = 0; i < dist.length; i++) {
    let t = dist[i] / rows;
    t = t < 0 ? 0 : t > 1 ? 1 : t; // (Infinity -> 1)
    out[i] = t * t * (3 - 2 * t);
  }
  return out;
}

/** The design's weights for one iteration: the schedule, then attractors (eq. 8-9), group rules,
 * intrinsic rules and function layers, each refining the previous result. */
export function makeProvider(design) {
  const att = attractors.makeProvider(design);
  return (mesh, level, spec) => {
    let W = att(mesh, level, spec);
    W = intrinsic.applyGroupRules(design, mesh, level, W);
    W = intrinsic.applyRules(design, mesh, level, W);
    return layers.applyWeightLayers(design, mesh, level, W);
  };
}

export class Pipeline {
  /** ctx: { objText(path) -> string, objHash(path) -> string } for imported meshes. */
  constructor({ faceBudget = DEFAULT_FACE_BUDGET, maxCached = 32, maxBytes = 600_000_000, ctx = {} } = {}) {
    this.faceBudget = faceBudget;
    this.maxCached = maxCached;
    this.maxBytes = maxBytes;
    this.ctx = ctx;
    this.cache = new Map();
    this.sizes = new Map();
    this.bytes = 0;
  }

  get(key) {
    const m = this.cache.get(key);
    if (m !== undefined) {
      this.cache.delete(key);
      this.cache.set(key, m);
    }
    return m;
  }

  put(key, mesh) {
    if (this.cache.has(key)) {
      this.bytes -= this.sizes.get(key);
      this.cache.delete(key);
    }
    this.cache.set(key, mesh);
    const size = mesh.nbytes() + mesh.faceIdx.byteLength;
    this.sizes.set(key, size);
    this.bytes += size;
    while (this.cache.size > 1 && (this.cache.size > this.maxCached || this.bytes > this.maxBytes)) {
      const old = this.cache.keys().next().value;
      this.cache.delete(old);
      this.bytes -= this.sizes.get(old);
      this.sizes.delete(old);
    }
  }

  clear() {
    this.cache.clear();
    this.sizes.clear();
    this.bytes = 0;
  }

  objHash(design) {
    return design.base.shape === "obj" && this.ctx.objHash ? this.ctx.objHash(design.base.path) : null;
  }

  /** The (cached) input mesh of a design. */
  base(design) {
    const key = baseKey(design, this.objHash(design));
    let mesh = this.get(key);
    if (mesh === undefined) {
      mesh = makeBase(design.base, this.ctx).shareTopology(); // same faces, new positions: tables reused
      mesh.vattr.rest = Float64Array.from(mesh.V); // input-mesh position, carried through subdivision
      intrinsic.decorateBase(design, mesh); // tags, locks, original vertex/edge markers
      if (design.boundary === "locked") mesh.vattr.bdist = mesh.boundaryDistance();
      this.put(key, mesh);
    }
    return mesh;
  }

  /** Run to `depth`. onLevel(level, mesh) is called after each computed level (progress). */
  run(design, depth, { weightProvider = null, onLevel = null } = {}) {
    const t0 = now();
    weightProvider = weightProvider || makeProvider(design);
    let mesh = this.base(design);
    const relative = design.extrusion === "relative";
    const lock = design.boundary === "locked";
    const keys = levelKeys(design, depth, this.objHash(design));
    let hits = 0, reached = 0, capped = false;
    for (let level = 0; level < keys.length; level++) {
      const spec = design.iterations[level], key = keys[level];
      const cached = this.get(key);
      if (cached !== undefined) {
        mesh = cached;
        hits++;
        reached = level + 1;
        continue;
      }
      const module = SCHEMES[spec.scheme];
      if (module.predictedFaces(mesh) > this.faceBudget) {
        capped = true;
        break;
      }
      let vmask = null;
      if (lock && mesh.vattr.bdist && mesh.vattr.bdist.some((x) => Number.isFinite(x))) {
        vmask = boundaryFade(mesh.vattr.bdist, design.fade_rows);
      }
      let vlock = null;
      if (mesh.vattr.lock) {
        vlock = new Uint8Array(mesh.nVerts);
        for (let i = 0; i < vlock.length; i++) vlock[i] = mesh.vattr.lock[i] > level ? 1 : 0;
      }
      const vU = spec.scheme === "cc" ? intrinsic.motifU(design, mesh) : null;
      mesh = module.subdivide(mesh, weightProvider(mesh, level, spec), { relative, lockBoundary: lock, vmask, vlock, vU });
      if (lock && mesh.vattr.bdist) vmask = boundaryFade(mesh.vattr.bdist, design.fade_rows);
      mesh = layers.postProcess(design, mesh, level, relative, vmask);
      mesh = merge.maybeMerge(design, mesh, level);
      this.put(key, mesh);
      reached = level + 1;
      if (onLevel) onLevel(level, mesh);
    }
    let relief = null;
    if (vessel.active(design) && mesh.nFaces) {
      const plain = reached ? this.run(vessel.plainDesign(design, normalizeDesign), reached).mesh : null;
      relief = mesh;
      mesh = vessel.build(design, mesh, plain);
    }
    return { mesh, depthReached: reached, depthRequested: depth, cappedByBudget: capped, seconds: (now() - t0) / 1000, cacheHits: hits, relief };
  }
}

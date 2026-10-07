// Vertex merging and porosity (Hansmeyer 2010, discussion of motifs); port of hansmeyer/merge.py.
//
// After an iteration, pairs of vertices closer than `distance` are welded, but only vertices that do not
// already share a face (parts of the surface that have grown into contact). With a relative distance the
// threshold of a pair is distance x the smaller of the two local edge lengths. Each vertex joins at most one
// partner (nearest pairs first). Faces that collapse are removed; where a welded vertex would exceed
// `max_valence`, the faces that joined it are not formed (holes). A cleanup pass then removes faces until
// every edge is manifold and consistently oriented, so the next iteration can subdivide the result.

import { PolyMesh } from "./mesh.js";
import { percentile } from "./util.js";

export const DEFAULT_MERGE = { enabled: false, distance: 0.15, relative: true, max_valence: 0, from: 1, to: 10 };

export function normalize(m) {
  const out = { ...DEFAULT_MERGE };
  for (const [k, v] of Object.entries(m || {})) if (k in DEFAULT_MERGE) out[k] = v;
  return out;
}

export function active(design, level) {
  const m = design.merge;
  return !!m.enabled && m.from <= level + 1 && level + 1 <= m.to;
}

export function signature(design, level) {
  return active(design, level) ? { ...design.merge } : null;
}

/** Vertex pairs (i < j) closer than thr, via a uniform grid of cell size thr, as [a0, b0, a1, b1, ...]
 * sorted like np.unique(axis=0). At most `cap` points per neighbouring cell are examined. */
export function closePairs(V, thr, cap = 48) {
  const n = V.length / 3;
  const cx = new Int32Array(n), cy = new Int32Array(n), cz = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    cx[i] = Math.floor(V[3 * i] / thr);
    cy[i] = Math.floor(V[3 * i + 1] / thr);
    cz[i] = Math.floor(V[3 * i + 2] / thr);
  }
  // cell ids through an open-addressing table on the integer cell coordinates
  let size = 1;
  while (size < 2 * n + 16) size <<= 1;
  const mask = size - 1;
  const slot = new Int32Array(size).fill(-1);
  const ck = new Int32Array(3 * n); // coordinates of each cell id
  const cellOf = new Int32Array(n);
  let C = 0;
  const hash = (x, y, z) => (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & mask;
  const find = (x, y, z) => {
    let h = hash(x, y, z);
    for (;;) {
      const id = slot[h];
      if (id < 0) return -1 - h;
      if (ck[3 * id] === x && ck[3 * id + 1] === y && ck[3 * id + 2] === z) return id;
      h = (h + 1) & mask;
    }
  };
  for (let i = 0; i < n; i++) {
    let id = find(cx[i], cy[i], cz[i]);
    if (id < 0) {
      slot[-1 - id] = C;
      ck[3 * C] = cx[i]; ck[3 * C + 1] = cy[i]; ck[3 * C + 2] = cz[i];
      id = C++;
    }
    cellOf[i] = id;
  }
  // points per cell, in index order (the stable argsort of mesh.py)
  const start = new Int32Array(C + 1);
  for (let i = 0; i < n; i++) start[cellOf[i] + 1]++;
  for (let c = 0; c < C; c++) start[c + 1] += start[c];
  const fill = start.slice(0, C), pts = new Int32Array(n);
  for (let i = 0; i < n; i++) pts[fill[cellOf[i]]++] = i;
  // a pair (a < b) is found from exactly one offset (b's cell relative to a's): no duplicates
  const found = [];
  for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
    for (let a = 0; a < n; a++) {
      const id = find(cx[a] + dx, cy[a] + dy, cz[a] + dz);
      if (id < 0) continue;
      const s = start[id], lim = Math.min(start[id + 1] - s, cap);
      for (let j = 0; j < lim; j++) {
        const b = pts[s + j];
        if (a < b) found.push(a * n + b);
      }
    }
  }
  const keys = Float64Array.from(found).sort();
  const out = [];
  for (const k of keys) {
    const a = Math.floor(k / n), b = k - a * n;
    const ddx = V[3 * a] - V[3 * b], ddy = V[3 * a + 1] - V[3 * b + 1], ddz = V[3 * a + 2] - V[3 * b + 2];
    if (Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) < thr) out.push(a, b);
  }
  return out;
}

/** True for vertex pairs that are corners of a common face (topological neighbours). */
function shareAFace(mesh, pairs) {
  const N = mesh.nVerts, p = mesh.facePtr, idx = mesh.faceIdx, hf = mesh.heFace;
  // faces around each vertex (CSR)
  const vptr = new Int32Array(N + 1);
  for (let h = 0; h < idx.length; h++) vptr[idx[h] + 1]++;
  for (let v = 0; v < N; v++) vptr[v + 1] += vptr[v];
  const fill = vptr.slice(0, N), vf = new Int32Array(idx.length);
  for (let h = 0; h < idx.length; h++) vf[fill[idx[h]]++] = hf[h];
  const out = new Uint8Array(pairs.length / 2);
  for (let k = 0; k < out.length; k++) {
    const a = pairs[2 * k], b = pairs[2 * k + 1];
    search: for (let i = vptr[a]; i < vptr[a + 1]; i++) {
      const f = vf[i];
      for (let h = p[f]; h < p[f + 1]; h++) if (idx[h] === b) { out[k] = 1; break search; }
    }
  }
  return out;
}

/** Greedy matching, nearest pairs first: each vertex joins at most one partner. Returns labels (the
 * smaller index of each matched pair). */
function match(n, pairs, dist) {
  const lab = new Int32Array(n);
  for (let i = 0; i < n; i++) lab[i] = i;
  const used = new Uint8Array(n);
  const order = Array.from(dist, (_, i) => i).sort((i, j) => dist[i] - dist[j] || i - j);
  for (const k of order) {
    const a = pairs[2 * k], b = pairs[2 * k + 1];
    if (!used[a] && !used[b]) {
      used[a] = used[b] = 1;
      lab[b] = lab[a] = Math.min(a, b);
    }
  }
  return lab;
}

/** Drop faces until every edge has at most one face per direction (manifold, oriented). */
export function makeManifold(faces, keep) {
  keep = Uint8Array.from(keep);
  for (let it = 0; it < 10; it++) {
    const owner = new Map();
    const bad = new Set();
    for (let fi = 0; fi < faces.length; fi++) {
      if (!keep[fi]) continue;
      const f = faces[fi], k = f.length;
      for (let i = 0; i < k; i++) {
        const key = `${f[i]},${f[(i + 1) % k]}`;
        if (owner.has(key)) bad.add(fi); // this direction is taken: drop the later face
        else owner.set(key, fi);
      }
    }
    if (!bad.size) return keep;
    for (const fi of bad) keep[fi] = 0;
  }
  return keep;
}

/** Weld vertex pairs closer than `distance` (x the pair's smaller localScale, if given). */
export function mergeVertices(mesh, distance, maxValence = 0, localScale = null) {
  const stats = { merged: 0, dropped: 0 };
  let pairs, limit;
  if (localScale === null) {
    pairs = closePairs(mesh.V, distance);
    limit = new Float64Array(pairs.length / 2).fill(distance);
  } else {
    pairs = closePairs(mesh.V, distance * percentile(localScale, 90));
    limit = new Float64Array(pairs.length / 2);
    for (let k = 0; k < limit.length; k++) limit[k] = distance * Math.min(localScale[pairs[2 * k]], localScale[pairs[2 * k + 1]]);
  }
  let d = [];
  if (pairs.length) {
    const share = shareAFace(mesh, pairs), V = mesh.V, kept = [];
    for (let k = 0; k < pairs.length / 2; k++) {
      const a = pairs[2 * k], b = pairs[2 * k + 1];
      const dd = Math.sqrt((V[3 * a] - V[3 * b]) ** 2 + (V[3 * a + 1] - V[3 * b + 1]) ** 2 + (V[3 * a + 2] - V[3 * b + 2]) ** 2);
      if (dd < limit[k] && !share[k]) { kept.push(a, b); d.push(dd); }
    }
    pairs = kept;
  }
  if (pairs.length === 0) return [mesh, stats];
  const N = mesh.nVerts;
  const lab = match(N, pairs, d);
  // np.unique(lab, return_inverse, return_counts)
  const reps = [...new Set(lab)].sort((a, b) => a - b);
  const repIndex = new Int32Array(N).fill(-1);
  reps.forEach((r, i) => { repIndex[r] = i; });
  const inv = new Int32Array(N);
  const sizes = new Int32Array(reps.length);
  for (let v = 0; v < N; v++) { inv[v] = repIndex[lab[v]]; sizes[inv[v]]++; }
  const R = reps.length;
  stats.merged = N - R;

  // merged positions / attributes = cluster means (locks: the longest lock wins)
  const cmean = (x) => {
    const dim = x.length / N, acc = new Float64Array(R * dim);
    for (let v = 0; v < N; v++) for (let c = 0; c < dim; c++) acc[inv[v] * dim + c] += x[v * dim + c];
    for (let r = 0; r < R; r++) for (let c = 0; c < dim; c++) acc[r * dim + c] /= sizes[r];
    return acc;
  };
  const cmax = (x) => {
    const acc = new Float64Array(R).fill(-Infinity);
    for (let v = 0; v < N; v++) if (x[v] > acc[inv[v]]) acc[inv[v]] = x[v];
    return acc;
  };
  const newV = cmean(mesh.V);
  const newVattr = {};
  for (const [k, a] of Object.entries(mesh.vattr)) newVattr[k] = k === "lock" ? cmax(a) : cmean(a);

  // only faces touching a welded vertex can change; the rest are remapped in bulk
  const welded = new Uint8Array(N);
  for (let v = 0; v < N; v++) welded[v] = sizes[inv[v]] > 1 ? 1 : 0;
  const p = mesh.facePtr, idx = mesh.faceIdx, F = mesh.nFaces;
  const faceAff = new Uint8Array(F);
  const aff = [];
  for (let f = 0; f < F; f++) {
    for (let h = p[f]; h < p[f + 1]; h++) if (welded[idx[h]]) { faceAff[f] = 1; break; }
    if (faceAff[f]) aff.push(f);
  }
  const faces = new Map(), keep = new Map();
  for (const fi of aff) {
    const g0 = [];
    for (let h = p[fi]; h < p[fi + 1]; h++) g0.push(inv[idx[h]]);
    const g = g0.filter((x, i) => x !== g0[(i - 1 + g0.length) % g0.length]); // drop consecutive repeats
    faces.set(fi, g);
    keep.set(fi, g.length >= 3 && new Set(g).size === g.length);
  }

  if (maxValence && maxValence > 0) {
    const edges = new Set();
    for (const fi of aff) {
      if (!keep.get(fi)) continue;
      const f = faces.get(fi);
      for (let i = 0; i < f.length; i++) {
        const a = f[i], b = f[(i + 1) % f.length];
        edges.add(Math.min(a, b) * R + Math.max(a, b));
      }
    }
    const val = new Int32Array(R);
    for (const e of edges) {
      const a = Math.floor(e / R), b = e - a * R;
      val[a]++;
      val[b]++;
    }
    const isRep = new Uint8Array(N);
    for (const r of reps) isRep[r] = 1;
    for (const fi of aff) {
      if (!keep.get(fi)) continue;
      // the face reached an over-valence vertex through a non-representative member: not formed
      for (let h = p[fi]; h < p[fi + 1]; h++) {
        const v = idx[h];
        if (val[inv[v]] > maxValence && welded[v] && !isRep[v]) { keep.set(fi, false); break; }
      }
    }
  }

  const affList = aff.map((fi) => faces.get(fi));
  const kmask = makeManifold(affList, aff.map((fi) => (keep.get(fi) ? 1 : 0)));
  let dropped = 0;
  for (const k of kmask) if (!k) dropped++;
  stats.dropped = dropped;

  // assemble: untouched faces then the surviving affected faces
  const order = [];
  const fsizes = [];
  const parts = [];
  for (let f = 0; f < F; f++) {
    if (faceAff[f]) continue;
    order.push(f);
    fsizes.push(p[f + 1] - p[f]);
    for (let h = p[f]; h < p[f + 1]; h++) parts.push(inv[idx[h]]);
  }
  aff.forEach((fi, i) => {
    if (!kmask[i]) return;
    const g = faces.get(fi);
    order.push(fi);
    fsizes.push(g.length);
    for (const x of g) parts.push(x);
  });
  const usedFlag = new Uint8Array(R);
  for (const x of parts) usedFlag[x] = 1;
  const remap = new Int32Array(R).fill(-1);
  const used = [];
  for (let r = 0; r < R; r++) if (usedFlag[r]) { remap[r] = used.length; used.push(r); }
  const U = used.length;
  const outV = new Float64Array(3 * U);
  used.forEach((r, i) => { outV[3 * i] = newV[3 * r]; outV[3 * i + 1] = newV[3 * r + 1]; outV[3 * i + 2] = newV[3 * r + 2]; });
  const facePtr = new Int32Array(fsizes.length + 1);
  for (let i = 0; i < fsizes.length; i++) facePtr[i + 1] = facePtr[i] + fsizes[i];
  const faceIdx = Int32Array.from(parts, (x) => remap[x]);
  const vtype = Int8Array.from(used, (r) => mesh.vtype[reps[r]]);
  const fclass = Int8Array.from(order, (f) => mesh.fclass[f]);
  const vattr = {};
  for (const [k, a] of Object.entries(newVattr)) {
    const dim = a.length / R, out = new Float64Array(U * dim);
    used.forEach((r, i) => { for (let c = 0; c < dim; c++) out[i * dim + c] = a[r * dim + c]; });
    vattr[k] = out;
  }
  const fattr = {};
  for (const [k, a] of Object.entries(mesh.fattr)) fattr[k] = a.constructor.from(order, (f) => a[f]);
  const out = new PolyMesh(outV, facePtr, faceIdx, { vtype, fclass, vattr, fattr, info: { ...mesh.info, merge: stats } });
  return [out, stats];
}

export function maybeMerge(design, mesh, level) {
  if (!active(design, level)) return mesh;
  const m = design.merge;
  const scale = m.relative ? mesh.vertMean(mesh.faceScale) : null;
  const [out, stats] = mergeVertices(mesh, Math.max(Number(m.distance), 1e-12), Math.trunc(m.max_valence), scale);
  if (out.nFaces < 0.5 * mesh.nFaces) { // safety net: never let merging eat the form
    return new PolyMesh(mesh.V, mesh.facePtr, mesh.faceIdx, {
      vtype: mesh.vtype, fclass: mesh.fclass, vattr: mesh.vattr, fattr: mesh.fattr, topo: mesh.topo,
      info: { ...mesh.info, merge: { ...stats, skipped: true } },
    });
  }
  return out;
}

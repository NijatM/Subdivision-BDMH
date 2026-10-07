// Extended Catmull-Clark subdivision (Hansmeyer 2010, eq. 1-4); port of hansmeyer/catmull_clark.py.
//
// Weights may be scalars or per-face arrays (non-uniform weights). Edge weights are the mean of the two
// adjacent faces, vertex weights the mean of incident faces. With all weights at zero this is exactly
// standard Catmull-Clark (incl. the standard boundary rules).
//
// Boundaries: by default the standard (smooth) boundary rules apply. With lockBoundary boundary vertices
// stay fixed and boundary edge points are plain midpoints without extrusion, so identical open panels
// tile seamlessly. vmask (per-vertex, 0..1) scales all extrusion (relief fading out near a locked boundary).
//
// Locking (paper Fig. 9): vertices in vlock keep their position; an edge with both ends locked gets its
// plain midpoint and a face with all corners locked its centroid. Motif attraction (eq. 10-11): with a
// per-vertex value vU, face and edge points are pulled toward (U > 0) or pushed away from (U < 0) their
// vertices, scaled by w6 / w7.
//
// Extension beyond the paper: eq. 1 and 3 are applied to n-gons and any valence; eq. 4 applies to quads
// with the corner/edge/face/edge provenance pattern produced by a previous CC step.

import { FCLASS_CC, VTYPE_CORNER, VTYPE_EDGE, VTYPE_FACE, PolyMesh, Topology } from "./mesh.js";
import { anyNonZero, segSum } from "./util.js";

/** A weight as a per-face array (scalars are broadcast). */
export function perFace(W, name, n) {
  const v = W[name];
  if (v === undefined || v === null) return new Float64Array(n);
  if (typeof v === "number") return new Float64Array(n).fill(v);
  return v;
}

/** Eq. 1 (centroid) and, where the provenance pattern allows, eq. 4: before extrusion. */
export function facePoints(m, W) {
  const F = m.nFaces;
  const Fp = Float64Array.from(m.faceCentroid);
  const w3 = perFace(W, "w3", F), w4 = perFace(W, "w4", F);
  if (!(anyNonZero(w3) || anyNonZero(w4))) return Fp;
  const p = m.facePtr, idx = m.faceIdx, t = m.vtype, V = m.V;
  for (let f = 0; f < F; f++) {
    if (p[f + 1] - p[f] !== 4) continue;
    const s = p[f];
    for (let r = 0; r < 4; r++) {
      const a = idx[s + r], b = idx[s + ((r + 1) % 4)], c = idx[s + ((r + 2) % 4)], d = idx[s + ((r + 3) % 4)];
      if (t[a] === VTYPE_CORNER && t[c] === VTYPE_FACE && t[b] === VTYPE_EDGE && t[d] === VTYPE_EDGE) {
        const w3q = w3[f], w4q = w4[f];
        for (let k = 0; k < 3; k++) {
          const Pv = V[3 * a + k], Pe1 = V[3 * b + k], Pf = V[3 * c + k], Pe2 = V[3 * d + k];
          Fp[3 * f + k] = ((Pv * (1 + w3q) + Pf * (1 - w3q)) * (1 + w4q) + (Pe1 + Pe2) * (1 - w4q)) / 4.0;
        }
      }
    }
  }
  return Fp;
}

export function subdivide(m, W, opts = {}) {
  const relative = opts.relative !== false;
  const lockBoundary = !!opts.lockBoundary;
  let vlock = opts.vlock || null;
  const vmask = opts.vmask || null;
  const vU = opts.vU || null;
  const N = m.nVerts, E = m.nEdges, F = m.nFaces;
  const V = m.V;
  const ev = m.edgeVerts;
  const p = m.facePtr, idx = m.faceIdx;
  if (vlock && !anyNonZero(vlock)) vlock = null;
  let lockE = null, lockF = null;
  if (vlock) {
    lockE = new Uint8Array(E);
    for (let e = 0; e < E; e++) lockE[e] = vlock[ev[2 * e]] && vlock[ev[2 * e + 1]] ? 1 : 0;
    lockF = new Uint8Array(F);
    for (let f = 0; f < F; f++) {
      let all = 1;
      for (let h = p[f]; h < p[f + 1]; h++) if (!vlock[idx[h]]) { all = 0; break; }
      lockF[f] = all;
    }
  }
  let maskF = null, maskE = null;
  if (vmask) {
    maskF = m.faceMeanOfVerts(vmask);
    maskE = new Float64Array(E);
    for (let e = 0; e < E; e++) maskE[e] = 0.5 * (vmask[ev[2 * e]] + vmask[ev[2 * e + 1]]);
  }

  const wF = perFace(W, "w_f", F);
  const wEf = perFace(W, "w_e", F);
  const wCf = perFace(W, "w_c", F);
  const w1f = perFace(W, "w1", F);
  const w2f = perFace(W, "w2", F);
  const scaleF = relative ? m.faceScale : new Float64Array(F).fill(1);

  // ---- eq. 1 / 4: face points
  let Fp = facePoints(m, W);
  if (anyNonZero(wF)) {
    const n = m.faceNormal;
    for (let f = 0; f < F; f++) {
      const amount = maskF ? wF[f] * scaleF[f] * maskF[f] : wF[f] * scaleF[f];
      Fp[3 * f] += n[3 * f] * amount;
      Fp[3 * f + 1] += n[3 * f + 1] * amount;
      Fp[3 * f + 2] += n[3 * f + 2] * amount;
    }
  }
  if (vlock) {
    const C = m.faceCentroid;
    for (let f = 0; f < F; f++) if (lockF[f]) { Fp[3 * f] = C[3 * f]; Fp[3 * f + 1] = C[3 * f + 1]; Fp[3 * f + 2] = C[3 * f + 2]; }
  }

  // ---- eq. 2: edge points
  const ef = m.edgeFaces;
  const w1 = m.edgeMean(w1f);
  let Ep = new Float64Array(3 * E);
  for (let e = 0; e < E; e++) {
    const a = 3 * ev[2 * e], b = 3 * ev[2 * e + 1], f0 = ef[2 * e], f1 = ef[2 * e + 1];
    if (f1 >= 0) {
      const w = w1[e];
      for (let k = 0; k < 3; k++) {
        const Fsum = Fp[3 * f0 + k] + Fp[3 * f1 + k];
        Ep[3 * e + k] = (Fsum * (1 + w) + (V[a + k] + V[b + k]) * (1 - w)) / 4.0;
      }
    } else {
      for (let k = 0; k < 3; k++) Ep[3 * e + k] = 0.5 * (V[a + k] + V[b + k]);
    }
  }
  const wE = m.edgeMean(wEf);
  if (anyNonZero(wE)) {
    const sE = m.edgeMean(scaleF), n = m.edgeNormal;
    for (let e = 0; e < E; e++) {
      let amount = wE[e] * sE[e];
      if (maskE) amount = amount * maskE[e];
      if (lockBoundary && ef[2 * e + 1] < 0) amount = 0.0;
      Ep[3 * e] += n[3 * e] * amount;
      Ep[3 * e + 1] += n[3 * e + 1] * amount;
      Ep[3 * e + 2] += n[3 * e + 2] * amount;
    }
  }
  if (vlock) {
    for (let e = 0; e < E; e++) {
      if (!lockE[e]) continue;
      const a = 3 * ev[2 * e], b = 3 * ev[2 * e + 1];
      for (let k = 0; k < 3; k++) Ep[3 * e + k] = 0.5 * (V[a + k] + V[b + k]);
    }
  }

  // ---- eq. 3: corner points
  const val = m.valence;
  const Fbar = m.vertMean(Fp, 3);
  const Esum = new Float64Array(3 * N);
  for (let e = 0; e < E; e++) {
    const a = ev[2 * e], b = ev[2 * e + 1];
    for (let k = 0; k < 3; k++) {
      const mid = 0.5 * (V[3 * a + k] + V[3 * b + k]);
      Esum[3 * a + k] += mid;
    }
  }
  for (let e = 0; e < E; e++) {
    const a = ev[2 * e], b = ev[2 * e + 1];
    for (let k = 0; k < 3; k++) {
      const mid = 0.5 * (V[3 * a + k] + V[3 * b + k]);
      Esum[3 * b + k] += mid;
    }
  }
  const w2 = m.vertMean(w2f);
  const Cp = new Float64Array(3 * N);
  for (let v = 0; v < N; v++) {
    const i = val[v], w = w2[v], dv = i > 1 ? i : 1, ebd = i > 1 ? i : 1;
    for (let k = 0; k < 3; k++) {
      const Ebar = Esum[3 * v + k] / ebd;
      Cp[3 * v + k] = (Fbar[3 * v + k] * (1 + w) + 2 * Ebar * (1 - w / 2) + (i - 3) * V[3 * v + k]) / dv;
    }
  }

  const bnd = m.vertIsBoundary;
  let anyBnd = false;
  for (let v = 0; v < N; v++) if (bnd[v]) { anyBnd = true; break; }
  if (anyBnd) {
    // standard boundary rule: cubic B-spline along the boundary, (P- + 6P + P+) / 8
    const isB = m.edgeIsBoundary;
    const nbSum = new Float64Array(3 * N), nbCnt = new Int32Array(N);
    // scatter order of mesh.py: all first endpoints (getting the second's position), then all second ones
    for (let e = 0; e < E; e++) {
      if (!isB[e]) continue;
      const a = ev[2 * e], b = ev[2 * e + 1];
      for (let k = 0; k < 3; k++) nbSum[3 * a + k] += V[3 * b + k];
    }
    for (let e = 0; e < E; e++) {
      if (!isB[e]) continue;
      const a = ev[2 * e], b = ev[2 * e + 1];
      for (let k = 0; k < 3; k++) nbSum[3 * b + k] += V[3 * a + k];
      nbCnt[a]++;
      nbCnt[b]++;
    }
    for (let v = 0; v < N; v++) {
      if (!bnd[v]) continue;
      const regular = nbCnt[v] === 2 && !lockBoundary;
      for (let k = 0; k < 3; k++) {
        Cp[3 * v + k] = regular ? (nbSum[3 * v + k] + 6 * V[3 * v + k]) / 8.0 : V[3 * v + k];
      }
    }
  }
  const wC = m.vertMean(wCf);
  if (anyNonZero(wC)) {
    const sV = m.vertMean(scaleF), n = m.vertNormal;
    for (let v = 0; v < N; v++) {
      let amount = wC[v] * sV[v];
      if (vmask) amount = amount * vmask[v];
      if (lockBoundary && bnd[v]) amount = 0.0;
      Cp[3 * v] += n[3 * v] * amount;
      Cp[3 * v + 1] += n[3 * v + 1] * amount;
      Cp[3 * v + 2] += n[3 * v + 2] * amount;
    }
  }
  if (vlock) {
    for (let v = 0; v < N; v++) if (vlock[v]) { Cp[3 * v] = V[3 * v]; Cp[3 * v + 1] = V[3 * v + 1]; Cp[3 * v + 2] = V[3 * v + 2]; }
  }

  // ---- eq. 10 / 11: motif attraction, applied after eq. 1-2
  if (vU) {
    const w6 = perFace(W, "w6", F);
    const w7 = m.edgeMean(perFace(W, "w7", F));
    if (anyNonZero(w6)) {
      const H = idx.length, hf = m.heFace, ph = new Float64Array(3 * H);
      for (let h = 0; h < H; h++) {
        const v = idx[h], f = hf[h], u = vU[v];
        ph[3 * h] = (V[3 * v] - Fp[3 * f]) * u;
        ph[3 * h + 1] = (V[3 * v + 1] - Fp[3 * f + 1]) * u;
        ph[3 * h + 2] = (V[3 * v + 2] - Fp[3 * f + 2]) * u;
      }
      const pull = m.faceReduce(ph, 3);
      const next = Float64Array.from(Fp);
      for (let f = 0; f < F; f++) {
        if (lockF && lockF[f]) continue;
        next[3 * f] += w6[f] * pull[3 * f];
        next[3 * f + 1] += w6[f] * pull[3 * f + 1];
        next[3 * f + 2] += w6[f] * pull[3 * f + 2];
      }
      Fp = next;
    }
    if (anyNonZero(w7)) {
      const next = Float64Array.from(Ep);
      for (let e = 0; e < E; e++) {
        if (lockE && lockE[e]) continue;
        const a = ev[2 * e], b = ev[2 * e + 1], ua = vU[a], ub = vU[b];
        for (let k = 0; k < 3; k++) {
          next[3 * e + k] += w7[e] * ((V[3 * a + k] - Ep[3 * e + k]) * ua + (V[3 * b + k] - Ep[3 * e + k]) * ub);
        }
      }
      Ep = next;
    }
  }

  // ---- topology: one quad per half-edge (the same for every run from the same input: built once)
  const topo = m.topo.child("cc", () => childTopology(m));
  const { vtype, fclass } = topo.extra;
  const newV = new Float64Array(3 * (N + E + F));
  newV.set(Cp, 0);
  newV.set(Ep, 3 * N);
  newV.set(Fp, 3 * (N + E));
  const vattr = {};
  for (const [k, a] of Object.entries(m.vattr)) vattr[k] = propagateAttr(m, a, k);
  const fattr = {};
  const hf = m.heFace;
  for (const [k, a] of Object.entries(m.fattr)) {
    const out = new a.constructor(hf.length);
    for (let h = 0; h < hf.length; h++) out[h] = a[hf[h]];
    fattr[k] = out;
  }
  return new PolyMesh(newV, topo.facePtr, topo.faceIdx, { vtype, fclass, vattr, fattr, topo });
}

/** Connectivity of the subdivided mesh (corners, then edge points, then face points). */
export function childTopology(m) {
  const N = m.nVerts, E = m.nEdges, F = m.nFaces, H = m.nHalfedges;
  const from = m.faceIdx, he = m.heEdge, hf = m.heFace, prev = m.hePrev;
  const quads = new Int32Array(4 * H);
  for (let h = 0; h < H; h++) {
    quads[4 * h] = from[h];
    quads[4 * h + 1] = N + he[h];
    quads[4 * h + 2] = N + E + hf[h];
    quads[4 * h + 3] = N + he[prev[h]];
  }
  const ptr = new Int32Array(H + 1);
  for (let i = 0; i <= H; i++) ptr[i] = 4 * i;
  const vtype = new Int8Array(N + E + F);
  vtype.fill(VTYPE_CORNER, 0, N);
  vtype.fill(VTYPE_EDGE, N, N + E);
  vtype.fill(VTYPE_FACE, N + E);
  return new Topology(ptr, quads, N + E + F, { vtype, fclass: new Int8Array(H).fill(FCLASS_CC) });
}

/** Carry a per-vertex attribute to the next level by plain averaging: corners keep their value, edge
 * points take the endpoint mean, face points the face mean. Special attributes: "tv" (original vertex)
 * only survives on corners; "te" (on an original edge) also on edge points of such edges; "lock"
 * (iterations to stay locked) passes to edge/face points only if all their parents are locked. */
export function propagateAttr(m, a, name = "") {
  const N = m.nVerts, E = m.nEdges, F = m.nFaces, ev = m.edgeVerts, p = m.facePtr, idx = m.faceIdx;
  const dim = a.length / N;
  const out = new Float64Array((N + E + F) * dim);
  out.set(a, 0);
  if (name === "tv") return out;
  if (name === "te") {
    for (let e = 0; e < E; e++) out[N + e] = a[ev[2 * e]] * a[ev[2 * e + 1]];
    return out;
  }
  if (name === "lock") {
    for (let e = 0; e < E; e++) {
      const x = a[ev[2 * e]], y = a[ev[2 * e + 1]];
      out[N + e] = x > 0 && y > 0 ? Math.min(x, y) : 0.0;
    }
    for (let f = 0; f < F; f++) {
      let mn = Infinity;
      for (let h = p[f]; h < p[f + 1]; h++) if (a[idx[h]] < mn) mn = a[idx[h]];
      out[N + E + f] = mn;
    }
    return out;
  }
  for (let e = 0; e < E; e++) {
    const x = ev[2 * e] * dim, y = ev[2 * e + 1] * dim;
    for (let c = 0; c < dim; c++) out[(N + e) * dim + c] = 0.5 * (a[x + c] + a[y + c]);
  }
  for (let f = 0; f < F; f++) {
    const k = p[f + 1] - p[f];
    for (let c = 0; c < dim; c++) out[(N + E + f) * dim + c] = segSum(a, p[f], p[f + 1], idx, dim, c) / k;
  }
  return out;
}

export function predictedFaces(m) {
  return m.nHalfedges;
}

// Extended Doo-Sabin subdivision (Hansmeyer 2010, eq. 5-6); port of hansmeyer/doo_sabin.py.
//
// Each face's corner i gets a new point  sum_d alpha_d(w1) * P_(i+d)  + n_f * w_f. The standard masks are
//   alpha_0 = (k+5)/(4k),   alpha_d = (3 + 2 cos(2 pi d / k)) / (4k).
// w1 moves  w1 * (k-2)/k  of weight from the two adjacent vertices onto the corner's own vertex. For k = 4
// and k = 3 this reproduces the paper's eq. 5 and 6 exactly; other k is our generalisation.
// Faces are classed as face-, edge- or vertex-derived after a DS step, and each class can carry its own
// (w1, w_f). Open meshes: Doo-Sabin has no boundary rule; boundary edges and vertices get no E-/V-faces,
// so the boundary recedes.

import { FCLASS_DS_EDGE, FCLASS_DS_FACE, FCLASS_DS_VERT, VTYPE_DS, PolyMesh, Topology } from "./mesh.js";
import { perFace } from "./catmullClark.js";
import { anyNonZero } from "./util.js";

/** Mask weights for a face of size k: alpha[d] multiplies the vertex d steps ahead. */
export function dsAlphas(k, w1) {
  const a = new Float64Array(k);
  for (let d = 0; d < k; d++) a[d] = (3 + 2 * Math.cos((2 * Math.PI * d) / k)) / (4 * k);
  a[0] = (k + 5) / (4 * k);
  const shift = (w1 * (k - 2)) / k;
  a[0] += shift;
  a[1] -= shift / 2;
  a[k - 1] -= shift / 2;
  return a;
}

function classWeights(m, W) {
  const F = m.nFaces, fc = m.fclass;
  const g = (n) => perFace(W, n, F);
  const w1f = g("ds_w1_face"), w1e = g("ds_w1_edge"), w1v = g("ds_w1_vert");
  const wff = g("ds_wf_face"), wfe = g("ds_wf_edge"), wfv = g("ds_wf_vert");
  const w1 = new Float64Array(F), wf = new Float64Array(F);
  for (let f = 0; f < F; f++) {
    const c = fc[f];
    if (c === FCLASS_DS_EDGE) { w1[f] = w1e[f]; wf[f] = wfe[f]; }
    else if (c === FCLASS_DS_VERT) { w1[f] = w1v[f]; wf[f] = wfv[f]; }
    else { w1[f] = w1f[f]; wf[f] = wff[f]; }
  }
  return [w1, wf];
}

/** Closed fans of outgoing half-edges around interior vertices, grouped by size (ascending), each as
 * { k, fans: Int32Array(n*k) }. rot(h) = twin(prev(h)) steps counter-clockwise around the vertex, so each
 * fan lists the corners in the order of a correctly oriented polygon. */
export function vertexFans(m) {
  const twin = m.heTwin, prev = m.hePrev, from = m.faceIdx, N = m.nVerts;
  const bnd = m.vertIsBoundary, cnt = m.vertFaceCount;
  const start = new Int32Array(N).fill(-1);
  for (let h = from.length - 1; h >= 0; h--) start[from[h]] = h; // the first outgoing half-edge per vertex
  const byK = new Map();
  for (let v = 0; v < N; v++) {
    if (bnd[v] || start[v] < 0) continue;
    const k = cnt[v];
    if (k < 3) continue;
    let list = byK.get(k);
    if (!list) byK.set(k, (list = []));
    list.push(v);
  }
  const out = [];
  for (const k of [...byK.keys()].sort((x, y) => x - y)) {
    const vs = byK.get(k);
    const buf = new Int32Array(vs.length * k);
    let n = 0;
    const col = new Int32Array(k);
    for (const v of vs) {
      col[0] = start[v];
      let ok = true;
      for (let j = 1; j < k; j++) {
        const h = col[j - 1];
        col[j] = h >= 0 ? twin[prev[h]] : -1;
      }
      for (let j = 0; j < k; j++) if (col[j] < 0) { ok = false; break; }
      if (ok && twin[prev[col[k - 1]]] !== col[0]) ok = false;
      if (!ok) continue;
      buf.set(col, n * k);
      n++;
    }
    if (n) out.push({ k, fans: buf.slice(0, n * k) });
  }
  return out;
}

/** Apply the (extended) Doo-Sabin masks to a per-vertex array X (dim values each) -> one value per half-edge. */
export function cornerValues(m, X, w1, dim = 3) {
  const H = m.nHalfedges, p = m.facePtr, idx = m.faceIdx, F = m.nFaces;
  const out = new Float64Array(H * dim);
  const cache = new Map();
  for (let f = 0; f < F; f++) {
    const s = p[f], k = p[f + 1] - s;
    const w = w1[f];
    let a;
    if (w === 0) {
      a = cache.get(k);
      if (!a) cache.set(k, (a = dsAlphas(k, 0)));
    } else {
      a = dsAlphas(k, w);
    }
    for (let i = 0; i < k; i++) {
      for (let c = 0; c < dim; c++) {
        let acc = 0;
        for (let d = 0; d < k; d++) acc += a[d] * X[idx[s + ((i + d) % k)] * dim + c];
        out[(s + i) * dim + c] = acc;
      }
    }
  }
  return out;
}

/** Locks and motif values are Catmull-Clark features; Doo-Sabin ignores them but passes each vertex's
 * lock on to the corners cut from it. */
export function subdivide(m, W, opts = {}) {
  const relative = opts.relative !== false;
  const vmask = opts.vmask || null;
  const H = m.nHalfedges, F = m.nFaces;
  const [w1, wf] = classWeights(m, W);
  const scaleF = relative ? m.faceScale : new Float64Array(F).fill(1);

  // ---- new corner points: one per half-edge (face corner)
  const newV = cornerValues(m, m.V, w1, 3);
  if (anyNonZero(wf)) {
    const n = m.faceNormal, hf = m.heFace;
    const maskF = vmask ? m.faceMeanOfVerts(vmask) : null;
    const amount = new Float64Array(F);
    for (let f = 0; f < F; f++) amount[f] = maskF ? wf[f] * scaleF[f] * maskF[f] : wf[f] * scaleF[f];
    for (let h = 0; h < H; h++) {
      const f = hf[h];
      newV[3 * h] += n[3 * f] * amount[f];
      newV[3 * h + 1] += n[3 * f + 1] * amount[f];
      newV[3 * h + 2] += n[3 * f + 2] * amount[f];
    }
  }
  const zero = new Float64Array(F);
  const vattr = {};
  for (const [k, a] of Object.entries(m.vattr)) {
    if (k === "lock") {
      const out = new Float64Array(H), from = m.faceIdx;
      for (let h = 0; h < H; h++) out[h] = a[from[h]];
      vattr[k] = out;
    } else if (k === "tv" || k === "te") {
      vattr[k] = new Float64Array(H);
    } else {
      vattr[k] = cornerValues(m, a, zero, a.length / m.nVerts);
    }
  }

  // ---- faces: F-faces, E-faces (interior edges), V-faces (interior vertices), built once per input
  const topo = m.topo.child("ds", () => childTopology(m));
  const { fclass, eh, et, fans } = topo.extra;
  const hf = m.heFace;
  const fattr = {};
  for (const [k, a] of Object.entries(m.fattr)) {
    const isInt = !(a instanceof Float64Array || a instanceof Float32Array);
    const comb = isInt ? (x, y) => x | y : (x, y) => (x > y ? x : y);
    const out = new a.constructor(topo.facePtr.length - 1);
    out.set(a, 0);
    let o = a.length;
    for (let i = 0; i < eh.length; i++) out[o++] = comb(a[hf[eh[i]]], a[hf[et[i]]]);
    for (const { k: kk, fans: fb } of fans) {
      for (let i = 0; i < fb.length; i += kk) {
        let acc = a[hf[fb[i]]];
        for (let j = 1; j < kk; j++) acc = comb(acc, a[hf[fb[i + j]]]);
        out[o++] = acc;
      }
    }
    fattr[k] = out;
  }
  return new PolyMesh(newV, topo.facePtr, topo.faceIdx, {
    vtype: new Int8Array(H).fill(VTYPE_DS), fclass, vattr, fattr, topo,
  });
}

/** Connectivity of the subdivided mesh; extra = (fclass, interior edge half-edges, their twins, vertex fans). */
export function childTopology(m) {
  const H = m.nHalfedges, F = m.nFaces, nxt = m.heNext, twin = m.heTwin, p = m.facePtr;
  const all = m.edgeHe;
  let nInner = 0;
  for (let e = 0; e < all.length; e++) if (twin[all[e]] >= 0) nInner++;
  const eh = new Int32Array(nInner), et = new Int32Array(nInner);
  for (let e = 0, i = 0; e < all.length; e++) {
    const h = all[e];
    if (twin[h] >= 0) { eh[i] = h; et[i] = twin[h]; i++; }
  }
  const fans = vertexFans(m);
  let nV = 0, fanH = 0;
  for (const { k, fans: fb } of fans) { nV += fb.length / k; fanH += fb.length; }
  const nF = F + nInner + nV;
  const facePtr = new Int32Array(nF + 1);
  const faceIdx = new Int32Array(H + 4 * nInner + fanH);
  for (let h = 0; h < H; h++) faceIdx[h] = h;
  facePtr.set(p, 0);
  let o = H, f = F;
  for (let i = 0; i < nInner; i++) {
    faceIdx[o++] = nxt[et[i]];
    faceIdx[o++] = et[i];
    faceIdx[o++] = nxt[eh[i]];
    faceIdx[o++] = eh[i];
    facePtr[++f] = o;
  }
  for (const { k, fans: fb } of fans) {
    for (let i = 0; i < fb.length; i += k) {
      for (let j = 0; j < k; j++) faceIdx[o++] = fb[i + j];
      facePtr[++f] = o;
    }
  }
  const fclass = new Int8Array(nF);
  fclass.fill(FCLASS_DS_FACE, 0, F);
  fclass.fill(FCLASS_DS_EDGE, F, F + nInner);
  fclass.fill(FCLASS_DS_VERT, F + nInner);
  return new Topology(facePtr, faceIdx, H, { fclass, eh, et, fans });
}

export function predictedFaces(m) {
  let ie = 0, iv = 0;
  for (const b of m.edgeIsBoundary) if (!b) ie++;
  for (const b of m.vertIsBoundary) if (!b) iv++;
  return m.nFaces + ie + iv;
}

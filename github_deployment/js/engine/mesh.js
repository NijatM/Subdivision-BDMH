// General polygon mesh with vectorised half-edge connectivity (port of hansmeyer/mesh.py).
//
// Faces are stored in CSR form: facePtr (F+1) indexes into faceIdx (H), so face f has vertices
// faceIdx[facePtr[f] .. facePtr[f+1]) in counter-clockwise order (normals point outward by the
// right-hand rule). Every slot of faceIdx is one half-edge going from that vertex to the next
// vertex of the same face. Positions are packed xyz in a Float64Array.

import { EPS, hashArray, segSum } from "./util.js";

// Vertex provenance: how a vertex was created in the most recent iteration.
export const VTYPE_BASE = -1;
export const VTYPE_CORNER = 0;
export const VTYPE_EDGE = 1;
export const VTYPE_FACE = 2;
export const VTYPE_DS = 3;

// Face provenance: what a face was created from in the most recent iteration.
export const FCLASS_BASE = -1;
export const FCLASS_CC = 0;
export const FCLASS_DS_FACE = 1;
export const FCLASS_DS_EDGE = 2;
export const FCLASS_DS_VERT = 3;

const WeakRefImpl = typeof WeakRef !== "undefined" ? WeakRef : class { constructor(v) { this.v = v; } deref() { return this.v; } };

/** The connectivity of a mesh (its faces and vertex count) with the tables derived from it. Meshes with
 * the same faces share one Topology, so the half-edge and edge tables are built once. A subdivision step
 * keeps the next level's connectivity in `children` for as long as some mesh uses it. */
export class Topology {
  constructor(facePtr, faceIdx, nVerts, extra = null) {
    this.facePtr = facePtr instanceof Int32Array ? facePtr : Int32Array.from(facePtr);
    this.faceIdx = faceIdx instanceof Int32Array ? faceIdx : Int32Array.from(faceIdx);
    this.nVerts = nVerts | 0;
    this.cache = {};
    this.children = new Map();
    this.extra = extra;
  }

  child(key, build) {
    const ref = this.children.get(key);
    let topo = ref ? ref.deref() : undefined;
    if (topo === undefined) {
      topo = build();
      this.children.set(key, new WeakRefImpl(topo));
    }
    return topo;
  }

  nbytes() {
    let n = this.facePtr.byteLength + this.faceIdx.byteLength;
    for (const v of Object.values(this.cache)) {
      if (ArrayBuffer.isView(v)) n += v.byteLength;
      else if (v && typeof v === "object") for (const x of Object.values(v)) if (ArrayBuffer.isView(x)) n += x.byteLength;
    }
    return n;
  }
}

const SHARED = new Map();

/** The Topology for these faces, reused while the same connectivity comes back (input meshes: a size
 * slider rebuilds the base mesh with new positions but the same faces). For small meshes only. */
export function sharedTopology(facePtr, faceIdx, nVerts, keep = 4) {
  const key = `${nVerts}|${facePtr.length}|${faceIdx.length}|${hashArray(facePtr)}|${hashArray(faceIdx)}`;
  let topo = SHARED.get(key);
  if (topo === undefined) {
    topo = new Topology(Int32Array.from(facePtr), Int32Array.from(faceIdx), nVerts);
    SHARED.set(key, topo);
    while (SHARED.size > keep) SHARED.delete(SHARED.keys().next().value);
  } else {
    SHARED.delete(key);
    SHARED.set(key, topo);
  }
  return topo;
}

function shared(mesh, name, fn) {
  const cache = mesh.topo.cache;
  let v = cache[name];
  if (v === undefined) v = cache[name] = fn();
  return v;
}

export class PolyMesh {
  /**
   * @param {Float64Array} V packed xyz
   * @param {Int32Array} facePtr
   * @param {Int32Array} faceIdx
   */
  constructor(V, facePtr, faceIdx, opts = {}) {
    this.V = V instanceof Float64Array ? V : Float64Array.from(V);
    const nV = this.V.length / 3;
    if (opts.topo) {
      if (opts.topo.nVerts !== nV || opts.topo.facePtr.length !== facePtr.length) {
        throw new Error("topology does not match the vertices / faces");
      }
      this.topo = opts.topo;
    } else {
      this.topo = new Topology(facePtr, faceIdx, nV);
    }
    this.facePtr = this.topo.facePtr;
    this.faceIdx = this.topo.faceIdx;
    this.vtype = opts.vtype || new Int8Array(nV).fill(VTYPE_BASE);
    this.fclass = opts.fclass || new Int8Array(this.nFaces).fill(FCLASS_BASE);
    this.vattr = opts.vattr ? { ...opts.vattr } : {};
    this.fattr = opts.fattr ? { ...opts.fattr } : {};
    this.info = opts.info ? { ...opts.info } : {};
    this._c = {};
  }

  // ---------------------------------------------------------------- build
  /** Build from positions (packed or [[x,y,z]]) and a list of faces (lists of vertex indices). */
  static fromFaces(V, faces) {
    let flatV = V;
    if (Array.isArray(V) && V.length && Array.isArray(V[0])) {
      flatV = new Float64Array(V.length * 3);
      V.forEach((p, i) => { flatV[3 * i] = p[0]; flatV[3 * i + 1] = p[1]; flatV[3 * i + 2] = p[2]; });
    }
    let H = 0;
    for (const f of faces) H += f.length;
    const ptr = new Int32Array(faces.length + 1);
    const idx = new Int32Array(H);
    let o = 0;
    faces.forEach((f, i) => {
      for (const v of f) idx[o++] = v;
      ptr[i + 1] = o;
    });
    return new PolyMesh(Float64Array.from(flatV), ptr, idx);
  }

  /** Same topology and attributes, new positions. */
  withPositions(V) {
    return new PolyMesh(V, this.facePtr, this.faceIdx, {
      vtype: this.vtype, fclass: this.fclass, vattr: this.vattr, fattr: this.fattr, info: { ...this.info }, topo: this.topo,
    });
  }

  shareTopology() {
    this.topo = sharedTopology(this.facePtr, this.faceIdx, this.nVerts);
    this.facePtr = this.topo.facePtr;
    this.faceIdx = this.topo.faceIdx;
    return this;
  }

  nbytes() {
    let n = this.V.byteLength + this.vtype.byteLength + this.fclass.byteLength;
    for (const a of Object.values(this.vattr)) n += a.byteLength;
    for (const a of Object.values(this.fattr)) n += a.byteLength;
    for (const a of Object.values(this._c)) if (ArrayBuffer.isView(a)) n += a.byteLength;
    return n;
  }

  // ------------------------------------------------------------ basic info
  get nVerts() { return this.V.length / 3; }
  get nFaces() { return this.facePtr.length - 1; }
  get nHalfedges() { return this.faceIdx.length; }

  get faceSize() {
    return shared(this, "faceSize", () => {
      const F = this.nFaces, out = new Int32Array(F), p = this.facePtr;
      for (let f = 0; f < F; f++) out[f] = p[f + 1] - p[f];
      return out;
    });
  }

  faceVerts(f) {
    return Array.from(this.faceIdx.subarray(this.facePtr[f], this.facePtr[f + 1]));
  }

  facesList() {
    const out = [];
    for (let f = 0; f < this.nFaces; f++) out.push(this.faceVerts(f));
    return out;
  }

  isAllQuads() {
    return this.nFaces > 0 && this.nHalfedges === 4 * this.nFaces && this.faceSize.every((k) => k === 4);
  }

  // ------------------------------------------------------------ half-edges
  get heFace() {
    return shared(this, "heFace", () => {
      const out = new Int32Array(this.nHalfedges), p = this.facePtr;
      for (let f = 0; f < this.nFaces; f++) for (let h = p[f]; h < p[f + 1]; h++) out[h] = f;
      return out;
    });
  }

  get heNext() {
    return shared(this, "heNext", () => {
      const out = new Int32Array(this.nHalfedges), p = this.facePtr;
      for (let f = 0; f < this.nFaces; f++) {
        const a = p[f], b = p[f + 1];
        for (let h = a; h < b - 1; h++) out[h] = h + 1;
        if (b > a) out[b - 1] = a;
      }
      return out;
    });
  }

  get hePrev() {
    return shared(this, "hePrev", () => {
      const nxt = this.heNext, out = new Int32Array(this.nHalfedges);
      for (let h = 0; h < nxt.length; h++) out[nxt[h]] = h;
      return out;
    });
  }

  get heFrom() { return this.faceIdx; }

  get heTo() {
    return shared(this, "heTo", () => {
      const nxt = this.heNext, idx = this.faceIdx, out = new Int32Array(this.nHalfedges);
      for (let h = 0; h < out.length; h++) out[h] = idx[nxt[h]];
      return out;
    });
  }

  /** Edges numbered in order of their (lower, higher) vertex pair, as mesh.py's one-sort edge table. */
  get _edgeData() {
    return shared(this, "_edgeData", () => {
      const H = this.nHalfedges, N = this.nVerts;
      const a = this.faceIdx, b = this.heTo;
      const start = new Int32Array(N + 1);
      for (let h = 0; h < H; h++) start[(a[h] < b[h] ? a[h] : b[h]) + 1]++;
      for (let v = 0; v < N; v++) start[v + 1] += start[v];
      const fill = start.slice(0, N);
      const order = new Int32Array(H);
      const hiOf = new Int32Array(H);
      for (let h = 0; h < H; h++) {
        const lo = a[h] < b[h] ? a[h] : b[h];
        const hi = a[h] < b[h] ? b[h] : a[h];
        const pos = fill[lo]++;
        order[pos] = h;
        hiOf[pos] = hi;
      }
      // sort each bucket by the higher vertex (buckets are small: insertion sort)
      for (let v = 0; v < N; v++) {
        const s = start[v], e = start[v + 1];
        for (let i = s + 1; i < e; i++) {
          const kh = hiOf[i], kv = order[i];
          let j = i - 1;
          while (j >= s && (hiOf[j] > kh || (hiOf[j] === kh && order[j] > kv))) {
            hiOf[j + 1] = hiOf[j];
            order[j + 1] = order[j];
            j--;
          }
          hiOf[j + 1] = kh;
          order[j + 1] = kv;
        }
      }
      const heEdge = new Int32Array(H);
      const twin = new Int32Array(H).fill(-1);
      const edgeHeTmp = new Int32Array(H);
      let E = 0;
      let v = 0;
      for (let i = 0; i < H; ) {
        while (start[v + 1] <= i) v++;
        const end = start[v + 1];
        let j = i + 1;
        while (j < end && hiOf[j] === hiOf[i]) j++;
        const count = j - i;
        if (count > 2) throw new Error("non-manifold mesh: an edge is shared by more than two faces");
        const h0 = order[i];
        heEdge[h0] = E;
        if (count === 2) {
          const h1 = order[i + 1];
          heEdge[h1] = E;
          if (a[h0] === a[h1]) throw new Error("inconsistently oriented mesh: neighbouring faces disagree on winding");
          twin[h0] = h1;
          twin[h1] = h0;
          edgeHeTmp[E] = h0 < h1 ? h0 : h1;
        } else {
          edgeHeTmp[E] = h0;
        }
        E++;
        i = j;
      }
      return { heEdge, twin, edgeHe: edgeHeTmp.slice(0, E), nEdges: E };
    });
  }

  get heEdge() { return this._edgeData.heEdge; }
  get heTwin() { return this._edgeData.twin; }
  /** A representative half-edge for each undirected edge. */
  get edgeHe() { return this._edgeData.edgeHe; }
  get nEdges() { return this._edgeData.nEdges; }

  /** (E*2) packed [from, to] of each edge's representative half-edge. */
  get edgeVerts() {
    return shared(this, "edgeVerts", () => {
      const eh = this.edgeHe, from = this.faceIdx, to = this.heTo, out = new Int32Array(2 * eh.length);
      for (let e = 0; e < eh.length; e++) {
        out[2 * e] = from[eh[e]];
        out[2 * e + 1] = to[eh[e]];
      }
      return out;
    });
  }

  /** (E*2) adjacent faces; the second is -1 on boundary edges. */
  get edgeFaces() {
    return shared(this, "edgeFaces", () => {
      const eh = this.edgeHe, tw = this.heTwin, hf = this.heFace, out = new Int32Array(2 * eh.length);
      for (let e = 0; e < eh.length; e++) {
        out[2 * e] = hf[eh[e]];
        const t = tw[eh[e]];
        out[2 * e + 1] = t >= 0 ? hf[t] : -1;
      }
      return out;
    });
  }

  get edgeIsBoundary() {
    return shared(this, "edgeIsBoundary", () => {
      const eh = this.edgeHe, tw = this.heTwin, out = new Uint8Array(eh.length);
      for (let e = 0; e < eh.length; e++) out[e] = tw[eh[e]] < 0 ? 1 : 0;
      return out;
    });
  }

  get vertIsBoundary() {
    return shared(this, "vertIsBoundary", () => {
      const out = new Uint8Array(this.nVerts), ev = this.edgeVerts, bnd = this.edgeIsBoundary;
      for (let e = 0; e < bnd.length; e++) if (bnd[e]) { out[ev[2 * e]] = 1; out[ev[2 * e + 1]] = 1; }
      return out;
    });
  }

  hasBoundary() {
    return this.edgeIsBoundary.some((x) => x);
  }

  /** Number of incident edges per vertex. */
  get valence() {
    return shared(this, "valence", () => {
      const out = new Int32Array(this.nVerts), ev = this.edgeVerts;
      for (let i = 0; i < ev.length; i++) out[ev[i]]++;
      return out;
    });
  }

  get vertFaceCount() {
    return shared(this, "vertFaceCount", () => {
      const out = new Int32Array(this.nVerts), idx = this.faceIdx;
      for (let h = 0; h < idx.length; h++) out[idx[h]]++;
      return out;
    });
  }

  eulerCharacteristic() {
    return this.nVerts - this.nEdges + this.nFaces;
  }

  /** Vertex adjacency in CSR form (via the edges). */
  get vertAdjacency() {
    return shared(this, "vertAdjacency", () => {
      const N = this.nVerts, ev = this.edgeVerts, E = ev.length / 2;
      const ptr = new Int32Array(N + 1);
      for (let i = 0; i < ev.length; i++) ptr[ev[i] + 1]++;
      for (let v = 0; v < N; v++) ptr[v + 1] += ptr[v];
      const fill = ptr.slice(0, N), nb = new Int32Array(2 * E);
      for (let e = 0; e < E; e++) {
        const a = ev[2 * e], b = ev[2 * e + 1];
        nb[fill[a]++] = b;
        nb[fill[b]++] = a;
      }
      return { ptr, nb };
    });
  }

  /** Edge-hop distance from each vertex to the nearest boundary vertex (Infinity on closed meshes). */
  boundaryDistance() {
    return this.hopDistance(this.vertIsBoundary);
  }

  /** Edge-hop (topological) distance from each vertex to the nearest seed vertex. */
  hopDistance(seeds) {
    const N = this.nVerts, dist = new Float64Array(N).fill(Infinity);
    const queue = new Int32Array(N);
    let qh = 0, qt = 0;
    for (let v = 0; v < N; v++) if (seeds[v]) { dist[v] = 0; queue[qt++] = v; }
    if (qt === 0) return dist;
    const { ptr, nb } = this.vertAdjacency;
    while (qh < qt) {
      const v = queue[qh++], d = dist[v] + 1;
      for (let k = ptr[v]; k < ptr[v + 1]; k++) {
        const w = nb[k];
        if (dist[w] > d) { dist[w] = d; queue[qt++] = w; }
      }
    }
    return dist;
  }

  // -------------------------------------------------------------- geometry
  /** Sum a per-half-edge quantity (dim values per half-edge) over each face (numpy's add.reduceat order). */
  faceReduce(perHe, dim = 1) {
    const F = this.nFaces, p = this.facePtr, out = new Float64Array(F * dim);
    for (let f = 0; f < F; f++) for (let c = 0; c < dim; c++) out[f * dim + c] = segSum(perHe, p[f], p[f + 1], null, dim, c);
    return out;
  }

  /** Mean of a per-vertex quantity over each face's corners (face_reduce(a[face_idx]) / face_size). */
  faceMeanOfVerts(perVert, dim = 1) {
    const F = this.nFaces, p = this.facePtr, idx = this.faceIdx, out = new Float64Array(F * dim);
    for (let f = 0; f < F; f++) {
      const k = p[f + 1] - p[f];
      for (let c = 0; c < dim; c++) out[f * dim + c] = segSum(perVert, p[f], p[f + 1], idx, dim, c) / k;
    }
    return out;
  }

  get faceCentroid() {
    return this._c.faceCentroid || (this._c.faceCentroid = this.faceMeanOfVerts(this.V, 3));
  }

  /** Newell area vector (direction = normal, length = area): robust for non-planar n-gons. */
  get faceAreaVec() {
    if (this._c.faceAreaVec) return this._c.faceAreaVec;
    const H = this.nHalfedges, idx = this.faceIdx, nxt = this.heNext, V = this.V;
    const cr = new Float64Array(3 * H);
    for (let h = 0; h < H; h++) {
      const i = 3 * idx[h], j = 3 * idx[nxt[h]];
      const px = V[i], py = V[i + 1], pz = V[i + 2], qx = V[j], qy = V[j + 1], qz = V[j + 2];
      cr[3 * h] = py * qz - pz * qy;
      cr[3 * h + 1] = pz * qx - px * qz;
      cr[3 * h + 2] = px * qy - py * qx;
    }
    const out = this.faceReduce(cr, 3);
    for (let i = 0; i < out.length; i++) out[i] = 0.5 * out[i];
    return (this._c.faceAreaVec = out);
  }

  get faceNormal() {
    return this._c.faceNormal || (this._c.faceNormal = normalized(this.faceAreaVec));
  }

  get edgeLength() {
    if (this._c.edgeLength) return this._c.edgeLength;
    const ev = this.edgeVerts, V = this.V, E = ev.length / 2, out = new Float64Array(E);
    for (let e = 0; e < E; e++) {
      const i = 3 * ev[2 * e], j = 3 * ev[2 * e + 1];
      const dx = V[j] - V[i], dy = V[j + 1] - V[i + 1], dz = V[j + 2] - V[i + 2];
      out[e] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return (this._c.edgeLength = out);
  }

  /** Mean edge length of each face (the local length unit for relative extrusion). */
  get faceScale() {
    if (this._c.faceScale) return this._c.faceScale;
    const F = this.nFaces, p = this.facePtr, out = new Float64Array(F);
    const L = this.edgeLength, he = this.heEdge;
    for (let f = 0; f < F; f++) out[f] = segSum(L, p[f], p[f + 1], he) / (p[f + 1] - p[f]);
    return (this._c.faceScale = out);
  }

  /** Average of the adjacent face normals (paper: n_e). */
  get edgeNormal() {
    if (this._c.edgeNormal) return this._c.edgeNormal;
    const ef = this.edgeFaces, n = this.faceNormal, E = ef.length / 2, out = new Float64Array(3 * E);
    for (let e = 0; e < E; e++) {
      const f0 = ef[2 * e], f1 = ef[2 * e + 1];
      let x = n[3 * f0], y = n[3 * f0 + 1], z = n[3 * f0 + 2];
      if (f1 >= 0) { x += n[3 * f1]; y += n[3 * f1 + 1]; z += n[3 * f1 + 2]; }
      out[3 * e] = x; out[3 * e + 1] = y; out[3 * e + 2] = z;
    }
    return (this._c.edgeNormal = normalized(out, true));
  }

  /** Area-weighted average of incident face normals (paper: n_p). */
  get vertNormal() {
    if (this._c.vertNormal) return this._c.vertNormal;
    const N = this.nVerts, idx = this.faceIdx, hf = this.heFace, A = this.faceAreaVec, out = new Float64Array(3 * N);
    for (let h = 0; h < idx.length; h++) {
      const v = 3 * idx[h], f = 3 * hf[h];
      out[v] += A[f]; out[v + 1] += A[f + 1]; out[v + 2] += A[f + 2];
    }
    return (this._c.vertNormal = normalized(out, true));
  }

  /** Average a per-face quantity onto edges. */
  edgeMean(perFace) {
    const ef = this.edgeFaces, E = ef.length / 2, out = new Float64Array(E);
    if (typeof perFace === "number") return out.fill(perFace);
    for (let e = 0; e < E; e++) {
      const f1 = ef[2 * e + 1];
      out[e] = f1 >= 0 ? 0.5 * (perFace[ef[2 * e]] + perFace[f1]) : perFace[ef[2 * e]];
    }
    return out;
  }

  /** Average a per-face quantity (dim values per face) onto vertices. */
  vertMean(perFace, dim = 1) {
    const N = this.nVerts, idx = this.faceIdx, hf = this.heFace, cnt = this.vertFaceCount;
    const out = new Float64Array(N * dim);
    if (typeof perFace === "number") return out.fill(perFace);
    if (dim === 1) {
      for (let h = 0; h < idx.length; h++) out[idx[h]] += perFace[hf[h]];
      for (let v = 0; v < N; v++) out[v] /= cnt[v] > 1 ? cnt[v] : 1;
    } else {
      for (let h = 0; h < idx.length; h++) {
        const v = idx[h] * dim, f = hf[h] * dim;
        for (let c = 0; c < dim; c++) out[v + c] += perFace[f + c];
      }
      for (let v = 0; v < N; v++) {
        const k = cnt[v] > 1 ? cnt[v] : 1;
        for (let c = 0; c < dim; c++) out[v * dim + c] /= k;
      }
    }
    return out;
  }

  /** Vertex motif (paper Fig. 6) encoded as faces*100 + edges, e.g. 404 = "4F4E". */
  motifCodes() {
    const fc = this.vertFaceCount, val = this.valence, out = new Int32Array(this.nVerts);
    for (let v = 0; v < out.length; v++) out[v] = fc[v] * 100 + val[v];
    return out;
  }

  /** Largest distance of a face's vertices from its mean plane, relative to the face size (0 = planar). */
  get facePlanarity() {
    if (this._c.facePlanarity) return this._c.facePlanarity;
    const F = this.nFaces, p = this.facePtr, idx = this.faceIdx, V = this.V;
    const C = this.faceCentroid, n = this.faceNormal, s = this.faceScale, out = new Float64Array(F);
    for (let f = 0; f < F; f++) {
      let m = -Infinity;
      for (let h = p[f]; h < p[f + 1]; h++) {
        const i = 3 * idx[h];
        const d = Math.abs((V[i] - C[3 * f]) * n[3 * f] + (V[i + 1] - C[3 * f + 1]) * n[3 * f + 1]
          + (V[i + 2] - C[3 * f + 2]) * n[3 * f + 2]);
        if (d > m) m = d;
      }
      out[f] = m / Math.max(s[f], EPS);
    }
    return (this._c.facePlanarity = out);
  }

  /** Mean (1 - cos) of the dihedral angles to neighbouring faces: 0 = flat, 1 = folded at 90 deg. */
  get faceBend() {
    if (this._c.faceBend) return this._c.faceBend;
    const F = this.nFaces, H = this.nHalfedges, p = this.facePtr, tw = this.heTwin, hf = this.heFace, n = this.faceNormal;
    const val = new Float64Array(H), has = new Float64Array(H);
    for (let h = 0; h < H; h++) {
      const t = tw[h];
      if (t < 0) continue;
      const f = 3 * hf[h], g = 3 * hf[t];
      val[h] = 1.0 - (n[f] * n[g] + n[f + 1] * n[g + 1] + n[f + 2] * n[g + 2]);
      has[h] = 1;
    }
    const out = new Float64Array(F);
    for (let f = 0; f < F; f++) {
      const c = segSum(has, p[f], p[f + 1]);
      out[f] = segSum(val, p[f], p[f + 1]) / (c > 1 ? c : 1);
    }
    return (this._c.faceBend = out);
  }

  /** Divergence-theorem volume; positive for a closed, outward-oriented mesh. */
  signedVolume() {
    const A = this.faceAreaVec, C = this.faceCentroid;
    let s = 0;
    for (let i = 0; i < A.length; i++) s += A[i] * C[i];
    return s / 3.0;
  }

  // --------------------------------------------------------------- display
  /** Fan triangulation (for display / STL export), packed (T*3). */
  triangles() {
    return shared(this, "triangles", () => {
      const F = this.nFaces, p = this.facePtr, idx = this.faceIdx;
      let T = 0;
      for (let f = 0; f < F; f++) T += p[f + 1] - p[f] - 2;
      const out = new Int32Array(3 * Math.max(T, 0));
      let o = 0;
      for (let f = 0; f < F; f++) {
        const s = p[f], k = p[f + 1] - s;
        for (let t = 0; t < k - 2; t++) {
          out[o++] = idx[s];
          out[o++] = idx[s + t + 1];
          out[o++] = idx[s + t + 2];
        }
      }
      return out;
    });
  }

  /** For each display triangle, the polygon it belongs to. */
  triangleFace() {
    return shared(this, "triangleFace", () => {
      const F = this.nFaces, p = this.facePtr;
      let T = 0;
      for (let f = 0; f < F; f++) T += p[f + 1] - p[f] - 2;
      const out = new Int32Array(Math.max(T, 0));
      let o = 0;
      for (let f = 0; f < F; f++) for (let t = 0; t < p[f + 1] - p[f] - 2; t++) out[o++] = f;
      return out;
    });
  }
}

/** Normalise packed xyz vectors (in place when `inPlace`). */
export function normalized(v, inPlace = false) {
  const out = inPlace ? v : new Float64Array(v.length);
  for (let i = 0; i < v.length; i += 3) {
    const x = v[i], y = v[i + 1], z = v[i + 2];
    const n = Math.sqrt(x * x + y * y + z * z);
    const m = n > EPS ? n : EPS;
    out[i] = x / m; out[i + 1] = y / m; out[i + 2] = z / m;
  }
  return out;
}

// Mesh import / export (port of hansmeyer/meshio.py): OBJ in (with cleanup), OBJ / STL / PLY out, and an
// STL validator (complete and watertight). Writers return ArrayBuffers / strings; the page downloads them.

import { PolyMesh } from "./mesh.js";

export class MeshImportError extends Error {}

// ------------------------------------------------------------------- import
/** Positions and polygon faces from OBJ text (v / f only; handles v/vt/vn and negative indices). */
export function readObj(text, name = "mesh.obj") {
  const verts = [], faces = [];
  for (const line of text.split(/\r?\n/)) {
    const parts = line.trim().split(/\s+/);
    if (!parts[0]) continue;
    if (parts[0] === "v" && parts.length >= 4) {
      verts.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
    } else if (parts[0] === "f" && parts.length >= 4) {
      const idx = [];
      for (const p of parts.slice(1)) {
        const i = parseInt(p.split("/")[0], 10);
        if (Number.isNaN(i)) throw new MeshImportError(`${name}: bad face index '${p}'`);
        idx.push(i > 0 ? i - 1 : verts.length + i);
      }
      faces.push(idx);
    }
  }
  if (!verts.length || !faces.length) throw new MeshImportError(`${name}: no vertices/faces found`);
  return [verts, faces];
}

/** Weld duplicate vertices, drop degenerate/duplicate faces and unused vertices. */
export function cleanMesh(V, faces, weld = true) {
  if (weld) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of V) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], p[a]); hi[a] = Math.max(hi[a], p[a]); }
    const diag = Math.sqrt((hi[0] - lo[0]) ** 2 + (hi[1] - lo[1]) ** 2 + (hi[2] - lo[2]) ** 2) || 1.0;
    const q = diag * 1e-7;
    const keys = V.map((p) => p.map((x) => Math.round(x / q)));
    // np.unique(axis=0): rows sorted lexicographically; `first` = first occurrence of each
    const order = keys.map((_, i) => i).sort((i, j) => keys[i][0] - keys[j][0] || keys[i][1] - keys[j][1]
      || keys[i][2] - keys[j][2] || i - j);
    const remap = new Array(V.length);
    const first = [];
    let prev = null;
    for (const i of order) {
      const k = keys[i];
      if (!prev || k[0] !== prev[0] || k[1] !== prev[1] || k[2] !== prev[2]) {
        first.push(i);
        prev = k;
      }
      remap[i] = first.length - 1;
    }
    V = first.map((i) => V[i]);
    faces = faces.map((f) => f.map((i) => remap[i]));
  }
  const out = [], seen = new Set();
  for (const f of faces) {
    const g = f.filter((v, i) => v !== f[(i - 1 + f.length) % f.length]); // drop repeated consecutive indices
    if (g.length < 3 || new Set(g).size !== g.length) continue;
    const key = [...g].sort((a, b) => a - b).join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(g);
  }
  const usedSet = new Set();
  for (const f of out) for (const v of f) usedSet.add(v);
  const used = [...usedSet].sort((a, b) => a - b);
  const newIndex = new Map(used.map((v, i) => [v, i]));
  return [used.map((v) => V[v]), out.map((f) => f.map((v) => newIndex.get(v)))];
}

/** Make winding consistent per connected component; closed components face outward. Throws for
 * non-manifold edges (shared by >2 faces) or non-orientable surfaces (e.g. a Moebius strip). */
export function orientFaces(V, faces) {
  const edgeFaces = new Map();
  const ek = (a, b) => (a < b ? a * 4294967296 + b : b * 4294967296 + a);
  faces.forEach((f, fi) => {
    for (let i = 0; i < f.length; i++) {
      const k = ek(f[i], f[(i + 1) % f.length]);
      let l = edgeFaces.get(k);
      if (!l) edgeFaces.set(k, (l = []));
      l.push(fi);
    }
  });
  let bad = 0;
  for (const l of edgeFaces.values()) if (l.length > 2) bad++;
  if (bad) {
    throw new MeshImportError(`non-manifold mesh: ${bad} edge(s) are shared by more than two faces `
      + "(e.g. T-junction walls or touching parts): clean it up in Rhino/Blender first");
  }
  faces = faces.map((f) => [...f]);
  const hasDirected = (f, a, b) => {
    for (let i = 0; i < f.length; i++) if (f[i] === a && f[(i + 1) % f.length] === b) return true;
    return false;
  };
  const done = new Uint8Array(faces.length);
  for (let seed = 0; seed < faces.length; seed++) {
    if (done[seed]) continue;
    const comp = [seed], queue = [seed];
    done[seed] = 1;
    for (let qi = 0; qi < queue.length; qi++) {
      const fi = queue[qi], f = faces[fi];
      for (let i = 0; i < f.length; i++) {
        const a = f[i], b = f[(i + 1) % f.length];
        for (const gj of edgeFaces.get(ek(a, b))) {
          if (gj === fi) continue;
          const same = hasDirected(faces[gj], a, b);
          if (done[gj]) {
            if (same) throw new MeshImportError("non-orientable surface (Moebius-like): cannot be subdivided");
            continue;
          }
          if (same) faces[gj].reverse();
          done[gj] = 1;
          comp.push(gj);
          queue.push(gj);
        }
      }
    }
    // closed component -> outward (positive signed volume)
    const cnt = new Map();
    for (const fi of comp) {
      const f = faces[fi];
      for (let i = 0; i < f.length; i++) {
        const k = ek(f[i], f[(i + 1) % f.length]);
        cnt.set(k, (cnt.get(k) || 0) + 1);
      }
    }
    let closed = true;
    for (const c of cnt.values()) if (c !== 2) { closed = false; break; }
    if (closed) {
      let vol = 0.0;
      for (const fi of comp) {
        const p = faces[fi].map((i) => V[i]);
        for (let i = 1; i < p.length - 1; i++) {
          const b = p[i], c = p[i + 1];
          vol += p[0][0] * (b[1] * c[2] - b[2] * c[1]) + p[0][1] * (b[2] * c[0] - b[0] * c[2]) + p[0][2] * (b[0] * c[1] - b[1] * c[0]);
        }
      }
      if (vol < 0) for (const fi of comp) faces[fi].reverse();
    }
  }
  return faces;
}

export function loadObj(text, name = "mesh.obj", { normalize = true, weld = true, radius = 3 ** 0.5 } = {}) {
  let [V, faces] = readObj(text, name);
  [V, faces] = cleanMesh(V, faces, weld);
  if (!faces.length) throw new MeshImportError(`${name}: no valid faces after cleanup`);
  faces = orientFaces(V, faces);
  if (normalize) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of V) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], p[a]); hi[a] = Math.max(hi[a], p[a]); }
    const c = [0, 1, 2].map((a) => 0.5 * (lo[a] + hi[a]));
    V = V.map((p) => [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
    let r = 0;
    for (const p of V) r = Math.max(r, Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]));
    if (r > 0) V = V.map((p) => p.map((x) => x * (radius / r)));
  }
  const m = PolyMesh.fromFaces(V, faces);
  void m.heTwin; // validate
  return m;
}

// ------------------------------------------------------------------- export
function fmt6(x) {
  const s = x.toFixed(6);
  return s === "-0.000000" ? "-0.000000" : s;
}

/** OBJ text: faces grouped by size, like the desktop app's writer. */
export function objText(m, header = "# Hansmeyer subdivision engine") {
  const parts = [header + "\n"];
  const V = m.V;
  let chunk = [];
  for (let i = 0; i < V.length; i += 3) {
    chunk.push(`v ${fmt6(V[i])} ${fmt6(V[i + 1])} ${fmt6(V[i + 2])}`);
    if (chunk.length >= 50000) { parts.push(chunk.join("\n") + "\n"); chunk = []; }
  }
  if (chunk.length) parts.push(chunk.join("\n") + "\n");
  chunk = [];
  const sizes = m.faceSize, p = m.facePtr, idx = m.faceIdx;
  const ks = [...new Set(sizes)].sort((a, b) => a - b);
  for (const k of ks) {
    for (let f = 0; f < sizes.length; f++) {
      if (sizes[f] !== k) continue;
      let line = "f";
      for (let h = p[f]; h < p[f + 1]; h++) line += " " + (idx[h] + 1);
      chunk.push(line);
      if (chunk.length >= 50000) { parts.push(chunk.join("\n") + "\n"); chunk = []; }
    }
  }
  if (chunk.length) parts.push(chunk.join("\n") + "\n");
  return parts;
}

/** Binary STL from positions (packed, any float array) and triangles (packed). */
export function stlFromTriangles(V, T, header = "Hansmeyer subdivision engine") {
  const n = T.length / 3;
  const buf = new ArrayBuffer(84 + 50 * n);
  const dv = new DataView(buf);
  for (let i = 0; i < 80; i++) dv.setUint8(i, i < header.length ? header.charCodeAt(i) : 32);
  dv.setUint32(80, n, true);
  const f32 = Math.fround;
  let o = 84;
  for (let t = 0; t < n; t++) {
    const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], c = 3 * T[3 * t + 2];
    const ax = f32(V[a]), ay = f32(V[a + 1]), az = f32(V[a + 2]);
    const bx = f32(V[b]), by = f32(V[b + 1]), bz = f32(V[b + 2]);
    const cx = f32(V[c]), cy = f32(V[c + 1]), cz = f32(V[c + 2]);
    const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const nn = Math.max(Math.sqrt(nx * nx + ny * ny + nz * nz), 1e-20);
    nx /= nn; ny /= nn; nz /= nn;
    dv.setFloat32(o, nx, true); dv.setFloat32(o + 4, ny, true); dv.setFloat32(o + 8, nz, true);
    dv.setFloat32(o + 12, ax, true); dv.setFloat32(o + 16, ay, true); dv.setFloat32(o + 20, az, true);
    dv.setFloat32(o + 24, bx, true); dv.setFloat32(o + 28, by, true); dv.setFloat32(o + 32, bz, true);
    dv.setFloat32(o + 36, cx, true); dv.setFloat32(o + 40, cy, true); dv.setFloat32(o + 44, cz, true);
    dv.setUint16(o + 48, 0, true);
    o += 50;
  }
  return buf;
}

export function stlBuffer(m) {
  return stlFromTriangles(m.V, m.triangles());
}

/** Binary little-endian PLY with the original polygons (quads stay quads). */
export function plyBuffer(m) {
  const header = "ply\nformat binary_little_endian 1.0\ncomment Hansmeyer subdivision engine\n"
    + `element vertex ${m.nVerts}\nproperty float x\nproperty float y\nproperty float z\n`
    + `element face ${m.nFaces}\nproperty list uchar int vertex_indices\nend_header\n`;
  const hb = new TextEncoder().encode(header);
  const size = hb.length + 12 * m.nVerts + m.nFaces + 4 * m.nHalfedges;
  const buf = new ArrayBuffer(size);
  new Uint8Array(buf).set(hb, 0);
  const dv = new DataView(buf);
  let o = hb.length;
  for (let i = 0; i < m.V.length; i++, o += 4) dv.setFloat32(o, m.V[i], true);
  const p = m.facePtr, idx = m.faceIdx;
  for (let f = 0; f < m.nFaces; f++) {
    dv.setUint8(o, p[f + 1] - p[f]);
    o += 1;
    for (let h = p[f]; h < p[f + 1]; h++, o += 4) dv.setInt32(o, idx[h], true);
  }
  return buf;
}

/** Re-read a binary STL: complete file? watertight (every edge shared by exactly two oppositely
 * oriented triangles)? Returns {triangles, complete, watertight, bad_edges}. */
export function validateStl(buf) {
  const size = buf.byteLength;
  if (size < 84) return { triangles: 0, complete: false, watertight: false, bad_edges: -1 };
  const dv = new DataView(buf);
  const n = dv.getUint32(80, true);
  if (size !== 84 + 50 * n) return { triangles: n, complete: false, watertight: false, bad_edges: -1 };
  // weld identical float32 corners (open addressing on their bit patterns)
  const nc = 3 * n;
  let cap = 1;
  while (cap < 2 * nc + 16) cap <<= 1;
  const slotId = new Int32Array(cap).fill(-1);
  const keys = new Uint32Array(3 * nc);
  const T = new Int32Array(nc);
  let nv = 0;
  for (let t = 0; t < n; t++) {
    for (let c = 0; c < 3; c++) {
      const o = 84 + 50 * t + 12 + 12 * c;
      const x = dv.getUint32(o, true), y = dv.getUint32(o + 4, true), z = dv.getUint32(o + 8, true);
      let h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & (cap - 1);
      for (;;) {
        const id = slotId[h];
        if (id < 0) {
          slotId[h] = nv;
          keys[3 * nv] = x; keys[3 * nv + 1] = y; keys[3 * nv + 2] = z;
          T[3 * t + c] = nv++;
          break;
        }
        if (keys[3 * id] === x && keys[3 * id + 1] === y && keys[3 * id + 2] === z) {
          T[3 * t + c] = id;
          break;
        }
        h = (h + 1) & (cap - 1);
      }
    }
  }
  const fwd = new Float64Array(nc), rev = new Float64Array(nc);
  for (let t = 0; t < n; t++) {
    for (let c = 0; c < 3; c++) {
      const a = T[3 * t + c], b = T[3 * t + ((c + 1) % 3)];
      fwd[3 * t + c] = a * nv + b;
      rev[3 * t + c] = b * nv + a;
    }
  }
  fwd.sort();
  rev.sort();
  let dup = 0, missing = 0;
  for (let i = 0; i < nc; ) {
    let j = i + 1;
    while (j < nc && fwd[j] === fwd[i]) j++;
    if (j - i > 1) dup++; // the same directed edge twice: non-manifold or flipped
    i = j;
  }
  for (let i = 0, j = 0; i < nc; i++) { // an edge without its opposite: a hole
    while (j < nc && fwd[j] < rev[i]) j++;
    if (j >= nc || fwd[j] !== rev[i]) missing++;
  }
  const bad = dup + missing;
  return { triangles: n, complete: true, watertight: bad === 0, bad_edges: bad };
}

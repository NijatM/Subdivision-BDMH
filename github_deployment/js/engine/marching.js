// Marching cubes for the print model (replaces skimage.measure.marching_cubes).
//
// The case table is generated, not typed in: on every cube face the surface crosses the face's edges in
// segments chosen by one fixed rule (ambiguous faces, two diagonal corners inside, always separate the
// inside corners). Both cubes sharing a face apply the same rule to the same four values, so their
// polygons meet edge to edge, and every surface vertex sits on a grid edge shared by its four cubes: the
// result is a closed, manifold, consistently oriented surface (normals point from high to low values).

// corner c sits at (c & 1, c >> 1 & 1, c >> 2 & 1)
const CORNER = Array.from({ length: 8 }, (_, c) => [c & 1, (c >> 1) & 1, (c >> 2) & 1]);
// edges: (corner, corner with bit a set, axis a)
export const EDGES = [];
for (let a = 0; a < 3; a++) for (let c = 0; c < 8; c++) if (!(c & (1 << a))) EDGES.push([c, c | (1 << a), a]);
const edgeOf = (c0, c1) => EDGES.findIndex(([p, q]) => (p === c0 && q === c1) || (p === c1 && q === c0));

// faces: the four corners of each cube face in cyclic order
const FACES = [];
for (let a = 0; a < 3; a++) {
  const b = (a + 1) % 3, c = (a + 2) % 3;
  for (const side of [0, 1]) {
    const cyc = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => (side << a) | (u << b) | (v << c));
    FACES.push(cyc);
  }
}

function buildTable() {
  const table = [];
  for (let cfg = 0; cfg < 256; cfg++) {
    const inside = (c) => (cfg >> c) & 1;
    const segs = [];
    for (const cyc of FACES) {
      const ins = cyc.map(inside);
      const n = ins.reduce((s, x) => s + x, 0);
      if (n === 0 || n === 4) continue;
      const cut = (i) => segs.push([edgeOf(cyc[(i + 3) % 4], cyc[i]), edgeOf(cyc[i], cyc[(i + 1) % 4])]);
      if (n === 1) cut(ins.indexOf(1));
      else if (n === 3) cut(ins.indexOf(0));
      else if (ins[0] === ins[2]) { // diagonal: separate the inside corners
        for (let i = 0; i < 4; i++) if (ins[i]) cut(i);
      } else { // adjacent pair: one segment across the face
        const cross = [];
        for (let i = 0; i < 4; i++) if (ins[i] !== ins[(i + 1) % 4]) cross.push(edgeOf(cyc[i], cyc[(i + 1) % 4]));
        segs.push(cross);
      }
    }
    // chain segments into loops (each crossing edge has exactly two segments)
    const nb = new Map();
    for (const [p, q] of segs) {
      if (!nb.has(p)) nb.set(p, []);
      if (!nb.has(q)) nb.set(q, []);
      nb.get(p).push(q);
      nb.get(q).push(p);
    }
    const used = new Set(), tris = [];
    for (const start of nb.keys()) {
      if (used.has(start)) continue;
      const loop = [start];
      used.add(start);
      let prev = -1, cur = start;
      for (;;) {
        const next = nb.get(cur).find((e) => e !== prev && !used.has(e));
        if (next === undefined) break;
        loop.push(next);
        used.add(next);
        prev = cur;
        cur = next;
      }
      // orient: the normal points away from the inside corners of the loop's edges
      const pts = loop.map((e) => {
        const [p, q] = EDGES[e];
        return [0, 1, 2].map((k) => 0.5 * (CORNER[p][k] + CORNER[q][k]));
      });
      const nrm = [0, 0, 0], cen = [0, 0, 0], inn = [0, 0, 0];
      for (let i = 0; i < pts.length; i++) {
        const P = pts[i], Q = pts[(i + 1) % pts.length];
        nrm[0] += P[1] * Q[2] - P[2] * Q[1];
        nrm[1] += P[2] * Q[0] - P[0] * Q[2];
        nrm[2] += P[0] * Q[1] - P[1] * Q[0];
        for (let k = 0; k < 3; k++) cen[k] += P[k] / pts.length;
        const [p, q] = EDGES[loop[i]];
        const ci = inside(p) ? p : q;
        for (let k = 0; k < 3; k++) inn[k] += CORNER[ci][k] / pts.length;
      }
      const dot = nrm[0] * (cen[0] - inn[0]) + nrm[1] * (cen[1] - inn[1]) + nrm[2] * (cen[2] - inn[2]);
      if (dot < 0) loop.reverse();
      for (let i = 1; i < loop.length - 1; i++) tris.push(loop[0], loop[i], loop[i + 1]);
    }
    table.push(tris);
  }
  return table;
}

let TABLE = null;

/** Isosurface of vol (Float32Array, C order: z fastest) at `level`. Vertices are in index units times
 * `spacing`; triangles face from inside (> level) to outside. Returns { V: Float64Array, F: Int32Array }. */
export function marchingCubes(vol, nx, ny, nz, level = 0.5, spacing = 1.0) {
  if (!TABLE) TABLE = buildTable();
  const sx = ny * nz, sy = nz;
  let V = new Float64Array(3 * 65536), nV = 0;
  let F = new Int32Array(3 * 131072), nF = 0;
  const plane = ny * nz;
  let ex = new Int32Array(plane);
  let ey0 = new Int32Array(plane).fill(-1), ez0 = new Int32Array(plane).fill(-1);
  let ey1 = new Int32Array(plane), ez1 = new Int32Array(plane);
  const val = new Float64Array(8);
  const ids = new Int32Array(12);

  const addVertex = (gx, gy, gz, a, v0, v1) => {
    if (3 * (nV + 1) > V.length) {
      const nv = new Float64Array(V.length * 2);
      nv.set(V);
      V = nv;
    }
    const t = (level - v0) / (v1 - v0);
    const o = 3 * nV;
    V[o] = gx * spacing;
    V[o + 1] = gy * spacing;
    V[o + 2] = gz * spacing;
    V[o + a] += t * spacing;
    return nV++;
  };

  for (let x = 0; x < nx - 1; x++) {
    ex.fill(-1);
    ey1.fill(-1);
    ez1.fill(-1);
    for (let y = 0; y < ny - 1; y++) {
      for (let z = 0; z < nz - 1; z++) {
        const base = x * sx + y * sy + z;
        let cfg = 0;
        for (let c = 0; c < 8; c++) {
          const v = vol[base + (c & 1) * sx + ((c >> 1) & 1) * sy + ((c >> 2) & 1)];
          val[c] = v;
          if (v > level) cfg |= 1 << c;
        }
        if (cfg === 0 || cfg === 255) continue;
        const tris = TABLE[cfg];
        ids.fill(-1);
        for (let i = 0; i < tris.length; i++) {
          const e = tris[i];
          if (ids[e] >= 0) continue;
          const [c0, c1, a] = EDGES[e];
          const dx = c0 & 1, dy = (c0 >> 1) & 1, dz = (c0 >> 2) & 1;
          let arr, k;
          if (a === 0) { arr = ex; k = (y + dy) * nz + (z + dz); }
          else if (a === 1) { arr = dx ? ey1 : ey0; k = y * nz + (z + dz); }
          else { arr = dx ? ez1 : ez0; k = (y + dy) * nz + z; }
          let id = arr[k];
          if (id < 0) {
            id = addVertex(x + dx, y + dy, z + dz, a, val[c0], val[c1]);
            arr[k] = id;
          }
          ids[e] = id;
        }
        if (nF + tris.length > F.length) {
          const nf = new Int32Array(F.length * 2);
          nf.set(F);
          F = nf;
        }
        for (let i = 0; i < tris.length; i++) F[nF++] = ids[tris[i]];
      }
    }
    let t = ey0; ey0 = ey1; ey1 = t;
    t = ez0; ez0 = ez1; ez1 = t;
  }
  return { V: V.slice(0, 3 * nV), F: F.slice(0, nF) };
}

export function caseTable() {
  if (!TABLE) TABLE = buildTable();
  return TABLE;
}

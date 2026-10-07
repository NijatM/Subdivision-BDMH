// 3D noise: improved Perlin, fBm, ridged, domain-warped and Worley (cellular); port of hansmeyer/noise.py.
// Inputs and outputs are packed: points are Float64Array(n*3), values Float64Array(n).

import { PERM_B64, PERM_SEEDS } from "./noisePerm.js";

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

function decodeBase64(s) {
  const out = new Uint8Array(Math.floor((s.length * 3) / 4));
  let o = 0, buf = 0, bits = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "=") break;
    buf = (buf << 6) | B64.indexOf(c);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[o++] = (buf >> bits) & 255;
    }
  }
  return out.subarray(0, o);
}

let TABLE = null;
const PERMS = new Map();

/** numpy's default_rng(seed).permutation(256), doubled to 512 entries. Seeds beyond the shipped table use
 * a deterministic shuffle of our own (they differ from the desktop app's, which never uses them). */
function perm(seed) {
  let p = PERMS.get(seed);
  if (p) return p;
  p = new Int32Array(512);
  if (seed >= 0 && seed < PERM_SEEDS) {
    if (!TABLE) TABLE = decodeBase64(PERM_B64);
    for (let i = 0; i < 256; i++) p[i] = p[i + 256] = TABLE[seed * 256 + i];
  } else {
    const q = Array.from({ length: 256 }, (_, i) => i);
    let s = (seed * 2654435761) >>> 0;
    for (let i = 255; i > 0; i--) {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      const j = (((t ^ (t >>> 14)) >>> 0) % (i + 1));
      [q[i], q[j]] = [q[j], q[i]];
    }
    for (let i = 0; i < 256; i++) p[i] = p[i + 256] = q[i];
  }
  PERMS.set(seed, p);
  return p;
}

const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);

function grad(h, x, y, z) {
  h &= 15;
  const u = h < 8 ? x : y;
  const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
  return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

const lerp = (t, a, b) => a + t * (b - a);

/** Ken Perlin's improved noise, roughly in [-1, 1]. */
export function perlin(p, seed = 0) {
  const P = perm(Math.trunc(seed));
  const n = p.length / 3, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const px = p[3 * i], py = p[3 * i + 1], pz = p[3 * i + 2];
    const fx = Math.floor(px), fy = Math.floor(py), fz = Math.floor(pz);
    const X = fx & 255, Y = fy & 255, Z = fz & 255;
    const x = px - fx, y = py - fy, z = pz - fz;
    const u = fade(x), v = fade(y), w = fade(z);
    const A = P[X] + Y, B = P[X + 1] + Y;
    const AA = P[A] + Z, AB = P[A + 1] + Z, BA = P[B] + Z, BB = P[B + 1] + Z;
    out[i] = lerp(w,
      lerp(v, lerp(u, grad(P[AA], x, y, z), grad(P[BA], x - 1, y, z)),
        lerp(u, grad(P[AB], x, y - 1, z), grad(P[BB], x - 1, y - 1, z))),
      lerp(v, lerp(u, grad(P[AA + 1], x, y, z - 1), grad(P[BA + 1], x - 1, y, z - 1)),
        lerp(u, grad(P[AB + 1], x, y - 1, z - 1), grad(P[BB + 1], x - 1, y - 1, z - 1))));
  }
  return out;
}

export function fbm(p, octaves = 4, lacunarity = 2.0, gain = 0.5, seed = 0) {
  const n = p.length / 3, total = new Float64Array(n);
  let amp = 1.0, norm = 0.0, q = p;
  for (let o = 0; o < Math.trunc(octaves); o++) {
    const v = perlin(q, seed + o);
    for (let i = 0; i < n; i++) total[i] += amp * v[i];
    norm += amp;
    amp *= gain;
    const nq = new Float64Array(q.length);
    for (let i = 0; i < q.length; i++) nq[i] = q[i] * lacunarity;
    q = nq;
  }
  const d = Math.max(norm, 1e-9);
  for (let i = 0; i < n; i++) total[i] /= d;
  return total;
}

/** Sharp ridges: 1 - |noise| per octave, mapped to [-1, 1]. */
export function ridged(p, octaves = 4, lacunarity = 2.0, gain = 0.5, seed = 0) {
  const n = p.length / 3, total = new Float64Array(n);
  let amp = 1.0, norm = 0.0, q = p;
  for (let o = 0; o < Math.trunc(octaves); o++) {
    const v = perlin(q, seed + o);
    for (let i = 0; i < n; i++) total[i] += amp * (1.0 - Math.abs(v[i])) ** 2;
    norm += amp;
    amp *= gain;
    const nq = new Float64Array(q.length);
    for (let i = 0; i < q.length; i++) nq[i] = q[i] * lacunarity;
    q = nq;
  }
  const d = Math.max(norm, 1e-9);
  for (let i = 0; i < n; i++) total[i] = (2.0 * total[i]) / d - 1.0;
  return total;
}

/** Domain-warped fBm (noise fed by noise): marbled, flowing patterns. */
export function warped(p, warp = 1.0, octaves = 4, seed = 0) {
  const n = p.length / 3;
  const offsets = [[0, 0, 0], [5.2, 1.3, 2.8], [1.7, 9.2, 4.1]];
  const q = offsets.map((o) => {
    const s = new Float64Array(p.length);
    for (let i = 0; i < n; i++) { s[3 * i] = p[3 * i] + o[0]; s[3 * i + 1] = p[3 * i + 1] + o[1]; s[3 * i + 2] = p[3 * i + 2] + o[2]; }
    return fbm(s, octaves, 2.0, 0.5, seed + 7);
  });
  const s = new Float64Array(p.length);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) s[3 * i + k] = p[3 * i + k] + warp * q[k][i];
  return fbm(s, octaves, 2.0, 0.5, seed);
}

// ---------------------------------------------------------------- 64-bit hash (numpy uint64 arithmetic)
const TWO32 = 4294967296;

/** (hi, lo) * (bh, bl) mod 2^64 -> [hi, lo] */
function mul64(ah, al, bh, bl) {
  const a0 = al & 0xffff, a1 = al >>> 16, b0 = bl & 0xffff, b1 = bl >>> 16;
  const p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const mid = p01 + p10;
  const low = p00 + (mid % 65536) * 65536;
  const lo = low >>> 0;
  const carry = Math.floor(low / TWO32);
  let hi = (p11 + Math.floor(mid / 65536) + carry) >>> 0;
  hi = (hi + Math.imul(ah, bl) + Math.imul(al, bh)) >>> 0;
  return [hi, lo];
}

function add64(ah, al, bh, bl) {
  const lo = al + bl;
  const carry = lo >= TWO32 ? 1 : 0;
  return [(ah + bh + carry) >>> 0, lo >>> 0];
}

// 64-bit constants as [hi, lo]
const MUL = [0x5851f42d, 0x4c957f2d]; // 6364136223846793005
const INC = [0x14057b7e, 0xf767814f]; // 1442695040888963407

/** Deterministic pseudo-random point in [0,1)^3 per integer cell (noise.py _hash3), written into out[o..o+2]. */
function hash3(cx, cy, cz, seedTerm, out, o) {
  const toU = (c) => [c < 0 ? 0xffffffff : 0, c >>> 0];
  const [xh, xl] = toU(cx), [yh, yl] = toU(cy), [zh, zl] = toU(cz);
  const [ph, pl] = mul64(xh, xl, 0, 73856093);
  const [qh, ql] = mul64(yh, yl, 0, 19349663);
  const [rh, rl] = mul64(zh, zl, 0, 83492791);
  let h = (ph ^ qh ^ rh) >>> 0, l = (pl ^ ql ^ rl ^ seedTerm) >>> 0;
  for (let k = 0; k < 3; k++) {
    [h, l] = mul64(h, l, MUL[0], MUL[1]);
    const [ih, il] = add64(INC[0], INC[1], 0, k);
    [h, l] = add64(h, l, ih, il);
    out[o + k] = ((h >>> 1) & 0xffffff) / 0x1000000; // (h >> 33) & 0xFFFFFF
  }
}

function seedTermOf(seed) {
  // seed * 2654435761 % 2**32 (Python integers: exact)
  return Number((BigInt(Math.trunc(seed)) * 2654435761n) % 4294967296n);
}

/** Distances to the nearest and second-nearest feature point (F1, F2) of a jittered grid. */
export function worley(p, seed = 0, jitter = 1.0) {
  const n = p.length / 3;
  const f1 = new Float64Array(n), f2 = new Float64Array(n);
  if (n === 0) return [f1, f2];
  const st = seedTermOf(seed);
  const base = new Int32Array(3 * n);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 3 * n; i++) {
    base[i] = Math.floor(p[i]);
    const a = i % 3;
    if (base[i] < lo[a]) lo[a] = base[i];
    if (base[i] > hi[a]) hi[a] = base[i];
  }
  for (let a = 0; a < 3; a++) lo[a] -= 1;
  const dims = [0, 1, 2].map((a) => hi[a] + 2 - lo[a]);
  const cells = dims[0] * dims[1] * dims[2];
  const h = new Float64Array(3);
  if (cells > Math.max(8 * n, 4096)) {
    // sparse points over a huge range: per point instead
    for (let i = 0; i < n; i++) {
      let d1 = Infinity, d2 = Infinity;
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
        const cx = base[3 * i] + a, cy = base[3 * i + 1] + b, cz = base[3 * i + 2] + c;
        hash3(cx, cy, cz, st, h, 0);
        const dx = cx + 0.5 + jitter * (h[0] - 0.5) - p[3 * i];
        const dy = cy + 0.5 + jitter * (h[1] - 0.5) - p[3 * i + 1];
        const dz = cz + 0.5 + jitter * (h[2] - 0.5) - p[3 * i + 2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) d2 = d;
      }
      f1[i] = d1;
      f2[i] = d2;
    }
    return [f1, f2];
  }
  // one feature point per grid cell for the block of cells around the points (built once)
  const feat = new Float64Array(3 * cells);
  let o = 0;
  for (let x = 0; x < dims[0]; x++) for (let y = 0; y < dims[1]; y++) for (let z = 0; z < dims[2]; z++) {
    const cx = lo[0] + x, cy = lo[1] + y, cz = lo[2] + z;
    hash3(cx, cy, cz, st, feat, o);
    feat[o] = cx + 0.5 + jitter * (feat[o] - 0.5);
    feat[o + 1] = cy + 0.5 + jitter * (feat[o + 1] - 0.5);
    feat[o + 2] = cz + 0.5 + jitter * (feat[o + 2] - 0.5);
    o += 3;
  }
  const s1 = dims[1] * dims[2], s2 = dims[2];
  for (let i = 0; i < n; i++) {
    const lin = (base[3 * i] - lo[0]) * s1 + (base[3 * i + 1] - lo[1]) * s2 + (base[3 * i + 2] - lo[2]);
    const px = p[3 * i], py = p[3 * i + 1], pz = p[3 * i + 2];
    let d1 = Infinity, d2 = Infinity;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      const k = 3 * (lin + a * s1 + b * s2 + c);
      const dx = feat[k] - px, dy = feat[k + 1] - py, dz = feat[k + 2] - pz;
      const d = dx * dx + dy * dy + dz * dz;
      d2 = Math.min(d2, Math.max(d1, d));
      d1 = Math.min(d1, d);
    }
    f1[i] = Math.sqrt(d1);
    f2[i] = Math.sqrt(d2);
  }
  return [f1, f2];
}

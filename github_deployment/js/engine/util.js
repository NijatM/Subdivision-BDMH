// Small numeric helpers shared by the engine (pure: runs in the page, in workers and in the test shell).

export const EPS = 1e-12;

/** Python's round(): halves go to the even neighbour (round(2.5) == 2). */
export function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** numpy.round(x, decimals) (half to even on the scaled value). */
export function npRound(x, decimals = 0) {
  const k = 10 ** decimals;
  return pyRound(x * k) / k;
}

export function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}

/** numpy.linspace(a, b, n) */
export function linspace(a, b, n, endpoint = true) {
  const out = new Float64Array(n);
  if (n === 1) {
    out[0] = a;
    return out;
  }
  const div = endpoint ? n - 1 : n;
  const step = (b - a) / div;
  for (let i = 0; i < n; i++) out[i] = a + i * step;
  if (endpoint) out[n - 1] = b;
  return out;
}

/** numpy.percentile with the default linear interpolation. */
export function percentile(values, pct) {
  const n = values.length;
  if (n === 0) return NaN;
  const s = Float64Array.from(values).sort();
  const pos = (pct / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  const t = pos - lo;
  // numpy's _lerp: a + (b - a) * t, with the symmetric form for t >= 0.5
  const a = s[lo], b = s[hi];
  const diff = b - a;
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

// ------------------------------------------------ numpy's summation order
// numpy reduces an array as  a[0] + pairwise_sum(a[1:]),  where pairwise_sum adds fewer than 8 values one
// by one (from 0.0), up to 128 values in 8 interleaved partial sums, and splits longer runs in two. The
// engine sums in the same order, so results match the desktop app's bit for bit, which matters where a
// value is a tiny residue (a centroid on an axis, then fed to atan2).

/** numpy's pairwise_sum of vals[off + i*step] for i < n (step: index stride; idx maps through an index array). */
export function pairwiseSum(vals, off, n, stride = 1, idx = null, dim = 1, c = 0) {
  const get = idx ? (j) => vals[idx[off + j * stride] * dim + c] : (j) => vals[(off + j * stride) * dim + c];
  return pw(get, 0, n);
}

function pw(get, o, n) {
  if (n < 8) {
    let res = 0.0;
    for (let i = 0; i < n; i++) res += get(o + i);
    return res;
  }
  if (n <= 128) {
    let r0 = get(o), r1 = get(o + 1), r2 = get(o + 2), r3 = get(o + 3), r4 = get(o + 4), r5 = get(o + 5), r6 = get(o + 6), r7 = get(o + 7);
    let i = 8;
    const lim = n - (n % 8);
    for (; i < lim; i += 8) {
      r0 += get(o + i); r1 += get(o + i + 1); r2 += get(o + i + 2); r3 += get(o + i + 3);
      r4 += get(o + i + 4); r5 += get(o + i + 5); r6 += get(o + i + 6); r7 += get(o + i + 7);
    }
    let res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; i++) res += get(o + i);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pw(get, o, n2) + pw(get, o + n2, n - n2);
}

/** numpy add.reduce of the segment [s, e) of vals (through idx when given; component c of dim). */
export function segSum(vals, s, e, idx = null, dim = 1, c = 0) {
  const k = e - s;
  if (k <= 0) return 0.0;
  if (k < 9) { // the common case (faces): first value plus the rest added one by one from 0.0
    if (idx) {
      let rest = 0.0;
      for (let j = s + 1; j < e; j++) rest += vals[idx[j] * dim + c];
      return vals[idx[s] * dim + c] + rest;
    }
    let rest = 0.0;
    for (let j = s + 1; j < e; j++) rest += vals[j * dim + c];
    return vals[s * dim + c] + rest;
  }
  const first = idx ? vals[idx[s] * dim + c] : vals[s * dim + c];
  return first + pairwiseSum(vals, s + 1, k - 1, 1, idx, dim, c);
}

/** np.linalg.norm(v, axis=1) of a 3-vector (numpy adds a row's three squares in order). */
export function norm3(x, y, z) {
  return Math.sqrt(x * x + y * y + z * z);
}

export function anyNonZero(a) {
  if (typeof a === "number") return a !== 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== 0) return true;
  return false;
}

export function minMax(a, stride = 1, offset = 0) {
  let lo = Infinity, hi = -Infinity;
  for (let i = offset; i < a.length; i += stride) {
    const v = a[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}

/** Bounding box of packed xyz positions: [[minx, miny, minz], [maxx, maxy, maxz]]. */
export function bounds3(V) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < V.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = V[i + a];
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
    }
  }
  return [lo, hi];
}

/** JSON with sorted object keys (stable input for cache keys and snapshots). */
export function stableStringify(obj) {
  if (obj === null || typeof obj !== "object") {
    if (typeof obj === "number" && !Number.isFinite(obj)) return "null";
    return JSON.stringify(obj === undefined ? null : obj);
  }
  if (ArrayBuffer.isView(obj)) return stableStringify(Array.from(obj));
  if (Array.isArray(obj)) return "[" + obj.map(stableStringify).join(",") + "]";
  const keys = Object.keys(obj).filter((k) => obj[k] !== undefined).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + stableStringify(obj[k])).join(",") + "}";
}

/** A 64-bit string hash (two independent 32-bit lanes), as 16 hex digits. */
export function hashString(str, seed = 0) {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (h2 >>> 0).toString(16).padStart(8, "0") + (h1 >>> 0).toString(16).padStart(8, "0");
}

export function digest(obj) {
  return hashString(stableStringify(obj));
}

/** Hash of a typed array's contents (for shared base topologies). */
export function hashArray(a) {
  let h1 = 0x811c9dc5 ^ a.length, h2 = 0x01000193;
  for (let i = 0; i < a.length; i++) {
    const v = a[i] | 0;
    h1 = Math.imul(h1 ^ v, 16777619);
    h2 = Math.imul(h2 ^ (v + i), 2246822519);
  }
  return (h1 >>> 0).toString(16) + (h2 >>> 0).toString(16);
}

export function deepCopy(x) {
  return x === undefined ? undefined : JSON.parse(JSON.stringify(x));
}

/** Stable argsort of a numeric array (ascending). */
export function argsortStable(values) {
  const n = values.length;
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  const arr = Array.from(idx);
  arr.sort((a, b) => values[a] - values[b] || a - b);
  return Int32Array.from(arr);
}

export function concatTyped(Type, parts) {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Type(n);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

export function now() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

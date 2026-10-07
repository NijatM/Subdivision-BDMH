// Parity test: the web engine against reference results from the desktop (Python) engine.
//   1. python github_deployment/tests/make_reference.py REF_DIR
//   2. jsc -m github_deployment/tests/parity.mjs -- REF_DIR      (macOS: jsc lives in
//      /System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc; run from the repo root)
// Every preset (and a few extra designs) must give the same faces, in the same order, and positions within
// a tiny tolerance of the desktop app's.

import { normalizeDesign } from "../js/engine/design.js";
import { Pipeline } from "../js/engine/pipeline.js";
import { loadPlugins } from "../js/engine/functions.js";
import { hashString } from "../js/engine/util.js";

const ARGS = globalThis.arguments || [];
const REF = ARGS[0] || "ref";
const SITE = "github_deployment";
const plugins = JSON.parse(readFile(`${SITE}/functions/index.json`)).map((f) => ({ name: f, source: readFile(`${SITE}/functions/${f}`) }));
const errors = loadPlugins(plugins);
if (errors.length) print("plug-in errors:", errors);

const ctx = {
  objText: (p) => readFile(`${SITE}/${p}`),
  objHash: (p) => hashString(readFile(`${SITE}/${p}`)),
};
const manifest = JSON.parse(readFile(`${REF}/manifest.json`));
let failed = 0;
const only = ARGS[1];
for (const item of manifest) {
  if (only && item.name !== only) continue;
  const t0 = Date.now();
  let r;
  try {
    r = new Pipeline({ ctx }).run(normalizeDesign(item.design), item.depth);
  } catch (e) {
    print(`FAIL ${item.name}: ${e && e.stack ? e.stack : e}`);
    failed++;
    continue;
  }
  const ms = Date.now() - t0;
  const m = r.mesh;
  const bytes = readFile(`${REF}/${item.name}.bin`, "binary");
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const nv = item.n_verts, nf = item.n_faces, nh = item.n_he;
  const V = new Float64Array(buf, 0, 3 * nv);
  const ptr = new Int32Array(buf, 24 * nv, nf + 1);
  const idx = new Int32Array(buf, 24 * nv + 4 * (nf + 1), nh);
  const problems = [];
  if (r.depthReached !== item.reached) problems.push(`depth ${r.depthReached} vs ${item.reached}`);
  if (m.nVerts !== nv || m.nFaces !== nf || m.nHalfedges !== nh) {
    problems.push(`size ${m.nVerts}/${m.nFaces}/${m.nHalfedges} vs ${nv}/${nf}/${nh}`);
  } else {
    let badF = 0;
    for (let i = 0; i <= nf; i++) if (ptr[i] !== m.facePtr[i]) badF++;
    for (let i = 0; i < nh; i++) if (idx[i] !== m.faceIdx[i]) badF++;
    if (badF) problems.push(`${badF} face entries differ`);
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], maxd = 0, at = -1;
    for (let i = 0; i < V.length; i++) {
      const a = i % 3;
      lo[a] = Math.min(lo[a], V[i]);
      hi[a] = Math.max(hi[a], V[i]);
      const d = Math.abs(V[i] - m.V[i]);
      if (!(d <= maxd)) { maxd = d; at = i; }
    }
    const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    if (!(maxd <= 1e-9 * diag)) problems.push(`max position error ${maxd.toExponential(2)} (diag ${diag.toFixed(3)}) at vertex ${Math.floor(at / 3)}`);
    item.err = maxd / diag;
  }
  const info = JSON.stringify(m.info);
  if (problems.length) {
    failed++;
    print(`FAIL ${item.name.padEnd(26)} ${problems.join("; ")}  ${info}`);
  } else {
    print(`ok   ${item.name.padEnd(26)} ${String(nf).padStart(8)} faces  ${String(ms).padStart(5)} ms  rel.err ${item.err.toExponential(1)}  ${info === "{}" ? "" : info}`);
  }
}
print(failed ? `${failed} FAILED` : "all passed");

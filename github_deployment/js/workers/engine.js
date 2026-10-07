// The engine worker: runs the subdivision pipeline off the page's thread (previews and bakes), keeps its
// own level cache, and answers colour / export requests for the last mesh it computed.

import { normalizeDesign } from "../engine/design.js";
import { Pipeline } from "../engine/pipeline.js";
import * as functions from "../engine/functions.js";
import * as attractors from "../engine/attractors.js";
import * as intrinsic from "../engine/intrinsic.js";
import * as layers from "../engine/layers.js";
import * as vessel from "../engine/vessel.js";
import { objText, stlBuffer, plyBuffer } from "../engine/meshio.js";
import { hashString, bounds3 } from "../engine/util.js";

const files = {};
const pipe = new Pipeline({
  ctx: { objText: (p) => files[p], objHash: (p) => (files[p] === undefined ? "missing" : hashString(files[p])) },
});
let last = null; // { mesh, design }
const topoIds = new WeakMap();
let topoCounter = 0;

function topoId(topo) {
  let id = topoIds.get(topo);
  if (id === undefined) topoIds.set(topo, (id = ++topoCounter));
  return id;
}

function colourValues(mesh, design, c) {
  if (!c || c.mode === "none") return null;
  let face = null, range = [0, 1];
  const mode = c.mode;
  if (mode === "influence") {
    const a = design.attractors[c.sel];
    if (!a) return null;
    face = attractors.influence(a, attractors.facePositions(mesh, design.attractor_space));
    range = [0, Math.max(Number(a.strength), 1e-6)];
  } else if (mode === "layer") {
    const ly = design.layers[c.layer];
    if (!ly || ly.target === "fold") return null;
    face = functions.evaluate(ly.function, attractors.facePositions(mesh, ly.space), ly.params, ly.domain);
    range = [-1, 1];
  } else if (mode === "light") {
    if (!mesh.vattr.wall_mm) return null;
    face = vessel.light(mesh);
  } else if (mode.startsWith("measure:")) {
    face = intrinsic.measure(mesh, mode.slice(8));
  } else if (mode === "tags") {
    if (!mesh.fattr.tags) return null;
    face = new Float64Array(mesh.nFaces);
    for (let f = 0; f < face.length; f++) face[f] = c.group < 0 ? (intrinsic.anyTag(mesh.fattr, f) ? 1 : 0) : (intrinsic.hasTag(mesh.fattr, f, c.group) ? 1 : 0);
  } else return null;
  const vert = mesh.vertMean(face);
  const out = new Float32Array(vert.length);
  const span = Math.max(range[1] - range[0], 1e-12);
  for (let i = 0; i < vert.length; i++) out[i] = Math.min(Math.max((vert[i] - range[0]) / span, 0), 1);
  return out;
}

function summary(r, design) {
  const m = r.mesh;
  let bnd = 0;
  for (const b of m.edgeIsBoundary) bnd += b;
  return {
    depthReached: r.depthReached, depthRequested: r.depthRequested, capped: r.cappedByBudget, seconds: r.seconds,
    cacheHits: r.cacheHits, nFaces: m.nFaces, nVerts: m.nVerts, info: m.info, euler: m.eulerCharacteristic(),
    boundaryEdges: bnd, motifs: intrinsic.motifCounts(r.relief || m), bounds: bounds3(m.V),
    finite: m.V.every((x) => Number.isFinite(x)),
  };
}

function display(m, wantEdges) {
  const V = new Float32Array(m.V);
  const T = new Uint32Array(m.triangles());
  const E = wantEdges ? new Uint32Array(m.edgeVerts) : null;
  return { V, T, E, topo: topoId(m.topo) };
}

self.onmessage = (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "plugins") {
      const errors = functions.loadPlugins(msg.list);
      pipe.clear();
      self.postMessage({ type: "plugins", errors, registry: functions.describe() });
    } else if (msg.type === "files") {
      Object.assign(files, msg.files);
    } else if (msg.type === "run") {
      const design = normalizeDesign(msg.design);
      pipe.faceBudget = msg.faceBudget || pipe.faceBudget;
      const r = pipe.run(design, msg.depth);
      last = { mesh: r.mesh, design };
      const disp = display(r.mesh, msg.wantEdges);
      const colour = colourValues(r.mesh, design, msg.colour);
      const transfer = [disp.V.buffer, disp.T.buffer];
      if (disp.E) transfer.push(disp.E.buffer);
      if (colour) transfer.push(colour.buffer);
      self.postMessage({ type: "result", id: msg.id, ok: true, summary: summary(r, design), display: disp, colour }, transfer);
    } else if (msg.type === "colour") {
      const colour = last ? colourValues(last.mesh, normalizeDesign(msg.design || last.design), msg.colour) : null;
      self.postMessage({ type: "colour", id: msg.id, colour }, colour ? [colour.buffer] : []);
    } else if (msg.type === "edges") {
      const E = last ? new Uint32Array(last.mesh.edgeVerts) : null;
      self.postMessage({ type: "edges", id: msg.id, E }, E ? [E.buffer] : []);
    } else if (msg.type === "export") {
      if (!last) throw new Error("nothing computed yet");
      const m = last.mesh;
      let data;
      if (msg.format === "obj") data = objText(m);
      else if (msg.format === "stl") data = stlBuffer(m);
      else data = plyBuffer(m);
      self.postMessage({ type: "export", id: msg.id, data }, data instanceof ArrayBuffer ? [data] : []);
    } else if (msg.type === "meshData") {
      if (!last) throw new Error("nothing computed yet");
      const m = last.mesh;
      const out = {
        V: new Float64Array(m.V), facePtr: new Int32Array(m.facePtr), faceIdx: new Int32Array(m.faceIdx),
        wall: m.vattr.wall_mm ? new Float64Array(m.vattr.wall_mm) : null, closed: !m.hasBoundary(),
      };
      self.postMessage({ type: "meshData", id: msg.id, data: out });
    }
  } catch (e) {
    self.postMessage({ type: msg.type === "run" ? "result" : msg.type, id: msg.id, ok: false, error: `${e && e.name ? e.name : "Error"}: ${e && e.message ? e.message : e}` });
  }
};

// 09 print: printer and size, and the print model. Vessels are already clean closed solids and export
// exactly (no voxel remesh), like the desktop app. The voxel remesh for every other form (watertight
// rebuild, cuts into parts, pin holes, support check) comes in the next update of the web app.

import { PolyMesh } from "../engine/mesh.js";
import { stlFromTriangles, validateStl } from "../engine/meshio.js";
import * as VS from "../engine/vessel.js";
import { download } from "./ui.js";

export const PRINTERS = [["Bambu Lab A1 / P1S / X1C", [256, 256, 256]], ["Bambu Lab A1 mini", [180, 180, 180]],
  ["Prusa MK4 / MK4S", [250, 210, 220]], ["Prusa CORE One", [250, 220, 270]], ["Prusa MINI", [180, 180, 180]], ["Custom", null]];
const NOZZLES = [0.25, 0.4, 0.6, 0.8];
const UP_KEYS = ["x", "-x", "y", "-y", "z", "-z"];
const upLabel = (u) => (u.startsWith("-") ? u : "+" + u).toUpperCase();

/** Smallest rotation taking the model direction `up` to the printer's +Z (never a mirror), row-major. */
export function upRotation(up) {
  const sign = up.startsWith("-") ? -1 : 1, u = [0, 0, 0];
  u["xyz".indexOf(up.replace(/^[+-]/, ""))] = sign;
  const c = u[2];
  if (c < -1 + 1e-9) return [1, 0, 0, 0, -1, 0, 0, 0, -1];
  const v = [u[1], -u[0], 0]; // u x z
  const K = [0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0];
  const KK = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) KK[3 * i + j] += K[3 * i + k] * K[3 * k + j];
  return [0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (i % 4 === 0 ? 1 : 0) + K[i] + KK[i] / (1 + c));
}

export function toPrint(V, up, scale = 1) {
  const R = upRotation(up), out = new Float32Array(V.length);
  for (let i = 0; i < V.length; i += 3) {
    const x = V[i], y = V[i + 1], z = V[i + 2];
    out[i] = (R[0] * x + R[1] * y + R[2] * z) * scale;
    out[i + 1] = (R[3] * x + R[4] * y + R[5] * z) * scale;
    out[i + 2] = (R[6] * x + R[7] * y + R[8] * z) * scale;
  }
  return out;
}

export class PrintPanel {
  constructor(app) {
    this.app = app;
    this.printer = app.settings.printer ?? 0;
    this.custom = app.settings.customBed || [220, 220, 250];
    this.nozzle = 1;
    this.up = "y";
    this.sizeMM = 100;
    this.showing = false;
    this.result = null;
    this.busy = false;
  }

  reset() {
    this.result = null;
    if (this.showing) this.showForm();
    this.defaultsForShape();
  }

  defaultsForShape() {
    const d = this.app.design;
    if (VS.active(d)) { this.up = "-y"; return; }
    const m = this.app.baseMesh?.();
    this.up = m && m.hasBoundary() && d.base.shape !== "sphere_open" ? "z" : "y";
  }

  state() {
    if (this.busy) return "…";
    return this.result ? "ready" : "";
  }

  bed() {
    return PRINTERS[this.printer][1] || this.custom;
  }

  dims() {
    const disp = this.app.lastDisplay;
    if (!disp) return null;
    const P = toPrint(disp.V, this.up);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i++) { const a = i % 3; lo[a] = Math.min(lo[a], P[i]); hi[a] = Math.max(hi[a], P[i]); }
    const ext = [0, 1, 2].map((a) => hi[a] - lo[a]);
    const k = this.sizeMM / Math.max(Math.max(...ext), 1e-12);
    return ext.map((x) => x * k);
  }

  fits(dims) {
    const bed = this.bed();
    const xy = [...dims.slice(0, 2)].sort((a, b) => a - b), bxy = [...bed.slice(0, 2)].sort((a, b) => a - b);
    return xy[0] <= bxy[0] + 1e-6 && xy[1] <= bxy[1] + 1e-6 && dims[2] <= bed[2] + 1e-6;
  }

  frame() {
    const v = this.app.viewer, b = v.bounds();
    if (!b) return;
    const c = b.getCenter(b.min.clone()), ext = b.getSize(b.min.clone()).length();
    v.camera.up.set(0, 0, 1);
    v.look(c.clone().set(c.x + ext * 0.8, c.y - 1.4 * ext * 0.8, c.z + 0.9 * ext * 0.8), c);
  }

  showPrint() {
    const r = this.result;
    if (!r) return;
    const col = new Float32Array(r.V.length);
    for (let i = 0; i < r.V.length / 3; i++) {
      const red = r.thin[i];
      col[3 * i] = red ? 0.9 : 0.84; col[3 * i + 1] = red ? 0.25 : 0.84; col[3 * i + 2] = red ? 0.2 : 0.83;
    }
    this.app.viewer.setPrintMesh(r.V, r.F, col);
    this.app.viewer.setFormVisible(false);
    this.showing = true;
    this.frame();
  }

  showForm() {
    this.app.viewer.setPrintMesh(null);
    this.showing = false;
    this.app.viewer.setFormVisible(this.app.grp.pick === "off");
    this.app.viewer.setView(this.app.design.view);
  }

  /** Print model straight from a mesh that is already a clean closed solid (the vessel): exact. */
  async prepareExact() {
    const app = this.app;
    this.busy = true;
    app.renderInspector();
    if (!(await app.bakeNow())) { this.busy = false; return; }
    const r = await app.result.worker.call({ type: "meshData" });
    this.busy = false;
    if (!r.ok && r.error) { app.setError(r.error); return; }
    const { V, facePtr, faceIdx, wall, closed } = r.data;
    const D = app.design.vessel.diameter_mm;
    const m = new PolyMesh(V, facePtr, faceIdx);
    const P = toPrint(V, this.up, D / 2);
    let zmin = Infinity;
    const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) {
      zmin = Math.min(zmin, P[i + 2]);
      for (let a = 0; a < 2; a++) { lo[a] = Math.min(lo[a], P[i + a]); hi[a] = Math.max(hi[a], P[i + a]); }
    }
    for (let i = 0; i < P.length; i += 3) { P[i] -= 0.5 * (lo[0] + hi[0]); P[i + 1] -= 0.5 * (lo[1] + hi[1]); P[i + 2] -= zmin; }
    let F = new Uint32Array(m.triangles());
    let vol = 0;
    for (let t = 0; t < F.length; t += 3) {
      const a = 3 * F[t], b = 3 * F[t + 1], c = 3 * F[t + 2];
      vol += P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) + P[a + 1] * (P[b + 2] * P[c] - P[b] * P[c + 2]) + P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c]);
    }
    vol /= 6;
    if (vol < 0) {
      for (let t = 0; t < F.length; t += 3) [F[t + 1], F[t + 2]] = [F[t + 2], F[t + 1]];
      vol = -vol;
    }
    const noz = NOZZLES[this.nozzle];
    const thin = new Uint8Array(P.length / 3);
    let nThin = 0;
    if (wall) for (let i = 0; i < thin.length; i++) if (wall[i] < 2 * noz - 1e-6) { thin[i] = 1; nThin++; }
    const notes = [];
    if (!closed) notes.push("the mesh is not closed: switch the vessel off and use the voxel print model");
    const share = nThin / Math.max(thin.length, 1);
    if (share > 0.02) notes.push(`${(100 * share).toFixed(0)}% of the wall is thinner than ${(2 * noz).toFixed(1)} mm (2 nozzle widths): it may print with gaps`);
    const ext = [0, 1, 2].map((a) => { let mn = Infinity, mx = -Infinity; for (let i = a; i < P.length; i += 3) { mn = Math.min(mn, P[i]); mx = Math.max(mx, P[i]); } return mx - mn; });
    this.result = { V: P, F, thin, dims: ext, volume: vol / 1000, notes, exact: true, triangles: F.length / 3 };
    app.status(`print model ready: ${ext.map((x) => x.toFixed(0)).join(" x ")} mm`);
    this.showPrint();
    app.renderAll();
  }

  exportSTL() {
    const r = this.result;
    if (!r) return;
    const buf = stlFromTriangles(r.V, r.F);
    const info = validateStl(buf);
    const name = `${(this.app.presetName || this.app.saveName || "form").replace(/[^A-Za-z0-9_-]/g, "")}_print_${Math.round(this.app.design.vessel.diameter_mm)}mm.stl`;
    download(name, buf, "model/stl");
    this.lastExport = [name, info];
    if (info.complete && info.watertight) this.app.status(`exported ${name}: verified complete and watertight`);
    else this.app.setError(`Exported file did not verify: ${JSON.stringify(info)}`);
    this.app.renderInspector();
  }

  ui(ui) {
    const app = this.app, d = app.design;
    ui.subhead("printer");
    ui.select("printer", this.printer, PRINTERS.map(([n], i) => [i, n]), (i) => { this.printer = i; app.remember("printer", i); app.renderInspector(); }, "Sets the build volume the size is checked against.");
    const b = this.bed();
    if (PRINTERS[this.printer][1]) ui.value("bed", `${b[0]} x ${b[1]} x ${b[2]} mm`, "dim");
    else ui.vec3("W x D x H", this.custom, (v) => { this.custom = v.map((x) => Math.max(20, x)); app.remember("customBed", this.custom); app.renderInspector(); }, "Build volume in mm.");
    ui.choice("nozzle", NOZZLES.map((n, i) => [i, String(n)]), this.nozzle, (i) => { this.nozzle = i; app.renderInspector(); }, "Nozzle diameter in mm.");

    if (VS.active(d)) {
      ui.explain("The vessel is already one clean, closed solid, so it is exported exactly (no voxel remesh) and the outside stays a perfect sphere.");
      ui.subhead("size");
      ui.slider("diameter", d.vessel.diameter_mm, 30, 300, { fmt: "%.0f mm", help: "Outer diameter of the sphere. The walls stay as set in mm.", onInput: (v) => { d.vessel.diameter_mm = v; app.edited(); }, onChange: (v) => { d.vessel.diameter_mm = v; app.edited(true); } });
      ui.inline(() => {
        for (const mm of [80, 100, 120, 150, 200]) ui.toggle(String(mm), Math.abs(d.vessel.diameter_mm - mm) < 1e-6, () => { d.vessel.diameter_mm = mm; app.edited(true); }, `${mm} mm`);
        ui.button("fit bed", () => {
          const bb = this.bed().map((x) => x - 4), h = 0.5 * (1 + Math.cos(((d.base.opening ?? 40) * Math.PI) / 180));
          d.vessel.diameter_mm = Math.floor(Math.min(bb[0], bb[1], bb[2] / h));
          app.edited(true);
        }, { help: "The largest size that fits the printer bed." });
      });
      const D = d.vessel.diameter_mm, H = 0.5 * D * (1 + Math.cos(((d.base.opening ?? 40) * Math.PI) / 180));
      const ok = this.fits([D, D, H]);
      ui.value("print size", `${D.toFixed(0)} x ${D.toFixed(0)} x ${H.toFixed(0)} mm`, ok ? "hi" : "warn", "W x D x H, on the rim.");
      if (!ok) ui.warn("Bigger than the bed.");
      ui.subhead("orientation");
      ui.choice("up", UP_KEYS.map((k) => [k, upLabel(k)]), this.up, (u) => { this.up = u; app.renderInspector(); }, "Which model axis points up on the printer.");
      if (this.up === "-y") ui.note("Upside down: the opening's rim stands on the plate, the dome prints on top.");
      ui.gap(8);
      if (this.busy) ui.warn("baking the full depth …");
      else ui.button("prepare print model", () => this.prepareExact(), { primary: true, help: "Bakes the full depth first, then builds the exact print model." });
    } else {
      ui.explain("Rebuilds the form as closed solids a slicer can read: self-intersections merged, open skins thickened, sized in mm. Cutting it into parts lets each print with far less support.");
      ui.subhead("size");
      ui.number("longest side", this.sizeMM, (v) => { this.sizeMM = Math.min(Math.max(v, 5), 2000); app.renderInspector(); }, { step: 1, unit: "mm" });
      ui.choice("up", UP_KEYS.map((k) => [k, upLabel(k)]), this.up, (u) => { this.up = u; app.renderInspector(); }, "Which model axis points up on the printer.");
      const dims = this.dims();
      if (dims) {
        const ok = this.fits(dims);
        ui.value("print size", dims.map((x) => x.toFixed(1)).join(" x ") + " mm", ok ? "hi" : "warn", "W x D x H");
        if (!ok) ui.warn(`Bigger than the ${b[0]} x ${b[1]} x ${b[2]} bed.`);
      }
      ui.gap(6);
      ui.empty("The watertight voxel print model (self-intersections merged, open skins thickened, cuts into parts with pin holes, support check) is being ported to the web and arrives in the next update. Until then: 10 export gives the raw mesh, or use the desktop app's print section with a preset downloaded from 00 presets.");
    }

    const r = this.result;
    if (r) {
      ui.subhead("print model");
      ui.card(() => {
        ui.spaced("READY", "dim");
        ui.add(Object.assign(document.createElement("div"), { className: "medium hi", textContent: r.dims.map((x) => x.toFixed(1)).join(" x ") + " mm" }));
        ui.small(`${r.triangles.toLocaleString()} triangles · ${r.volume.toFixed(1)} cm3 · ~${(r.volume * 1.24).toFixed(0)} g PLA (all wall) · exact mesh`, "dim");
      });
      for (const n of r.notes) ui.warn(n);
      if (!r.notes.length) ui.ok("Thickness check passed.");
      ui.inline(() => {
        if (this.showing) ui.button("show form", () => this.showForm(), { help: "Back to the subdivision form." });
        else ui.button("show print model", () => this.showPrint());
        ui.button("export print STL", () => this.exportSTL(), { primary: true, help: "Downloads the STL, verified complete and watertight." });
      });
      if (this.lastExport) {
        const [name, info] = this.lastExport;
        ui.small(`${name}: ${info.triangles.toLocaleString()} triangles, ${info.complete && info.watertight ? "verified complete and watertight" : "NOT valid"}`, info.complete && info.watertight ? "dim" : "warn");
      }
    }
  }
}

// SubdivisionEngine_BDMH on the web: the app shell (sidebar, inspector, toolbar, bottom bar), the design
// state with undo / redo, and the two engine workers (live preview, background bake).

import { UI, installTooltips, download, fmt } from "./ui.js";
import { Viewer, VIEWS, cmapCss } from "./viewer.js";
import { normalizeDesign, toDict, presetText, copyDesign } from "../engine/design.js";
import { CC_WEIGHTS, DS_WEIGHTS, MAX_ITERATIONS, ALL_WEIGHTS, normalizeIteration } from "../engine/weights.js";
import { SHAPES, defaultSpec, makeBase } from "../engine/shapes.js";
import * as A from "../engine/attractors.js";
import * as L from "../engine/layers.js";
import * as I from "../engine/intrinsic.js";
import * as FN from "../engine/functions.js";
import * as VS from "../engine/vessel.js";
import { stableStringify, hashString } from "../engine/util.js";
import { PrintPanel } from "./print.js";

const TITLE = "SubdivisionEngine_BDMH";
const DEFAULT_PRESET = "cube_six_arms";
const AUTO_BAKE_DELAY = 1500;
const SHARP = { w1: -1.0, w2: -2.0 };
const LS = "bdmh.";

const SECTIONS = [
  ["FORM", "presets", "Presets", "Start from a recipe. Presets are JSON files, the same format as the desktop app."],
  ["FORM", "base mesh", "Base mesh", "The coarse input the subdivision starts from."],
  ["FORM", "schedule", "Iteration schedule", "The weights of every subdivision step: where the form extrudes, bulges and creases."],
  ["GROWTH", "attractors", "Attractors", "Points and curves that change the weights near them."],
  ["GROWTH", "layers", "Function layers", "Mathematical fields and folds that drive weights or move the surface."],
  ["GROWTH", "groups", "Groups", "Tag faces or vertices of the input mesh, then lock them or give them rules."],
  ["GROWTH", "intrinsic", "Intrinsic rules", "Weights read from the mesh itself: vertex motifs and measures."],
  ["GROWTH", "porosity", "Porosity", "Weld parts of the surface that grow into contact, opening holes."],
  ["MAKE", "vessel", "Vessel", "A lithophane: smooth sphere outside, the relief inside, glowing where thin."],
  ["MAKE", "print", "Print", "A watertight STL in mm for FDM: size, orientation, cuts into parts, support."],
  ["MAKE", "export", "Export & render", "Mesh files, screenshots and turntable animations."],
];
const [PRESETS, BASE, SCHEDULE, ATTRACTORS, LAYERS, GROUPS, INTRINSIC, POROSITY, VESSEL, PRINT, EXPORT] = SECTIONS.map((_, i) => i);
const FAMILIES = [["cube", "CUBE"], ["column", "COLUMN"], ["panel", "PANEL"], ["solid", "SOLID"], ["cage", "CAGE"], ["vessel", "VESSEL"]];
const VIEW_LIST = [["diagonal", "diag"], ["front", "front"], ["top", "top"], ["three_quarter", "3/4"]];
const WEIGHT_SHORT = { w_f: "w_f  face", w_e: "w_e  edge", w_c: "w_c  corner", w1: "w1   edge bias", w2: "w2   corner bias", w3: "w3   V/F bias", w4: "w4   diag bias", w6: "w6   motif face", w7: "w7   motif edge" };
const COLOUR_MODES = [["none", "plain"], ["influence", "attractor influence"], ["layer", "selected layer field"], ["tags", "group tags"], ["light", "light through wall"],
  ...Object.entries(I.MEASURES).map(([k, v]) => [`measure:${k}`, v])];
const WEIGHT_OPTS = ALL_WEIGHTS.map((w) => [w.name, w.label.split(/\s+/).join(" ")]);

const store = {
  get(k, d) { try { const v = localStorage.getItem(LS + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(LS + k, JSON.stringify(v)); } catch { /* private mode / full */ } },
};

const presetFamily = (name) => {
  const prefix = name.split("_")[0];
  const f = FAMILIES.find(([k]) => k === prefix);
  return f ? f[1] : "SAVED";
};

class History {
  constructor(limit = 120) { this.stack = []; this.pos = -1; this.limit = limit; }
  push(s) {
    if (this.pos >= 0 && this.stack[this.pos] === s) return false;
    this.stack.splice(this.pos + 1);
    this.stack.push(s);
    if (this.stack.length > this.limit) this.stack.shift();
    this.pos = this.stack.length - 1;
    return true;
  }
  canUndo() { return this.pos > 0; }
  canRedo() { return this.pos < this.stack.length - 1; }
  undo() { return this.canUndo() ? this.stack[--this.pos] : null; }
  redo() { return this.canRedo() ? this.stack[++this.pos] : null; }
}

// ======================================================================== engine workers
class EngineWorker {
  constructor(app, name) {
    this.app = app;
    this.name = name;
    this.seq = 0;
    this.pending = new Map();
    this.start();
  }
  start() {
    this.w = new Worker(new URL("../workers/engine.js", import.meta.url), { type: "module" });
    this.w.onmessage = (e) => {
      const m = e.data;
      const p = this.pending.get(m.id);
      if (m.type === "plugins") return;
      if (p) { this.pending.delete(m.id); p(m); }
    };
    this.w.onerror = (e) => this.app.setError(`${this.name} worker: ${e.message || "failed"}`);
    this.busy = false;
    this.job = null;
    this.sync();
  }
  sync() {
    this.w.postMessage({ type: "plugins", list: this.app.pluginList() });
    this.w.postMessage({ type: "files", files: this.app.objFiles });
  }
  restart() {
    this.w.terminate();
    for (const p of this.pending.values()) p({ ok: false, cancelled: true });
    this.pending.clear();
    this.start();
  }
  call(msg, transfer = []) {
    const id = ++this.seq;
    return new Promise((res) => {
      this.pending.set(id, res);
      this.w.postMessage({ ...msg, id }, transfer);
    });
  }
}

// ======================================================================== the app
class App {
  constructor() {
    this.settings = store.get("settings", {});
    this.theme = this.settings.theme || "dark";
    this.help = !!this.settings.help;
    this.section = (this.settings.section ?? PRESETS) % SECTIONS.length;
    const m = /^#(\d\d)/.exec(location.hash); // e.g. #03 opens the attractors section
    if (m) this.section = Math.min(+m[1], SECTIONS.length - 1);
    this.objFiles = store.get("objFiles", {});
    this.userPlugins = store.get("plugins", []);
    this.bundledPlugins = [];
    this.savedPresets = store.get("presets", {});
    this.bundled = {};
    this.history = new History();
    this.design = normalizeDesign({});
    this.presetName = null;
    this.loadedSnap = null;
    this.modified = false;
    this.saveName = "my_form_edit";
    this.log = [];
    this.error = "";
    this.result = null; // { summary, display, depth, baked, worker }
    this.baked = false;
    this.dirty = true;
    this.lastEdit = performance.now();
    this.autoBake = true;
    this.wantBake = false;
    this.faceBudget = 2_000_000;
    this.showEdges = false;
    this.colourMode = "none";
    this.needViewReset = true;
    this.busyDrag = 0;
    this.scheduleTab = 0;
    this.panelsHidden = false;
    this.cut = { on: false, axis: 2, pos: 0.5, flip: false };
    this.att = { sel: -1, ctrl: 0, iterTab: 0, show: true };
    this.lay = { sel: -1 };
    this.grp = { sel: -1, pick: "off", normalDir: "+y", normalAngle: 30, bandAxis: 1, band: [0, 0.2], kth: 2, motif: 0 };
    this.tt = { frames: 72, seconds: 6, size: 640, elevation: 20 };
    this.exportFmt = "obj";
    this.baseCache = { key: null, mesh: null, error: null };
  }

  // ---------------------------------------------------------------- start
  async init() {
    document.documentElement.dataset.theme = this.theme;
    document.body.classList.toggle("help", this.help);
    installTooltips();
    this.buildShell();
    this.viewer = new Viewer(this.$view, {
      onPick: (hit) => this.pick(hit),
      onGizmo: (pos) => this.gizmoMoved(pos),
      onGizmoEnd: () => { this.snapDue = true; this.renderInspector(); },
    });
    this.viewer.setTheme(this.theme);
    await Promise.all([this.loadBundledPresets(), this.loadBundledPlugins(), this.loadBundledInputs()]);
    FN.loadPlugins(this.pluginList());
    this.preview = new EngineWorker(this, "preview");
    this.baker = new EngineWorker(this, "bake");
    this.print = new PrintPanel(this);
    const last = store.get("lastSession", null);
    const want = new URLSearchParams(location.search).get("preset");
    if (want && this.presetData(want)) this.loadPreset(want);
    else if (last && last.data) this.restoreLastSession(true);
    else this.loadPreset(this.bundled[DEFAULT_PRESET] ? DEFAULT_PRESET : Object.keys(this.bundled)[0]);
    this.installKeys();
    setInterval(() => this.tick(), 40);
    this.renderAll();
  }

  async loadBundledPresets() {
    try {
      const names = await (await fetch("presets/index.json")).json();
      const all = await Promise.all(names.map(async (n) => [n, await (await fetch(`presets/${n}.json`)).json()]));
      for (const [n, d] of all) this.bundled[n] = d;
    } catch (e) {
      this.setError(`Could not load the presets: ${e.message}`);
    }
  }

  async loadBundledPlugins() {
    try {
      const names = await (await fetch("functions/index.json")).json();
      this.bundledPlugins = await Promise.all(names.map(async (n) => ({ name: n, source: await (await fetch(`functions/${n}`)).text() })));
    } catch { this.bundledPlugins = []; }
  }

  async loadBundledInputs() {
    for (const p of ["inputs/l_block.obj"]) {
      if (this.objFiles[p]) continue;
      try { this.objFiles[p] = await (await fetch(p)).text(); } catch { /* optional */ }
    }
  }

  pluginList() {
    return [...this.bundledPlugins, ...this.userPlugins];
  }

  presetNames() {
    return [...Object.keys(this.bundled), ...Object.keys(this.savedPresets).filter((n) => !(n in this.bundled))].sort();
  }

  presetData(name) {
    return this.savedPresets[name] || this.bundled[name] || null;
  }

  // ---------------------------------------------------------------- messages
  status(msg) {
    if (!msg) return;
    const t = new Date().toTimeString().slice(0, 8);
    this.log.push([t, msg]);
    if (this.log.length > 40) this.log.shift();
    this.renderStatus();
  }

  setError(msg) {
    if (msg && msg !== this.error) this.status("! " + msg);
    this.error = msg || "";
    this.renderStatus();
  }

  remember(k, v) {
    this.settings[k] = v;
    store.set("settings", this.settings);
  }

  // ---------------------------------------------------------------- design
  snapshot(d = this.design) {
    const x = toDict(d);
    delete x.view;
    return stableStringify(x);
  }

  install(design) {
    const view = this.design.view;
    this.design = design;
    this.att.sel = design.attractors.length ? 0 : -1;
    this.att.ctrl = 0;
    this.lay.sel = design.layers.length ? 0 : -1;
    this.grp.sel = design.groups.length ? 0 : -1;
    this.setPick("off");
    this.scheduleTab = 0;
    this.print?.reset();
    this.edited();
    return view;
  }

  loadPreset(name) {
    const data = this.presetData(name);
    if (!data) { this.install(normalizeDesign({})); return; }
    let d;
    try { d = normalizeDesign(data); } catch (e) { this.setError(`Could not load ${name}: ${e.message}`); return; }
    this.install(d);
    this.needViewReset = true;
    this.presetName = name;
    this.loadedSnap = this.snapshot();
    this.modified = false;
    this.saveName = name.endsWith("_edit") ? name : name + "_edit";
    this.remember("preset", name);
    this.status(`loaded ${name}`);
    this.renderAll();
  }

  restoreLastSession(startup = false) {
    const last = store.get("lastSession", null);
    if (!last) return;
    let d;
    try { d = normalizeDesign(last.data); } catch (e) { this.setError(`Could not restore the last session: ${e.message}`); return; }
    this.install(d);
    this.needViewReset = true;
    this.presetName = last.preset || null;
    const src = this.presetName && this.presetData(this.presetName);
    this.loadedSnap = src ? this.snapshot(normalizeDesign(src)) : null;
    this.modified = this.loadedSnap !== null && this.loadedSnap !== this.snapshot();
    this.saveName = (this.presetName || "my_form").replace(/_edit$/, "") + "_edit";
    const when = new Date(last.time).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    this.status(`${startup ? "reopened" : "restored"} the last session (saved ${when})`);
    this.renderAll();
  }

  autosave() {
    const data = toDict(this.design);
    const text = JSON.stringify(data);
    if (text === this._autosaved) return;
    store.set("lastSession", { data, preset: this.presetName, time: Date.now() });
    this._autosaved = text;
  }

  savePreset() {
    let name = this.saveName.trim().replace(/[^A-Za-z0-9_-]/g, "").replace(/^_+/, "") || "untitled";
    if (this.design.name === "" || this.design.name === "Untitled") this.design.name = name;
    this.savedPresets[name] = toDict(this.design);
    store.set("presets", this.savedPresets);
    this.presetName = name;
    this.loadedSnap = this.snapshot();
    this.modified = false;
    this.remember("preset", name);
    this.status(`saved ${name} in this browser (download it to keep a file)`);
    this.renderAll();
  }

  edited(rerender = false) {
    this.dirty = true;
    this.baked = false;
    this.lastEdit = performance.now();
    this.snapDue = true;
    if (this.baker && this.baker.busy) { this.baker.restart(); }
    if (rerender) this.renderInspector();
    this.renderNav();
  }

  historyTick() {
    if (!this.snapDue || this.busyDrag || this.viewer?.gizmoDragging) return;
    this.snapDue = false;
    const s = this.snapshot();
    this.history.push(s);
    const m = this.loadedSnap !== null && s !== this.loadedSnap;
    if (m !== this.modified) { this.modified = m; this.renderNav(); }
    this.renderStatus();
  }

  undo(redo = false) {
    const s = redo ? this.history.redo() : this.history.undo();
    if (s === null) return;
    const view = this.install(normalizeDesign(JSON.parse(s)));
    this.design.view = view;
    this.snapDue = false;
    this.modified = this.loadedSnap !== null && s !== this.loadedSnap;
    this.status(redo ? "redo" : "undo");
    this.renderAll();
  }

  // ---------------------------------------------------------------- compute
  baseMesh() {
    const key = stableStringify({ base: this.design.base, obj: this.design.base.shape === "obj" ? hashString(this.objFiles[this.design.base.path] || "") : null });
    if (this.baseCache.key !== key) {
      this.baseCache.key = key;
      try {
        this.baseCache.mesh = makeBase(this.design.base, { objText: (p) => this.objFiles[p] });
        this.baseCache.error = null;
      } catch (e) {
        this.baseCache.mesh = null;
        this.baseCache.error = e.message;
      }
    }
    return this.baseCache.mesh;
  }

  colourRequest() {
    return { mode: this.colourMode, sel: this.att.sel, layer: this.lay.sel, group: this.grp.sel };
  }

  tick() {
    this.historyTick();
    if (this.dirty && !this.preview.busy) {
      this.dirty = false;
      this.runPreview();
    }
    const d = this.design;
    if (!this.dirty && !this.preview.busy && !this.baker.busy && !this.baked && !this.error && this.result) {
      const idle = performance.now() - this.lastEdit > AUTO_BAKE_DELAY && !this.busyDrag;
      if (this.wantBake || (this.autoBake && d.full_depth > d.preview_depth && idle)) {
        this.wantBake = false;
        this.runBake();
      }
    }
    if (this.preview.busy || this.baker.busy) this.renderStatus();
  }

  async runPreview() {
    const w = this.preview;
    w.busy = true;
    w.started = performance.now();
    const design = toDict(this.design);
    const depth = this.design.preview_depth;
    const r = await w.call({ type: "run", design, depth, faceBudget: this.faceBudget, colour: this.colourRequest(), wantEdges: this.showEdges });
    w.busy = false;
    if (r.cancelled) return;
    if (!r.ok) { this.setError(r.error); return; }
    if (this.dirty && performance.now() - w.started < 30) return; // superseded already: the next run follows
    this.show(r, w, depth, depth >= this.design.full_depth);
    if (depth >= this.design.full_depth && !this.dirty) { this.baked = true; this.autosave(); }
  }

  async runBake() {
    const w = this.baker;
    w.busy = true;
    w.started = performance.now();
    const snap = this.snapshot();
    const depth = this.design.full_depth;
    const r = await w.call({ type: "run", design: toDict(this.design), depth, faceBudget: this.faceBudget, colour: this.colourRequest(), wantEdges: this.showEdges });
    w.busy = false;
    if (r.cancelled) return;
    if (!r.ok) { this.setError(r.error); return; }
    if (snap !== this.snapshot() || this.dirty) return;
    this.show(r, w, depth, true);
    this.baked = true;
    this.autosave();
    this.renderAll();
  }

  /** Bake now and wait for it (export and print need the full depth). */
  async bakeNow() {
    if (this.baked && this.result) return true;
    while (this.preview.busy) await new Promise((r) => setTimeout(r, 30));
    if (this.baker.busy) this.baker.restart();
    this.dirty = false;
    await this.runBake();
    return this.baked;
  }

  show(r, worker, depth, baked) {
    this.error = "";
    this.result = { summary: r.summary, display: { topo: r.display.topo }, depth, baked, worker };
    this.lastDisplay = r.display;
    this.viewer.showMesh(r.display, r.colour, this.colourMode === "light" ? "inferno" : "viridis");
    if (this.showEdges && r.display.E) this.viewer.setEdges(r.display.E);
    else if (!this.showEdges) this.viewer.setEdges(null);
    this.viewer.setFormVisible(this.grp.pick === "off" && !this.print.showing);
    if (this.needViewReset) {
      this.viewer.setView(this.design.view);
      this.needViewReset = false;
    }
    if (this.cut.on) this.applyCut();
    if (!r.summary.finite) this.setError("Non-finite vertices: reduce the weights");
    this.updateAttractorOverlay();
    this.renderStatus();
    this.renderNav();
    if (!this.busyDrag && !this.inputFocused()) this.renderInspector(true);
  }

  async refreshColours() {
    if (!this.result) return;
    const r = await this.result.worker.call({ type: "colour", design: toDict(this.design), colour: this.colourRequest() });
    this.viewer.setColours(r.colour || null, this.colourMode === "light" ? "inferno" : "viridis");
  }

  setColourMode(mode) {
    this.colourMode = mode;
    this.refreshColours();
    this.renderToolbar();
    this.renderInspector();
  }

  async setWire(on) {
    this.showEdges = on;
    if (on && this.result) {
      const r = await this.result.worker.call({ type: "edges" });
      if (r.E) this.viewer.setEdges(r.E);
    } else this.viewer.setEdges(null);
    this.renderToolbar();
  }

  setView(view) {
    this.design.view = view;
    if (this.print.showing) this.print.frame();
    else this.viewer.setView(view);
    this.renderToolbar();
  }

  setTheme(name) {
    this.theme = name;
    document.documentElement.dataset.theme = name;
    this.viewer.setTheme(name);
    this.remember("theme", name);
    this.updateAttractorOverlay();
    this.renderAll();
  }

  setSection(i) {
    i = ((i % SECTIONS.length) + SECTIONS.length) % SECTIONS.length;
    if (i === this.section) return;
    if (this.section === GROUPS && this.grp.pick !== "off") this.setPick("off");
    this.section = i;
    this.remember("section", i);
    this.$inspector.scrollTop = 0;
    this.updateAttractorOverlay();
    this.renderAll();
  }

  inputFocused() {
    const a = document.activeElement;
    return a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.tagName === "SELECT") && this.$inspector.contains(a);
  }

  // ---------------------------------------------------------------- shell
  buildShell() {
    const app = document.getElementById("app");
    app.innerHTML = `
      <nav id="nav"></nav>
      <main id="stage">
        <div id="view"></div>
        <header id="toolbar"></header>
        <footer id="statusbar"></footer>
        <div id="hint">tab · show panels</div>
      </main>
      <aside id="inspector"></aside>`;
    this.$nav = app.querySelector("#nav");
    this.$view = app.querySelector("#view");
    this.$toolbar = app.querySelector("#toolbar");
    this.$status = app.querySelector("#statusbar");
    this.$inspector = app.querySelector("#inspector");
    this.ui = (root) => new UI(root, { busy: (on) => { this.busyDrag += on ? 1 : -1; if (!on) setTimeout(() => this.renderInspector(true), 0); } });
  }

  renderAll() {
    this.renderNav();
    this.renderToolbar();
    this.renderInspector();
    this.renderStatus();
  }

  sectionState(i) {
    const d = this.design;
    if (i === PRESETS) return [(this.presetName || "unsaved") + (this.modified ? "*" : ""), false];
    if (i === BASE) return [String(d.base.shape || "cube").replace(/_/g, " "), false];
    if (i === SCHEDULE) return [`d${d.preview_depth}/${d.full_depth}`, false];
    const counts = {
      [ATTRACTORS]: d.attractors.filter((a) => a.enabled).length, [LAYERS]: d.layers.filter((l) => l.enabled).length,
      [GROUPS]: d.groups.filter((g) => g.enabled).length,
      [INTRINSIC]: Object.values(d.motifs).filter((v) => v).length + d.intrinsic.filter((r) => r.enabled).length,
    };
    if (i in counts) return [counts[i] ? String(counts[i]) : "—", counts[i] === 0];
    if (i === POROSITY) return d.merge.enabled ? ["on", false] : ["off", true];
    if (i === VESSEL) return VS.active(d) ? [`${d.vessel.diameter_mm.toFixed(0)} mm`, false] : [d.base.shape === "sphere_open" ? "off" : "—", true];
    if (i === PRINT) return [this.print ? this.print.state() : "", false];
    return ["", false];
  }

  renderNav() {
    if (!this.$nav) return;
    this.$nav.innerHTML = "";
    const ui = this.ui(this.$nav);
    const head = ui.group("navhead", () => {
      ui.add(Object.assign(document.createElement("div"), { className: "name", textContent: "Nijat Mahamaliyev" }));
      ui.small("Subdivision as a generative system", "dim");
      ui.small("HTMAA 2026 · week 03", "dim");
    });
    void head;
    let group = null;
    SECTIONS.forEach(([grp, name, title, intro], i) => {
      if (grp !== group) {
        group = grp;
        ui.gap(18);
        ui.spaced(grp);
        ui.gap(4);
      }
      const [state, faint] = this.sectionState(i);
      ui.listRow(name, state, { num: String(i).padStart(2, "0"), selected: i === this.section, faint, help: `${title}: ${intro}\n[ / ] previous / next section`, onClick: () => this.setSection(i) });
    });
    ui.group("navfoot", () => {
      ui.small("after Hansmeyer (2010)", "faint");
      const a = document.createElement("a");
      a.href = "https://nijatmahamaliyev.com";
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = "nijatmahamaliyev.com";
      a.className = "small faint";
      ui.add(a);
    });
  }

  renderToolbar() {
    if (!this.$toolbar) return;
    this.$toolbar.innerHTML = "";
    const ui = this.ui(this.$toolbar);
    ui.group("tbline", () => {
      ui.add(Object.assign(document.createElement("div"), { className: "prompt", innerHTML: `<b>root@nijat:~$</b> ./${TITLE}<span class="cursor"></span>` }));
      ui.group("tbright", () => {
        ui.toggle("?", this.help, () => { this.help = !this.help; document.body.classList.toggle("help", this.help); this.remember("help", this.help); this.renderToolbar(); }, "Show the longer explanations inline (H).");
        ui.toggle(this.theme === "dark" ? "light" : "dark", false, () => this.setTheme(this.theme === "dark" ? "light" : "dark"), "Switch between the dark and the light palette.");
        ui.toggle("shadow", this.viewer?.shadows, () => { this.viewer.setShadows(!this.viewer.shadows); this.renderToolbar(); }, "The soft ground shadow under the form.");
      });
    });
    ui.group("tbline", () => {
      ui.group("seg joined", () => {
        VIEW_LIST.forEach(([view, label], i) => ui.toggle(label, this.design.view === view, () => this.setView(view),
          `${view.replace("_", " ")} view (${i + 1}). F frames the form again; double-click a point of the form to orbit and zoom around it.`));
      });
      ui.check("wire", this.showEdges, (v) => this.setWire(v), "Show the mesh edges (W).");
      ui.group("colourby", () => {
        ui.add(Object.assign(document.createElement("span"), { className: "dim", textContent: "colour" }));
        const s = ui.selectEl(this.colourMode, COLOUR_MODES, (m) => this.setColourMode(m));
        s.dataset.tip = "Colour the form by a field or a measure, to see where things act.";
        ui.add(s);
        if (this.colourMode !== "none") {
          const g = document.createElement("span");
          g.className = "legend";
          g.style.background = cmapCss(this.colourMode === "light" ? "inferno" : "viridis");
          ui.add(g);
        }
      });
      ui.group("cutbox", () => {
        ui.toggle(this.cut.on ? `cut ${"xyz"[this.cut.axis]} ${Math.round(100 * this.cut.pos)}%` : "cut", this.cut.on, () => { this.cut.on = !this.cut.on; this.applyCut(); this.renderToolbar(); },
          "Section cut: hide one side of a plane to see inside (also cuts screenshots and turntables). Click … for settings.");
        ui.button("…", () => { this.cutOpen = !this.cutOpen; this.renderToolbar(); }, { help: "Section settings" });
        if (this.cutOpen) {
          ui.group("popover", () => {
            ui.spaced("SECTION CUT", "dim");
            ui.check("cut the view open", this.cut.on, (v) => { this.cut.on = v; this.applyCut(); this.renderToolbar(); });
            ui.choice("plane across", [[0, "X"], [1, "Y"], [2, "Z"]], this.cut.axis, (a) => { this.cut.axis = a; this.cut.on = true; this.applyCut(); this.renderToolbar(); });
            ui.slider("position", 100 * this.cut.pos, 0, 100, { fmt: "%.0f %%", onInput: (v) => { this.cut.pos = v / 100; this.applyCut(); }, onChange: (v) => { this.cut.pos = v / 100; this.applyCut(); this.renderToolbar(); } });
            ui.inline(() => {
              ui.button("flip side", () => { this.cut.flip = !this.cut.flip; this.applyCut(); }, { help: "Keep the other half." });
              ui.button("face the cut", () => this.faceTheCut(), { help: "Point the camera straight at the cut face." });
              ui.button("close", () => { this.cutOpen = false; this.renderToolbar(); });
            });
          });
        }
      });
    });
  }

  applyCut() {
    this.viewer.setCut(this.cut.on, this.cut.axis, this.cut.pos, this.cut.flip);
  }

  faceTheCut() {
    this.cut.on = true;
    this.applyCut();
    const b = this.viewer.bounds();
    if (!b) return;
    const c = b.getCenter(b.min.clone()), size = b.getSize(b.min.clone()).length();
    const a = this.cut.axis;
    c.setComponent(a, b.min.getComponent(a) + this.cut.pos * (b.max.getComponent(a) - b.min.getComponent(a)));
    const n = [0, 0, 0];
    n[a] = this.cut.flip ? 1 : -1;
    const up = [0, 0, 0];
    up[a !== 1 ? 1 : 2] = 1;
    this.viewer.camera.up.set(...(a !== 1 ? [0, 1, 0] : [0, 0, -1]));
    const eye = c.clone();
    eye.x -= n[0] * 1.6 * size - up[0] * 0.25 * size;
    eye.y -= n[1] * 1.6 * size - up[1] * 0.25 * size;
    eye.z -= n[2] * 1.6 * size - up[2] * 0.25 * size;
    this.viewer.look(eye, c);
    this.renderToolbar();
  }

  renderStatus() {
    if (!this.$status) return;
    this.$status.innerHTML = "";
    const ui = this.ui(this.$status);
    const d = this.design;
    ui.group("stline", () => {
      ui.button("‹ undo", () => this.undo(), { help: "Undo the last design change (Ctrl/Cmd+Z).", disabled: !this.history.canUndo() });
      ui.button("redo ›", () => this.undo(true), { help: "Redo (Ctrl/Cmd+Shift+Z).", disabled: !this.history.canRedo() });
      ui.stepper("preview", d.preview_depth, 1, MAX_ITERATIONS, (v) => { d.preview_depth = v; d.full_depth = Math.max(d.full_depth, v); this.edited(); this.renderAll(); }, "Depth recomputed live while you edit.");
      ui.stepper("full", d.full_depth, 1, MAX_ITERATIONS, (v) => { d.full_depth = v; d.preview_depth = Math.min(d.preview_depth, v); this.edited(); this.renderAll(); },
        "Depth of the baked result, used for export and print. Faces x4 per Catmull-Clark step.");
      ui.check("auto-bake", this.autoBake, (v) => { this.autoBake = v; this.renderStatus(); }, "Bake the full depth after 1.5 s without edits.");
      ui.button("bake", () => { this.wantBake = true; this.baked = false; }, { primary: true, help: "Compute the full depth now (B)." });
    });
    const s = this.result?.summary;
    ui.group("stline small", () => {
      if (s) {
        const size = VS.active(d) ? ` · ${d.vessel.diameter_mm.toFixed(0)} mm sphere` : "";
        ui.add(Object.assign(document.createElement("span"), { className: "fg", textContent: `${this.baked ? "baked" : "preview"} · depth ${s.depthReached} · ${s.nFaces.toLocaleString()} faces · ${(s.seconds * 1000).toFixed(0)} ms${size}` }));
        const job = this.baker.busy ? ["baking", this.design.full_depth, this.baker.started] : this.preview.busy && performance.now() - this.preview.started > 300 ? ["updating", this.design.preview_depth, this.preview.started] : null;
        if (job) ui.add(Object.assign(document.createElement("span"), { className: "dim", textContent: `${job[0]} depth ${job[1]} … ${((performance.now() - job[2]) / 1000).toFixed(1)} s` }));
        if (s.capped) ui.add(Object.assign(document.createElement("span"), { className: "warn", textContent: `capped at depth ${s.depthReached} by the face budget (schedule section)` }));
      } else ui.add(Object.assign(document.createElement("span"), { className: "dim", textContent: "computing…" }));
    });
    ui.group("stline small", () => {
      if (this.error) {
        const e = ui.add(Object.assign(document.createElement("span"), { className: "warn trunc", textContent: "! " + this.error }));
        e.dataset.tip = this.error;
      } else if (this.log.length) {
        const [t, m] = this.log[this.log.length - 1];
        const e = ui.add(Object.assign(document.createElement("span"), { className: "dim trunc", textContent: "> " + m }));
        e.dataset.tip = this.log.slice(-12).map(([a, b]) => `${a}  ${b}`).join("\n");
        ui.add(Object.assign(document.createElement("span"), { className: "faint", textContent: t }));
      }
    });
  }

  renderInspector(keepScroll = false) {
    if (!this.$inspector) return;
    if (this.busyDrag || (keepScroll && this.inputFocused())) return;
    const scroll = this.$inspector.scrollTop;
    this.$inspector.innerHTML = "";
    const ui = this.ui(this.$inspector);
    const [group, , title, intro] = SECTIONS[this.section];
    ui.heading(`${String(this.section).padStart(2, "0")} · ${group}`, title);
    ui.note(intro);
    try {
      [this.presetsUI, this.baseUI, this.scheduleUI, this.attractorsUI, this.layersUI, this.groupsUI, this.intrinsicUI,
        this.porosityUI, this.vesselUI, (u) => this.print.ui(u), this.exportUI][this.section].call(this, ui);
    } catch (e) {
      console.error(e);
      ui.warn(`This section failed to draw: ${e.message}`);
    }
    ui.gap(30);
    this.$inspector.scrollTop = scroll;
  }

  // ---------------------------------------------------------------- 00 presets
  presetsUI(ui) {
    const d = this.design;
    ui.card(() => {
      ui.spaced(this.modified ? "EDITED" : this.presetName ? "LOADED" : "UNSAVED", "dim");
      ui.add(Object.assign(document.createElement("div"), { className: "medium hi", textContent: (this.presetName || d.name || "untitled") + (this.modified ? "*" : "") }));
      if (d.description) ui.small(d.description, "dim");
    });
    const last = store.get("lastSession", null);
    if (last) {
      const when = new Date(last.time).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
      ui.button(`restore last session · ${when}`, () => this.restoreLastSession(), { help: "The design as it was when you last baked (it reopens automatically). Undo brings back what you have now." });
    }
    ui.subhead("save as");
    ui.inline(() => {
      const inp = document.createElement("input");
      inp.className = "txt grow";
      inp.value = this.saveName;
      inp.dataset.tip = "Name (letters, digits, - and _). Ctrl/Cmd+S saves in this browser.";
      inp.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") { this.saveName = inp.value; this.savePreset(); } });
      inp.addEventListener("input", () => { this.saveName = inp.value; });
      ui.add(inp);
      const exists = !!this.savedPresets[this.saveName.trim()];
      ui.button(exists ? "overwrite" : "save", () => this.savePreset(), { primary: !exists, help: "Saves the preset in this browser." });
    });
    ui.inline(() => {
      ui.button("download .json", () => download(`${(this.presetName || this.saveName || "form").replace(/[^A-Za-z0-9_-]/g, "")}.json`, presetText(this.design), "application/json"),
        { help: "The preset as a file: put it in the desktop app's presets/ folder, or load it here later." });
      ui.file("load .json", ".json,application/json", (name, text) => {
        try {
          const data = JSON.parse(text);
          const n = name.replace(/\.json$/i, "").replace(/[^A-Za-z0-9_-]/g, "") || "loaded";
          normalizeDesign(data);
          this.savedPresets[n] = data;
          store.set("presets", this.savedPresets);
          this.loadPreset(n);
        } catch (e) { this.setError(`Could not load ${name}: ${e.message}`); }
      }, { help: "Load a preset file (from this app or the desktop app)." });
    });
    const order = [...FAMILIES.map(([, l]) => l), "SAVED"];
    for (const fam of order) {
      const names = this.presetNames().filter((n) => (this.savedPresets[n] && !this.bundled[n] ? "SAVED" : presetFamily(n)) === fam);
      if (!names.length) continue;
      ui.gap(12);
      ui.spaced(fam);
      ui.list(() => {
        for (const name of names) {
          const short = fam !== "SAVED" && name.includes("_") ? name.split("_").slice(1).join("_") : name;
          ui.listRow(short, name === this.presetName && this.modified ? "*" : this.savedPresets[name] && !this.bundled[name] ? "saved" : "",
            { selected: name === this.presetName, help: `presets/${name}.json`, onClick: () => this.loadPreset(name) });
        }
      });
    }
    const mine = Object.keys(this.savedPresets).filter((n) => !this.bundled[n]);
    if (mine.length) {
      ui.gap(6);
      ui.button(`delete saved '${this.presetName}'`, () => {
        delete this.savedPresets[this.presetName];
        store.set("presets", this.savedPresets);
        this.status(`deleted ${this.presetName} from this browser`);
        this.renderInspector();
      }, { disabled: !this.presetName || !this.savedPresets[this.presetName] || !!this.bundled[this.presetName] });
    }
    ui.gap(8);
    ui.note("Loading replaces the design; undo (Ctrl/Cmd+Z) brings it back.");
  }

  // ---------------------------------------------------------------- 01 base mesh
  baseUI(ui) {
    const d = this.design;
    const keys = Object.keys(SHAPES);
    const cur = d.base.shape || "cube";
    ui.select("shape", cur, keys.map((k) => [k, SHAPES[k].label]), (name) => {
      if (name === cur) return;
      d.base = defaultSpec(name);
      if (name === "obj") d.base.path = Object.keys(this.objFiles)[0] || "";
      d.view = SHAPES[name].view;
      this.needViewReset = true;
      this.print.defaultsForShape();
      this.edited(true);
    }, SHAPES[cur]?.help);
    const shape = SHAPES[d.base.shape || "cube"];
    ui.explain(shape.help);
    for (const p of shape.params) {
      ui.slider(p.label, d.base[p.name] ?? p.default, p.lo, p.hi, {
        int: p.integer, fmt: p.integer ? "%d" : "%.2f", help: p.help || undefined,
        onInput: (v) => { d.base[p.name] = v; this.edited(); },
      });
    }
    if (shape.name === "sphere_open") this.vesselSizeUI(ui);
    if (shape.name === "obj") this.objUI(ui);
    const m = this.baseMesh();
    if (m) {
      const open = m.hasBoundary();
      ui.value("input", `${m.nVerts} verts · ${m.nFaces} faces · ${open ? "open" : "closed"}`, "dim");
      if (open) this.boundaryUI(ui);
    } else if (this.baseCache.error) ui.warn(this.baseCache.error);
  }

  vesselSizeUI(ui) {
    const d = this.design;
    ui.subhead("size");
    if (!VS.active(d)) {
      ui.note("Turn on the vessel (08) to give the sphere a size in mm.");
      ui.button("open vessel →", () => this.setSection(VESSEL));
      return;
    }
    ui.slider("diameter", d.vessel.diameter_mm, 30, 300, {
      fmt: "%.0f mm", help: "The real, printed size of the sphere (double-click to type). Walls stay as set in mm.",
      onInput: (v) => { d.vessel.diameter_mm = Math.min(Math.max(v, 20), 400); this.edited(); },
      onChange: (v) => { d.vessel.diameter_mm = Math.min(Math.max(v, 20), 400); this.edited(true); },
    });
    const D = d.vessel.diameter_mm, H = 0.5 * D * (1 + Math.cos(((d.base.opening ?? 40) * Math.PI) / 180));
    ui.value("print size", `${D.toFixed(0)} x ${D.toFixed(0)} x ${H.toFixed(0)} mm`, "hi", "Standing on its rim.");
  }

  objUI(ui) {
    const d = this.design;
    ui.subhead("obj file");
    const files = Object.keys(this.objFiles);
    if (files.length) ui.select("file", d.base.path, files.map((f) => [f, f]), (p) => { d.base.path = p; this.needViewReset = true; this.edited(true); });
    else ui.empty("No .obj files yet: load one.");
    ui.inline(() => {
      ui.file("load .obj", ".obj", (name, text) => {
        const path = `inputs/${name}`;
        this.objFiles[path] = text;
        store.set("objFiles", this.objFiles);
        this.preview.sync();
        this.baker.sync();
        d.base.path = path;
        this.needViewReset = true;
        this.edited(true);
        this.status(`loaded ${path}`);
      }, { primary: true, help: "Any polygon mesh (quads recommended). Kept in this browser." });
    });
    ui.inline(() => {
      ui.check("fit to view", d.base.normalize !== false, (v) => { d.base.normalize = v; this.edited(true); }, "Scale and centre the mesh to a unit size.");
      ui.check("weld", d.base.weld !== false, (v) => { d.base.weld = v; this.edited(true); }, "Merge duplicate vertices (needed for meshes exported as separate faces).");
    });
  }

  boundaryUI(ui) {
    const d = this.design;
    ui.subhead("open boundary");
    ui.choice("boundary", [["smooth", "smooth", "Standard Catmull-Clark rules: the edge relaxes into a B-spline curve."],
      ["locked", "locked / tile", "The boundary stays fixed and relief fades out near it, so identical tiles meet seamlessly."]],
    d.boundary, (v) => { d.boundary = v; this.edited(true); });
    if (d.boundary === "locked") {
      ui.slider("fade rows", d.fade_rows, 0, 6, { fmt: "%.1f", help: "Extrusion fades in over this many base-mesh rows from the boundary.", onInput: (v) => { d.fade_rows = v; this.edited(); } });
      if (d.iterations.slice(0, d.full_depth).some((it) => it.scheme === "ds")) ui.warn("Doo-Sabin steps shrink open boundaries: the result will not tile.");
    }
  }

  // ---------------------------------------------------------------- 02 schedule
  scheduleUI(ui) {
    const d = this.design;
    ui.choice("extrusion", [["relative", "relative", "Displacement = w x local edge length: the same w behaves the same at every depth."],
      ["absolute", "absolute", "Paper-literal: displacement = w in model units (unit normals)."]], d.extrusion, (v) => { d.extrusion = v; this.edited(true); }, "How far a weight of 1 pushes.");
    ui.subhead("iteration");
    this.scheduleTab = Math.min(this.scheduleTab, d.full_depth - 1);
    ui.segmented(Array.from({ length: d.full_depth }, (_, k) => [k, `${k + 1}${d.iterations[k].scheme === "ds" ? "ds" : ""}${k >= d.preview_depth ? "*" : ""}`]),
      this.scheduleTab, (k) => { this.scheduleTab = k; this.renderInspector(); });
    ui.small("* only in the baked (full-depth) result", "dim");
    this.iterationUI(ui, this.scheduleTab);
    ui.subhead("overview");
    const its = d.iterations.slice(0, d.full_depth);
    let shown = false;
    for (const [title, defs] of [["Catmull-Clark", CC_WEIGHTS], ["Doo-Sabin", DS_WEIGHTS]]) {
      for (const wd of defs) {
        const v = its.map((it) => ((it.scheme === "cc") === (defs === CC_WEIGHTS) ? it.weights[wd.name] : 0));
        if (!v.some((x) => x)) continue;
        shown = true;
        const label = defs === CC_WEIGHTS ? wd.name : wd.label.split(/\s+/).join(" ");
        ui.bars(label, v, Math.max(Math.max(...v.map(Math.abs)), 0.25), this.scheduleTab,
          `${title} ${wd.name} over iterations 1-${its.length}:\n` + v.map((x, k) => `${k + 1}: ${x >= 0 ? "+" : ""}${x.toFixed(2)}`).join("  "));
      }
    }
    if (!shown) ui.empty("All weights are zero: plain subdivision.");
    ui.subhead("limits");
    ui.number("face budget", this.faceBudget, (v) => { this.faceBudget = Math.max(10000, Math.round(v)); this.edited(); }, { step: 100000, help: "Stop subdividing before the mesh would pass this many faces." });
  }

  iterationUI(ui, k) {
    const d = this.design, spec = d.iterations[k];
    ui.choice("scheme", [["cc", "Catmull-Clark", "Quads: faces x4 per step (paper eq. 1-4)."], ["ds", "Doo-Sabin", "Corner-cutting: new faces from faces, edges and vertices (eq. 5-6)."]],
      spec.scheme, (v) => { spec.scheme = v; this.edited(true); });
    const defs = spec.scheme === "cc" ? CC_WEIGHTS : DS_WEIGHTS;
    for (const wd of defs) {
      ui.slider(WEIGHT_SHORT[wd.name] || wd.label.split(/\s+/).join(" "), spec.weights[wd.name], wd.lo, wd.hi, {
        help: `${wd.label.split(/\s+/).join(" ")}\n${wd.help}\nDouble-click to type a value.`,
        onInput: (v) => { spec.weights[wd.name] = v; this.edited(); },
        onChange: (v) => { spec.weights[wd.name] = v; this.edited(true); },
      });
    }
    ui.inline(() => {
      ui.button("reset", () => { for (const n of Object.keys(spec.weights)) spec.weights[n] = 0; this.edited(true); }, { help: "All weights of this iteration to 0 (standard subdivision)." });
      if (spec.scheme === "cc") ui.button("sharp", () => { Object.assign(spec.weights, SHARP); this.edited(true); }, { help: "w1 = -1, w2 = -2: switches off Catmull-Clark smoothing, so extrusions accumulate." });
      ui.button("copy → next", () => { d.iterations[k + 1] = normalizeIteration(JSON.parse(JSON.stringify(spec))); this.edited(true); }, { help: "Copy this iteration to the next one.", disabled: k + 1 >= MAX_ITERATIONS });
      ui.button("copy → all", () => { for (let j = 0; j < MAX_ITERATIONS; j++) if (j !== k) d.iterations[j] = normalizeIteration(JSON.parse(JSON.stringify(spec))); this.edited(true); }, { help: "Copy this iteration to every other one." });
    });
  }

  // ---------------------------------------------------------------- shared editors
  rulesEditor(ui, rules, onChange) {
    let remove = -1;
    rules.forEach((m, i) => {
      ui.itemHeader(`rule ${i + 1}`, { onRemove: () => { remove = i; rules.splice(i, 1); onChange(true); } });
      ui.select("weight", m.weight, WEIGHT_OPTS, (w) => { m.weight = w; onChange(true); });
      ui.choice("operation", [["scale", "scale x"], ["offset", "offset +"]], m.op, (op) => { m.op = op; m.value = op === "scale" ? 2.0 : 0.3; onChange(true); });
      const [lo, hi] = m.op === "scale" ? [0, 5] : [-1.5, 1.5];
      ui.slider("value", m.value, lo, hi, { onInput: (v) => { m.value = v; onChange(false); } });
      ui.range("iterations", m.from, m.to, 1, MAX_ITERATIONS, (a, b) => { m.from = a; m.to = b; onChange(true); });
    });
    void remove;
    if (!rules.length) ui.empty("No rules yet.");
    ui.button("+ rule", () => { rules.push({ weight: "w_f", op: "scale", value: 2.0, from: 1, to: MAX_ITERATIONS }); onChange(true); }, { help: "Scale or offset one weight over a range of iterations." });
  }

  paramsEditor(ui, fd, params, onChange) {
    for (const p of fd.params) {
      ui.slider(p.name, params[p.name] ?? p.default, p.lo, p.hi, { int: p.integer, fmt: p.integer ? "%d" : "%.3f", onInput: (v) => { params[p.name] = v; onChange(false); } });
    }
  }

  // ---------------------------------------------------------------- 03 attractors
  sceneBox() {
    const m = this.baseMesh();
    if (!m) return [[0, 0, 0], 2];
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < m.V.length; i++) { const a = i % 3; lo[a] = Math.min(lo[a], m.V[i]); hi[a] = Math.max(hi[a], m.V[i]); }
    return [[0, 1, 2].map((a) => 0.5 * (lo[a] + hi[a])), Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])];
  }

  selAtt() {
    const a = this.design.attractors;
    return this.att.sel >= 0 && this.att.sel < a.length ? a[this.att.sel] : null;
  }

  anchor(a) {
    if (a.kind === "point") return a.position.map(Number);
    const c = a.curve;
    if (c.type === "polyline" && c.points.length) {
      this.att.ctrl = Math.min(this.att.ctrl, c.points.length - 1);
      return c.points[this.att.ctrl].map(Number);
    }
    if (c.type === "line") return c.start.map((s, k) => 0.5 * (Number(s) + Number(c.end[k])));
    return c.center.map(Number);
  }

  gizmoMoved(pos) {
    const a = this.selAtt();
    if (!a) return;
    const old = this.anchor(a), delta = pos.map((x, k) => x - old[k]);
    if (a.kind === "point") a.position = pos;
    else {
      const c = a.curve;
      if (c.type === "polyline" && c.points.length) c.points[this.att.ctrl] = pos;
      else if (c.type === "line") { c.start = c.start.map((s, k) => Number(s) + delta[k]); c.end = c.end.map((s, k) => Number(s) + delta[k]); }
      else c.center = pos;
    }
    this.edited();
    this.updateAttractorOverlay(false);
  }

  updateAttractorOverlay(gizmo = true) {
    if (!this.viewer) return;
    const v = this.viewer, atts = this.design.attractors;
    const visible = this.att.show && this.section === ATTRACTORS;
    const col = (a, i) => (!a.enabled ? v.helper("off") : i === this.att.sel ? v.helper("main") : v.helper(a.payload === "set" ? "set" : "mod"));
    if (!visible || !atts.length) {
      v.setOverlay("attractors", null);
      v.showGizmo(null);
      return;
    }
    const grp = new (v.overlays.constructor)();
    const pts = atts.map((a, i) => [a, i]).filter(([a]) => a.kind === "point");
    if (pts.length) grp.add(v.pointsObject(pts.map(([a]) => a.position.map(Number)), pts.map(([a, i]) => col(a, i)), 0.012));
    atts.forEach((a, i) => {
      if (a.kind !== "curve") return;
      grp.add(v.lineObject(A.curvePoints(a.curve, 160), col(a, i)));
      if (a.curve.type === "polyline" && i === this.att.sel && a.curve.points.length) grp.add(v.pointsObject(a.curve.points.map((p) => p.map(Number)), v.helper("main"), 0.008));
    });
    v.setOverlay("attractors", grp);
    if (gizmo) {
      const a = this.selAtt();
      v.showGizmo(a ? this.anchor(a) : null);
    }
  }

  attractorsUI(ui) {
    const d = this.design;
    ui.inline(() => {
      ui.check("show in view", this.att.show, (v) => { this.att.show = v; this.updateAttractorOverlay(); this.renderInspector(); }, "Draw the attractors and the drag handle.");
      ui.check("influence map", this.colourMode === "influence", (on) => this.setColourMode(on ? "influence" : "none"), "Colour the form by the selected attractor's influence (strength x falloff).");
    });
    ui.choice("measure at", [["current", "current", "Distance from each face's current position: growth reacts to where it has moved."], ["rest", "original", "Distance from where the face sat on the input mesh: stable zoning."]],
      d.attractor_space, (v) => { d.attractor_space = v; this.edited(true); });
    ui.slider("background", d.background, 0, 3, { help: "Influence of the main schedule in the weight-set blend (paper eq. 8-9). 0 = paper-pure: only the sets count, and a single set then applies fully everywhere inside its radius. Use > 0 for a gradual effect.", onInput: (v) => { d.background = v; this.edited(); } });
    ui.explain("Weight sets carry a whole schedule of their own and are blended by distance (paper eq. 8-9, Fig. 4). Modifiers scale or offset chosen weights near them. Drag the handle in the view to move the selected attractor.");
    ui.subhead("attractors");
    const [centre, diag] = this.sceneBox();
    const r3 = (x) => Math.round(x * 1000) / 1000;
    ui.inline(() => {
      ui.button("+ point", () => this.addAttractor({ kind: "point", position: [r3(centre[0] + 0.35 * diag), r3(centre[1]), r3(centre[2])], radius: r3(0.35 * diag) }), { help: "A point attractor: influence falls off with distance from it." });
      ui.button("+ curve", () => this.addAttractor({ kind: "curve", radius: r3(0.2 * diag), curve: { type: "helix", center: centre.map(r3), radius: r3(0.25 * diag), height: r3(0.6 * diag), length: r3(0.6 * diag), size: r3(0.3 * diag) } }), { help: "A curve attractor (circle, helix, sine, line, polyline, ...)." });
    });
    ui.list(() => {
      d.attractors.forEach((a, i) => {
        const what = a.kind === "point" ? "point" : a.curve.type;
        ui.listRow(`${a.name}  ${what} · ${a.payload === "set" ? "set" : "modifier"}`, a.enabled ? "on" : "off", { selected: i === this.att.sel, faint: !a.enabled, onClick: () => { this.att.sel = i; this.att.ctrl = 0; this.updateAttractorOverlay(); if (this.colourMode === "influence") this.refreshColours(); this.renderInspector(); } });
      });
    });
    if (!d.attractors.length) { ui.empty("No attractors: the weights are the same everywhere in each iteration."); return; }
    const a = this.selAtt();
    if (a) this.attractorEditor(ui, a, diag);
  }

  addAttractor(spec) {
    const d = this.design;
    const names = new Set(d.attractors.map((a) => a.name));
    let n = d.attractors.length + 1;
    while (names.has(`A${n}`)) n++;
    const a = A.normalize({ name: `A${n}`, ...spec });
    if (!a.mods.length) a.mods = [{ weight: "w_f", op: "scale", value: 2.0, from: 1, to: MAX_ITERATIONS }];
    d.attractors.push(a);
    this.att.sel = d.attractors.length - 1;
    this.att.ctrl = 0;
    this.edited(true);
    this.updateAttractorOverlay();
  }

  attractorEditor(ui, a, diag) {
    const d = this.design;
    const changed = (re = false) => { this.edited(re); this.updateAttractorOverlay(); if (this.colourMode === "influence" && re) this.refreshColours(); };
    ui.subhead(`edit ${a.name}`);
    ui.text("name", a.name, (v) => { a.name = v; this.edited(true); });
    ui.inline(() => {
      ui.check("on", a.enabled, (v) => { a.enabled = v; changed(true); }, "Switch it off without deleting it.");
      ui.button("duplicate", () => { const dup = JSON.parse(JSON.stringify(a)); dup.name += "'"; d.attractors.splice(this.att.sel + 1, 0, dup); this.att.sel++; changed(true); });
      ui.button("delete", () => { d.attractors.splice(this.att.sel, 1); this.att.sel = Math.min(this.att.sel, d.attractors.length - 1); changed(true); });
    });
    ui.choice("kind", [["point", "point"], ["curve", "curve"]], a.kind, (k) => { a.kind = k; changed(true); });
    if (a.kind === "point") ui.vec3("position", a.position, (v) => { a.position = v; changed(true); }, "Type here, or drag the handle in the view.");
    else this.curveUI(ui, a, changed);
    ui.choice("payload", [["set", "weight set", "A full per-iteration weight schedule, blended with the others by distance (eq. 8-9)."], ["modifier", "modifier", "Scales or offsets chosen weights near the attractor."]],
      a.payload, (p) => { a.payload = p; changed(true); });
    ui.slider("strength", a.strength, 0, 5, { help: "h in eq. 8.", onInput: (v) => { a.strength = v; changed(); } });
    ui.slider("radius", a.radius, 0.01, Math.max(2 * diag, 1), { help: "Distance over which the falloff runs from 1 to its end value.", onInput: (v) => { a.radius = v; changed(); } });
    const f = a.falloff;
    ui.select("falloff", f.type, A.FALLOFFS.map((x) => [x, x]), (t) => { f.type = t; changed(true); }, "power = the paper's (1 - d)^t; spline = draw your own with the 5 bars.");
    if (f.type === "power") ui.slider("tightness t", f.tightness, 0.1, 8, { onInput: (v) => { f.tightness = v; changed(); }, onChange: (v) => { f.tightness = v; changed(true); } });
    if (f.type === "spline") ui.vsliders("shape", f.spline, (vals) => { f.spline = vals; changed(true); }, "Influence at d = 0, 0.25, 0.5, 0.75, 1 x radius.");
    ui.plot("curve", Array.from({ length: 64 }, (_, i) => A.falloff1((i / 63) * 1.25, f)), "Influence (up) against distance / radius (across).");
    if (a.payload === "set") this.setUI(ui, a, changed);
    else {
      ui.subhead("modifier rules");
      ui.note("Applied near this attractor, after the weight-set blend.");
      this.rulesEditor(ui, a.mods, (re) => changed(re));
    }
  }

  curveUI(ui, a, changed) {
    const c = a.curve;
    ui.select("curve", c.type, A.CURVE_TYPES.map((t) => [t, t]), (t) => {
      if (t === c.type) return;
      if (t === "polyline" && !c.points.length) Object.assign(c, A.toPolyline(c));
      c.type = t;
      this.att.ctrl = 0;
      changed(true);
    });
    if (c.type === "line") {
      ui.vec3("start", c.start, (v) => { c.start = v; changed(true); });
      ui.vec3("end", c.end, (v) => { c.end = v; changed(true); });
    } else if (c.type === "polyline") {
      const pts = c.points;
      if (pts.length) {
        ui.slider("control point", this.att.ctrl, 0, pts.length - 1, { int: true, onInput: (v) => { this.att.ctrl = v; this.updateAttractorOverlay(); } });
        ui.vec3("point xyz", pts[Math.min(this.att.ctrl, pts.length - 1)], (v) => { pts[this.att.ctrl] = v; changed(true); }, "Or drag the handle in the view.");
      }
      ui.inline(() => {
        ui.button("+ point after", () => {
          if (pts.length) {
            const i = this.att.ctrl;
            const nxt = i + 1 < pts.length ? pts[i + 1] : pts[i].map((x, k) => Number(x) + (k === 0 ? 0.3 : 0));
            pts.splice(i + 1, 0, pts[i].map((x, k) => 0.5 * (Number(x) + Number(nxt[k]))));
            this.att.ctrl++;
          } else pts.push([0, 0, 0]);
          changed(true);
        });
        ui.button("- remove point", () => { pts.splice(this.att.ctrl, 1); this.att.ctrl = Math.min(this.att.ctrl, pts.length - 1); changed(true); }, { disabled: pts.length <= 2 });
      });
      ui.inline(() => {
        ui.check("closed", !!c.closed, (v) => { c.closed = v; changed(true); });
        ui.check("smooth", c.smooth !== false, (v) => { c.smooth = v; changed(true); }, "Catmull-Rom spline through the control points.");
        ui.file("import curve", ".csv,.txt,.obj", (name, text) => {
          try { c.points = A.loadPolyline(text, name).map((p) => p.map((x) => Math.round(x * 1e5) / 1e5)); this.att.ctrl = 0; this.status(`imported ${c.points.length} points`); changed(true); } catch (e) { this.setError(`Import failed: ${e.message}`); }
        }, { help: "Polyline from Rhino / Blender: .obj (v + l) or .csv / .txt with x y z per line. Try inputs/sample_curve.csv." });
      });
    } else {
      ui.vec3("centre", c.center, (v) => { c.center = v; changed(true); }, "Or drag the handle in the view.");
      if (c.type !== "lissajous") ui.choice("axis", A.AXES.map((x) => [x, x]), c.axis, (x) => { c.axis = x; changed(true); });
      const params = { circle: [["radius", 0.01, 10]], helix: [["radius", 0.01, 10], ["height", 0.01, 20], ["turns", 0.1, 12]], sine: [["length", 0.01, 20], ["amplitude", 0, 5], ["waves", 0.1, 12]], lissajous: [["size", 0.01, 10], ["phase", 0, 2]] }[c.type];
      for (const [k, lo, hi] of params) ui.slider(k, Number(c[k]), lo, hi, { onInput: (v) => { c[k] = v; changed(); } });
      if (c.type === "lissajous") for (const k of ["a", "b", "c"]) ui.slider(`freq ${k}`, Number(c[k]), 1, 8, { int: true, onInput: (v) => { c[k] = v; changed(); } });
      ui.button("convert to polyline", () => { Object.assign(c, A.toPolyline(c)); this.att.ctrl = 0; changed(true); }, { help: "Turns the curve into control points you can drag one by one." });
    }
  }

  setUI(ui, a, changed) {
    const d = this.design;
    ui.subhead("weight set");
    ui.note("The weights this set pulls toward (the scheme follows the main schedule).");
    this.att.iterTab = Math.min(this.att.iterTab, d.full_depth - 1);
    ui.segmented(Array.from({ length: d.full_depth }, (_, k) => [k, `${k + 1}${d.iterations[k].scheme === "ds" ? "ds" : ""}`]), this.att.iterTab, (k) => { this.att.iterTab = k; this.renderInspector(); });
    const k = this.att.iterTab, defs = d.iterations[k].scheme === "cc" ? CC_WEIGHTS : DS_WEIGHTS, w = a.weights[k];
    for (const wd of defs) ui.slider(wd.label.split(/\s+/).slice(0, 2).join(" "), w[wd.name], wd.lo, wd.hi, { help: `${wd.label.split(/\s+/).join(" ")}\n${wd.help}`, onInput: (v) => { w[wd.name] = v; changed(); } });
    ui.inline(() => {
      ui.button("from schedule", () => { for (let j = 0; j < MAX_ITERATIONS; j++) a.weights[j] = { ...d.iterations[j].weights }; changed(true); }, { help: "Copy the main schedule into this set." });
      ui.button("sharp", () => { Object.assign(w, SHARP); changed(true); }, { disabled: d.iterations[k].scheme !== "cc", help: "w1 = -1, w2 = -2 for this iteration." });
      ui.button("→ next", () => { a.weights[k + 1] = { ...w }; changed(true); }, { disabled: k + 1 >= MAX_ITERATIONS });
      ui.button("→ all", () => { for (let j = 0; j < MAX_ITERATIONS; j++) a.weights[j] = { ...w }; changed(true); });
    });
    if (d.attractors.filter((x) => x.payload === "set" && x.enabled).length === 1 && d.background <= 0) ui.warn("A single set with background 0 applies fully inside its radius: raise background for a gradient.");
  }

  // ---------------------------------------------------------------- 04 layers
  layersUI(ui) {
    const d = this.design;
    const TARGET_HELP = {
      weight: "Drives one weight per face before an iteration (pattern -> where folds/spikes grow).",
      displacement: "Moves vertices along their normals after an iteration (adds relief directly).",
      fold: "Deforms the whole mesh with a fold map after an iteration (Mandelbox-style space folding).",
    };
    ui.explain("Our extension of Hansmeyer's process: a field f(p) (gyroid, noise, Worley cells, ...) is evaluated on every face and blended into one weight, or moves the surface along its normals after a step; a fold deforms the whole mesh. Plug-ins add your own.");
    ui.inline(() => {
      for (const [t, label] of [["weight", "+ weight field"], ["displacement", "+ displacement"], ["fold", "+ fold"]]) ui.button(label, () => this.addLayer(t), { help: TARGET_HELP[t] });
    });
    ui.subhead("stack");
    ui.list(() => {
      d.layers.forEach((ly, i) => {
        const fd = FN.REGISTRY[ly.function];
        const what = ly.target === "weight" ? ly.weight : ly.target;
        ui.listRow(`${ly.name}  ${fd ? fd.label : ly.function + " (missing)"} → ${what}`, `${ly.from}-${ly.to}`, { selected: i === this.lay.sel, faint: !ly.enabled, help: "Layers apply top to bottom.", onClick: () => { this.lay.sel = i; if (this.colourMode === "layer") this.refreshColours(); this.renderInspector(); } });
      });
    });
    if (!d.layers.length) ui.empty("No layers: the schedule alone shapes the form.");
    const ly = d.layers[this.lay.sel];
    if (ly) {
      ui.inline(() => {
        const s = this.lay.sel;
        ui.button("↑", () => { [d.layers[s - 1], d.layers[s]] = [d.layers[s], d.layers[s - 1]]; this.lay.sel--; this.edited(true); }, { disabled: s <= 0, help: "Move up (applies earlier)." });
        ui.button("↓", () => { [d.layers[s + 1], d.layers[s]] = [d.layers[s], d.layers[s + 1]]; this.lay.sel++; this.edited(true); }, { disabled: s >= d.layers.length - 1, help: "Move down (applies later)." });
        ui.button("duplicate", () => { const dup = JSON.parse(JSON.stringify(ly)); dup.name += "'"; d.layers.splice(s + 1, 0, dup); this.lay.sel++; this.edited(true); });
        ui.button("delete", () => { d.layers.splice(s, 1); this.lay.sel = Math.min(s, d.layers.length - 1); this.edited(true); });
      });
      this.layerEditor(ui, ly, TARGET_HELP);
    }
    ui.subhead("plug-ins");
    ui.note("JavaScript files that register fields or folds: see functions/README.md (ripples.js and twist.js are built in).");
    ui.inline(() => {
      ui.file("load plug-in .js", ".js,text/javascript", (name, source) => {
        this.userPlugins = this.userPlugins.filter((p) => p.name !== name).concat([{ name, source }]);
        store.set("plugins", this.userPlugins);
        this.reloadPlugins();
      }, { help: "Adds a plug-in to this browser." });
      if (this.userPlugins.length) ui.button("remove my plug-ins", () => { this.userPlugins = []; store.set("plugins", []); this.reloadPlugins(); });
    });
    for (const p of this.userPlugins) ui.small(`loaded: ${p.name}`, "dim");
    for (const e of FN.PLUGIN_ERRORS) ui.warn(e);
  }

  reloadPlugins() {
    FN.loadPlugins(this.pluginList());
    this.preview.sync();
    this.baker.sync();
    this.edited(true);
    this.status(`plug-ins reloaded (${Object.values(FN.REGISTRY).filter((f) => f.source).length} functions)`);
  }

  addLayer(target) {
    const d = this.design;
    const names = new Set(d.layers.map((l) => l.name));
    let n = d.layers.length + 1;
    while (names.has(`L${n}`)) n++;
    const defaults = { weight: { function: "gyroid", amplitude: 0.3 }, displacement: { function: "worley", amplitude: 0.15, params: { mode: 1 } },
      fold: { function: FN.REGISTRY.twist ? "twist" : "kaleido", amplitude: 0.3, from: 1, to: 2 } }[target];
    d.layers.push(L.normalize({ name: `L${n}`, target, ...defaults }));
    this.lay.sel = d.layers.length - 1;
    this.edited(true);
  }

  layerEditor(ui, ly, TARGET_HELP) {
    const d = this.design;
    const ch = (re = false) => { this.edited(re); if (this.colourMode === "layer" && re) this.refreshColours(); };
    ui.subhead(`edit ${ly.name} · ${ly.target}`);
    ui.text("name", ly.name, (v) => { ly.name = v; this.edited(true); });
    ui.inline(() => {
      ui.check("on", ly.enabled, (v) => { ly.enabled = v; ch(true); }, "Switch it off without deleting it.");
      ui.check("show field", this.colourMode === "layer", (on) => this.setColourMode(on ? "layer" : "none"), "Colour the form by this layer's field.");
    });
    ui.explain(TARGET_HELP[ly.target]);
    const kind = ly.target === "fold" ? "fold" : "field";
    const funcs = Object.values(FN.REGISTRY).filter((f) => f.kind === kind).sort((a, b) => (a.family + a.label).localeCompare(b.family + b.label));
    ui.select("function", ly.function, funcs.map((f) => [f.name, `${f.family}: ${f.label}`]), (n) => { ly.function = n; ly.params = {}; ch(true); });
    const fd = FN.REGISTRY[ly.function];
    if (!fd) { ui.warn(`Function '${ly.function}' not found (plug-in missing?)`); return; }
    if (fd.help) ui.small(fd.help, "dim");
    this.paramsEditor(ui, fd, ly.params, (re) => ch(re));
    if (ly.target === "weight") {
      ui.select("weight", ly.weight, WEIGHT_OPTS, (w) => { ly.weight = w; ch(true); }, "The weight this field drives.");
      ui.select("blend", ly.blend, L.BLENDS.map((b) => [b, b]), (b) => { ly.blend = b; ch(true); }, "add: w + v    multiply: w x (1 + v)    replace: v    min / max: min(w, v) / max(w, v)\nwhere v = amplitude x f(p) + offset, faded by the mask.");
    }
    const fold = ly.target === "fold";
    ui.slider(fold ? "amount" : "amplitude", ly.amplitude, fold ? 0 : -2, fold ? 1 : 2, { fmt: "%.3f", help: fold ? "0 = no change, 1 = fully folded." : "v = amplitude x f(p) + offset. Displacement is in local edge lengths (relative extrusion) or model units (absolute).", onInput: (v) => { ly.amplitude = v; ch(); } });
    if (!fold) ui.slider("offset", ly.offset, -2, 2, { fmt: "%.3f", onInput: (v) => { ly.offset = v; ch(); } });
    ui.range("iterations", ly.from, ly.to, 1, MAX_ITERATIONS, (a, b) => { ly.from = a; ly.to = b; ch(true); });
    if (!fold) ui.choice("evaluate at", [["rest", "original", "The pattern sticks to the input mesh like a texture."], ["current", "current", "The pattern is fixed in space and the growing form moves through it."]], ly.space, (s) => { ly.space = s; ch(true); });
    const masks = [["", "everywhere"], ...d.attractors.map((a) => [a.name, a.name]), ...Object.keys(L.FACING).map((f) => [f, f])];
    ui.select("mask", ly.mask, masks, (m) => { ly.mask = m; ch(true); }, "Limit the layer to an attractor's reach, or to surfaces facing a direction: 'facing +y' = detail on top, plain undersides (easier to print).");
    if (!fold) {
      ui.subhead("domain fold");
      ui.explain("Folds the field's input space before evaluating it: p <- scale x fold(p) + c x p0, repeated. A Mandelbox step x 3-5 with c = 1 gives fractal ornament.");
      const dom = ly.domain;
      const folds = Object.values(FN.REGISTRY).filter((f) => f.kind === "fold");
      ui.select("fold", dom.fold, [["none", "none"], ...folds.map((f) => [f.name, f.label])], (f) => { dom.fold = f; dom.params = {}; ch(true); }, "Fractal patterns from simple fields: fold the space the field is evaluated in.");
      if (dom.fold && dom.fold !== "none") {
        ui.slider("repeat", dom.repeat, 1, 8, { int: true, onInput: (v) => { dom.repeat = v; ch(); } });
        ui.slider("scale", dom.scale, -3, 3, { onInput: (v) => { dom.scale = v; ch(); } });
        ui.slider("c (add p0)", dom.c, 0, 1, { onInput: (v) => { dom.c = v; ch(); } });
        const dfd = FN.REGISTRY[dom.fold];
        if (dfd) this.paramsEditor(ui, dfd, dom.params, (re) => ch(re));
      }
    }
  }

  // ---------------------------------------------------------------- 05 groups
  selGroup() {
    const g = this.design.groups;
    return this.grp.sel >= 0 && this.grp.sel < g.length ? g[this.grp.sel] : null;
  }

  setPick(mode) {
    this.grp.pick = mode;
    if (!this.viewer) return;
    if (mode === "off") {
      this.viewer.setPickMesh(null);
      this.viewer.setOverlay("groupverts", null);
      this.viewer.setFormVisible(!this.print?.showing);
    } else this.drawPickOverlay();
  }

  drawPickOverlay() {
    const m = this.baseMesh();
    if (!m || this.grp.pick === "off") return;
    const g = this.selGroup(), v = this.viewer;
    const base = new (v.scene.background.constructor)(v.helper("base")), picked = new (v.scene.background.constructor)(v.helper("main"));
    const T = m.triangles(), tf = m.triangleFace(), sel = new Set(g ? g.faces : []);
    const colors = Array.from(tf, (f) => (sel.has(f) ? [picked.r, picked.g, picked.b] : [base.r, base.g, base.b]));
    v.setPickMesh(Float32Array.from(m.V), T, colors);
    v.setFormVisible(false);
    const verts = (g ? g.verts : []).filter((i) => i < m.nVerts);
    v.setOverlay("groupverts", verts.length ? v.pointsObject(verts.map((i) => [m.V[3 * i], m.V[3 * i + 1], m.V[3 * i + 2]]), v.helper("main"), 0.012) : null);
  }

  pick(hit) {
    const g = this.selGroup(), m = this.baseMesh();
    if (!g || !m) return;
    const f = m.triangleFace()[hit.faceIndex];
    const toggle = (arr, i) => { const k = arr.indexOf(i); if (k >= 0) arr.splice(k, 1); else { arr.push(i); arr.sort((a, b) => a - b); } };
    if (this.grp.pick === "faces") toggle(g.faces, f);
    else {
      let best = -1, bd = Infinity;
      for (const v of m.faceVerts(f)) {
        const d = Math.hypot(m.V[3 * v] - hit.point.x, m.V[3 * v + 1] - hit.point.y, m.V[3 * v + 2] - hit.point.z);
        if (d < bd) { bd = d; best = v; }
      }
      toggle(g.verts, best);
    }
    this.edited(true);
    this.drawPickOverlay();
  }

  groupsUI(ui) {
    const d = this.design;
    ui.explain("Paper Fig. 9: tag faces or vertices of the input mesh. Locked vertices keep their position for a number of iterations (locked edges become creases, locked points spikes); a group's weight rules change the weights of its faces.");
    ui.button("+ group", () => { d.groups.push(I.normalizeGroup({ name: `G${d.groups.length + 1}` })); this.grp.sel = d.groups.length - 1; this.edited(true); });
    ui.list(() => {
      d.groups.forEach((x, i) => {
        const detail = `${x.faces.length}f ${x.verts.length}v` + (x.lock ? ` · lock ${x.lock}` : "") + (x.rules.length ? ` · ${x.rules.length} rules` : "");
        ui.listRow(`${x.name}  ${detail}`, x.enabled ? "on" : "off", { selected: i === this.grp.sel, faint: !x.enabled, onClick: () => { this.grp.sel = i; this.drawPickOverlay(); this.renderInspector(); } });
      });
    });
    if (!d.groups.length) { ui.empty("No groups. Add one, then click faces or vertices of the input mesh in the view."); return; }
    const g = this.selGroup();
    if (!g) return;
    const ch = () => { this.edited(true); this.drawPickOverlay(); };
    ui.subhead(`edit ${g.name}`);
    ui.text("name", g.name, (v) => { g.name = v; this.edited(true); });
    ui.inline(() => {
      ui.check("on", g.enabled, (v) => { g.enabled = v; ch(); }, "Switch it off without deleting it.");
      ui.button("delete", () => { d.groups.splice(this.grp.sel, 1); this.grp.sel = Math.min(this.grp.sel, d.groups.length - 1); this.setPick("off"); this.edited(true); });
    });
    ui.choice("click to tag", [["off", "off"], ["faces", "faces"], ["verts", "vertices"]], this.grp.pick, (m) => { this.setPick(m); this.renderInspector(); }, "Shows the input mesh; click faces / vertices in the view to add or remove them.");
    this.selectTools(ui, g, ch);
    ui.subhead("lock");
    ui.slider("iterations", g.lock, 0, MAX_ITERATIONS, { int: true, help: "Group vertices keep their position for this many iterations (paper Fig. 9): locked edges become sharp creases, locked points spikes.", onInput: (v) => { g.lock = v; this.edited(); } });
    ui.check("+ face corners", g.lock_faces, (v) => { g.lock_faces = v; ch(); }, "Also lock every corner of the group's faces (keeps whole regions flat).");
    ui.subhead("weight rules");
    ui.note("For the group's faces.");
    this.rulesEditor(ui, g.rules, (re) => this.edited(re));
  }

  selectTools(ui, g, ch) {
    const m = this.baseMesh();
    if (!m) return;
    const verts = this.grp.pick === "verts";
    const add = (idx, fromFaces) => {
      if (verts && fromFaces) { const s = new Set(g.verts); for (const f of idx) for (const v of m.faceVerts(f)) s.add(v); g.verts = [...s].sort((a, b) => a - b); }
      else if (verts) g.verts = [...new Set([...g.verts, ...idx])].sort((a, b) => a - b);
      else g.faces = [...new Set([...g.faces, ...idx])].sort((a, b) => a - b);
      ch();
    };
    ui.subhead("select by rule");
    ui.small(`Adds to the group's ${verts ? "vertices" : "faces"} (switch 'click to tag' to choose).`, "dim");
    ui.select("facing", this.grp.normalDir, Object.keys(I.AXIS_VECTORS).map((k) => [k, k]), (v) => { this.grp.normalDir = v; this.renderInspector(); });
    ui.slider("within", this.grp.normalAngle, 1, 90, { fmt: "%.0f deg", onInput: (v) => { this.grp.normalAngle = v; } });
    ui.button("add facing", () => add(I.selectByNormal(m, this.grp.normalDir, this.grp.normalAngle), true));
    ui.choice("band axis", [[0, "x"], [1, "y"], [2, "z"]], this.grp.bandAxis, (a) => { this.grp.bandAxis = a; this.renderInspector(); });
    ui.slider("band from", this.grp.band[0], 0, 1, { onInput: (v) => { this.grp.band[0] = Math.min(v, this.grp.band[1]); } });
    ui.slider("band to", this.grp.band[1], 0, 1, { onInput: (v) => { this.grp.band[1] = Math.max(v, this.grp.band[0]); } });
    ui.button("add band", () => add(I.selectByHeight(m, this.grp.bandAxis, this.grp.band[0], this.grp.band[1], !verts), !verts));
    ui.slider("every k-th", this.grp.kth, 2, 12, { int: true, onInput: (v) => { this.grp.kth = v; } });
    ui.button("add every k-th", () => add(I.selectEveryKth(verts ? m.nVerts : m.nFaces, this.grp.kth), !verts));
    const labels = Object.keys(I.motifCounts(m));
    if (labels.length) {
      this.grp.motif = Math.min(this.grp.motif, labels.length - 1);
      ui.select("motif", labels[this.grp.motif], labels.map((l) => [l, l]), (l) => { this.grp.motif = labels.indexOf(l); this.renderInspector(); });
      ui.button("add motif vertices", () => { g.verts = [...new Set([...g.verts, ...I.selectByMotif(m, labels[this.grp.motif])])].sort((a, b) => a - b); ch(); });
    }
    ui.inline(() => {
      ui.button("clear", () => { if (verts) g.verts = []; else g.faces = []; ch(); }, { help: `Remove all the group's ${verts ? "vertices" : "faces"}.` });
      ui.button("invert", () => {
        const n = verts ? m.nVerts : m.nFaces, cur = new Set(verts ? g.verts : g.faces), inv = [];
        for (let i = 0; i < n; i++) if (!cur.has(i)) inv.push(i);
        if (verts) g.verts = inv; else g.faces = inv;
        ch();
      });
    });
  }

  // ---------------------------------------------------------------- 06 intrinsic
  intrinsicUI(ui) {
    const d = this.design;
    ui.subhead("motifs");
    ui.note("Vertices classed by their incident faces / edges (3F3E, 4F4E, ...). A motif's U attracts (+) or deflects (-) nearby points through w6 / w7 in the schedule.");
    ui.explain("Paper Fig. 7, eq. 10-11: the same weights act differently on differently connected vertices, so a column's capital and base differentiate from the mesh's own topology.");
    const m = this.baseMesh();
    const counts = m ? I.motifCounts(m) : {};
    if (this.result?.summary?.motifs) for (const k of Object.keys(this.result.summary.motifs)) if (!(k in counts)) counts[k] = 0;
    const labels = [...new Set([...Object.keys(counts), ...Object.keys(d.motifs)])].sort((a, b) => { const x = I.parseLabel(a), y = I.parseLabel(b); return Math.floor(x / 100) - Math.floor(y / 100) || (x % 100) - (y % 100); });
    for (const label of labels) {
      ui.slider(`U ${label}`, d.motifs[label] ?? 0, -1, 1, { help: `${counts[label] ?? 0} vertices of motif ${label} on the input mesh.`, onInput: (v) => { d.motifs[label] = v; this.edited(); } });
    }
    if (Object.keys(d.motifs).length && !d.iterations.slice(0, d.full_depth).some((it) => it.weights.w6 || it.weights.w7)) ui.warn("Set w6 / w7 in some iteration (schedule) for the motifs to act.");
    ui.subhead("measure rules");
    ui.note("A per-face measure t in [0, 1] sets, scales or adds to a weight, from 'at 0' to 'at 1'.");
    ui.explain("Paper Fig. 8: two sub-values interpolated by distance or curvature. Colour the form by the measure to see where it is high.");
    d.intrinsic.forEach((r, i) => {
      ui.itemHeader(`rule ${i + 1}`, { enabled: r.enabled, onToggle: (v) => { r.enabled = v; this.edited(true); }, onRemove: () => { d.intrinsic.splice(i, 1); this.edited(true); } });
      ui.select("measure", r.measure, Object.entries(I.MEASURES), (x) => { r.measure = x; this.edited(true); });
      ui.select("weight", r.weight, WEIGHT_OPTS, (w) => { r.weight = w; this.edited(true); });
      ui.choice("operation", I.RULE_OPS.map((o) => [o, o]), r.op, (o) => { r.op = o; this.edited(true); });
      ui.slider("at 0", r.a, -2, 2, { onInput: (v) => { r.a = v; this.edited(); } });
      ui.slider("at 1", r.b, -2, 2, { onInput: (v) => { r.b = v; this.edited(); } });
      ui.slider("gamma", r.gamma, 0.2, 5, { help: "Shapes the ramp: t^gamma.", onInput: (v) => { r.gamma = v; this.edited(); } });
      ui.range("iterations", r.from, r.to, 1, MAX_ITERATIONS, (a, b) => { r.from = a; r.to = b; this.edited(true); });
      const mode = `measure:${r.measure}`;
      ui.toggle("show measure", this.colourMode === mode, () => this.setColourMode(this.colourMode === mode ? "none" : mode), "Colour the form by this measure.");
    });
    if (!d.intrinsic.length) ui.empty("No measure rules.");
    ui.button("+ measure rule", () => { d.intrinsic.push(I.normalizeRule({})); this.edited(true); });
  }

  // ---------------------------------------------------------------- 07 porosity
  porosityUI(ui) {
    const mg = this.design.merge;
    ui.explain("Vertices of different parts of the surface that grow into contact are welded. Where a welded vertex would exceed the max valence, its faces are dropped: the surface opens into holes and handles (watch the Euler characteristic).");
    ui.check("merging on", mg.enabled, (v) => { mg.enabled = v; this.edited(true); });
    ui.slider("distance", mg.distance, 0.01, 1, { fmt: "%.3f", help: "x local edge length (relative) or model units (absolute).", onInput: (v) => { mg.distance = v; this.edited(); } });
    ui.check("relative distance", mg.relative, (v) => { mg.relative = v; this.edited(true); }, "Measure the distance in local edge lengths instead of model units.");
    ui.slider("max valence", mg.max_valence, 0, 12, { int: true, help: "0 = unlimited (welds only). 4-6 opens holes where sheets meet.", onInput: (v) => { mg.max_valence = v; this.edited(); } });
    ui.range("iterations", mg.from, mg.to, 1, MAX_ITERATIONS, (a, b) => { mg.from = a; mg.to = b; this.edited(true); });
    const s = this.result?.summary;
    if (s && mg.enabled) {
      ui.subhead("last step");
      const info = s.info.merge;
      if (info) {
        ui.value("welded", `${info.merged} vertices, ${info.dropped} faces dropped`);
        if (info.skipped) ui.warn("Skipped: it would have destroyed the form.");
      }
      ui.value("topology", `Euler ${s.euler} · ${s.boundaryEdges} hole edges`);
    }
  }

  // ---------------------------------------------------------------- 08 vessel
  useSphere() {
    const d = this.design;
    d.base = defaultSpec("sphere_open");
    d.boundary = "locked";
    d.view = "three_quarter";
    this.needViewReset = true;
    this.print.defaultsForShape();
    this.edited(true);
  }

  vesselUI(ui) {
    const d = this.design, v = d.vessel;
    ui.explain("Turns the form into the inside of a thin shell: an exact, smooth sphere outside, the subdivision relief inside, exactly what the schedule's weights build. The parts reaching out furthest press against the shell as thin windows that glow when lit from inside; deeper relief is thicker and darker.");
    if (d.base.shape !== "sphere_open") {
      ui.empty("Made for the 'Sphere with opening' base shape.");
      ui.button("use the open sphere", () => this.useSphere(), { primary: true, help: "Switches the base mesh (01) to the open sphere." });
      if (!v.enabled) return;
    }
    ui.check("vessel on", v.enabled, (on) => { v.enabled = on; if (on && d.base.shape !== "sphere_open") this.useSphere(); this.print.defaultsForShape(); this.edited(true); });
    if (!v.enabled) return;
    const e = () => this.edited();
    ui.subhead("size");
    ui.slider("diameter", v.diameter_mm, 30, 300, { fmt: "%.0f mm", help: "The real, printed size (also in base mesh and print). Walls stay as set in mm.", onInput: (x) => { v.diameter_mm = Math.min(Math.max(x, 20), 400); e(); } });
    ui.subhead("relief");
    ui.slider("depth", v.depth, 0.1, 4, { fmt: "x %.2f", help: "1 = the form's true proportions; 2 = twice as deep into the sphere.", onInput: (x) => { v.depth = x; e(); } });
    ui.slider("glowing share", v.glow, 1, 60, { fmt: "%.0f %%", help: "How much of the inside (the parts reaching out furthest) presses against the shell as thin, glowing windows.", onInput: (x) => { v.glow = x; e(); } });
    ui.check("turn the relief inside out", v.invert, (x) => { v.invert = x; this.edited(true); });
    ui.subhead("walls");
    ui.slider("window", v.min_wall_mm, 0.4, 3, { fmt: "%.2f mm", help: "The thinnest wall: the glowing windows. Keep it >= 2 nozzle widths (0.8 mm for a 0.4 nozzle).", onInput: (x) => { v.min_wall_mm = x; v.max_wall_mm = Math.max(v.max_wall_mm, x + 0.1); e(); } });
    ui.slider("deepest", v.max_wall_mm, 1, 30, { fmt: "%.1f mm", help: "A cap on how far the relief may reach into the sphere (it flattens anything deeper).", onInput: (x) => { v.max_wall_mm = Math.max(x, v.min_wall_mm + 0.1); e(); } });
    ui.slider("rim", v.rim_mm, 0.8, 8, { fmt: "%.1f mm", help: "Solid ring around the opening: what the print stands on.", onInput: (x) => { v.rim_mm = x; e(); } });
    const info = this.result?.summary?.info?.vessel;
    if (info) {
      ui.subhead("result");
      const [lo, hi] = info.wall_mm;
      ui.value("wall", `${lo.toFixed(2)} - ${hi.toFixed(2)} mm`);
      if (hi >= v.max_wall_mm - 1e-6) ui.warn("The relief reaches the cap: deeper parts are flattened.");
      const op = info.opening_deg;
      if (op !== null && op !== undefined) {
        ui.value("opening", `${(info.diameter_mm * Math.sin((op * Math.PI) / 180)).toFixed(0)} mm across (${op.toFixed(0)} deg from the top)`);
        if (op >= 35) ui.ok(`Prints upside down on the rim: the outside starts at ${op.toFixed(0)} deg from flat, no support needed.`);
        else ui.warn(`The outside starts only ${op.toFixed(0)} deg from flat near the rim: it may need support there. Widen the opening (base mesh) to 35 deg or more.`);
      }
    }
    ui.inline(() => {
      ui.toggle("light preview", this.colourMode === "light", () => this.setColourMode(this.colourMode === "light" ? "none" : "light"), "Colours the sphere by how much light gets through the wall when lit from inside (bright = thin).");
      ui.button("cut it open", () => { this.cut.on = true; this.applyCut(); this.renderToolbar(); }, { help: "Turn on the section cut (toolbar) to see inside." });
    });
  }

  // ---------------------------------------------------------------- 10 export
  stem() {
    const name = (this.presetName || this.saveName || "form").replace(/[^A-Za-z0-9_-]/g, "") || "form";
    return `${name}_d${this.result ? this.result.summary.depthReached : 0}`;
  }

  async exportMesh() {
    if (!(await this.bakeNow())) return;
    const fmtName = this.exportFmt;
    this.status(`writing ${fmtName.toUpperCase()} …`);
    const r = await this.result.worker.call({ type: "export", format: fmtName });
    if (!r.ok && r.error) { this.setError(`Export failed: ${r.error}`); return; }
    download(`${this.stem()}.${fmtName}`, r.data, fmtName === "obj" ? "text/plain" : "application/octet-stream");
    this.status(`exported ${this.stem()}.${fmtName}${fmtName === "stl" ? " (raw surface: use print for a printable STL)" : ""}`);
  }

  async screenshot() {
    const blob = await this.viewer.capture();
    download(`${this.stem()}.png`, blob, "image/png");
    this.status(`saved ${this.stem()}.png`);
  }

  exportUI(ui) {
    ui.subhead("mesh");
    ui.choice("format", [["obj", "OBJ"], ["stl", "STL"], ["ply", "PLY"]], this.exportFmt, (f) => { this.exportFmt = f; this.renderInspector(); }, "OBJ and PLY keep quads; STL is triangulated.");
    ui.button("export mesh", () => this.exportMesh(), { primary: true, help: "Bakes the full depth first, then downloads the file (E). For a printable, watertight STL use the print section." });
    ui.explain("This is the raw subdivision surface: it may pass through itself. The print section (09) rebuilds it as a clean solid in mm.");
    ui.subhead("screenshot");
    ui.button("save png", () => this.screenshot(), { help: "The 3D view without the panels." });
    ui.subhead("turntable");
    const t = this.tt;
    ui.slider("frames", t.frames, 12, 240, { int: true, onInput: (v) => { t.frames = v; } });
    ui.slider("seconds", t.seconds, 1, 20, { fmt: "%.1f s", onInput: (v) => { t.seconds = v; } });
    ui.slider("size", t.size, 256, 1600, { int: true, fmt: "%d px", onInput: (v) => { t.size = v; } });
    ui.slider("elevation", t.elevation, -60, 80, { fmt: "%.0f deg", onInput: (v) => { t.elevation = v; } });
    ui.button(this.turning ? "rendering …" : "render turntable", () => this.turntable(), { primary: true, disabled: !!this.turning, help: "Orbits the camera around whatever is shown (form or print model) and downloads a video (MP4 where the browser can record it, else WebM)." });
  }

  async turntable() {
    const v = this.viewer, b = v.bounds();
    if (!b || this.turning) return;
    this.turning = true;
    this.renderInspector();
    const t = this.tt;
    const center = b.getCenter(b.min.clone()), radius = 0.5 * b.getSize(b.min.clone()).length();
    const zUp = !!(v.printMesh && v.printMesh.visible);
    const saved = { pos: v.camera.position.clone(), target: v.target.clone(), up: v.camera.up.clone() };
    const canvas = v.renderer.domElement;
    const size = Math.round(t.size / 16) * 16;
    const pr = v.renderer.getPixelRatio();
    v.renderer.setPixelRatio(1);
    v.renderer.setSize(size, size, false);
    v.camera.aspect = 1;
    v.camera.updateProjectionMatrix();
    const fps = t.frames / Math.max(t.seconds, 0.1);
    const type = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"].find((x) => window.MediaRecorder && MediaRecorder.isTypeSupported(x));
    if (!type) { this.setError("This browser cannot record video"); this.turning = false; return; }
    const stream = canvas.captureStream(0);
    const track = stream.getVideoTracks()[0];
    const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 8e6 });
    const chunks = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const done = new Promise((res) => { rec.onstop = res; });
    rec.start();
    v.gizmoHelper.visible = false;
    const el = (t.elevation * Math.PI) / 180;
    v.camera.up.set(...(zUp ? [0, 0, 1] : [0, 1, 0]));
    for (let k = 0; k < t.frames; k++) {
      const az = (2 * Math.PI * k) / t.frames;
      const ring = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az)], h = Math.sin(el);
      const d = zUp ? [ring[0], ring[1], h] : [ring[0], h, ring[1]];
      v.camera.position.set(center.x + radius * 3 * d[0], center.y + radius * 3 * d[1], center.z + radius * 3 * d[2]);
      v.target.copy(center);
      v.renderNow();
      if (track.requestFrame) track.requestFrame();
      await new Promise((r) => setTimeout(r, 1000 / fps));
    }
    rec.stop();
    await done;
    v.renderer.setPixelRatio(pr);
    v.resize();
    v.camera.position.copy(saved.pos);
    v.target.copy(saved.target);
    v.camera.up.copy(saved.up);
    v.gizmoHelper.visible = v.gizmo.enabled;
    v.request();
    const ext = type.startsWith("video/mp4") ? "mp4" : "webm";
    download(`${(this.presetName || this.saveName || "form").replace(/[^A-Za-z0-9_-]/g, "")}_turntable.${ext}`, new Blob(chunks, { type }));
    this.turning = false;
    this.status(`saved turntable (${ext.toUpperCase()})`);
    this.renderInspector();
  }

  // ---------------------------------------------------------------- keys
  installKeys() {
    window.addEventListener("keydown", (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      const mod = e.ctrlKey || e.metaKey;
      const k = e.key.toLowerCase();
      if (mod) {
        if (k === "z") { e.preventDefault(); this.undo(e.shiftKey); }
        else if (k === "y") { e.preventDefault(); this.undo(true); }
        else if (k === "s") { e.preventDefault(); this.savePreset(); }
        return;
      }
      if (e.key === "Tab") { e.preventDefault(); this.panelsHidden = !this.panelsHidden; document.body.classList.toggle("hidden-panels", this.panelsHidden); setTimeout(() => this.viewer.resize(), 0); }
      else if (k === "b") { this.wantBake = true; this.baked = false; }
      else if (k === "e") this.exportMesh();
      else if (k === "w") this.setWire(!this.showEdges);
      else if (k === "f") this.setView(this.design.view);
      else if (k === "h") { this.help = !this.help; document.body.classList.toggle("help", this.help); this.remember("help", this.help); this.renderToolbar(); }
      else if (e.key === "]") this.setSection(this.section + 1);
      else if (e.key === "[") this.setSection(this.section - 1);
      else if (["1", "2", "3", "4"].includes(e.key)) this.setView(VIEW_LIST[+e.key - 1][0]);
    });
  }
}

const app = new App();
window.bdmh = app;
app.init().catch((e) => {
  console.error(e);
  document.getElementById("app").innerHTML = `<pre class="fatal">Could not start: ${e.message}</pre>`;
});

export { fmt, copyDesign };

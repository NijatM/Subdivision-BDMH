// Widgets in the look of nijatmahamaliyev.com: monochrome, IBM Plex Mono, 1px lines, bar sliders, [x]
// checks, segmented choices. Panels are rebuilt from scratch (immediate-mode style) whenever their
// structure changes; slider drags update values in place and never rebuild under the pointer.

export function fmt(spec, v) {
  if (!spec) spec = "%.2f";
  return spec.replace(/%(\.(\d+))?([fd%])/g, (_, __, p, t) => {
    if (t === "%") return "%";
    if (t === "d") return String(Math.round(v));
    return Number(v).toFixed(p === undefined ? 6 : +p);
  });
}

function h(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined && text !== null) el.textContent = text;
  return el;
}

export class UI {
  constructor(root, hooks = {}) {
    this.root = root;
    this.stack = [root];
    this.hooks = hooks; // { busy(on) } while a slider is held
  }

  get el() {
    return this.stack[this.stack.length - 1];
  }

  add(el) {
    this.el.appendChild(el);
    return el;
  }

  tip(el, help) {
    if (help) el.dataset.tip = help;
    return el;
  }

  // ------------------------------------------------------------ text
  heading(kicker, title) {
    this.add(h("div", "kicker", kicker));
    this.add(h("div", "title", title));
    this.add(h("div", "dash"));
  }

  subhead(text) {
    const d = this.add(h("div", "subhead"));
    d.append(h("span", "", text.toUpperCase()), h("i"));
    return d;
  }

  spaced(text, cls = "faint") {
    return this.add(h("div", `spaced ${cls}`, text.toUpperCase()));
  }

  note(text) { return this.add(h("div", "note", text)); }
  small(text, cls = "dim") { return this.add(h("div", `small ${cls}`, text)); }
  explain(text) { return this.add(h("div", "explain", text)); }
  warn(text) { return this.add(h("div", "warn", text)); }
  ok(text) { return this.add(h("div", "okline", text)); }
  empty(text) { return this.add(h("div", "empty", text)); }

  value(label, text, cls = "fg", help) {
    const r = this.row(label, help);
    r.append(h("div", `val ${cls}`, text));
    return r;
  }

  row(label, help, cls = "") {
    const r = this.add(h("div", `row ${cls}`));
    const l = h("label", "", label);
    this.tip(l, help);
    r.append(l);
    return r;
  }

  group(cls, fn) {
    const d = this.add(h("div", cls));
    this.stack.push(d);
    try { fn(d); } finally { this.stack.pop(); }
    return d;
  }

  inline(fn) {
    return this.group("inline", fn);
  }

  card(fn, cls = "") {
    return this.group(`card ${cls}`, fn);
  }

  // ------------------------------------------------------------ controls
  /** A bar slider: drag, arrow keys, or double-click / ctrl+click to type. opts: fmt, help, int, step,
   * onInput(v) while dragging, onChange(v) when released (defaults to onInput). */
  slider(label, value, lo, hi, opts = {}) {
    const r = this.row(label, opts.help);
    const bar = h("div", "bar");
    bar.tabIndex = 0;
    const fill = h("div", "fill"), mark = h("div", "mark"), zero = h("div", "zero"), val = h("span", "barval");
    bar.append(fill, zero, mark);
    r.append(bar, val);
    let v = Number(value);
    const isInt = !!opts.int;
    const f = opts.fmt || (isInt ? "%d" : "%.2f");
    const draw = () => {
      const t = (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo || 1);
      const z = lo < 0 && hi > 0 ? (0 - lo) / (hi - lo) : 0;
      const a = Math.min(t, z), b = Math.max(t, z);
      fill.style.left = `${a * 100}%`;
      fill.style.width = `${(b - a) * 100}%`;
      mark.style.left = `calc(${t * 100}% - 1px)`;
      zero.style.display = lo < 0 && hi > 0 ? "" : "none";
      zero.style.left = `${z * 100}%`;
      val.textContent = fmt(f, v);
    };
    draw();
    const set = (x, final) => {
      if (isInt) x = Math.round(x);
      if (opts.step) x = Math.round(x / opts.step) * opts.step;
      x = Math.min(Math.max(x, opts.min ?? -Infinity), opts.max ?? Infinity);
      if (x === v && !final) return;
      v = x;
      draw();
      if (final) (opts.onChange || opts.onInput)?.(v);
      else opts.onInput?.(v);
    };
    const fromX = (e) => {
      const rc = bar.getBoundingClientRect();
      return lo + Math.min(Math.max((e.clientX - rc.left) / rc.width, 0), 1) * (hi - lo);
    };
    bar.addEventListener("pointerdown", (e) => {
      if (e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      bar.setPointerCapture(e.pointerId);
      bar.classList.add("active");
      this.hooks.busy?.(true);
      const start = v;
      set(fromX(e), false);
      const move = (ev) => set(fromX(ev), false);
      const up = () => {
        bar.removeEventListener("pointermove", move);
        bar.removeEventListener("pointerup", up);
        bar.removeEventListener("pointercancel", up);
        bar.classList.remove("active");
        this.hooks.busy?.(false);
        if (v !== start || opts.onChange) set(v, true);
      };
      bar.addEventListener("pointermove", move);
      bar.addEventListener("pointerup", up);
      bar.addEventListener("pointercancel", up);
    });
    const type = () => {
      const inp = h("input", "num");
      inp.type = "text";
      inp.value = isInt ? String(Math.round(v)) : String(+v.toFixed(4));
      r.replaceChild(inp, bar);
      inp.focus();
      inp.select();
      let done = false;
      const finish = (ok) => {
        if (done) return;
        done = true;
        const x = parseFloat(inp.value);
        r.replaceChild(bar, inp);
        if (ok && Number.isFinite(x)) set(x, true);
      };
      inp.addEventListener("keydown", (e) => {
        if (e.key === "Enter") finish(true);
        else if (e.key === "Escape") finish(false);
        e.stopPropagation();
      });
      inp.addEventListener("blur", () => finish(true));
    };
    bar.addEventListener("dblclick", type);
    bar.addEventListener("click", (e) => { if (e.ctrlKey || e.metaKey) type(); });
    val.addEventListener("dblclick", type);
    bar.addEventListener("keydown", (e) => {
      const step = opts.step || (isInt ? 1 : (hi - lo) / 100);
      if (e.key === "ArrowRight" || e.key === "ArrowUp") { set(v + step * (e.shiftKey ? 10 : 1), true); e.preventDefault(); }
      else if (e.key === "ArrowLeft" || e.key === "ArrowDown") { set(v - step * (e.shiftKey ? 10 : 1), true); e.preventDefault(); }
      else if (e.key === "Enter") { type(); e.preventDefault(); }
    });
    return r;
  }

  check(label, value, onChange, help) {
    const b = h("button", `check${value ? " on" : ""}`);
    b.append(h("span", "box", value ? "[x]" : "[ ]"), h("span", "", " " + label));
    this.tip(b, help);
    b.addEventListener("click", () => onChange(!value));
    return this.add(b);
  }

  /** Labelled segmented choice. options: [[value, label, help?], ...] */
  choice(label, options, current, onChange, help) {
    const r = this.row(label, help);
    r.append(this.segmentedEl(options, current, onChange));
    return r;
  }

  segmented(options, current, onChange) {
    return this.add(this.segmentedEl(options, current, onChange, true));
  }

  segmentedEl(options, current, onChange, full = false) {
    const s = h("div", `seg${full ? " full" : ""}`);
    for (const [value, label, help] of options) {
      const b = h("button", value === current ? "on" : "", label);
      this.tip(b, help);
      b.addEventListener("click", () => onChange(value));
      s.append(b);
    }
    return s;
  }

  /** A select. options: [[value, label], ...] */
  select(label, current, options, onChange, help) {
    const r = this.row(label, help);
    r.append(this.selectEl(current, options, onChange));
    return r;
  }

  selectEl(current, options, onChange, cls = "") {
    const s = h("select", cls);
    options.forEach(([value, label], i) => {
      const o = h("option", "", label);
      o.value = String(i);
      if (value === current) o.selected = true;
      s.append(o);
    });
    s.addEventListener("change", () => onChange(options[+s.value][0]));
    return s;
  }

  button(label, onClick, opts = {}) {
    const b = h("button", `btn${opts.primary ? " primary" : ""}${opts.on ? " on" : ""}${opts.cls ? " " + opts.cls : ""}`, label);
    this.tip(b, opts.help);
    if (opts.disabled) b.disabled = true;
    b.addEventListener("click", (e) => { if (!b.disabled) onClick(e); });
    return this.add(b);
  }

  toggle(label, on, onClick, help) {
    return this.button(label, onClick, { on, help });
  }

  text(label, value, onChange, help, opts = {}) {
    const r = this.row(label, help);
    const inp = h("input", "txt");
    inp.type = "text";
    inp.value = value;
    if (opts.placeholder) inp.placeholder = opts.placeholder;
    inp.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") inp.blur(); });
    inp.addEventListener("change", () => onChange(inp.value));
    r.append(inp);
    return inp;
  }

  number(label, value, onChange, opts = {}) {
    const r = this.row(label, opts.help);
    r.append(this.numberEl(value, onChange, opts));
    if (opts.unit) r.append(h("span", "unit", opts.unit));
    return r;
  }

  numberEl(value, onChange, opts = {}) {
    const inp = h("input", "num");
    inp.type = "number";
    inp.step = opts.step ?? "any";
    if (opts.min !== undefined) inp.min = opts.min;
    if (opts.max !== undefined) inp.max = opts.max;
    inp.value = opts.digits !== undefined ? Number(value).toFixed(opts.digits) : value;
    inp.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") inp.blur(); });
    inp.addEventListener("change", () => {
      let x = parseFloat(inp.value);
      if (!Number.isFinite(x)) return;
      if (opts.min !== undefined) x = Math.max(x, opts.min);
      if (opts.max !== undefined) x = Math.min(x, opts.max);
      onChange(x);
    });
    return inp;
  }

  vec3(label, v, onChange, help) {
    const r = this.row(label, help);
    const g = h("div", "vec3");
    const cur = v.map(Number);
    for (let i = 0; i < 3; i++) {
      g.append(this.numberEl(+cur[i].toFixed(4), (x) => { cur[i] = x; onChange([...cur]); }, { step: 0.01 }));
    }
    r.append(g);
    return r;
  }

  /** Integer range [from, to] as two steppers. */
  range(label, from, to, lo, hi, onChange, help) {
    const r = this.row(label, help || "First and last iteration this applies to.");
    const g = h("div", "range");
    const a = this.numberEl(from, (x) => onChange(Math.round(Math.min(x, to)), to), { step: 1, min: lo, max: hi });
    const b = this.numberEl(to, (x) => onChange(from, Math.round(Math.max(x, from))), { step: 1, min: lo, max: hi });
    g.append(a, h("span", "dim", " – "), b);
    r.append(g);
    return r;
  }

  stepper(label, v, lo, hi, onChange, help) {
    const s = h("div", "stepper");
    this.tip(s, help);
    const minus = h("button", "", "‹"), plus = h("button", "", "›");
    if (v <= lo) minus.disabled = true;
    if (v >= hi) plus.disabled = true;
    minus.addEventListener("click", () => onChange(Math.max(lo, v - 1)));
    plus.addEventListener("click", () => onChange(Math.min(hi, v + 1)));
    s.append(h("span", "dim", label + " "), minus, h("span", "hi", ` ${v} `), plus);
    return this.add(s);
  }

  list(fn) {
    return this.group("list", fn);
  }

  listRow(left, right, { selected = false, num = null, faint = false, help = null, onClick } = {}) {
    const r = h("div", `lrow${selected ? " sel" : ""}${faint ? " faint" : ""}`);
    if (num !== null) r.append(h("span", "lnum", num));
    r.append(h("span", "left", (selected ? "> " : "") + left), h("span", "right", right || ""));
    this.tip(r, help);
    if (onClick) r.addEventListener("click", onClick);
    return this.add(r);
  }

  /** One weight over the iterations: a bar per iteration, up for +, down for -. */
  bars(label, values, rng, highlight, help) {
    const r = this.row(label, help, "barsrow");
    const g = h("div", "bars");
    values.forEach((x, i) => {
      const c = h("div", `b${i === highlight ? " hl" : ""}`);
      const bar = h("i");
      const t = Math.min(Math.abs(x) / rng, 1) * 50;
      bar.style.height = `${t}%`;
      if (x >= 0) bar.style.bottom = "50%";
      else bar.style.top = "50%";
      c.append(bar);
      g.append(c);
    });
    r.append(g);
    return r;
  }

  plot(label, values, help) {
    const r = this.row(label, help);
    const c = h("canvas", "plot");
    r.append(c);
    requestAnimationFrame(() => {
      const w = c.clientWidth || 240, hh = c.clientHeight || 52, dpr = window.devicePixelRatio || 1;
      c.width = w * dpr;
      c.height = hh * dpr;
      const g = c.getContext("2d");
      g.scale(dpr, dpr);
      const css = getComputedStyle(document.body);
      g.strokeStyle = css.getPropertyValue("--line").trim();
      g.strokeRect(0.5, 0.5, w - 1, hh - 1);
      g.strokeStyle = css.getPropertyValue("--hi").trim();
      g.lineWidth = 1.2;
      g.beginPath();
      values.forEach((v, i) => {
        const x = (i / (values.length - 1)) * (w - 4) + 2, y = hh - 3 - v * (hh - 6);
        if (i) g.lineTo(x, y);
        else g.moveTo(x, y);
      });
      g.stroke();
    });
    return r;
  }

  vsliders(label, values, onChange, help) {
    const r = this.row(label, help);
    const g = h("div", "vsl");
    values.forEach((v, i) => {
      const s = h("div", "vs");
      const f = h("i");
      f.style.height = `${v * 100}%`;
      s.append(f);
      const set = (e) => {
        const rc = s.getBoundingClientRect();
        const x = Math.min(Math.max(1 - (e.clientY - rc.top) / rc.height, 0), 1);
        f.style.height = `${x * 100}%`;
        values[i] = x;
      };
      s.addEventListener("pointerdown", (e) => {
        s.setPointerCapture(e.pointerId);
        this.hooks.busy?.(true);
        set(e);
        const mv = (ev) => set(ev);
        const up = () => {
          s.removeEventListener("pointermove", mv);
          s.removeEventListener("pointerup", up);
          this.hooks.busy?.(false);
          onChange([...values]);
        };
        s.addEventListener("pointermove", mv);
        s.addEventListener("pointerup", up);
      });
      g.append(s);
    });
    r.append(g);
    return r;
  }

  gap(px = 6) {
    const d = this.add(h("div"));
    d.style.height = px + "px";
    return d;
  }

  /** "TITLE ----- [x] on  x" above an item in a list of rules. */
  itemHeader(title, { enabled = null, onToggle = null, onRemove = null } = {}) {
    const d = this.add(h("div", "ihead"));
    d.append(h("span", "spaced dim", title.toUpperCase()), h("i"));
    if (enabled !== null) {
      const b = h("button", `check${enabled ? " on" : ""}`);
      b.append(h("span", "box", enabled ? "[x]" : "[ ]"), h("span", "", " on"));
      b.addEventListener("click", () => onToggle(!enabled));
      d.append(b);
    }
    if (onRemove) {
      const x = h("button", "btn x", "×");
      x.dataset.tip = "Delete";
      x.addEventListener("click", onRemove);
      d.append(x);
    }
    return d;
  }

  file(label, accept, onFile, opts = {}) {
    const inp = h("input");
    inp.type = "file";
    inp.accept = accept;
    if (opts.multiple) inp.multiple = true;
    inp.style.display = "none";
    inp.addEventListener("change", async () => {
      for (const f of inp.files) onFile(f.name, await f.text(), f);
      inp.value = "";
    });
    this.add(inp);
    return this.button(label, () => inp.click(), opts);
  }
}

// ------------------------------------------------------------ tooltips
export function installTooltips() {
  const tip = h("div", "tooltip");
  document.body.append(tip);
  let timer = 0, cur = null;
  document.addEventListener("pointerover", (e) => {
    const t = e.target.closest?.("[data-tip]");
    if (t === cur) return;
    cur = t;
    clearTimeout(timer);
    tip.classList.remove("show");
    if (!t || e.pointerType === "touch") return;
    timer = setTimeout(() => {
      tip.textContent = t.dataset.tip;
      const r = t.getBoundingClientRect();
      tip.classList.add("show");
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let x = Math.min(r.left, window.innerWidth - tw - 8), y = r.bottom + 6;
      if (y + th > window.innerHeight - 8) y = r.top - th - 6;
      tip.style.left = `${Math.max(8, x)}px`;
      tip.style.top = `${Math.max(8, y)}px`;
    }, 450);
  });
  document.addEventListener("pointerdown", () => { clearTimeout(timer); tip.classList.remove("show"); });
}

export function download(name, data, type = "application/octet-stream") {
  const blob = data instanceof Blob ? data : new Blob(Array.isArray(data) ? data : [data], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

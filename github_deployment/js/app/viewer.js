// The 3D view: Three.js scene, clay look, camera (turntable orbit, even scroll zoom), section cut, overlays.

import * as THREE from "three";
import { TransformControls } from "../vendor/TransformControls.js";

export const THEMES = {
  dark: { bg: 0x0b0b0b, edge: 0x1a1a1a, shadow: 0.32, helper: { main: 0xffffff, set: 0xdbdbdb, mod: 0x8f8f8f, off: 0x474747, plane: 0xebebeb, base: 0x525252 } },
  light: { bg: 0xf4f4f2, edge: 0x4d4d4d, shadow: 0.16, helper: { main: 0x000000, set: 0x242424, mod: 0x737373, off: 0xb8b8b8, plane: 0x1a1a1a, base: 0xb3b3b3 } },
};
const MESH_COLOR = new THREE.Color(0.84, 0.84, 0.83);
const BACK = 0.48; // back faces (the surface folded through itself) are drawn darker grey

export const VIEWS = {
  diagonal: { dir: [1, 1, 1], up: [0, 1, 0] },
  front: { dir: [0, 0, 1], up: [0, 1, 0] },
  top: { dir: [0, 1, 0.0001], up: [0, 0, -1] },
  three_quarter: { dir: [1, 0.6, 1.4], up: [0, 1, 0] },
};

// ------------------------------------------------------------------ colour maps
const CMAPS = {
  viridis: ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
  inferno: ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"],
};
const CMAP_RGB = {};
for (const [k, hex] of Object.entries(CMAPS)) CMAP_RGB[k] = hex.map((h) => new THREE.Color(h));

export function cmap(name, t, out) {
  const stops = CMAP_RGB[name] || CMAP_RGB.viridis;
  const x = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
  out[0] = stops[i].r + (stops[i + 1].r - stops[i].r) * f;
  out[1] = stops[i].g + (stops[i + 1].g - stops[i].g) * f;
  out[2] = stops[i].b + (stops[i + 1].b - stops[i].b) * f;
  return out;
}

export function cmapCss(name) {
  return `linear-gradient(90deg, ${CMAPS[name].join(", ")})`;
}

function matcapTexture() {
  const n = 256, c = document.createElement("canvas");
  c.width = c.height = n;
  const g = c.getContext("2d"), img = g.createImageData(n, n);
  const L = new THREE.Vector3(-0.45, 0.65, 0.62).normalize(), F = new THREE.Vector3(0.6, -0.25, 0.75).normalize();
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = (i + 0.5) / n * 2 - 1, y = 1 - (j + 0.5) / n * 2, r2 = x * x + y * y;
    const z = Math.sqrt(Math.max(1 - r2, 0));
    const d = Math.max(x * L.x + y * L.y + z * L.z, 0), f = Math.max(x * F.x + y * F.y + z * F.z, 0);
    const h = new THREE.Vector3(L.x, L.y, L.z + 1).normalize();
    const spec = Math.pow(Math.max(x * h.x + y * h.y + z * h.z, 0), 40) * 0.12;
    let v = 0.2 + 0.68 * d + 0.16 * f + spec + 0.06 * Math.pow(1 - z, 3);
    v = Math.min(v, 1);
    const k = 4 * (j * n + i);
    img.data[k] = img.data[k + 1] = img.data[k + 2] = Math.round(255 * Math.pow(v, 1 / 1.25));
    img.data[k + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function clayMaterial(matcap, opts = {}) {
  const m = new THREE.MeshMatcapMaterial({ matcap, color: opts.color || MESH_COLOR, flatShading: opts.flat !== false, side: THREE.DoubleSide, vertexColors: !!opts.vertexColors });
  m.onBeforeCompile = (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace("#include <opaque_fragment>",
      `if (!gl_FrontFacing) outgoingLight *= ${BACK.toFixed(3)};\n#include <opaque_fragment>`);
  };
  m.customProgramCacheKey = () => "clay" + (opts.vertexColors ? "vc" : "") + (opts.flat !== false ? "f" : "s");
  m.polygonOffset = true;
  m.polygonOffsetFactor = 1;
  m.polygonOffsetUnits = 1;
  return m;
}

// ------------------------------------------------------------------ the view
export class Viewer {
  constructor(host, { onPick, onGizmo, onGizmoEnd } = {}) {
    this.host = host;
    this.onPick = onPick;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.localClippingEnabled = true;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1000);
    this.target = new THREE.Vector3();
    this.size = 1;
    this.matcap = matcapTexture();
    this.theme = "dark";
    this.clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this.clipping = [];

    this.light = new THREE.DirectionalLight(0xffffff, 0);
    this.light.castShadow = true;
    this.light.shadow.mapSize.set(2048, 2048);
    this.light.shadow.bias = -0.0005;
    this.light.shadow.radius = 6;
    this.scene.add(this.light, this.light.target);
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({ opacity: 0.5 }));
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);
    this.shadows = true;

    this.formGeo = new THREE.BufferGeometry();
    this.formMat = clayMaterial(this.matcap);
    this.formMatVC = clayMaterial(this.matcap, { vertexColors: true });
    this.form = new THREE.Mesh(this.formGeo, this.formMat);
    this.form.castShadow = true;
    this.form.frustumCulled = false;
    this.scene.add(this.form);
    this.edgeMat = new THREE.LineBasicMaterial({ color: THEMES.dark.edge });
    this.edges = new THREE.LineSegments(new THREE.BufferGeometry(), this.edgeMat);
    this.edges.visible = false;
    this.edges.frustumCulled = false;
    this.scene.add(this.edges);
    this.overlays = new THREE.Group();
    this.scene.add(this.overlays);
    this.topo = null;
    this.raycaster = new THREE.Raycaster();

    this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
    this.gizmo.setSize(0.8);
    this.gizmoTarget = new THREE.Object3D();
    this.scene.add(this.gizmoTarget);
    this.gizmoHelper = this.gizmo.getHelper();
    this.scene.add(this.gizmoHelper);
    this.gizmo.addEventListener("change", () => this.request());
    this.gizmo.addEventListener("objectChange", () => onGizmo && onGizmo(this.gizmoTarget.position.toArray()));
    this.gizmo.addEventListener("dragging-changed", (e) => { this.gizmoDragging = e.value; if (!e.value && onGizmoEnd) onGizmoEnd(); });
    this.gizmo.enabled = false;
    this.gizmoHelper.visible = false;

    this.setTheme("dark");
    this.installControls();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(host);
    this.request();
  }

  // ---------------------------------------------------------------- frame
  request() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      this.renderNow();
    });
  }

  renderNow() {
    this.camera.lookAt(this.target);
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = Math.max(this.host.clientWidth, 1), h = Math.max(this.host.clientHeight, 1);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + "px";
    this.renderer.domElement.style.height = h + "px";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.request();
  }

  setTheme(name) {
    this.theme = name;
    const t = THEMES[name];
    this.scene.background = new THREE.Color(t.bg);
    this.edgeMat.color.set(t.edge);
    this.ground.material.opacity = t.shadow;
    this.request();
  }

  helper(kind) {
    return THEMES[this.theme].helper[kind];
  }

  // ---------------------------------------------------------------- the form
  /** disp: { V: Float32Array, T: Uint32Array, topo } ; colours: Float32Array per vertex in [0,1] or null */
  showMesh(disp, colours = null, map = "viridis") {
    const g = this.formGeo;
    const same = this.topo !== null && disp.topo === this.topo && g.attributes.position && g.attributes.position.count * 3 === disp.V.length;
    if (same) {
      g.attributes.position.array.set(disp.V);
      g.attributes.position.needsUpdate = true;
    } else {
      g.setAttribute("position", new THREE.BufferAttribute(disp.V, 3));
      g.setIndex(new THREE.BufferAttribute(disp.T, 1));
      g.deleteAttribute("color");
      this.topo = disp.topo;
    }
    g.computeBoundingSphere();
    g.computeBoundingBox();
    this.setColours(colours, map);
    this.updateGround();
    this.request();
  }

  setColours(colours, map = "viridis") {
    const g = this.formGeo;
    if (colours && g.attributes.position && colours.length === g.attributes.position.count) {
      const n = colours.length;
      let attr = g.attributes.color;
      if (!attr || attr.count !== n) {
        attr = new THREE.BufferAttribute(new Float32Array(3 * n), 3);
        g.setAttribute("color", attr);
      }
      const c = [0, 0, 0], a = attr.array;
      for (let i = 0; i < n; i++) {
        cmap(map, colours[i], c);
        a[3 * i] = c[0]; a[3 * i + 1] = c[1]; a[3 * i + 2] = c[2];
      }
      attr.needsUpdate = true;
      this.form.material = this.formMatVC;
      this.formMatVC.color.set(0xffffff);
    } else {
      this.form.material = this.formMat;
    }
    this.form.material.clippingPlanes = this.clipping;
    this.request();
  }

  setEdges(E) {
    if (!E) {
      this.edges.visible = false;
      this.request();
      return;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", this.formGeo.attributes.position);
    g.setIndex(new THREE.BufferAttribute(E, 1));
    this.edges.geometry.dispose();
    this.edges.geometry = g;
    this.edges.visible = this.form.visible;
    this.edgeMat.clippingPlanes = this.clipping;
    this.request();
  }

  setFormVisible(on) {
    this.form.visible = on;
    if (!on) this.edges.visible = false;
    this.request();
  }

  bounds() {
    const g = this.shownGeometry();
    if (!g || !g.attributes.position) return null;
    if (!g.boundingBox) g.computeBoundingBox();
    return g.boundingBox;
  }

  shownGeometry() {
    if (this.printMesh && this.printMesh.visible) return this.printMesh.geometry;
    if (this.overlayMesh && this.overlayMesh.visible) return this.overlayMesh.geometry;
    return this.formGeo;
  }

  updateGround() {
    const b = this.bounds();
    if (!b || b.isEmpty()) return;
    const size = b.getSize(new THREE.Vector3()).length() || 1;
    const c = b.getCenter(new THREE.Vector3());
    const up = this.camera.up.clone();
    const zUp = Math.abs(up.z) > 0.5 && this.printMesh && this.printMesh.visible;
    this.ground.visible = this.shadows;
    this.light.visible = this.shadows;
    if (zUp) {
      this.ground.rotation.set(0, 0, 0);
      this.ground.position.set(c.x, c.y, b.min.z - 0.001 * size);
      this.light.position.set(c.x + 0.3 * size, c.y - 0.2 * size, c.z + 2 * size);
    } else {
      this.ground.rotation.set(-Math.PI / 2, 0, 0);
      this.ground.position.set(c.x, b.min.y - 0.02 * size, c.z);
      this.light.position.set(c.x + 0.25 * size, c.y + 2 * size, c.z + 0.15 * size);
    }
    this.ground.scale.set(size * 6, size * 6, 1);
    this.light.target.position.copy(c);
    const cam = this.light.shadow.camera;
    cam.left = cam.bottom = -size;
    cam.right = cam.top = size;
    cam.near = 0.01 * size;
    cam.far = 6 * size;
    cam.updateProjectionMatrix();
    this.size = size;
  }

  setShadows(on) {
    this.shadows = on;
    this.updateGround();
    this.request();
  }

  // ---------------------------------------------------------------- camera
  setView(name, up = null) {
    const v = VIEWS[name] || VIEWS.diagonal;
    const b = this.bounds();
    if (!b || b.isEmpty()) return;
    const c = b.getCenter(new THREE.Vector3());
    const radius = 0.5 * b.getSize(new THREE.Vector3()).length();
    const dir = new THREE.Vector3(...v.dir).normalize();
    this.camera.up.set(...(up || v.up));
    this.look(c.clone().addScaledVector(dir, radius * 3.2), c);
  }

  look(eye, target) {
    this.camera.position.copy(eye);
    this.target.copy(target);
    this.camera.lookAt(target);
    this.updateGround();
    this.request();
  }

  installControls() {
    const el = this.renderer.domElement;
    el.style.touchAction = "none";
    const pointers = new Map();
    let mode = null, last = null, pinch = null;
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("pointerdown", (e) => {
      if (this.gizmoDragging || (this.gizmo.enabled && this.gizmo.axis)) return;
      el.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      this.downAt = [e.clientX, e.clientY, performance.now(), e.button];
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), m: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
        mode = "pinch";
      } else {
        mode = e.button === 2 || e.button === 1 || e.shiftKey ? "pan" : "orbit";
      }
      last = [e.clientX, e.clientY];
    });
    el.addEventListener("pointermove", (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (mode === "pinch" && pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]), m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        this.zoomBy(Math.log(d / Math.max(pinch.d, 1)) / 0.12);
        this.pan(m[0] - pinch.m[0], m[1] - pinch.m[1]);
        pinch = { d, m };
        return;
      }
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      last = [e.clientX, e.clientY];
      if (mode === "orbit") this.orbit(dx, dy);
      else if (mode === "pan") this.pan(dx, dy);
    });
    const up = (e) => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2 && mode === "pinch") mode = null;
      if (!pointers.size) mode = null;
      const d = this.downAt;
      if (d && e.button === 0 && Math.hypot(e.clientX - d[0], e.clientY - d[1]) < 4 && performance.now() - d[2] < 400) {
        this.click(e);
      }
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      let notches = -e.deltaY / (e.deltaMode === 1 ? 3 : e.deltaMode === 2 ? 0.3 : 100);
      if (e.ctrlKey) notches *= 2.5; // trackpad pinch
      this.zoomBy(Math.max(Math.min(notches, 2.5), -2.5));
    }, { passive: false });
    el.addEventListener("dblclick", (e) => {
      const hit = this.raycast(e, this.shownMesh());
      if (!hit) return;
      const d = this.camera.position.distanceTo(hit.point);
      const dir = this.camera.position.clone().sub(hit.point).normalize();
      this.look(hit.point.clone().addScaledVector(dir, d * 0.7), hit.point);
    });
  }

  shownMesh() {
    if (this.printMesh && this.printMesh.visible) return this.printMesh;
    if (this.overlayMesh && this.overlayMesh.visible) return this.overlayMesh;
    return this.form;
  }

  click(e) {
    if (this.onPick && this.overlayMesh && this.overlayMesh.visible) {
      const hit = this.raycast(e, this.overlayMesh);
      if (hit) this.onPick(hit);
    }
  }

  raycast(e, obj) {
    if (!obj || !obj.visible || !obj.geometry.attributes.position) return null;
    const r = this.renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(p, this.camera);
    const hits = this.raycaster.intersectObject(obj, false).filter((h) => !this.clipping.length || this.clipping[0].distanceToPoint(h.point) >= 0);
    return hits[0] || null;
  }

  orbit(dx, dy) {
    const up = this.camera.up.clone().normalize();
    const off = this.camera.position.clone().sub(this.target);
    const k = 0.006;
    off.applyAxisAngle(up, -dx * k);
    const right = new THREE.Vector3().crossVectors(off, up).normalize();
    const pitched = off.clone().applyAxisAngle(right, -dy * k);
    if (Math.abs(pitched.clone().normalize().dot(up)) < 0.995) off.copy(pitched);
    this.camera.position.copy(this.target).add(off);
    this.request();
  }

  pan(dx, dy) {
    const h = this.renderer.domElement.clientHeight || 1;
    const dist = this.camera.position.distanceTo(this.target);
    const scale = (2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) / h;
    this.camera.updateMatrix();
    const x = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0).multiplyScalar(-dx * scale);
    const y = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1).multiplyScalar(dy * scale);
    this.camera.position.add(x).add(y);
    this.target.add(x).add(y);
    this.request();
  }

  /** Every notch covers the same share (~11 %) of the distance to the orbit centre, within 0.01-15x the size. */
  zoomBy(notches) {
    const off = this.camera.position.clone().sub(this.target);
    const d = off.length();
    const nd = Math.min(Math.max(d * Math.exp(-0.12 * notches), 0.01 * this.size), 15 * this.size);
    this.camera.position.copy(this.target).addScaledVector(off.normalize(), nd);
    this.request();
  }

  // ---------------------------------------------------------------- section cut
  setCut(on, axis = 2, pos = 0.5, keepPositive = false) {
    const b = this.bounds();
    if (!on || !b || b.isEmpty()) {
      this.clipping.length = 0;
    } else {
      const n = new THREE.Vector3();
      n.setComponent(axis, keepPositive ? 1 : -1);
      const at = b.min.getComponent(axis) + pos * (b.max.getComponent(axis) - b.min.getComponent(axis));
      this.clipPlane.normal.copy(n);
      this.clipPlane.constant = -n.getComponent(axis) * at;
      this.clipping.length = 0;
      this.clipping.push(this.clipPlane);
    }
    for (const m of [this.formMat, this.formMatVC, this.edgeMat]) { m.clippingPlanes = this.clipping; m.needsUpdate = true; }
    if (this.printMesh) { this.printMesh.material.clippingPlanes = this.clipping; this.printMesh.material.needsUpdate = true; }
    this.request();
  }

  // ---------------------------------------------------------------- overlays
  clearOverlay(name) {
    const old = this.overlays.getObjectByName(name);
    if (old) {
      this.overlays.remove(old);
      old.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    }
  }

  setOverlay(name, obj) {
    this.clearOverlay(name);
    if (obj) {
      obj.name = name;
      this.overlays.add(obj);
    }
    this.request();
  }

  pointsObject(points, color, size = 0.012) {
    const g = new THREE.Group();
    const r = size * Math.max(this.size, 1e-3);
    const geo = new THREE.SphereGeometry(r, 12, 8);
    for (let i = 0; i < points.length; i++) {
      const col = Array.isArray(color) ? color[i] : color;
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: col }));
      m.position.set(...points[i]);
      g.add(m);
    }
    return g;
  }

  lineObject(points, color, closed = false) {
    const g = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
    const m = new THREE.LineBasicMaterial({ color });
    return closed ? new THREE.LineLoop(g, m) : new THREE.Line(g, m);
  }

  /** The input mesh for tagging: faces coloured (per face), clickable. */
  setPickMesh(V, T, faceColor) {
    if (this.overlayMesh) {
      this.scene.remove(this.overlayMesh);
      this.overlayMesh.geometry.dispose();
      this.overlayMesh = null;
    }
    if (!V) {
      this.request();
      return;
    }
    // non-indexed: one colour per triangle
    const nT = T.length / 3, pos = new Float32Array(9 * nT), col = new Float32Array(9 * nT);
    for (let t = 0; t < nT; t++) for (let c = 0; c < 3; c++) {
      const v = T[3 * t + c];
      pos.set([V[3 * v], V[3 * v + 1], V[3 * v + 2]], 9 * t + 3 * c);
      col.set(faceColor[t], 9 * t + 3 * c);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.computeBoundingBox();
    g.computeBoundingSphere();
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
    this.overlayMesh = m;
    this.scene.add(m);
    this.request();
  }

  // ---------------------------------------------------------------- print model
  setPrintMesh(V, F, colours) {
    if (this.printMesh) {
      this.scene.remove(this.printMesh);
      this.printMesh.geometry.dispose();
      this.printMesh = null;
    }
    if (!V) {
      this.request();
      return;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(V, 3));
    g.setIndex(new THREE.BufferAttribute(F, 1));
    if (colours) g.setAttribute("color", new THREE.BufferAttribute(colours, 3));
    g.computeVertexNormals();
    g.computeBoundingBox();
    g.computeBoundingSphere();
    const mat = clayMaterial(this.matcap, { flat: false, vertexColors: !!colours });
    mat.color.set(colours ? 0xffffff : MESH_COLOR);
    mat.clippingPlanes = this.clipping;
    this.printMesh = new THREE.Mesh(g, mat);
    this.printMesh.castShadow = true;
    this.printMesh.frustumCulled = false;
    this.scene.add(this.printMesh);
    this.request();
  }

  // ---------------------------------------------------------------- gizmo
  showGizmo(pos) {
    if (!pos) {
      this.gizmo.detach();
      this.gizmo.enabled = false;
      this.gizmoHelper.visible = false;
      this.request();
      return;
    }
    this.gizmoTarget.position.set(...pos);
    if (this.gizmo.object !== this.gizmoTarget) this.gizmo.attach(this.gizmoTarget);
    this.gizmo.enabled = true;
    this.gizmoHelper.visible = true;
    this.request();
  }

  // ---------------------------------------------------------------- capture
  async capture(width = null, height = null) {
    const canvas = this.renderer.domElement;
    let w0, h0;
    if (width) {
      w0 = canvas.width; h0 = canvas.height;
      this.renderer.setSize(width / this.renderer.getPixelRatio(), height / this.renderer.getPixelRatio(), false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    this.gizmoHelper.visible = false;
    this.renderNow();
    const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
    if (width) this.resize();
    this.gizmoHelper.visible = this.gizmo.enabled;
    this.request();
    return blob;
  }
}

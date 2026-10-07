# Subdivision-BDMH: the web app

The subdivision engine of this repository (`hansmeyer/`, Python) ported to JavaScript, running entirely in
the browser: no server, no build step, no install. GitHub Pages serves this folder as it is.

- **Live:** `https://nijatm.github.io/Subdivision-BDMH/` (once the repository is named `Subdivision-BDMH` and Pages is on, see below)
- **Locally:** `python3 -m http.server 8765` in this folder, then open http://127.0.0.1:8765
  (module workers need http://, opening `index.html` as a file does not work)

## What is here

```
index.html            the page (Three.js through an import map)
css/app.css           the look: nijatmahamaliyev.com palette, IBM Plex Mono, dark / light
js/engine/            the engine, a line-by-line port of hansmeyer/ (pure JS, runs in workers and in tests)
js/workers/engine.js  runs the pipeline off the page's thread: live preview and background bake
js/app/               the app shell (main.js), widgets (ui.js), 3D view (viewer.js), print (print.js)
js/vendor/            three.js r169 (MIT)
presets/              the same JSON presets as the desktop app (index.json lists them)
functions/            plug-ins: JavaScript fields / folds (ripples.js, twist.js; see below)
inputs/               sample OBJ and curve files
fonts/                IBM Plex Mono (SIL Open Font Licence)
tests/                parity test against the Python engine (not deployed)
```

## Same results as the desktop app

`tests/parity.mjs` runs every preset (and extra designs covering Doo-Sabin, open panels, OBJ import,
every layer type, groups and locks, motifs, measures and merging) through the web engine and compares the
mesh with the Python engine's: faces identical, positions identical to the last bit for 28 of 32 designs
and within 1.4e-16 for the rest.

```bash
python github_deployment/tests/make_reference.py /tmp/ref        # in the project's .venv
/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc \
    -m github_deployment/tests/parity.mjs -- /tmp/ref              # from the repository root
```

## Differences from the desktop app

- Presets you save live in this browser (localStorage); *download .json* gives a file the desktop app reads.
- Imported OBJ files and plug-ins are kept in this browser too.
- Plug-ins are JavaScript (one function per point) instead of Python:

```js
field("ripples", { label: "Ripples", family: "plug-in", params: { frequency: [0.2, 12, 3] } },
  (x, y, z, p) => Math.sin(2 * Math.PI * p.frequency * Math.hypot(x, z)));
fold("twist", { params: { turns: [-1, 1, 0.15] } }, (x, y, z, p) => { /* ... */ return [x2, y2, z2]; });
```

- Turntables are saved as MP4 (or WebM where the browser can't record MP4).
- **Print (09):** vessels export exactly (verified watertight STL). The voxel print model for other forms
  (watertight remesh, cuts into parts, pin holes, support check) is still being ported.

## Deploying (GitHub Pages)

`.github/workflows/pages.yml` publishes this folder on every push to `master` that touches it.

1. On GitHub: **Settings → General → Repository name** → `Subdivision-BDMH` (the Pages address uses the
   repository name, so this gives `https://nijatm.github.io/Subdivision-BDMH/`). GitHub redirects the old
   address; update the local remote with
   `git remote set-url origin https://github.com/NijatM/Subdivision-BDMH.git`.
2. **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Push. The **Actions** tab shows the deploy; the address appears in Settings → Pages.

All paths in the app are relative, so it works under any address.

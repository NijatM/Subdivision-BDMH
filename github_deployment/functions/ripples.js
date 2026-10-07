// Example plug-in field: concentric ripples around the y axis.
//
// Plug-ins listed in functions/index.json load when the app starts; your own files load with
// "load plug-in" in 04 layers. Register a per-point function with field() (or fold()) and it appears in the
// layer panel with one slider per parameter.

field("ripples", {
  label: "Ripples (plug-in example)",
  family: "plug-in",
  help: "Concentric waves around the y axis, fading with height.",
  params: { frequency: [0.2, 12.0, 3.0], decay: [0.0, 2.0, 0.3] },
}, (x, y, z, p) => Math.sin(2 * Math.PI * p.frequency * Math.hypot(x, z)) * Math.exp(-p.decay * Math.abs(y)));

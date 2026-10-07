// Example plug-in fold: twist space around the y axis.
//
// A fold returns the new position [x, y, z] of a point. Used as a "fold" layer it bends the mesh between
// iterations; used as a field's domain fold it twists the pattern instead.

fold("twist", {
  label: "Twist (plug-in example)",
  family: "plug-in",
  help: "Rotates each horizontal slice by `turns` per unit of height.",
  params: { turns: [-1.0, 1.0, 0.15] },
}, (x, y, z, p) => {
  const a = 2 * Math.PI * p.turns * y;
  const c = Math.cos(a), s = Math.sin(a);
  return [c * x - s * z, y, s * x + c * z];
});

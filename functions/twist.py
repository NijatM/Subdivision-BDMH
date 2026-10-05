"""Example plug-in fold: twist space around the y axis.

A fold returns new positions (N, 3). Used as a "fold" layer it bends the mesh between
iterations; used as a field's domain fold it twists the pattern instead.
"""

import numpy as np

from hansmeyer.functions import fold


@fold("twist", "Twist (plug-in example)", family="plug-in",
      help="Rotates each horizontal slice by `turns` per unit of height.",
      turns=(-1.0, 1.0, 0.15))
def twist(p, turns):
    a = 2 * np.pi * turns * p[:, 1]
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * p[:, 0] - s * p[:, 2], p[:, 1], s * p[:, 0] + c * p[:, 2]], 1)

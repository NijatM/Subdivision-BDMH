"""Example plug-in field: concentric ripples around the y axis.

Any .py file in this folder is loaded at start-up (and by "Reload plug-ins" in the
app). Decorate a function with @field (or @fold) and it appears in the Function
layers panel with one slider per parameter.
"""

import numpy as np

from hansmeyer.functions import field


@field("ripples", "Ripples (plug-in example)", family="plug-in",
       help="Concentric waves around the y axis, fading with height.",
       frequency=(0.2, 12.0, 3.0), decay=(0.0, 2.0, 0.3))
def ripples(p, frequency, decay):
    r = np.hypot(p[:, 0], p[:, 2])
    return np.sin(2 * np.pi * frequency * r) * np.exp(-decay * np.abs(p[:, 1]))

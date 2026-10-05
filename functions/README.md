# Function plug-ins

Every `.py` file in this folder is imported when the app starts (and when you press
**Reload plug-ins** in the *Function layers* panel). Register functions with the
decorators from `hansmeyer.functions`:

```python
import numpy as np
from hansmeyer.functions import field, fold

@field("my_field", "My field", family="plug-in", help="Shown as a tooltip.",
       frequency=(0.1, 10.0, 2.0),      # (min, max, default) -> float slider
       count=(1, 12, 4, "int"))         # (min, max, default, "int") -> integer slider
def my_field(p, frequency, count):
    # p: (N, 3) positions. Return (N,) values, ideally in [-1, 1].
    return np.sin(frequency * p[:, 0]) * np.cos(count * p[:, 1])

@fold("my_fold", "My fold", family="plug-in", amount=(0.0, 1.0, 0.5))
def my_fold(p, amount):
    # Return (N, 3) new positions.
    return p * (1 + amount * np.sin(p))
```

- **Fields** drive weights (e.g. `w_f`) or displace the surface.
- **Folds** deform the mesh between iterations, or fold the *domain* of a field (repeat a
  fold several times for fractal patterns).
- Saving a plug-in and pressing *Reload plug-ins* recomputes every design that uses it.
- Plug-ins are ordinary Python running inside the app, so only use files you trust.
- Files starting with `_` are ignored.

Examples here: `ripples.py` (field) and `twist.py` (fold).

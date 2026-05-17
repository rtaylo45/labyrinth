#!/usr/bin/env python
"""Examples showing how to use the color generator classes."""

from labyrinth.backgrounds.color_generators import RGBASampler, RGBSampler

# --- RGBSampler ---

# Default range: full [0, 255]
rgb_gen = RGBSampler()
r, g, b = rgb_gen()
print(f"RGB (full range):    R={r}, G={g}, B={b}")

# Restricted range: muted colors
muted_rgb_gen = RGBSampler(color_min=50, color_max=200)
r, g, b = muted_rgb_gen()
print(f"RGB (muted range):   R={r}, G={g}, B={b}")

# --- RGBASampler ---

# Default range: full [0, 255] for color and alpha
rgba_gen = RGBASampler()
r, g, b, a = rgba_gen()
print(f"RGBA (full range):   R={r}, G={g}, B={b}, A={a}")

# Constant alpha: always fully opaque
opaque_rgba_gen = RGBASampler(alpha_min=255, alpha_max=255)
r, g, b, a = opaque_rgba_gen()
print(f"RGBA (opaque):       R={r}, G={g}, B={b}, A={a}")

# Semi-transparent only
semi_rgba_gen = RGBASampler(alpha_min=100, alpha_max=200)
r, g, b, a = semi_rgba_gen()
print(f"RGBA (semi-trans):   R={r}, G={g}, B={b}, A={a}")

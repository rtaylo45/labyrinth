#!/usr/bin/env python
# Vidrovr Inc.

from __future__ import annotations

from typing import NewType

# Color spaces
ColorSpace = NewType("ColorSpace", tuple)

# Common spaces
R = NewType("R", int)
G = NewType("G", int)
B = NewType("B", int)
A = NewType("A", int)

RGB = tuple[R, G, B]
BRG = tuple[B, R, G]
RGBA = tuple[R, G, B, A]
BRGA = tuple[B, R, G, A]

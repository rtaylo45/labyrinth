#!/usr/bin/env python
# Vidrovr Inc.

from __future__ import annotations

from typing import NewType

# Dim names
N = NewType("N", int)
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)

# Common shapes
HWC = tuple[H, W, C]
WHC = tuple[W, H, C]
CHW = tuple[C, H, W]

# Batched shapes
NHWC = tuple[N, H, W, C]
NCHW = tuple[N, C, H, W]

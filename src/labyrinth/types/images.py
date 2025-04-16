#!/usr/bin/env python
# Vidrovr Inc.

from labyrinth.types.array import Array, ScalarType
from labyrinth.types.shapes import CHW, HWC, NCHW, NHWC, WHC

HWCImage = Array[HWC, ScalarType]
WHCImage = Array[WHC, ScalarType]
CHWImage = Array[CHW, ScalarType]
NHWCImage = Array[NHWC, ScalarType]
NCHWImage = Array[NCHW, ScalarType]

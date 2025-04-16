#!/usr/bin/env python
# Vidrovr Inc.

from __future__ import annotations

from typing import TypeVar

import numpy as np

ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
Shape = TypeVar("Shape", covariant=True, bound=tuple[int, ...])
Array = np.ndarray[Shape, np.dtype[ScalarType]]

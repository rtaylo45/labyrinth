#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Vidrovr Inc.

from typing import Protocol

import numpy as np

from labyrinth.types.array import Array, Shape


class ImageAugmentation(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(self, array: Array[Shape, np.uint8]) -> Array[Shape, np.uint8]: ...

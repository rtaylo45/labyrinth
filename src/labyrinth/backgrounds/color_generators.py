#!/usr/bin/env python
# Vidrovr Inc.

import numpy as np

from labyrinth.backgrounds.protocol import ColorGeneratorProtocol
from labyrinth.types.color_space import RGB, RGBA

rng = np.random.default_rng()


class RGBColorGenerator(ColorGeneratorProtocol):
    _color_min: int
    _color_max: int

    def __init__(self, color_min: int = 0, color_max: int = 255):
        if (color_min < 0) or (color_min > 255):
            raise ValueError(f"color_min ({color_min}) out of range [0, 255]")

        if (color_max < 0) or (color_max > 255):
            raise ValueError(f"color_max ({color_max}) out of range [0, 255]")

        if color_min >= color_max:
            raise ValueError(
                f"color_min ({color_min}) must be larger than color_max ({color_max})"
            )

        self._color_min = color_min
        self._color_max = color_max

    def __call__(
        self,
    ) -> RGB:
        r = int(rng.integers(self._color_min, self._color_max))
        g = int(rng.integers(self._color_min, self._color_max))
        b = int(rng.integers(self._color_min, self._color_max))

        return (r, g, b)


class RGBAColorGenerator(ColorGeneratorProtocol):
    _constant_alpha: bool
    _color_min: int
    _color_max: int
    _alpha_min: int
    _alpha_max: int

    def __init__(
        self,
        color_min: int = 0,
        color_max: int = 255,
        alpha_min: int = 0,
        alpha_max: int = 255,
    ):
        if (color_min < 0) or (color_min > 255):
            raise ValueError(f"color_min ({color_min}) out of range [0, 255]")

        if (color_max < 0) or (color_max > 255):
            raise ValueError(f"color_max ({color_max}) out of range [0, 255]")

        if (alpha_min < 0) or (alpha_min > 255):
            raise ValueError(f"alpha_min ({alpha_min}) out of range [0, 255]")

        if (alpha_max < 0) or (alpha_max > 255):
            raise ValueError(f"alpha_max ({alpha_max}) out of range [0, 255]")

        self._constant_alpha = False
        if alpha_min == alpha_max:
            self._constant_alpha = True

        self._color_min = color_min
        self._color_max = color_max
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max

    def __call__(
        self,
    ) -> RGBA:
        r = int(rng.integers(self._color_min, self._color_max))
        g = int(rng.integers(self._color_min, self._color_max))
        b = int(rng.integers(self._color_min, self._color_max))

        if self._constant_alpha:
            a = int(self._alpha_min)
        else:
            a = int(rng.integers(self._alpha_min, self._alpha_max))

        return (r, g, b, a)

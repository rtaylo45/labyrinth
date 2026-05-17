#!/usr/bin/env python
# Vidrovr Inc.

import numpy as np

from labyrinth.backgrounds.protocol import ColorSampler
from labyrinth.types.color_space import RGB, RGBA


def _validate_channel(value: int, name: str) -> None:
    if (value < 0) or (value > 255):
        raise ValueError(f"{name} ({value}) out of range [0, 255]")


class RGBSampler(ColorSampler):
    _color_min: int
    _color_max: int
    _rng: np.random.Generator

    def __init__(
        self, color_min: int = 0, color_max: int = 255, seed: int | None = None
    ):
        _validate_channel(color_min, "color_min")
        _validate_channel(color_max, "color_max")

        if color_min >= color_max:
            raise ValueError(
                f"color_max ({color_max}) must be larger than color_min ({color_min})"
            )

        self._color_min = color_min
        self._color_max = color_max
        self._rng = np.random.default_rng(seed)

    def __call__(self) -> RGB:
        r = int(self._rng.integers(self._color_min, self._color_max))
        g = int(self._rng.integers(self._color_min, self._color_max))
        b = int(self._rng.integers(self._color_min, self._color_max))

        return (r, g, b)


class RGBASampler(ColorSampler):
    _constant_alpha: bool
    _color_min: int
    _color_max: int
    _alpha_min: int
    _alpha_max: int
    _rng: np.random.Generator

    def __init__(
        self,
        color_min: int = 0,
        color_max: int = 255,
        alpha_min: int = 0,
        alpha_max: int = 255,
        seed: int | None = None,
    ):
        _validate_channel(color_min, "color_min")
        _validate_channel(color_max, "color_max")
        _validate_channel(alpha_min, "alpha_min")
        _validate_channel(alpha_max, "alpha_max")

        if color_min >= color_max:
            raise ValueError(
                f"color_max ({color_max}) must be larger than color_min ({color_min})"
            )

        if alpha_min > alpha_max:
            raise ValueError(
                f"alpha_max ({alpha_max}) must be greater than or equal to alpha_min ({alpha_min})"
            )

        self._constant_alpha = alpha_min == alpha_max
        self._color_min = color_min
        self._color_max = color_max
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._rng = np.random.default_rng(seed)

    def __call__(self) -> RGBA:
        r = int(self._rng.integers(self._color_min, self._color_max))
        g = int(self._rng.integers(self._color_min, self._color_max))
        b = int(self._rng.integers(self._color_min, self._color_max))

        if self._constant_alpha:
            a = int(self._alpha_min)
        else:
            a = int(self._rng.integers(self._alpha_min, self._alpha_max))

        return (r, g, b, a)

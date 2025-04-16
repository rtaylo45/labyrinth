#!/usr/bin/env python
# By: Zack Taylor

import numpy as np

from labyrinth.backgrounds import ColorGenerator
from labyrinth.types import Array

rng = np.random.default_rng()


class SolidBackgroundGenerator:
    _h_min: int
    _h_max: int
    _w_min: int
    _w_max: int

    def __init__(
        self,
        color_generator: ColorGenerator,
        height_min: int = 100,
        height_max: int = 720,
        width_min: int = 100,
        width_max: int = 1280,
    ):
        self.color_generator = color_generator
        self._h_min = height_min
        self._h_max = height_max
        self._w_min = width_min
        self._w_max = width_max

    def _sample_size(self) -> tuple[int, int]:
        width = rng.integers(self._w_min, self._w_max, dtype=int)
        height = rng.integers(self._h_min, self._h_max, dtype=int)

        return (height, width)

    def __call__(
        self,
    ) -> Array:
        color = self.color_generator()
        size = self._sample_size()

        channels = []
        for c in color:
            channel = np.full(size, c, dtype=np.uint8)
            channels.append(channel)

        array = np.stack(channels, axis=-1, dtype=np.uint8)

        return array

#!/usr/bin/env python
# By: Zack Taylor

from glob import glob

import numpy as np
from PIL import Image

from labyrinth.backgrounds.protocol import (
    BackgroundGeneratorProtocol,
    ColorSampler,
)
from labyrinth.types import Array

rng = np.random.default_rng()


class SolidBackgroundGenerator(BackgroundGeneratorProtocol):
    _color_generator: ColorSampler
    _h_min: int
    _h_max: int
    _w_min: int
    _w_max: int

    def __init__(
        self,
        color_generator: ColorSampler,
        height_min: int = 100,
        height_max: int = 720,
        width_min: int = 100,
        width_max: int = 1280,
    ):
        self._color_generator = color_generator
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
        color = self._color_generator()
        size = self._sample_size()

        channels = []
        for c in color:
            channel = np.full(size, c, dtype=np.uint8)
            channels.append(channel)

        array = np.stack(channels, axis=-1, dtype=np.uint8)

        return array


class FolderBackgroundGenerator(BackgroundGeneratorProtocol):
    _files: list[str]

    def __init__(
        self,
        image_folder: str,
        number_of_samples: int | None = None,
        glob_expression: str | None = None,
    ):
        expression = glob_expression if glob_expression is not None else "*"
        files = glob(f"{image_folder}/{expression}")

        if len(files) == 0:
            raise ValueError(f"No files found in folder. {image_folder}/{expression}")

        if number_of_samples is not None:
            files = list(rng.choice(files, size=number_of_samples))

        self._files = files

    def __call__(
        self,
    ) -> Array:
        file = rng.choice(self._files)

        img = Image.open(file)
        img = img.convert("RGB")

        return np.array(img, dtype=np.uint8)

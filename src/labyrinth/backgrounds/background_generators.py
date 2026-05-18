#!/usr/bin/env python
# By: Zack Taylor

from glob import glob

import numpy as np
from PIL import Image

from labyrinth.backgrounds.protocol import (
    BGSampler,
    ColorSampler,
)
from labyrinth.types import Array


class SolidBGSampler(BGSampler):
    _color_generator: ColorSampler
    _h_min: int
    _h_max: int
    _w_min: int
    _w_max: int
    _rng: np.random.Generator

    def __init__(
        self,
        color_generator: ColorSampler,
        height_min: int = 100,
        height_max: int = 720,
        width_min: int = 100,
        width_max: int = 1280,
        seed: int | None = None,
    ):
        self._color_generator = color_generator
        self._h_min = height_min
        self._h_max = height_max
        self._w_min = width_min
        self._w_max = width_max
        self._rng = np.random.default_rng(seed)

    def _sample_size(self) -> tuple[int, int]:
        width = self._rng.integers(self._w_min, self._w_max, dtype=int)
        height = self._rng.integers(self._h_min, self._h_max, dtype=int)

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


class FolderBGSampler(BGSampler):
    _files: list[str]
    _rng: np.random.Generator

    def __init__(
        self,
        image_folder: str,
        number_of_samples: int | None = None,
        glob_expression: str | None = None,
        seed: int | None = None,
    ):
        self._rng = np.random.default_rng(seed)

        expression = glob_expression if glob_expression is not None else "*"
        files = glob(f"{image_folder}/{expression}")

        if len(files) == 0:
            raise ValueError(f"No files found in folder. {image_folder}/{expression}")

        if number_of_samples is not None:
            files = list(self._rng.choice(files, size=number_of_samples))

        self._files = files

    def __call__(
        self,
    ) -> Array:
        file = self._rng.choice(self._files)

        img = Image.open(file)
        img = img.convert("RGB")

        return np.array(img, dtype=np.uint8)

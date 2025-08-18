#!/usr/bin/env python
# Vidrovr Inc.

import albumentations as A

from labyrinth.augmentations.protocol import ImageAugmentationProtocol
from labyrinth.types import Array


class DummyAugment(ImageAugmentationProtocol):
    def __init__(self) -> None:
        pass

    def __call__(self, array: Array) -> Array:
        return array


class AlbumAugmentation(ImageAugmentationProtocol):
    """Class that handels Albumentations augmentations."""

    _pipeline: A.Compose

    def __init__(self, pipeline: A.Compose) -> None:
        self._pipeline = pipeline

    def __call__(self, array: Array) -> Array:
        result = self._pipeline(image=array)
        return result["image"]

#!/usr/bin/env python
# Vidrovr Inc.

from typing import Tuple

from labyrinth.augmentations import ImageAugmentation
from labyrinth.backgrounds import BackgroundGenerator
from labyrinth.data_models.bounding_boxes import BoundingBox
from labyrinth.targets.sprite import SpritePlacer, SpriteSampler
from labyrinth.types import Array
from labyrinth.utils import with_timeout


class GenerateSample:
    _background_generator: BackgroundGenerator
    _sprite_sampler: SpriteSampler
    _sprite_placer: SpritePlacer
    _augment: ImageAugmentation | None

    def __init__(
        self,
        background_generator: BackgroundGenerator,
        sprite_sampler: SpriteSampler,
        sprite_placer: SpritePlacer,
        augment: ImageAugmentation | None = None,
    ) -> None:
        self._background_generator = background_generator
        self._sprite_sampler = sprite_sampler
        self._sprite_placer = sprite_placer
        self._augment = augment

    def __call__(
        self,
        label_id: int | None = None,
        timeout: int | None = None,
    ) -> Tuple[Array, list[int], list[BoundingBox]]:
        """Return an object detection training sample.

        Returns:
            image: Sample image with background and sprites
            labels: list of labels for each mask
            bboxs: list of bounding boxes for each mask
        """
        # Generate background
        background_array = self._background_generator()

        # Sample masks and labels
        mask_arrays, labels = self._sprite_sampler(label_id=label_id)

        # Augment the mask
        if self._augment:
            mask_arrays = [self._augment(mask_array) for mask_array in mask_arrays]

        # Place the masks and get the image and bounding boxes
        if timeout is not None:
            placed, bboxs = with_timeout(
                timeout, self._sprite_placer, mask_arrays, background_array
            )
        else:
            placed, bboxs = self._sprite_placer(mask_arrays, background_array)

        return placed, labels, bboxs

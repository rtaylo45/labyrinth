#!/usr/bin/env python
# Vidrovr Inc.

from typing import Tuple

from labyrinth.augmentations import ImageAugmentationProtocol
from labyrinth.backgrounds.models import (
    BackgroundComposition,
    BaseBackgroundGeneratorModel,
)
from labyrinth.data_models.bounding_boxes import BoundingBox
from labyrinth.modifiers import MaskBackgroundModifierProtocol
from labyrinth.targets.sprite.models import (
    BaseSpritePlacerModel,
    BaseSpriteSamplerModel,
)
from labyrinth.types import Array
from labyrinth.utils import with_timeout


class GenerateSample:
    _background_generator: BaseBackgroundGeneratorModel | BackgroundComposition
    _sprite_sampler: BaseSpriteSamplerModel
    _sprite_placer: BaseSpritePlacerModel
    _sprite_augment: ImageAugmentationProtocol | None
    _background_augment: ImageAugmentationProtocol | None
    _sample_augment: ImageAugmentationProtocol | None
    _mask_background_modifier: MaskBackgroundModifierProtocol | None

    def __init__(
        self,
        background_generator: BaseBackgroundGeneratorModel | BackgroundComposition,
        sprite_sampler: BaseSpriteSamplerModel,
        sprite_placer: BaseSpritePlacerModel,
        sprite_augment: ImageAugmentationProtocol | None = None,
        background_augment: ImageAugmentationProtocol | None = None,
        sample_augment: ImageAugmentationProtocol | None = None,
        mask_background_modifier: MaskBackgroundModifierProtocol | None = None,
    ) -> None:
        self._background_generator = background_generator
        self._sprite_sampler = sprite_sampler
        self._sprite_placer = sprite_placer
        self._sprite_augment = sprite_augment
        self._background_augment = background_augment
        self._sample_augment = sample_augment
        self._mask_background_modifier = mask_background_modifier

    def __call__(
        self,
        label_id: int | None = None,
        timeout: int | None = None,
    ) -> Tuple[Array, list[int], list[BoundingBox]]:
        """Return an object detection training sample.

        Args:
            label_id: ID for the specific sprite label you want.
            timeout: Max amount of time you want to spend generating the sample.

        Returns:
            image: Sample image with background and sprites
            labels: list of labels for each mask
            bboxs: list of bounding boxes for each mask
        """
        # Generate background
        background_array = self._background_generator()

        # Augment background
        if self._background_augment:
            background_array = self._background_augment(background_array)

        # Sample masks and labels
        mask_arrays, labels = self._sprite_sampler(label_id=label_id)

        # Augment the mask
        if self._sprite_augment:
            mask_arrays = [
                self._sprite_augment(mask_array) for mask_array in mask_arrays
            ]

        # Modify mask/backgrounds dependently
        if self._mask_background_modifier:
            mask_arrays, background_array = self._mask_background_modifier(
                mask_arrays, background_array
            )

        # Place the masks and get the image and bounding boxes
        if timeout is not None:
            placed, bboxs = with_timeout(
                timeout, self._sprite_placer, mask_arrays, background_array
            )
        else:
            placed, bboxs = self._sprite_placer(mask_arrays, background_array)

        # Augment the full sample image (background + mask) PIXEL LEVEL AUGMENTATIONS ONLY.. for now
        if self._sample_augment:
            placed = self._sample_augment(placed)

        return placed, labels, bboxs

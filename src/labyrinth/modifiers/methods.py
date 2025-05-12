#!/usr/bin/env python
# Vidrovr Inc.

from typing import List, Tuple

import albumentations as A

from labyrinth.types import Array


class AlbumMaskBackgroundModifier:
    _pipeline: A.Compose

    def __init__(self, pipeline: A.Compose) -> None:
        self._pipeline = pipeline

    def __call__(
        self,
        mask_arrays: List[Array],
        background_array: Array,
    ) -> Tuple[List[Array], Array]:
        transformed = self._pipeline(image=background_array, target_domain=mask_arrays)
        trans_img = transformed["image"]

        return mask_arrays, trans_img


class FourierDomainAdaptation:
    _fda: A.ImageOnlyTransform

    def __init__(
        self,
        beta_limit: Tuple[float, float] = (0, 0.1),
        p: float = 1.0,
    ) -> None:
        self._fda = A.FDA(
            beta_limit=beta_limit,
            p=p,
            metadata_key="target_domain",
            read_fn=lambda x: x,
        )

    def _resize_mask(
        self,
        mask_arrays: List[Array],
        height: int,
        width: int,
    ) -> List[Array]:
        T = A.Resize(height=height, width=width)

        transformed = [T(image=image_array)["image"] for image_array in mask_arrays]

        return transformed

    def __call__(
        self,
        mask_arrays: List[Array],
        background_array: Array,
    ) -> Tuple[List[Array], Array]:
        # Get background image dims
        height, width = background_array.shape[:2]

        # Resize the mask to the background image size so we can sample from the mask
        resized_masks = self._resize_mask(mask_arrays, height, width)

        # Apply transform on background array using the list of mask arrays as the
        # target domain. Interally Albumentations randomly picks one of these.
        transformed = self._fda(image=background_array, target_domain=resized_masks)
        trans_img = transformed["image"]

        return mask_arrays, trans_img


class HistogramMatching:
    _hist: A.ImageOnlyTransform

    def __init__(
        self,
        blend_ratio: Tuple[float, float] = (0.5, 1.0),
        p: float = 1.0,
    ) -> None:
        self._hist = A.HistogramMatching(
            blend_ratio=blend_ratio,
            p=p,
            metadata_key="target_domain",
            read_fn=lambda x: x,
        )

    def __call__(
        self,
        mask_arrays: List[Array],
        background_array: Array,
    ) -> Tuple[List[Array], Array]:
        transformed = self._hist(image=background_array, target_domain=mask_arrays)
        trans_img = transformed["image"]

        return mask_arrays, trans_img

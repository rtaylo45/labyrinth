#!/usr/bin/env python
# Vidrovr Inc.

from typing import List, Tuple

import numpy as np

from labyrinth.data_models.bounding_boxes import BoundingBox
from labyrinth.types import HWCImage
from labyrinth.utils.general import calculate_iou
from labyrinth.utils.sprite import bbox_squeeze, place_sprite

rng = np.random.default_rng()


class UniformSpritePlacer:
    _bbcls: BoundingBox
    _x_min: int | None
    _y_min: int | None
    _x_max: int | None
    _y_max: int | None

    def __init__(
        self,
        bbox_cls: BoundingBox,
        x_min: int | None = None,
        y_min: int | None = None,
        x_max: int | None = None,
        y_max: int | None = None,
    ) -> None:
        self._bbcls = bbox_cls
        self._x_min = x_min
        self._y_min = y_min
        self._x_max = x_max
        self._y_max = y_max

    def _check_clash(
        self, candidate: BoundingBox, bboxs: List[BoundingBox], iou_thresh: float = 0.0
    ):
        # First mask to place just return true
        if len(bboxs) == 0:
            return False

        # Calculate IOU's. Don't expect a lot of bboxs so i didn't vectorize it
        ious = [
            calculate_iou(candidate.to_xyxy_list(), bbox.to_xyxy_list())
            for bbox in bboxs
        ]

        # Loop and check ious
        for iou in ious:
            if iou > iou_thresh:
                return True

        return False

    def __call__(
        self,
        mask_arrays: List[HWCImage[np.uint8]],
        background_array: HWCImage[np.uint8],
    ) -> Tuple[HWCImage[np.uint8], List[BoundingBox]]:
        """Places mask in an array and returns the array with bounding boxes.

        Args:
            mask_arrays: A list of the mask arrays to place
            background_array: The background to place the mask

        Returns:
            placed: A single image with all mask placed
            bboxs: A list of bounding boxes. The order is the same as the mask_arrays
        """
        # Collect sample range
        x_min = self._x_min if self._x_min else 0
        y_min = self._y_min if self._y_min else 0

        # Hold the xywh
        bboxs = []
        placed = background_array

        # Loop over mask to build the sample
        for mask_array in mask_arrays:
            # Need to do a while loop to place the object in a valid location
            box_clash = True
            x_start, y_start = 0, 0
            xywh = None
            mask_array = bbox_squeeze(mask_array)
            while box_clash:
                x_max = (
                    self._x_max
                    if self._x_max
                    else background_array.shape[1] - mask_array.shape[1]
                )
                y_max = (
                    self._y_max
                    if self._y_max
                    else background_array.shape[0] - mask_array.shape[0]
                )

                # Sample from the range
                try:
                    x_start = rng.integers(low=x_min, high=x_max)
                    y_start = rng.integers(low=y_min, high=y_max)
                except ValueError:
                    continue

                # This part assumes that the mask array is a cropped bounding box of the object
                h, w = mask_array.shape[:2]
                dict_ = {"x": x_start, "y": y_start, "width": w, "height": h}
                xywh = self._bbcls.from_dict(dict_)

                # If false break the while loop
                box_clash = self._check_clash(xywh, bboxs)

            # Place the mask and save bbox
            placed = place_sprite(x_start, y_start, placed, mask_array)
            if xywh is not None:
                bboxs.append(xywh)

        return placed, bboxs

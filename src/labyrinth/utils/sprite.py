#!/usr/bin/env python
# Vidrovr Inc.

import numpy as np
from pycocotools.mask import decode, frPyObjects

from labyrinth.data_models.annotations import SegmentationRLE
from labyrinth.types import Array, HWCImage


def sprite_read(annotation: SegmentationRLE) -> Array:
    """Converts the pydantic Annotation to a binary numpy array sprite.

    Args:
        annotation: Pydantic model for AnnotationRLE. This represents a segmentation
            annotation in the RLE uncompressed format

    Returns:
        sprite: Binary numpy array, indecated 0 or 1 for in the mask is located at
            those pixel locations

    """
    seg = {"counts": annotation.counts, "size": annotation.size}
    bbox = annotation.bbox
    h, w = annotation.size

    # pycoco is written in c and because of that i can't copy/past some
    # of their fuctions. Wish i could do that with this call
    rle = frPyObjects(seg, h, w)  # type: ignore

    # Decode RLE into numpy binary sprite mask
    sprite = decode(rle)

    # Crop the mask from the original image to a bounding box with only the sprite
    idxs = bbox.to_numpy_indices()

    return sprite[idxs]


def sprite_crop(
    image_array: HWCImage[np.uint8], binary_sprite: HWCImage[np.uint8]
) -> Array:
    """Returns a numpy array with sprite in forground and background is transparent.

    Args:
        image_array: Numpy array of the original image
        binary_sprite: Numpy array of the binary mask for the sprite

    Returns:
        transparent_sprite: Numpy array of RGBA image with a transparent background

    """
    binary_sprite_ = binary_sprite[..., None]

    transparent_mask = np.concatenate([image_array, binary_sprite_ * 255], axis=-1)

    return transparent_mask


def place_sprite(
    x_min: int,
    y_min: int,
    background_array: HWCImage[np.uint8],
    sprite_array: HWCImage[np.uint8],
) -> Array:
    """Places the sprite in the background.

    Inplace operation but I return the array anyway.

    Args:
        x_min: Upper left corner x value
        y_min: Upper left corner y value
        background_array: The background to place in the mask
        sprite_array: The sprite array

    Returns:
        background_array: The background with the sprite placed
    """
    # Get max pad dims
    mask_height, mask_width = sprite_array.shape[:2]
    background_height, background_width = background_array.shape[:2]
    y_max = np.min([background_height, y_min + mask_height])
    x_max = np.min([background_width, x_min + mask_width])
    patch_idx = np.s_[y_min:y_max, x_min:x_max, :3]

    # Get the background patch
    patch = background_array[patch_idx]
    sprite_alpha_idx = np.s_[: patch.shape[0], : patch.shape[1], 3]
    sprite_rgb_idx = np.s_[: patch.shape[0], : patch.shape[1], :3]

    # Convert 255 to 1 on alpha channel
    alpha_sprite = sprite_array[sprite_alpha_idx] // 255
    alpha_sprite = alpha_sprite[..., None]

    # Splice the patch together
    patch = sprite_array[sprite_rgb_idx] * alpha_sprite + patch * (1 - alpha_sprite)

    # Apply the mask
    background_array[patch_idx] = patch

    return background_array

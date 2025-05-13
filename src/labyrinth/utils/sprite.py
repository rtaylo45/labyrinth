#!/usr/bin/env python
# Vidrovr Inc.

from typing import Sequence

import numpy as np
from pycocotools.mask import decode, frPyObjects

from labyrinth.data_models.annotations import Annotation
from labyrinth.types import Array, HWCImage


def sprite_read(annotation: Annotation) -> Array:
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


def _is_converged(image: Array, guess: int, direction: str) -> bool:
    if direction[0] == "x":
        try:
            data_before = image[:, guess - 1]
        except IndexError:
            data_before = np.array([0, 0, 0])

        try:
            data_after = image[:, guess + 1]
        except IndexError:
            data_after = np.array([0, 0, 0])

        data = image[:, guess]

    else:
        try:
            data_before = image[guess - 1, :]
        except IndexError:
            data_before = np.array([0, 0, 0])

        try:
            data_after = image[guess + 1, :]
        except IndexError:
            data_after = np.array([0, 0, 0])

        data = image[guess, :]

    if direction[1] == "-":
        data_after_bool = False
        data_bool = False
        data_before_bool = True
    else:
        data_after_bool = True
        data_bool = False
        data_before_bool = False

    if (
        np.all(data_after == 0) == data_after_bool
        and np.all(data == 0) == data_bool
        and np.all(data_before == 0) == data_before_bool
    ):
        return True
    else:
        return False


def _run_convergence(min_: int, max_: int, image: Array, direction: str) -> int | None:
    max_iter = 10
    converged = False
    guess = None
    iter_ = 0
    while not converged:
        guess = int((max_ + min_) / 2)

        if direction[0] == "x":
            data = image[:, guess]
        else:
            data = image[guess, :]

        if not np.all(data == 0):
            converged = _is_converged(image, guess, direction)

            if direction[1] == "-":
                max_ = guess
            else:
                min_ = guess
        else:
            if direction[1] == "-":
                min_ = guess
            else:
                max_ = guess

        if min_ == max_ == guess:
            converged = True

        if iter_ >= max_iter:
            converged = True

        iter_ += 1

    return guess


def min_max_search(image: Array) -> Sequence:
    alpha = image[:, :, 3]
    height, width = alpha.shape

    height_mid = height // 2
    width_mid = width // 2

    options_dict = {
        "xmin": {"min": 0, "max": width_mid, "direction": "x-", "value": None},
        "xmax": {"min": width_mid, "max": width, "direction": "x+", "value": None},
        "ymin": {"min": 0, "max": height_mid, "direction": "y-", "value": None},
        "ymax": {"min": height_mid, "max": height, "direction": "y+", "value": None},
    }

    for _, options in options_dict.items():
        min_, max_ = options["min"], options["max"]
        value = _run_convergence(min_, max_, image, direction=options["direction"])
        options["value"] = value

    xmin = options_dict["xmin"]["value"]
    xmax = options_dict["xmax"]["value"]
    ymin = options_dict["ymin"]["value"]
    ymax = options_dict["ymax"]["value"]

    return np.s_[ymin:ymax, xmin:xmax, ...]


def bbox_squeeze(image: Array):
    """Reduces the image size to the smallest width/height which encloses the alpha layer.

    During sprite augmentation, it is possible to artifically increase the size of the sprite
    image. This probably mainly comes from using the SafeRotate augmentation. In order to keep
    the bounding box size as small as the object, it is necessary to reduce the mask image size
    to its smallest size. The fuction applies a search algorithm to find the smallest box which
    completly encloses the positive alpha layer.

    Args:
        image: Numpy image array of the mask.

    Returns:
        reduced_image: Numpy image array of the mask reduced to its smallest size.
    """
    idxs = min_max_search(image)
    reduced_image = image[idxs]

    return reduced_image

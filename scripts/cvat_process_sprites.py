#!/usr/bin/env python
# Vidrovr Inc.

import os

import numpy as np
from PIL import Image as PILImage

from labyrinth.data_models.annotations import Annotation, SegmentationRLE
from labyrinth.data_models.bounding_boxes import XYWH
from labyrinth.types import Array
from labyrinth.utils.loaders import coco_loader
from labyrinth.utils.sprite import sprite_crop, sprite_read


def get_subdir(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def remove_spaces(directory):
    """Renames folders/file within a directory to remove spaces from their names.

    Args:
        directory (str): The path to the directory containing the folders/files to rename.
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        new_name = item.replace(" ", "_")
        new_path = os.path.join(directory, new_name)
        if new_name != item:
            os.rename(item_path, new_path)

    return None


def generate_mask(anno: Annotation, image_file_name: str) -> PILImage.Image:
    """Returns a pil image of the mask with transparent background.

    Args:
        anno: Annotation from the COCO dataset
        image_file_name: Path to the image location

    Returns:
        image: transparent mask RGBA image
    """
    # Open image file and convert to array
    image_og = PILImage.open(image_file_name)
    image_og = image_og.convert("RGB")
    image_og_array: Array = np.array(image_og)

    # Crop the image based on the annotation bounding box
    idxs = anno.bbox.to_numpy_indices()
    image_og_array = image_og_array[idxs]

    # Read in the mask as a binary array, crop to a transparent array, and return pil image
    mask = sprite_read(anno)
    transparent_image_array = sprite_crop(image_og_array, mask)
    image = PILImage.fromarray(transparent_image_array)

    return image


def main(import_folder: str):
    """Creates and saves sprite mask.

    Args:
        import_folder: CVAT exported dataset path
    """
    cat_folders = [f"{import_folder}/{f}" for f in get_subdir(import_folder)]

    # Loop over cat folders
    for cat_folder in cat_folders:
        annotation_file = f"{cat_folder}/annotations/instances_default.json"

        # Coco model for the category
        coco = coco_loader(
            annotation_file,
            annotation_model=SegmentationRLE,  # type: ignore
            bbox_model=XYWH,  # type: ignore
        )

        # Cut stuff up so we can figure out the image folder name for the annotations
        anno_image_folder = f"{cat_folder}/images/default"
        print(f"Processing {os.path.basename(cat_folder)}")

        # Build annotation id to image id map
        anno_id_to_image = {
            value.id: value.image_id for _, value in coco.annotations.items()
        }

        # Loop over the map
        for anno_id, image_id in anno_id_to_image.items():
            annotation_orm = coco.annotations[anno_id]
            image_orm = coco.images[image_id]
            image_file_name = f"{anno_image_folder}/{image_orm.file_name}"
            anno = coco.annotations[anno_id]
            label_id = anno.category_id

            pil_img = generate_mask(annotation_orm, image_file_name)
            pil_img.save(
                f"{anno_image_folder}/{image_orm.file_name.split('.')[0]}_annoid_{anno_id}_labelid_{label_id}_mask.png",
                "png",
            )


if __name__ == "__main__":
    import fire

    fire.Fire(main)

#!/usr/bin/env python
# Vidrovr Inc.

import logging
import os
from glob import glob
from uuid import uuid4

from PIL import Image

from labyrinth.augmentations import DummyAugment
from labyrinth.backgrounds import (
    BackgroundGenerator,
    FolderBackgroundGenerator,
    RGBAColorGenerator,
    SolidBackgroundGenerator,
)
from labyrinth.data_models.annotations import SegmentationRLE
from labyrinth.data_models.bounding_boxes import CENTER_XYWH, XYWH
from labyrinth.sample_generators.object_detection import GenerateSample
from labyrinth.targets.sprite import COCOSpriteSampler, UniformSpritePlacer
from labyrinth.utils.exceptions import TimeoutException
from labyrinth.utils.loaders import coco_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("labyrinth")


def get_background_generator(name: str) -> BackgroundGenerator:
    background_gen = None
    if name == "folder":
        background_gen = FolderBackgroundGenerator(
            "/shared_data/oidv7/train", glob_expression="*.jpg"
        )

    elif name == "random":
        color_gen = RGBAColorGenerator(
            color_min=0,
            color_max=255,
            alpha_min=255,
            alpha_max=255,
        )

        background_gen = SolidBackgroundGenerator(
            color_generator=color_gen,
            height_min=2160 // 2,
            height_max=2160,
            width_min=3840 // 2,
            width_max=3840,
        )
    else:
        raise ValueError(f"Invalid background generator name {name}")

    return background_gen


def normalize_bbox(bbox, image_array) -> CENTER_XYWH:
    """Normalize bbox coords to the image size."""
    y_max, x_max = image_array.shape[:2]

    bbox.x = bbox.x / x_max
    bbox.y = bbox.y / y_max
    bbox.width = bbox.width / x_max
    bbox.height = bbox.height / y_max

    return bbox


def save_result(placed, labels, bboxs, output_dir) -> str:
    """Saves samples image and labels."""
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    image_id = uuid4()
    sample_fname = f"{image_id}.png"
    image_path = os.path.join(images_dir, sample_fname)
    img = Image.fromarray(placed)
    img.save(image_path, "png")

    label_fname = f"{image_id}.txt"
    label_path = os.path.join(labels_dir, label_fname)
    with open(label_path, "a") as f:
        for label, bbox in zip(labels, bboxs):
            line = f"{label} {bbox.x} {bbox.y} {bbox.width} {bbox.height}\n"
            f.write(line)

    return image_path


def get_mask_cat_counts(cocos) -> dict[int, int]:
    """Returns a dict of labels to number of mask examples."""
    cat_counts = {category.id - 1: 0 for category in cocos[0].categories}

    for coco in cocos:
        for _, anno in coco.annotations.items():
            label_id = anno.category_id - 1
            cat_counts[label_id] += 1

    return cat_counts


def make_folders(output_dir) -> None:
    """Creates all of the dataset folders."""
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")

    dirs = [
        output_dir,
        out_images_dir,
        out_labels_dir,
    ]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    return None


def dump_cats(categories, output_dir) -> None:
    """Saves the labels."""
    # Dump the categories
    cat_names = [cat.name for cat in categories]
    with open(f"{output_dir}/manifest.yaml", "a") as f:
        f.write("names:\n")
        for name in cat_names:
            line = f"- {name}\n"
            f.write(line)

    return None


def generate_samples(
    sample_gen,
    samples_per_cat,
    categories,
    mask_cat_counts,
    output_dir,
) -> None:
    total_samples = len(categories) * samples_per_cat
    cat_counts = {category.id - 1: 0 for category in categories}
    activate_samples = 0

    logger.info(f"Generate samples: {total_samples}")

    # Initial loop to generate samples
    while activate_samples < total_samples:
        # Do the damn thing
        try:
            placed, labels, bboxs = sample_gen(timeout=2)
            bboxs = [bbox.to_cxywh() for bbox in bboxs]
            labels = [label - 1 for label in labels]
            bboxs = [normalize_bbox(bbox, placed) for bbox in bboxs]

            save_result(placed, labels, bboxs, output_dir)

            for label in labels:
                cat_counts[label] += 1

            activate_samples += 1
        except TimeoutException:
            continue

    # More of a manual loop to make sure we get enough examples of each label type
    for name, _ in cat_counts.items():
        num_samples = cat_counts[name]
        num_mask = mask_cat_counts[name]

        if num_mask > 0:
            while num_samples < samples_per_cat:
                try:
                    placed, labels, bboxs = sample_gen(timeout=2, label_id=name)
                    bboxs = [bbox.to_cxywh() for bbox in bboxs]
                    labels = [label - 1 for label in labels]
                    bboxs = [normalize_bbox(bbox, placed) for bbox in bboxs]

                    save_result(placed, labels, bboxs, output_dir)

                    for label in labels:
                        cat_counts[label] += 1

                    num_samples += 1
                except TimeoutException:
                    continue


def main(
    output_dir: str,
    import_folder: str,
    min_samples_per_label: int,
    background_name: str = "random",
) -> None:
    """Generates the dataset.

    Args:
        output_dir: Directory to place the generated dataset
        import_folder: Directory with original dataset/mask
        min_samples_per_label: The min number of samples per label
        background_name: Type of background generator you want to use
    """
    # Gobble up all those annotation files
    annotations_folder = f"{import_folder}/annotations"
    annotation_files = glob(f"{annotations_folder}/*.json")

    # Create folders
    make_folders(output_dir)

    # Build the coco dataset
    cocos = [
        coco_loader(file, annotation_model=SegmentationRLE, bbox_model=XYWH)  # type: ignore
        for file in annotation_files
    ]

    # Objects
    background_gen = get_background_generator(background_name)
    mask_samplers = [
        COCOSpriteSampler(coco, import_folder, max_num_sprites=4) for coco in cocos
    ]
    mask_sampler = sum(mask_samplers[1:], mask_samplers[0])
    mask_placer = UniformSpritePlacer(bbox_cls=XYWH)  # type: ignore
    augment = DummyAugment()

    sample_gen = GenerateSample(
        background_generator=background_gen,
        sprite_sampler=mask_sampler,
        sprite_placer=mask_placer,
        augment=augment,
    )

    categories = cocos[0].categories
    mask_cat_counts = get_mask_cat_counts(cocos)

    # Generate the samples
    generate_samples(
        sample_gen, min_samples_per_label, categories, mask_cat_counts, output_dir
    )

    # Dump the categories
    dump_cats(categories, output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

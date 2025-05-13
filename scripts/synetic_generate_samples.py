#!/usr/bin/env python
# Vidrovr Inc.

import logging
import os
from uuid import uuid4

import albumentations as A
from PIL import Image

from labyrinth.augmentations import AlbumAugmentation
from labyrinth.backgrounds import (
    BackgroundGenerator,
    FolderBackgroundGenerator,
    RGBAColorGenerator,
    SolidBackgroundGenerator,
)
from labyrinth.data_models.bounding_boxes import CENTER_XYWH, XYWH
from labyrinth.modifiers import AlbumMaskBackgroundModifier
from labyrinth.sample_generators.object_detection import GenerateSample
from labyrinth.targets.sprite import FolderSpriteSampler, UniformSpritePlacer
from labyrinth.utils.exceptions import TimeoutException

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


def dump_cats(names, output_dir) -> None:
    """Saves the labels."""
    # Dump the categories
    with open(f"{output_dir}/manifest.yaml", "a") as f:
        f.write("names:\n")
        for name in names:
            line = f"- {name}\n"
            f.write(line)

    return None


def generate_samples(
    sample_gen,
    samples_per_cat,
    output_dir,
) -> None:
    labels = list(sample_gen._sprite_sampler._sprite_files.keys())
    total_samples = len(labels) * samples_per_cat
    cat_counts = {label: 0 for label in labels}
    activate_samples = 0

    logger.info(f"Generate samples: {total_samples}")

    # Initial loop to generate samples
    while activate_samples < total_samples:
        # Do the damn thing
        try:
            placed, labels, bboxs = sample_gen(timeout=2)
            bboxs = [bbox.to_cxywh() for bbox in bboxs]
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

        while num_samples < samples_per_cat:
            try:
                placed, labels, bboxs = sample_gen(timeout=2, label_id=name)
                bboxs = [bbox.to_cxywh() for bbox in bboxs]
                bboxs = [normalize_bbox(bbox, placed) for bbox in bboxs]

                save_result(placed, labels, bboxs, output_dir)

                for label in labels:
                    cat_counts[label] += 1

                num_samples += 1
            except TimeoutException:
                continue


def get_sprite_augmentation() -> A.Compose:
    pipeline = A.Compose(
        [
            A.RandomScale(scale_limit=(-0.9, -0.4), p=1.0),
            A.SafeRotate(limit=180, p=0.9),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(p=0.2),
            A.Downscale(p=0.2),
            A.CoarseDropout(
                num_holes_range=(2, 5),
                hole_height_range=(20, 90),
                hole_width_range=(20, 90),
                fill=(0, 0, 0, 0),
                p=0.2,
            ),
        ]
    )

    return pipeline


def get_background_augmentation() -> A.Compose:
    pipeline = A.Compose(
        [
            A.HistogramMatching(
                read_fn=lambda x: x,
                metadata_key="target_domain",
                p=1.0,
            ),
            A.FDA(
                read_fn=lambda x: x,
                metadata_key="target_domain",
                p=0.8,
            ),
        ]
    )

    return pipeline


def main(
    output_dir: str,
    mask_folder: str,
    min_samples_per_label: int,
    background_name: str = "random",
) -> None:
    """Generates the dataset.

    Args:
        output_dir: Directory to place the generated dataset
        mask_folder: Directory with mask images
        min_samples_per_label: The min number of samples per label
        background_name: Type of background generator you want to use
    """
    # Create folders
    make_folders(output_dir)

    # Objects
    background_gen = get_background_generator(background_name)
    mask_sampler = FolderSpriteSampler(mask_folder, max_num_sprites=10)
    mask_placer = UniformSpritePlacer(bbox_cls=XYWH)  # type: ignore
    sprite_aug_pipeline = get_sprite_augmentation()
    sprite_augment = AlbumAugmentation(sprite_aug_pipeline)
    background_aug_pipeline = get_background_augmentation()
    mask_background_mod = AlbumMaskBackgroundModifier(background_aug_pipeline)

    sample_gen = GenerateSample(
        background_generator=background_gen,
        sprite_sampler=mask_sampler,
        sprite_placer=mask_placer,
        sprite_augment=sprite_augment,
        mask_background_modifier=mask_background_mod,
    )

    # Generate the samples
    generate_samples(sample_gen, min_samples_per_label, output_dir)

    names = ["T-90"]
    # Dump the categories
    dump_cats(names, output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

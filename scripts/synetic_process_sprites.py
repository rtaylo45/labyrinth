import os
from functools import partial
from glob import glob

import numpy as np
from PIL import Image as PILImage

from labyrinth.utils.multiprocessing import run_imap


def check_path(path: str) -> None:
    """Checks to see if a path exists."""
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exists")


def read_anno(anno_path: str, img_path: str) -> dict[str, int]:
    """Returns annotation in XYWH format.

    Args:
        anno_path: Absolute path to annotation file
        img_path: Absolute path to image file

    Returns:
        anno: Dict mapping annotation properties to their values.
            label_id -> label id (int)
            x -> upper left x (int)
            y -> upper left y (int)
            w -> width (int)
            h -> height (int)
    """
    img_array = np.array(PILImage.open(img_path))
    height, width = img_array.shape[:2]

    with open(anno_path, "r") as f:
        line = f.readlines()
        if len(line) == 0:
            raise KeyError("Data not found")
        elif len(line) > 1:
            raise ValueError("Too many annotations in this file")

    line = line[0].split()
    line = [float(val) for val in line]
    label_id, rcx, rcy, rw, rh = line
    w = int(rw * width)
    h = int(rh * height)
    x = int((rcx * width) - (w / 2.0))
    y = int((rcy * height) - (h / 2.0))
    anno = {
        "label_id": int(label_id),
        "x": x,
        "y": y,
        "width": w,
        "height": h,
    }

    return anno


def crop_and_save(
    img_path: str,
    anno: dict[str, int],
    outputdir: str | None = None,
) -> None:
    """Crops the image to the bounding box and saves result."""
    label_id = anno["label_id"]
    x0 = anno["x"]
    y0 = anno["y"]
    x1 = x0 + anno["width"]
    y1 = y0 + anno["height"]

    idxs = np.s_[y0:y1, x0:x1, ...]

    img_array = np.array(PILImage.open(img_path))
    img_array = img_array[idxs]

    if outputdir is None:
        dirname = os.path.dirname(img_path)
    else:
        dirname = outputdir
    basename = os.path.basename(img_path).split(".")[0]
    basename = f"{basename}_labelid_{label_id}_mask.png"
    save_path = f"{dirname}/{basename}"

    img = PILImage.fromarray(img_array)
    img.save(save_path)

    # Remove file if its an inplace operation
    if outputdir is None:
        os.remove(img_path)

    return None


def load_image_to_annos(
    import_folder: str,
    subfolder: str = "trains",
    img_format: str = "png",
) -> dict[str, str]:
    image_folder = f"{import_folder}/images/{subfolder}"
    anno_folder = f"{import_folder}/labels/{subfolder}-det"

    check_path(import_folder)
    check_path(image_folder)
    check_path(anno_folder)

    image_files = glob(f"{image_folder}/*[!mask].{img_format}")

    if len(image_files) == 0:
        raise ValueError("No images found!")

    mapping = {}
    for file in image_files:
        base_name = os.path.basename(file).split(".")[0]
        anno_file = f"{anno_folder}/{base_name}.txt"
        check_path(anno_file)
        mapping[file] = anno_file

    return mapping


def process_image(
    image_path: str,
    image_anno_map: dict[str, str] | None = None,
    output_directory: str | None = None,
) -> None:
    if image_anno_map is not None:
        anno_path = image_anno_map[image_path]

        # They left in some annotation/image that are blank. No idea why
        try:
            anno = read_anno(anno_path, image_path)
            crop_and_save(image_path, anno, outputdir=output_directory)
        except KeyError:
            return None


def main(
    import_folder: str,
    num_processors: int,
    output_directory: str | None = None,
) -> None:
    image_anno_map = load_image_to_annos(import_folder)

    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

    image_names = list(image_anno_map.keys())
    proc = partial(
        process_image, image_anno_map=image_anno_map, output_directory=output_directory
    )

    run_imap(proc, image_names, num_processes=num_processors)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)

#!/usr/bin/env python
# Vidrovr Inc.

import contextlib
import os
from typing import Any, Protocol

from pycocotools.coco import COCO as PYCOCO

from labyrinth.data_models.coco import COCO, Category
from labyrinth.data_models.media import Image


class Annotation(Protocol):
    def from_pycoco(self, annotation: Any) -> Any: ...


def coco_loader(annotation_file: str, annotation_model: Annotation) -> COCO:
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            pycoco = PYCOCO(annotation_file=annotation_file)

    coco = COCO()
    coco.annotation_file = annotation_file
    coco.categories = [Category(**c) for c in pycoco.dataset["categories"]]
    coco.images = {i["id"]: Image(**i) for i in pycoco.dataset["images"]}
    coco.annotations = {
        a["id"]: annotation_model.from_pycoco(a) for a in pycoco.dataset["annotations"]
    }

    return coco

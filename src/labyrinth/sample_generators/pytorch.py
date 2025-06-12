#!/usr/bin/env python
# Vidrovr Inc.
from typing import Self, Tuple

from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes

from labyrinth.sample_generators.object_detection import GenerateSample
from labyrinth.types import Array


class LabyObjectSet(Dataset):
    _sample_generator: GenerateSample
    _max_samples: int
    _transform: v2.Compose | None
    _timeout: int | None

    def __init__(
        self: Self,
        sample_generator: GenerateSample,
        max_samples: int,
        transform: v2.Compose | None,
        timeout: int | None = None,
    ) -> None:
        """Pytorch dataset class wrapper for laby.

        Args:
            sample_generator: Sample generator for object detection.
                This sample generator must use the XYWH bbox_cls.
            max_samples: Max number of samples to generate.
            transform: Torchvision compose transform.
            timeout: Time out limit for generating the sample.

        """
        self._sample_generator = sample_generator
        self._max_samples = max_samples
        self._transform = transform
        self._timeout = timeout

    def __len__(
        self: Self,
    ):
        """Return max samples as this thing can go forever."""
        return self._max_samples

    def __getitem__(
        self: Self, idx: int
    ) -> Tuple[Array, list[int], list[BoundingBoxes]]:
        """Return an object detection training sample.

        Args:
            idx: Pytorch training sample index. Not used.

        Returns:
            image: Sample image with background and sprites
            labels: list of labels for each mask
            bboxs: list of bounding boxes for each mask
        """
        image, labels, xywh_bboxs = self._sample_generator(
            label_id=None, timeout=self._timeout
        )
        h, w, _ = image.shape
        raw_bboxs = [[b.x, b.y, b.width, b.height] for b in xywh_bboxs]  # type: ignore
        tv_bboxs = [
            BoundingBoxes(data=b, format="XYWH", canvas_size=(h, w))  # type: ignore
            for b in raw_bboxs
        ]

        if self._transform is not None:
            image, tv_bboxs = self._transform(image, tv_bboxs)

        return image, labels, tv_bboxs

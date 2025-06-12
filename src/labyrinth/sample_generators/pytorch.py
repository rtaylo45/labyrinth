#!/usr/bin/env python
# Vidrovr Inc.
from typing import Self, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes

from labyrinth.sample_generators.object_detection import GenerateSample
from labyrinth.utils.exceptions import TimeoutException


def list_collate_fn(
    batch: list[Tuple[torch.Tensor, list[int], list[BoundingBoxes]]],
) -> Tuple[list[torch.Tensor], list[list[int]], list[list[BoundingBoxes]]]:
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    bboxs = [item[2] for item in batch]

    return imgs, labels, bboxs


class LabyObjectSet(Dataset):
    _sample_generator: GenerateSample
    _max_samples: int
    _transform: v2.Compose | None
    _timeout: int | None

    def __init__(
        self: Self,
        sample_generator: GenerateSample,
        max_samples: int,
        transform: v2.Compose | None = None,
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
    ) -> Tuple[torch.Tensor, list[int], list[BoundingBoxes]]:
        """Return an object detection training sample.

        Args:
            idx: Pytorch training sample index. Not used.

        Returns:
            image: Sample image with background and sprites
            labels: list of labels for each mask
            bboxs: list of bounding boxes for each mask
        """
        if idx >= self._max_samples:
            raise IndexError

        no_sample = True
        image, labels, xywh_bboxs = None, None, None
        while no_sample:
            try:
                image, labels, xywh_bboxs = self._sample_generator(
                    label_id=None, timeout=self._timeout
                )
                no_sample = False
            except TimeoutException:
                continue

        # Adding this part because of typing but should never happen
        if (image is None) or (labels is None) or (xywh_bboxs is None):
            raise SystemError("Hellow super user! You have fucked up")

        h, w, _ = image.shape
        raw_bboxs = [[b.x, b.y, b.width, b.height] for b in xywh_bboxs]  # type: ignore
        tv_bboxs = [
            BoundingBoxes(data=b, format="XYWH", canvas_size=(h, w))  # type: ignore
            for b in raw_bboxs
        ]

        if self._transform is not None:
            image, tv_bboxs = self._transform(image, tv_bboxs)

        # Convert to torch tensor
        image = torch.Tensor(image)

        return image, labels, tv_bboxs

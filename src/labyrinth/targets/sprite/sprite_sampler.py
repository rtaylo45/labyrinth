#!/usr/bin/env python
# Vidrovr Inc.

import os
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

from labyrinth.data_models.coco import COCO
from labyrinth.types import Array

rng = np.random.default_rng()


class COCOSpriteSampler:
    _is_coco_seq: bool
    _coco: COCO | None
    _import_folder: str | None
    _max_num_sprites: int
    _mask_files: dict[int, List[str]]

    def __init__(
        self,
        coco: COCO | None,
        import_folder: str | None,
        max_num_sprites: int = 1,
    ) -> None:
        self._coco = coco
        self._import_folder = import_folder
        if (coco is not None) or (import_folder is not None):
            self._mask_files = self._get_sprite_files()
        self._max_num_sprites = max_num_sprites

    def __add__(self, other):
        if isinstance(other, COCOSpriteSampler):
            max_num_sprites = np.max([self._max_num_sprites, other._max_num_sprites])
            ret = COCOSpriteSampler(
                coco=None, import_folder=None, max_num_sprites=max_num_sprites
            )
            ret._mask_files = self._mask_files

            for key, value in other._mask_files.items():
                current_values = ret._mask_files.get(key)
                if current_values is None:
                    ret._mask_files[key] = value
                else:
                    values = []
                    for v in value:
                        values.append(v)
                    ret._mask_files[key] = values

            return ret
        else:
            raise TypeError("Unsupported operation.")

    def _get_sprite_files(self) -> dict[int, list[str]]:
        assert self._coco is not None

        anno_image_folder = (
            os.path.basename(self._coco.annotation_file)
            .split(".")[0]
            .replace("instances_", "")
        )

        mask_files = glob(
            f"{self._import_folder}/images/{anno_image_folder}/*_mask.png"
        )

        label_ids = [self._get_label_id(file) for file in mask_files]
        label_sprite_map = {label_id: [] for label_id in label_ids}

        for label_id, mask_file in zip(label_ids, mask_files):
            label_sprite_map[label_id].append(mask_file)

        return label_sprite_map

    def _get_id(self, file_name: str, keyword: str) -> int:
        base = os.path.basename(file_name)
        split = base.split("_")

        i = -1
        for i, s in enumerate(split):
            if s == keyword:
                break

        if i == (len(split) - 1):
            raise ValueError("No {keyword} found.")

        id = int(split[i + 1])

        return id

    def _get_anno_id(self, file_name: str):
        return self._get_id(file_name, keyword="annoid")

    def _get_label_id(self, file_name: str) -> int:
        return self._get_id(file_name, keyword="labelid")

    def _sample_files(self, label_id: int | None = None) -> list[str]:
        if label_id is not None:
            file_range = self._mask_files[label_id]
            num_mask = (
                rng.integers(low=1, high=self._max_num_sprites)
                if self._max_num_sprites != 1
                else 1
            )

            files = list(rng.choice(file_range, size=num_mask))
        else:
            label_range = list(self._mask_files.keys())
            num_mask = (
                rng.integers(low=1, high=self._max_num_sprites)
                if self._max_num_sprites != 1
                else 1
            )
            labels = list(rng.choice(label_range, size=num_mask))

            files = []
            for label in labels:
                mask_range = self._mask_files[label]
                files.append(rng.choice(mask_range))

        return files

    def _read_sprite(self, file: str) -> Array:
        return np.array(Image.open(file), dtype=np.uint8)

    def __call__(
        self,
        label_id: int | None = None,
    ) -> Tuple[List[Array], List[int]]:
        mask_files = self._sample_files(label_id=label_id)
        labels = [self._get_label_id(file_name) for file_name in mask_files]
        mask_arrays = [self._read_sprite(file_name) for file_name in mask_files]

        return mask_arrays, labels

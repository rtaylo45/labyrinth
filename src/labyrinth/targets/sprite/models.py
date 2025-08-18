from abc import ABC, abstractmethod
from typing import Annotated, Any, Callable, List, Literal, Tuple

import numpy as np
from pydantic import BaseModel, DirectoryPath, PositiveInt, confloat

from labyrinth.data_models.bounding_boxes import BoundingBox
from labyrinth.data_models.coco import COCO
from labyrinth.targets.sprite import (
    COCOSpriteSampler,
    FolderSpriteSampler,
    UniformSpritePlacer,
)
from labyrinth.types import HWCImage

ACCEPTED_METHODS = Literal["coco", "folder"]


class BaseSpriteSamplerModel(ABC, BaseModel, extra="allow"):
    @abstractmethod
    def model_post_init(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[Any, Any]: ...


class BaseSpritePlacerModel(
    ABC, BaseModel, extra="allow", arbitrary_types_allowed=True
):
    @abstractmethod
    def model_post_init(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def __call__(
        self, *args, **kwargs
    ) -> Tuple[HWCImage[np.uint8], List[BoundingBox]]: ...


class COCOSpriteSamplerModel(BaseSpriteSamplerModel):
    method: Literal["coco"] = "coco"
    coco: COCO | None = None
    import_folder: DirectoryPath | None = None
    max_num_sprites: PositiveInt = 1

    def model_post_init(self, _) -> None:
        sprite_sampler = COCOSpriteSampler(
            coco=self.coco,
            import_folder=str(self.import_folder),
            max_num_sprites=self.max_num_sprites,
        )
        self.sampler = sprite_sampler

        return None

    def __call__(self, label_id: int | None = None) -> Tuple[Any, Any]:
        return self.sampler(label_id=label_id)


class FolderSpriteSamplerModel(BaseSpriteSamplerModel):
    method: Literal["folder"] = "folder"
    import_folder: DirectoryPath | List[DirectoryPath]
    max_num_sprites: PositiveInt = 1
    glob_expression: str | None = None
    folder_ids: int | List[int] | None = None

    def model_post_init(self, _) -> None:
        folder = self.import_folder
        if type(folder) is not list:
            folder = [str(folder)]
        else:
            folder = [str(f) for f in folder]

        folder_ids = self.folder_ids
        if (type(folder_ids) is int) and (folder_ids is not None):
            folder_ids = [folder_ids]

        sprite_sampler = FolderSpriteSampler(
            folder=folder,
            max_num_sprites=self.max_num_sprites,
            glob_expression=self.glob_expression,
            folder_ids=folder_ids,
        )

        self.sampler = sprite_sampler

        return None

    def __call__(self, label_id: int | None = None) -> Tuple[Any, Any]:
        return self.sampler(label_id=label_id)


class UniformSpritePlacerModel(BaseSpritePlacerModel):
    bbcls: BoundingBox
    x_min: PositiveInt | None = None
    y_min: PositiveInt | None = None
    x_max: PositiveInt | None = None
    y_max: PositiveInt | None = None
    alpha_blend: Callable[[], Annotated[float, confloat(ge=0.0, le=1.0)]] | None = None

    def model_post_init(self, _) -> None:
        sprite_placer = UniformSpritePlacer(
            bbox_cls=self.bbcls,
            x_min=self.x_min,
            y_min=self.y_min,
            x_max=self.x_max,
            y_max=self.y_max,
            alpha_blend=self.alpha_blend,
        )

        self.sprite_placer = sprite_placer

        return None

    def __call__(
        self,
        mask_arrays: List[HWCImage[np.uint8]],
        background_array: HWCImage[np.uint8],
    ) -> Tuple[HWCImage[np.uint8], List[BoundingBox]]:
        return self.sprite_placer(
            mask_arrays=mask_arrays, background_array=background_array
        )

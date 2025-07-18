from abc import ABC, abstractmethod
from typing import Any, List, Literal

from pydantic import BaseModel, DirectoryPath, PositiveInt

from labyrinth.data_models.coco import COCO
from labyrinth.targets.sprite import COCOSpriteSampler, FolderSpriteSampler

ACCEPTED_METHODS = Literal["coco", "folder"]


class BaseSpriteSamplerModel(ABC, BaseModel, extra="allow"):
    @abstractmethod
    def model_post_init(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any: ...


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

    def __call__(self, label_id: int | None = None) -> Any:
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

    def __call__(self, label_id: int | None = None) -> Any:
        return self.sampler(label_id=label_id)

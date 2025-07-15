from typing import List, Literal, get_args

from pydantic import BaseModel, DirectoryPath, PositiveInt

from labyrinth.data_models.coco import COCO
from labyrinth.targets.sprite import COCOSpriteSampler, FolderSpriteSampler

ACCEPTED_METHODS = Literal["coco", "folder"]


class COCOSpriteSamplerModel(BaseModel):
    coco: COCO | None = None
    import_folder: DirectoryPath | None = None
    max_num_sprites: PositiveInt = 1

    def build_sampler(
        self,
    ):
        sprite_sampler = COCOSpriteSampler(
            coco=self.coco,
            import_folder=str(self.import_folder),
            max_num_sprites=self.max_num_sprites,
        )

        return sprite_sampler


class FolderSpriteSamplerModel(BaseModel):
    folder: DirectoryPath | List[DirectoryPath]
    max_num_sprites: PositiveInt = 1
    glob_expression: str | None = None
    folder_ids: int | List[int] | None = None

    def build_sampler(
        self,
    ) -> FolderSpriteSampler:
        sprite_sampler = FolderSpriteSampler(
            folder=str(self.folder),
            max_num_sprites=self.max_num_sprites,
            glob_expression=self.glob_expression,
            folder_ids=self.folder_ids,
        )

        return sprite_sampler


def sprite_sampler(
    method: ACCEPTED_METHODS,
    **kwargs,
) -> COCOSpriteSampler | FolderSpriteSampler:
    match method:
        case "coco":
            return COCOSpriteSamplerModel(**kwargs).build_sampler()
        case "folder":
            return FolderSpriteSamplerModel(**kwargs).build_sampler()
        case _:
            raise ValueError(
                f"method ({method}) must be one of {get_args(ACCEPTED_METHODS)}"
            )

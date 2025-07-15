from typing import Literal, get_args

from pydantic import BaseModel, DirectoryPath, PositiveInt

from labyrinth.backgrounds import (
    FolderBackgroundGenerator,
    RGBAColorGenerator,
    RGBColorGenerator,
    SolidBackgroundGenerator,
)

ACCEPTED_METHODS = Literal["solid", "folder"]
ACCEPTED_SOLID_MODES = Literal["RGB", "RGBA"]
MIN_HEIGHT = 100
MAX_HEIGHT = 720
MIN_WIDTH = 100
MAX_WIDTH = 1280


class SolidBackgroundGeneratorModel(BaseModel):
    mode: ACCEPTED_SOLID_MODES
    min_height: PositiveInt = MIN_HEIGHT
    max_height: PositiveInt = MAX_HEIGHT
    min_width: PositiveInt = MIN_WIDTH
    max_width: PositiveInt = MAX_HEIGHT

    def build_generator(
        self,
    ) -> SolidBackgroundGenerator:
        if self.mode == "RGB":
            color_gen = RGBColorGenerator()
        else:
            color_gen = RGBAColorGenerator()

        background_gen = SolidBackgroundGenerator(
            color_generator=color_gen,
            height_min=self.min_height,
            height_max=self.max_height,
            width_min=self.min_width,
            width_max=self.max_width,
        )

        return background_gen


class FolderBackgroundGeneratorModel(BaseModel):
    image_folder: DirectoryPath
    number_of_samples: PositiveInt | None = None
    glob_expression: str | None = None

    def build_generator(
        self,
    ) -> FolderBackgroundGenerator:
        background_gen = FolderBackgroundGenerator(
            image_folder=str(self.image_folder),
            number_of_samples=self.number_of_samples,
            glob_expression=self.glob_expression,
        )

        return background_gen


def background_generator(
    method: ACCEPTED_METHODS,
    **kwargs,
) -> SolidBackgroundGenerator | FolderBackgroundGenerator:
    match method:
        case "solid":
            return SolidBackgroundGeneratorModel(**kwargs).build_generator()
        case "folder":
            return FolderBackgroundGeneratorModel(**kwargs).build_generator()
        case _:
            raise ValueError(
                f"method ({method}) must be one of {get_args(ACCEPTED_METHODS)}"
            )

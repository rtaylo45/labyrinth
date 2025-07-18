import warnings
from abc import ABC, abstractmethod
from typing import Annotated, Any, List, Literal, Self

import cv2
import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    DirectoryPath,
    PositiveInt,
    confloat,
    model_validator,
)

from labyrinth.backgrounds import (
    FolderBackgroundGenerator,
    RGBAColorGenerator,
    RGBColorGenerator,
    SolidBackgroundGenerator,
)

rng = np.random.default_rng()


ACCEPTED_SOLID_MODES = Literal["RGB", "RGBA"]
BLEND_RESIZE_MODES = Literal["max_size", "min_size"]
MIN_HEIGHT = 100
MAX_HEIGHT = 720
MIN_WIDTH = 100
MAX_WIDTH = 1280


class BaseBackgroundGeneratorModel(ABC, BaseModel, extra="allow"):
    @abstractmethod
    def model_post_init(self, *args, **kwargs): ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any: ...


class SolidBackgroundGeneratorModel(BaseBackgroundGeneratorModel):
    method: Literal["solid"] = "solid"
    mode: ACCEPTED_SOLID_MODES = "RGB"
    min_height: PositiveInt = MIN_HEIGHT
    max_height: PositiveInt = MAX_HEIGHT
    min_width: PositiveInt = MIN_WIDTH
    max_width: PositiveInt = MAX_HEIGHT
    p_i: Annotated[float, confloat(ge=0.0, le=1.0)] = 1

    def model_post_init(self, _) -> None:
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

        self.gen = background_gen

        return None

    def __call__(
        self,
    ):
        return self.gen()


class FolderBackgroundGeneratorModel(BaseBackgroundGeneratorModel):
    method: Literal["folder"] = "folder"
    image_folder: DirectoryPath
    max_num_samples: PositiveInt | None = None
    glob_expression: str | None = None

    def model_post_init(self, _) -> None:
        background_gen = FolderBackgroundGenerator(
            image_folder=str(self.image_folder),
            number_of_samples=self.max_num_samples,
            glob_expression=self.glob_expression,
        )
        self.gen = background_gen

        return None

    def __call__(
        self,
    ):
        return self.gen()


def _probability_validator(
    p: List[Annotated[float, confloat(ge=0.0, le=1.0)]],
) -> List[Annotated[float, confloat(ge=0.0, le=1.0)]]:
    total_prob = sum(p)
    if not np.isclose(total_prob, 1.0):
        raise ValueError(
            f"Probabilities must equal 1. Sum of p_i's equals {total_prob}"
        )
    return p


class BackgroundComposition(BaseModel):
    generators: List[BaseBackgroundGeneratorModel]
    p: (
        Annotated[
            List[Annotated[float, confloat(ge=0.0, le=1.0)]],
            AfterValidator(_probability_validator),
        ]
        | None
    ) = None
    alpha_blend: Annotated[float, confloat(ge=0.0, le=1.0)] = 0.0
    max_blends: PositiveInt = 0
    p_blend: Annotated[float, confloat(ge=0.0, le=1.0)] = 0.0
    blend_resize_mode: Literal["max_size", "min_size"] = "max_size"
    resize_interpolation_method: int = cv2.INTER_LINEAR

    @model_validator(mode="after")
    def _check_max_blends(self) -> Self:
        if self.max_blends > 0 and np.isclose(self.alpha_blend, 0):
            warnings.warn(
                "max_blends is greater than 0, but alph_blend is set to zero. Background blending will not happen."
            )

        return self

    def _resize(self, img_arrays: List[Any]) -> List[Any]:
        areas = [img.size for img in img_arrays]
        if self.blend_resize_mode == "max_size":
            img_idx_size = areas.index(max(areas))
        else:
            img_idx_size = areas.index(min(areas))

        resize_shape = img_arrays[img_idx_size].shape[:2]

        resized_arrays = []
        for img_array in img_arrays:
            resized_arrays.append(
                cv2.resize(
                    img_array,
                    dsize=resize_shape[::-1],
                    interpolation=self.resize_interpolation_method,
                )
            )

        return resized_arrays

    def _alpha_blend_arrays(self, img_arrays: List[Any]):
        resized_arrays = self._resize(img_arrays)

        blended = resized_arrays[0]
        for array in resized_arrays[1:]:
            blended = (blended * self.alpha_blend) + (array * (1.0 - self.alpha_blend))
        return blended.astype(np.uint8)

    def gen(
        self,
    ) -> Any:
        num_to_blend = 0
        if self.max_blends > 0:
            num_to_blend = rng.integers(0, self.max_blends + 1)

        size = num_to_blend if num_to_blend != 0 else 1
        model_gens = rng.choice(self.generators, size=size, p=self.p)  # type: ignore
        blend = rng.choice([True, False], p=[self.p_blend, 1 - self.p_blend])

        if (self.alpha_blend > 0) and (blend):
            img_arrays = []
            for model_gen in model_gens:
                img_arrays.append(model_gen())
            return self._alpha_blend_arrays(img_arrays)
        else:
            return model_gens[0]()

    def __call__(
        self,
    ):
        return self.gen()

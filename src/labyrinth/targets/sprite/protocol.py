#!/usr/bin/env python
# By: Zack Taylor

from typing import List, Protocol, Tuple

from labyrinth.data_models.bounding_boxes import BoundingBox
from labyrinth.types import Array


class SpritePlacerProtocol(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(
        self,
        mask_arrays: List[Array],
        background_array: Array,
    ) -> Tuple[Array, List[BoundingBox]]: ...


class SpriteSamplerProtocol(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Tuple[List[Array], List[int]]: ...

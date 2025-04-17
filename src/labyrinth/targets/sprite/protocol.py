#!/usr/bin/env python
# By: Zack Taylor

from typing import List, Protocol, Tuple

from pydantic import BaseModel

from labyrinth.types import Array


class SpritePlacer(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(
        self,
        mask_arrays: List[Array],
        background_array: Array,
    ) -> Tuple[Array, BaseModel]: ...


class SpriteSampler(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Tuple[List[Array], List[int]]: ...

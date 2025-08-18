#!/usr/bin/env python
# Vidrovr Inc.

from typing import List, Protocol, Tuple

from labyrinth.types import Array


class MaskBackgroundModifierProtocol(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(
        self,
        mask_arrays: List[Array],
        background_array: Array,
    ) -> Tuple[List[Array], Array]: ...

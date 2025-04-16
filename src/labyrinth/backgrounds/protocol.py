#!/usr/bin/env python
# Vidrovr Inc.

from typing import Protocol

from labyrinth.types.array import Array


class BackgroundGenerator(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(
        self,
    ) -> Array: ...

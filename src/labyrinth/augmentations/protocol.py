#!/usr/bin/env python
# Vidrovr Inc.

from typing import Protocol

from labyrinth.types import Array


class ImageAugmentationProtocol(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(self, array: Array) -> Array: ...

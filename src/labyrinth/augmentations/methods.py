#!/usr/bin/env python
# Vidrovr Inc.

from labyrinth.types import Array


class DummyAugment:
    def __init__(self) -> None:
        pass

    def __call__(self, array: Array) -> Array:
        return array

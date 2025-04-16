#!/usr/bin/env python
# Vidrovr Inc.

from typing import Protocol

from labyrinth.types import Sample


class SampleGenerator(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(self, *args, **kwargs) -> Sample: ...

#!/usr/bin/env python
# Vidrovr Inc.

from typing import List, Protocol, Self, Sequence, Tuple

import numpy as np
from pydantic import BaseModel


class BoundingBox(Protocol):
    @classmethod
    def from_list(cls, bbox: List[float]) -> Self: ...

    @classmethod
    def from_dict(cls, dict_: dict) -> Self: ...

    def to_xyxy_list(self) -> Tuple[int, int, int, int]: ...

    def to_numpy_indices(self) -> Sequence: ...


class X4Y4(BaseModel):
    """Defines a bounding box with 4 (x, y) coord points."""

    top_left: Tuple[float, float]
    top_right: Tuple[float, float]
    bottom_right: Tuple[float, float]
    bottom_left: Tuple[float, float]


class CENTER_XYWH(BaseModel):
    """Defines a bounding box with x, y center points and width and height."""

    x: float
    y: float
    width: float
    height: float


class XYWH(BaseModel):
    """Defines a bounding box with x, y top left and width and height."""

    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_list(cls, bbox: List[float]) -> Self:
        return cls(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])

    @classmethod
    def from_dict(cls, bbox: dict[str, int | float]) -> Self:
        x = bbox["x"]
        y = bbox["y"]
        width = bbox["width"]
        height = bbox["height"]

        return cls(x=x, y=y, width=width, height=height)

    def to_numpy_indices(self) -> Sequence:
        y0 = int(self.y)
        y1 = int(self.y + self.height)
        x0 = int(self.x)
        x1 = int(self.x + self.width)

        return np.s_[y0:y1, x0:x1, ...]

    def to_xyxy_list(self) -> Tuple[int, int, int, int]:
        y0 = int(self.y)
        y1 = int(self.y + self.height)
        x0 = int(self.x)
        x1 = int(self.x + self.width)

        return (x0, y0, x1, y1)

    def to_cxywh(self) -> CENTER_XYWH:
        x = self.x + (self.width / 2.0)
        y = self.y + (self.height / 2.0)
        return CENTER_XYWH(x=x, y=y, width=self.width, height=self.height)


class XYXY(BaseModel):
    """Defines a bounding box in xyxy format."""

    left: float
    top: float
    right: float
    bottom: float


class YXYX(BaseModel):
    """Defines a bounding box in yxyx format."""

    top: float
    left: float
    bottom: float
    right: float

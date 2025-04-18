#!/usr/bin/env python
# Vidrovr Inc.

from typing import Any, List, Protocol, Self, runtime_checkable
from uuid import uuid4

from pydantic import UUID4, BaseModel, ConfigDict, Field

from labyrinth.data_models.bounding_boxes import BoundingBox


@runtime_checkable
class Annotation(Protocol):
    id: UUID4 | int
    image_id: UUID4 | int
    category_id: UUID4 | int
    counts: List[int]
    size: tuple[int, int]
    area: int
    bbox: BoundingBox

    @classmethod
    def from_pycoco(cls, annotation: Any, bbcls: BoundingBox) -> Self: ...


class SegmentationRLE(BaseModel):
    """Annotation info."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID4 | int = Field(default_factory=uuid4)
    image_id: UUID4 | int
    category_id: UUID4 | int
    counts: List[int]
    size: tuple[int, int]
    area: int
    bbox: BoundingBox

    @classmethod
    def from_pycoco(cls, annotation: dict[str, Any], bbcls: BoundingBox) -> Self:
        # Unpack annotation
        id = annotation["id"]
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        area = int(annotation["area"])
        counts = annotation["segmentation"]["counts"]
        size = annotation["segmentation"]["size"]
        bbox = bbcls.from_list(annotation["bbox"])

        return cls(
            id=id,
            image_id=image_id,
            category_id=category_id,
            counts=counts,
            size=size,
            area=area,
            bbox=bbox,
        )

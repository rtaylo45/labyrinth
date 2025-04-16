#!/usr/bin/env python
# Vidrovr Inc.

from typing import List
from uuid import uuid4

from pydantic import UUID4, BaseModel, Field

# Internal libraries
from labyrinth.data_models.media import Image


class Category(BaseModel):
    """Information about categories."""

    id: UUID4 | int = Field(default_factory=uuid4)
    name: str = ""
    supercategory: str = ""


class YOLO(BaseModel):
    categories: List[Category]
    images: dict[UUID4 | int, Image] = Field(default_factory=dict)
    annotations: dict[UUID4 | int, BaseModel] = Field(default_factory=dict)

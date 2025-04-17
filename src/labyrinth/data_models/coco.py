#!/usr/bin/env python
# Vidrovr Inc.

from datetime import datetime
from typing import List
from uuid import uuid4

from pydantic import UUID4, BaseModel, Field

from labyrinth.data_models.annotations import Annotation
from labyrinth.data_models.media import Image


class Info(BaseModel):
    """General information about the dataset."""

    year: int | str = ""
    version: str = ""
    description: str = ""
    contributor: str = ""
    url: str = ""
    date_created: datetime | str = ""


class License(BaseModel):
    """Information about dataset license."""

    id: UUID4 | int = Field(default_factory=uuid4)
    name: str = ""
    url: str = ""


class Category(BaseModel):
    """Information about categories."""

    id: UUID4 | int = Field(default_factory=uuid4)
    name: str = ""
    supercategory: str = ""


class COCO(BaseModel):
    annotation_file: str = ""
    info: Info | None = None
    licenses: List[License] | None = None
    categories: List[Category] | None = None
    images: dict[UUID4 | int, Image] = Field(default_factory=dict)
    annotations: dict[UUID4 | int, Annotation] = Field(default_factory=dict)

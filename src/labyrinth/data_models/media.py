#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Vidrovr Inc.
# By: Zack Taylor

# Standard libraries
from datetime import datetime
from uuid import uuid4

# External libraries
from pydantic import UUID4, BaseModel, Field

# Internal libraries


class Image(BaseModel):
    """Information about an image."""

    id: UUID4 | int = Field(default_factory=uuid4)
    width: int
    height: int
    file_name: str
    license: int | None = None
    date_captured: datetime | None = None
    flickr_url: str | None = None
    coco_url: str | None = None
    metadata: dict = {}


class Video(BaseModel):
    pass


class Audio(BaseModel):
    pass

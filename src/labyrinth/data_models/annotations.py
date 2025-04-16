#!/usr/bin/env python
# Vidrovr Inc.

from uuid import uuid4

from pydantic import UUID4, BaseModel, Field


class Annotation(BaseModel):
    """Annotation info."""

    id: UUID4 | int = Field(default_factory=uuid4)
    image_id: UUID4 | int
    category_id: UUID4 | int
    bbox: BaseModel

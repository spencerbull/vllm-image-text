"""API response schemas for the image-to-text service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextExtractionResponse(BaseModel):
    """Response payload for single-image extraction."""

    text: list[str] = Field(..., description="Extracted lines of text")


class BatchItemResponse(BaseModel):
    """Response item for a single image in a batch request."""

    file: str = Field(..., description="Original filename from the upload")
    text: list[str] = Field(..., description="Extracted lines of text")
    error: str | None = Field(None, description="Error message, if extraction failed")


class BatchExtractionResponse(BaseModel):
    """Response payload for batch image extraction."""

    results: list[BatchItemResponse]


class HealthResponse(BaseModel):
    """Health response for the service."""

    status: str
    model: str
    vllm_base_url: str

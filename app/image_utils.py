"""Image validation and encoding helpers for vision model inputs."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ImageValidationConfig:
    """Constraints for uploaded images used by the API."""

    max_bytes: int
    max_pixels: int
    allowed_formats: Iterable[str]


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata extracted from an image after validation."""

    format: str
    mime_type: str
    width: int
    height: int


class ImageValidationError(ValueError):
    """Raised when image validation fails."""


def _normalize_format(image_format: str) -> str:
    """Normalize a PIL format string into a canonical form."""
    normalized = image_format.upper()
    if normalized == "JPG":
        return "JPEG"
    return normalized


def _format_to_mime(image_format: str) -> str:
    """Convert an image format string into a MIME type."""
    normalized = _normalize_format(image_format)
    if normalized == "JPEG":
        return "image/jpeg"
    return f"image/{normalized.lower()}"


def validate_image_bytes(
    image_bytes: bytes, config: ImageValidationConfig
) -> ImageMetadata:
    """Validate image bytes and return metadata for downstream use."""
    if not image_bytes:
        raise ImageValidationError("Empty image payload")

    if len(image_bytes) > config.max_bytes:
        raise ImageValidationError(
            f"Image exceeds maximum size of {config.max_bytes} bytes"
        )

    try:
        # PIL verify checks the stream is an image without decoding pixels.
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        # Re-open to access metadata after verify() invalidates the image object.
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = _normalize_format(image.format or "PNG")
            width, height = image.size
    except UnidentifiedImageError as exc:
        raise ImageValidationError("Unsupported or corrupted image") from exc
    except Exception as exc:
        raise ImageValidationError(f"Image validation failed: {exc}") from exc

    if width * height > config.max_pixels:
        raise ImageValidationError(
            f"Image exceeds maximum pixel count of {config.max_pixels}"
        )

    allowed_formats = {_normalize_format(fmt) for fmt in config.allowed_formats}
    if image_format not in allowed_formats:
        raise ImageValidationError(
            f"Unsupported image format {image_format}. Allowed: {', '.join(sorted(allowed_formats))}"
        )

    return ImageMetadata(
        format=image_format,
        mime_type=_format_to_mime(image_format),
        width=width,
        height=height,
    )


def encode_image_to_data_uri(image_bytes: bytes, metadata: ImageMetadata) -> str:
    """Encode raw image bytes as a base64 data URI."""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{metadata.mime_type};base64,{encoded}"


def image_bytes_to_data_uri(
    image_bytes: bytes,
    config: ImageValidationConfig,
) -> tuple[str, ImageMetadata]:
    """Validate image bytes and return a data URI plus metadata."""
    metadata = validate_image_bytes(image_bytes, config)
    return encode_image_to_data_uri(image_bytes, metadata), metadata

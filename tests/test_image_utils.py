"""Tests for image validation and encoding utilities."""

import base64
from io import BytesIO

import pytest
from PIL import Image

from app.image_utils import (
    ImageMetadata,
    ImageValidationConfig,
    ImageValidationError,
    encode_image_to_data_uri,
    image_bytes_to_data_uri,
    validate_image_bytes,
)

# ─── Fixtures ───────────────────────────────────────────────────────────────


def _make_png(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG image in memory."""
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid JPEG image in memory."""
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(0, 255, 0)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def default_config() -> ImageValidationConfig:
    return ImageValidationConfig(
        max_bytes=10 * 1024 * 1024,  # 10MB
        max_pixels=30_000_000,
        allowed_formats=["PNG", "JPEG", "WEBP"],
    )


@pytest.fixture
def png_bytes() -> bytes:
    return _make_png()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return _make_jpeg()


# ─── validate_image_bytes ───────────────────────────────────────────────────


class TestValidateImageBytes:
    def test_valid_png(self, default_config, png_bytes):
        meta = validate_image_bytes(png_bytes, default_config)
        assert meta.format == "PNG"
        assert meta.mime_type == "image/png"
        assert meta.width == 100
        assert meta.height == 100

    def test_valid_jpeg(self, default_config, jpeg_bytes):
        meta = validate_image_bytes(jpeg_bytes, default_config)
        assert meta.format == "JPEG"
        assert meta.mime_type == "image/jpeg"

    def test_empty_payload_raises(self, default_config):
        with pytest.raises(ImageValidationError, match="Empty"):
            validate_image_bytes(b"", default_config)

    def test_exceeds_max_bytes(self, png_bytes):
        tiny_config = ImageValidationConfig(
            max_bytes=10,  # way too small
            max_pixels=30_000_000,
            allowed_formats=["PNG"],
        )
        with pytest.raises(ImageValidationError, match="maximum size"):
            validate_image_bytes(png_bytes, tiny_config)

    def test_exceeds_max_pixels(self, default_config):
        big_img = _make_png(6000, 6000)  # 36M pixels
        with pytest.raises(ImageValidationError, match="pixel count"):
            validate_image_bytes(big_img, default_config)

    def test_unsupported_format(self):
        config = ImageValidationConfig(
            max_bytes=10 * 1024 * 1024,
            max_pixels=30_000_000,
            allowed_formats=["JPEG"],  # PNG not allowed
        )
        png = _make_png()
        with pytest.raises(ImageValidationError, match="Unsupported image format"):
            validate_image_bytes(png, config)

    def test_corrupted_image(self, default_config):
        with pytest.raises(ImageValidationError):
            validate_image_bytes(b"not an image at all", default_config)


# ─── encode_image_to_data_uri ───────────────────────────────────────────────


class TestEncodeImageToDataUri:
    def test_produces_valid_data_uri(self, png_bytes):
        meta = ImageMetadata(format="PNG", mime_type="image/png", width=100, height=100)
        uri = encode_image_to_data_uri(png_bytes, meta)
        assert uri.startswith("data:image/png;base64,")
        # Verify base64 is decodable
        b64_part = uri.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == png_bytes

    def test_jpeg_mime(self, jpeg_bytes):
        meta = ImageMetadata(format="JPEG", mime_type="image/jpeg", width=100, height=100)
        uri = encode_image_to_data_uri(jpeg_bytes, meta)
        assert uri.startswith("data:image/jpeg;base64,")


# ─── image_bytes_to_data_uri (integration) ──────────────────────────────────


class TestImageBytesToDataUri:
    def test_end_to_end(self, default_config, png_bytes):
        uri, meta = image_bytes_to_data_uri(png_bytes, default_config)
        assert uri.startswith("data:image/png;base64,")
        assert meta.width == 100
        assert meta.height == 100

    def test_invalid_raises(self, default_config):
        with pytest.raises(ImageValidationError):
            image_bytes_to_data_uri(b"garbage", default_config)

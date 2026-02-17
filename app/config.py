"""Application configuration loaded from environment variables.

This module is intentionally dependency-free to keep configuration reusable
outside of FastAPI or any specific runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable


def _get_env(name: str, default: str | None = None) -> str:
    """Return a required or defaulted environment variable value."""
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe default."""
    raw_value = os.getenv(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {raw_value}") from exc


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable with a safe default."""
    raw_value = os.getenv(name, str(default))
    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for {name}: {raw_value}") from exc


def _get_list(name: str, default: Iterable[str]) -> list[str]:
    """Read a comma-separated environment variable into a list."""
    raw_value = os.getenv(name)
    if not raw_value:
        return [value.strip() for value in default]
    return [value.strip() for value in raw_value.split(",") if value.strip()]


@dataclass(frozen=True)
class AppConfig:
    """Container for all configuration values used by the service."""

    vllm_base_url: str
    model_name: str
    max_concurrent: int
    max_image_bytes: int
    max_image_pixels: int
    allowed_image_formats: list[str]
    request_timeout_s: float
    max_tokens: int
    temperature: float
    default_prompt: str
    log_level: str


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    return AppConfig(
        vllm_base_url=_get_env("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model_name=_get_env("MODEL_NAME", "google/gemma-3-27b-it"),
        max_concurrent=_get_int("MAX_CONCURRENT", 8),
        max_image_bytes=_get_int("MAX_IMAGE_BYTES", 10 * 1024 * 1024),
        max_image_pixels=_get_int("MAX_IMAGE_PIXELS", 30_000_000),
        allowed_image_formats=_get_list(
            "ALLOWED_IMAGE_FORMATS", ["PNG", "JPEG", "WEBP"]
        ),
        request_timeout_s=_get_float("REQUEST_TIMEOUT_S", 120.0),
        max_tokens=_get_int("MAX_TOKENS", 4096),
        temperature=_get_float("TEMPERATURE", 0.0),
        default_prompt=_get_env(
            "DEFAULT_PROMPT",
            "Extract all text from this image. Return each line of text on a separate line."
            " Return only the extracted text, nothing else.",
        ),
        log_level=_get_env("LOG_LEVEL", "INFO"),
    )

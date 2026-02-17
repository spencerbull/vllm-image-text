"""Logging configuration for the API service."""

from __future__ import annotations

import logging


def configure_logging(level: str) -> None:
    """Configure application-wide logging with a consistent format."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

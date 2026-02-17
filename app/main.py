"""vLLM Image-to-Text API service.

This module wires the FastAPI routes to reusable client and image utilities.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile

from config import load_config
from image_utils import (
    ImageValidationConfig,
    ImageValidationError,
    image_bytes_to_data_uri,
)
from logging_config import configure_logging
from schemas import (
    BatchExtractionResponse,
    BatchItemResponse,
    HealthResponse,
    TextExtractionResponse,
)
from vllm_client import VLLMClient, VLLMClientConfig, VLLMClientError


config = load_config()
configure_logging(config.log_level)
logger = logging.getLogger("vllm-image-text")

image_config = ImageValidationConfig(
    max_bytes=config.max_image_bytes,
    max_pixels=config.max_image_pixels,
    allowed_formats=config.allowed_image_formats,
)

vllm_config = VLLMClientConfig(
    base_url=config.vllm_base_url,
    model_name=config.model_name,
    image_part_type=config.image_part_type,
    max_tokens=config.max_tokens,
    temperature=config.temperature,
    request_timeout_s=config.request_timeout_s,
    max_concurrent=config.max_concurrent,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared HTTP resources for the app lifecycle."""
    # A single AsyncClient reuses connections efficiently across requests.
    http_client = httpx.AsyncClient()
    app.state.vllm_client = VLLMClient(vllm_config, http_client=http_client)
    try:
        yield
    finally:
        await http_client.aclose()


app = FastAPI(
    title="vLLM Image-to-Text",
    description="Extract text from images using Gemma 3 vision via vLLM",
    version="1.0.0",
    lifespan=lifespan,
)


def _normalize_prompt(prompt: str, default_prompt: str) -> str:
    """Return a non-empty prompt, falling back to the default when needed."""
    cleaned = prompt.strip()
    return cleaned if cleaned else default_prompt


def _format_vllm_error(error: VLLMClientError) -> str:
    """Format a vLLM client error into a user-facing message."""
    if error.status_code is None:
        return str(error)
    return f"vLLM error {error.status_code}: {error.response_text or str(error)}"


def _get_vllm_client(request: Request) -> VLLMClient:
    """Retrieve the shared vLLM client from app state."""
    return request.app.state.vllm_client


async def _process_upload(
    file: UploadFile,
    vllm_client: VLLMClient,
    prompt: str,
) -> BatchItemResponse:
    """Validate, encode, and send one uploaded image to vLLM."""
    filename = file.filename or "unnamed"
    contents = await file.read()

    if not contents:
        return BatchItemResponse(file=filename, text=[], error="Empty file")

    try:
        image_uri, metadata = image_bytes_to_data_uri(contents, image_config)
    except ImageValidationError as exc:
        return BatchItemResponse(file=filename, text=[], error=str(exc))

    # Log metadata for observability; helps debug unexpected OCR quality issues.
    logger.info(
        "Processing image %s (%s, %sx%s)",
        filename,
        metadata.format,
        metadata.width,
        metadata.height,
    )

    try:
        lines = await vllm_client.extract_text_lines(image_uri, prompt)
    except VLLMClientError as exc:
        return BatchItemResponse(file=filename, text=[], error=_format_vllm_error(exc))

    # The API always returns a list of text lines for consistency with batch output.
    return BatchItemResponse(file=filename, text=lines, error=None)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with configuration hints."""
    return HealthResponse(
        status="ok",
        model=config.model_name,
        vllm_base_url=config.vllm_base_url,
    )


@app.post("/extract-text", response_model=TextExtractionResponse)
async def extract_text(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Query(
        default=config.default_prompt,
        description="Custom prompt for text extraction",
    ),
) -> TextExtractionResponse:
    """Extract text from a single uploaded image."""
    vllm_client = _get_vllm_client(request)
    normalized_prompt = _normalize_prompt(prompt, config.default_prompt)

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image_uri, metadata = image_bytes_to_data_uri(contents, image_config)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "Processing image %s (%s, %sx%s)",
        file.filename or "unnamed",
        metadata.format,
        metadata.width,
        metadata.height,
    )

    try:
        lines = await vllm_client.extract_text_lines(image_uri, normalized_prompt)
    except VLLMClientError as exc:
        logger.error("vLLM request failed: %s", exc)
        raise HTTPException(status_code=502, detail=_format_vllm_error(exc)) from exc

    # Response is a stable list so clients can render lines directly.
    return TextExtractionResponse(text=lines)


@app.post("/extract-text/batch", response_model=BatchExtractionResponse)
async def extract_text_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    prompt: str = Query(
        default=config.default_prompt,
        description="Custom prompt for text extraction",
    ),
) -> BatchExtractionResponse:
    """Extract text from multiple images concurrently."""
    vllm_client = _get_vllm_client(request)
    normalized_prompt = _normalize_prompt(prompt, config.default_prompt)

    # Each file becomes its own task; vLLMClient enforces concurrency limits.
    tasks = [_process_upload(file, vllm_client, normalized_prompt) for file in files]
    results = await asyncio.gather(*tasks)

    # Batch responses preserve input ordering with per-item errors, if any.
    return BatchExtractionResponse(results=results)

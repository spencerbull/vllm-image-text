"""Reusable vLLM OpenAI-compatible client for vision chat completions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Any

import httpx


@dataclass(frozen=True)
class VLLMClientConfig:
    """Configuration for the vLLM client."""

    base_url: str
    model_name: str
    max_tokens: int
    temperature: float
    request_timeout_s: float
    max_concurrent: int


class VLLMClientError(RuntimeError):
    """Raised when vLLM returns an error or unexpected response."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class VLLMClient:
    """Async client wrapper for vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        config: VLLMClientConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._external_client = http_client
        self._client = http_client
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

    async def __aenter__(self) -> "VLLMClient":
        """Ensure the underlying HTTP client is initialized."""
        if self._client is None:
            self._client = self._build_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close resources when exiting an async context manager."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._client is not None and self._external_client is None:
            await self._client.aclose()
        self._client = None

    def _build_client(self) -> httpx.AsyncClient:
        """Build a configured httpx client with sane defaults."""
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        timeout = httpx.Timeout(self._config.request_timeout_s)
        return httpx.AsyncClient(limits=limits, timeout=timeout)

    def _build_payload(self, image_data_uri: str, prompt: str) -> dict[str, Any]:
        """Construct the OpenAI-compatible payload for vLLM."""
        return {
            "model": self._config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
        }

    async def extract_text_lines(
        self, image_data_uri: str, prompt: str, include_usage: bool = False
    ) -> list[str] | tuple[list[str], dict]:
        """
        Send an image to vLLM and return extracted text lines.

        If include_usage=True, returns (lines, usage_dict) where usage_dict
        contains prompt_tokens, completion_tokens, total_tokens from vLLM.
        """
        if self._client is None:
            self._client = self._build_client()

        payload = self._build_payload(image_data_uri, prompt)

        try:
            # Semaphore prevents too many simultaneous vLLM requests.
            async with self._semaphore:
                response = await self._client.post(
                    f"{self._config.base_url}/chat/completions",
                    json=payload,
                )
        except httpx.RequestError as exc:
            raise VLLMClientError(f"vLLM request failed: {exc}") from exc

        if response.status_code != 200:
            raise VLLMClientError(
                f"vLLM returned {response.status_code}",
                status_code=response.status_code,
                response_text=response.text,
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self._logger.error("Unexpected vLLM response: %s", response.text)
            raise VLLMClientError("Unexpected vLLM response shape") from exc

        text = str(content).strip()
        # Normalize the response into a stable list of non-empty lines.
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if include_usage:
            usage = data.get("usage", {})
            return lines, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        return lines

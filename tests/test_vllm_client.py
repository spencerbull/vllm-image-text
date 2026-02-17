"""Tests for the vLLM client wrapper."""

import json

import httpx
import pytest
import respx

from app.vllm_client import VLLMClient, VLLMClientConfig, VLLMClientError


@pytest.fixture
def client_config() -> VLLMClientConfig:
    return VLLMClientConfig(
        base_url="http://fake-vllm:8000/v1",
        model_name="google/gemma-3-27b-it",
        max_tokens=4096,
        temperature=0.0,
        request_timeout_s=30.0,
        max_concurrent=4,
    )


@pytest.fixture
def sample_image_uri() -> str:
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


def _mock_vllm_response(text: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> dict:
    """Build a mock vLLM chat completion response."""
    return {
        "id": "cmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class TestVLLMClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_extract_text_lines_success(self, client_config, sample_image_uri):
        """Successful extraction returns list of text lines."""
        respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=_mock_vllm_response("Hello World\nSecond Line\nThird Line")
            )
        )

        async with VLLMClient(client_config) as client:
            lines = await client.extract_text_lines(sample_image_uri, "Extract text")

        assert lines == ["Hello World", "Second Line", "Third Line"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_extract_text_filters_empty_lines(self, client_config, sample_image_uri):
        """Empty lines in response are filtered out."""
        respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=_mock_vllm_response("Line 1\n\n\nLine 2\n  \nLine 3")
            )
        )

        async with VLLMClient(client_config) as client:
            lines = await client.extract_text_lines(sample_image_uri, "Extract text")

        assert lines == ["Line 1", "Line 2", "Line 3"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_extract_with_usage(self, client_config, sample_image_uri):
        """include_usage=True returns (lines, usage_dict)."""
        respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=_mock_vllm_response("OCR text", prompt_tokens=200, completion_tokens=30)
            )
        )

        async with VLLMClient(client_config) as client:
            lines, usage = await client.extract_text_lines(
                sample_image_uri, "Extract text", include_usage=True
            )

        assert lines == ["OCR text"]
        assert usage["prompt_tokens"] == 200
        assert usage["completion_tokens"] == 30
        assert usage["total_tokens"] == 230

    @pytest.mark.asyncio
    @respx.mock
    async def test_vllm_error_raises(self, client_config, sample_image_uri):
        """Non-200 response raises VLLMClientError."""
        respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        async with VLLMClient(client_config) as client:
            with pytest.raises(VLLMClientError, match="500"):
                await client.extract_text_lines(sample_image_uri, "Extract text")

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_response_raises(self, client_config, sample_image_uri):
        """Malformed JSON response raises VLLMClientError."""
        respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )

        async with VLLMClient(client_config) as client:
            with pytest.raises(VLLMClientError, match="Unexpected"):
                await client.extract_text_lines(sample_image_uri, "Extract text")

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error_raises(self, client_config, sample_image_uri):
        """Network failure raises VLLMClientError."""
        respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        async with VLLMClient(client_config) as client:
            with pytest.raises(VLLMClientError, match="request failed"):
                await client.extract_text_lines(sample_image_uri, "Extract text")

    @pytest.mark.asyncio
    @respx.mock
    async def test_payload_structure(self, client_config, sample_image_uri):
        """Verify the request payload sent to vLLM is correct."""
        route = respx.post("http://fake-vllm:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_mock_vllm_response("text"))
        )

        async with VLLMClient(client_config) as client:
            await client.extract_text_lines(sample_image_uri, "My prompt")

        assert route.called
        payload = json.loads(route.calls[0].request.content)
        assert payload["model"] == "google/gemma-3-27b-it"
        assert payload["max_tokens"] == 4096
        assert payload["temperature"] == 0.0
        assert len(payload["messages"]) == 1
        content = payload["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == sample_image_uri
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "My prompt"

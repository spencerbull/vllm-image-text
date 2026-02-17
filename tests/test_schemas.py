"""Tests for API response schemas."""

from app.schemas import (
    BatchExtractionResponse,
    BatchItemResponse,
    HealthResponse,
    TextExtractionResponse,
)


class TestSchemas:
    def test_text_extraction_response(self):
        resp = TextExtractionResponse(text=["line1", "line2"])
        assert resp.text == ["line1", "line2"]
        assert resp.model_dump() == {"text": ["line1", "line2"]}

    def test_batch_item_response_success(self):
        item = BatchItemResponse(file="test.png", text=["hello"], error=None)
        assert item.file == "test.png"
        assert item.error is None

    def test_batch_item_response_error(self):
        item = BatchItemResponse(file="bad.png", text=[], error="Invalid image")
        assert item.error == "Invalid image"
        assert item.text == []

    def test_batch_extraction_response(self):
        resp = BatchExtractionResponse(
            results=[
                BatchItemResponse(file="a.png", text=["text"]),
                BatchItemResponse(file="b.png", text=[], error="fail"),
            ]
        )
        assert len(resp.results) == 2

    def test_health_response(self):
        resp = HealthResponse(
            status="ok",
            model="google/gemma-3-27b-it",
            vllm_base_url="http://localhost:8000/v1",
        )
        d = resp.model_dump()
        assert d["status"] == "ok"
        assert d["model"] == "google/gemma-3-27b-it"

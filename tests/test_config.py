"""Tests for application configuration loading."""

import os

import pytest

from app.config import AppConfig, load_config


class TestLoadConfig:
    def test_defaults(self, monkeypatch):
        """Config loads with sane defaults when no env vars set."""
        # Clear any existing env vars that might interfere
        for key in ["VLLM_BASE_URL", "MODEL_NAME", "MAX_CONCURRENT", "LOG_LEVEL"]:
            monkeypatch.delenv(key, raising=False)

        config = load_config()
        assert config.vllm_base_url == "http://localhost:8000/v1"
        assert config.model_name == "google/gemma-3-27b-it"
        assert config.image_part_type == "image_url"
        assert config.max_concurrent == 8
        assert config.max_image_bytes == 10 * 1024 * 1024
        assert config.temperature == 0.0
        assert config.log_level == "INFO"

    def test_custom_env_vars(self, monkeypatch):
        """Config reads from environment variables."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://myhost:9000/v1")
        monkeypatch.setenv("MODEL_NAME", "google/gemma-3-4b-it")
        monkeypatch.setenv("VLLM_IMAGE_PART_TYPE", "input_image")
        monkeypatch.setenv("MAX_CONCURRENT", "16")
        monkeypatch.setenv("TEMPERATURE", "0.5")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        config = load_config()
        assert config.vllm_base_url == "http://myhost:9000/v1"
        assert config.model_name == "google/gemma-3-4b-it"
        assert config.image_part_type == "input_image"
        assert config.max_concurrent == 16
        assert config.temperature == 0.5
        assert config.log_level == "DEBUG"

    def test_invalid_int_raises(self, monkeypatch):
        """Non-integer value for int config raises RuntimeError."""
        monkeypatch.setenv("MAX_CONCURRENT", "not_a_number")
        with pytest.raises(RuntimeError, match="Invalid integer"):
            load_config()

    def test_invalid_float_raises(self, monkeypatch):
        """Non-float value for float config raises RuntimeError."""
        monkeypatch.setenv("TEMPERATURE", "hot")
        with pytest.raises(RuntimeError, match="Invalid float"):
            load_config()

    def test_allowed_formats_from_env(self, monkeypatch):
        """Comma-separated formats are parsed correctly."""
        monkeypatch.setenv("ALLOWED_IMAGE_FORMATS", "PNG, JPEG, BMP")
        config = load_config()
        assert config.allowed_image_formats == ["PNG", "JPEG", "BMP"]

    def test_invalid_image_part_type_raises(self, monkeypatch):
        monkeypatch.setenv("VLLM_IMAGE_PART_TYPE", "image")
        with pytest.raises(RuntimeError, match="VLLM_IMAGE_PART_TYPE"):
            load_config()

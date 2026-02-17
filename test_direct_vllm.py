#!/usr/bin/env python3
"""Send a direct multimodal request to vLLM for debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import httpx

from app.config import load_config
from app.image_utils import ImageValidationConfig, ImageValidationError, image_bytes_to_data_uri


def _build_image_part(image_data_uri: str, image_part_type: str) -> dict:
    if image_part_type == "input_image":
        return {"type": "input_image", "image_url": {"url": image_data_uri}}
    return {"type": "image_url", "image_url": {"url": image_data_uri}}


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Send a direct vLLM chat completion request with an image."
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--base-url",
        default=config.vllm_base_url,
        help=f"vLLM base URL (default: {config.vllm_base_url})",
    )
    parser.add_argument(
        "--model",
        default=config.model_name,
        help=f"Model name (default: {config.model_name})",
    )
    parser.add_argument(
        "--prompt",
        default=config.default_prompt,
        help="Prompt to send with the image",
    )
    parser.add_argument(
        "--image-part-type",
        choices=["image_url", "input_image"],
        default=config.image_part_type,
        help="OpenAI multimodal part type to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.max_tokens,
        help=f"Max completion tokens (default: {config.max_tokens})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.temperature,
        help=f"Sampling temperature (default: {config.temperature})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=config.request_timeout_s,
        help=f"Request timeout seconds (default: {config.request_timeout_s})",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print full JSON response",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 2

    image_bytes = image_path.read_bytes()
    image_config = ImageValidationConfig(
        max_bytes=config.max_image_bytes,
        max_pixels=config.max_image_pixels,
        allowed_formats=config.allowed_image_formats,
    )

    try:
        image_uri, metadata = image_bytes_to_data_uri(image_bytes, image_config)
    except ImageValidationError as exc:
        print(f"Invalid image: {exc}")
        return 2

    image_part = _build_image_part(image_uri, args.image_part_type)
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    image_part,
                ],
            }
        ],
        "max_completion_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    url = f"{args.base_url}/chat/completions"
    print(f"Sending request to {url}")
    print(
        f"Image: {image_path.name} ({metadata.format}, {metadata.width}x{metadata.height}), "
        f"image_part_type={args.image_part_type}"
    )

    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(url, json=payload)
    except httpx.RequestError as exc:
        print(f"Request failed: {exc}")
        return 1

    if response.status_code != 200:
        print(f"vLLM returned {response.status_code}")
        print(response.text)
        return 1

    data = response.json()
    if args.raw:
        print(json.dumps(data, indent=2))
        return 0

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    print("\nResponse:\n")
    print(content)
    if usage:
        print(
            f"\nUsage: prompt_tokens={usage.get('prompt_tokens', 0)}, "
            f"completion_tokens={usage.get('completion_tokens', 0)}, "
            f"total_tokens={usage.get('total_tokens', 0)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

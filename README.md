# vLLM Image-to-Text

Extract text from images using a Gemma 3 vision model served by vLLM. This repository focuses on production quality: modular reusable components, explicit configuration, and predictable API behavior.

## Architecture

```mermaid
flowchart LR
  Client[Client
  cURL / SDKs] --> API[FastAPI
  :8080]
  API -->|OpenAI-compatible| VLLM[vLLM
  :8000]
  VLLM --> Model[Gemma 3 Vision
  (HF Hub)]
```

## Quick Start

```bash
# Set HuggingFace token (needed for gated models)
export HF_TOKEN=your_token_here

# Start everything
docker compose up -d

# Wait for vLLM to load the model (check health)
docker compose logs -f vllm

# Extract text from an image
curl -X POST http://localhost:8080/extract-text \
  -F "file=@screenshot.png"
```

## Usage Examples

### Single image (file on disk)

```bash
# Extract text from a local screenshot
curl -s -X POST http://localhost:8080/extract-text \
  -F "file=@/path/to/screenshot.png" | jq

# Response:
# {
#   "text": [
#     "Welcome to the Dashboard",
#     "Total Users: 1,234",
#     "Revenue: $56,789"
#   ]
# }
```

### With a custom prompt

```bash
# Ask for specific extraction (e.g., only numbers)
curl -s -X POST http://localhost:8080/extract-text \
  -F "file=@invoice.jpg" \
  -G --data-urlencode "prompt=Extract only dollar amounts from this image. Return one per line." | jq

# Response:
# {
#   "text": ["$1,250.00", "$89.99", "$1,339.99"]
# }
```

### Batch — multiple images at once

```bash
# Process several images concurrently
curl -s -X POST http://localhost:8080/extract-text/batch \
  -F "files=@page1.png" \
  -F "files=@page2.png" \
  -F "files=@page3.png" | jq

# Response:
# {
#   "results": [
#     {"file": "page1.png", "text": ["Header text", "Body text..."], "error": null},
#     {"file": "page2.png", "text": ["More text here"], "error": null},
#     {"file": "page3.png", "text": ["Final page content"], "error": null}
#   ]
# }
```

### From a script — process all images in a directory

```bash
# Extract text from every PNG in a folder
for img in ./documents/*.png; do
  echo "=== $(basename $img) ==="
  curl -s -X POST http://localhost:8080/extract-text \
    -F "file=@${img}" | jq -r '.text[]'
  echo
done
```

### Python requests example

```python
import requests

# Single image
with open("receipt.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8080/extract-text",
        files={"file": ("receipt.jpg", f, "image/jpeg")},
    )
lines = resp.json()["text"]
print(lines)  # ["Store: Walmart", "Date: 02/17/2026", "Total: $42.50"]
```

### Health check

```bash
curl -s http://localhost:8080/health | jq
# {"status": "ok", "model": "google/gemma-3-27b-it", "vllm_base_url": "http://vllm:8000/v1"}
```

---

## API

### `POST /extract-text`
Extract text from a single image. Always returns a list of text lines for a stable client contract.

**Parameters:**
- `file` (form) — Image file (PNG, JPEG, WEBP by default)
- `prompt` (query, optional) — Custom extraction prompt

**Response:**
```json
{
  "text": ["Line 1 of text", "Line 2 of text", "..."]
}
```

### `POST /extract-text/batch`
Extract text from multiple images concurrently. Each item includes an `error` field when extraction fails.

**Parameters:**
- `files` (form) — Multiple image files
- `prompt` (query, optional) — Custom extraction prompt

**Response:**
```json
{
  "results": [
    {"file": "img1.png", "text": ["..."], "error": null},
    {"file": "img2.png", "text": [], "error": "Unsupported image format TIFF"}
  ]
}
```

### `GET /health`
Health check endpoint with model/base URL metadata.

## Reusable Modules

These modules are dependency-light and can be copied into other projects as-is:

- `app/vllm_client.py` — OpenAI-compatible vLLM client for vision chat completions.
- `app/image_utils.py` — Image validation and base64 data URI encoding.

Example usage (outside FastAPI):

```python
import asyncio

from image_utils import ImageValidationConfig, image_bytes_to_data_uri
from vllm_client import VLLMClient, VLLMClientConfig


async def main() -> None:
    with open("screenshot.png", "rb") as handle:
        image_bytes = handle.read()

    image_uri, _ = image_bytes_to_data_uri(
        image_bytes,
        ImageValidationConfig(
            max_bytes=10 * 1024 * 1024,
            max_pixels=30_000_000,
            allowed_formats=["PNG", "JPEG", "WEBP"],
        ),
    )

    config = VLLMClientConfig(
        base_url="http://localhost:8000/v1",
        model_name="google/gemma-3-27b-it",
        max_tokens=4096,
        temperature=0.0,
        request_timeout_s=120.0,
        max_concurrent=8,
    )

    async with VLLMClient(config) as client:
        lines = await client.extract_text_lines(
            image_uri,
            "Extract all text. Return each line separately.",
        )
        print(lines)


asyncio.run(main())
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token for gated models |
| `VLLM_BASE_URL` | `http://vllm:8000/v1` | vLLM API base URL |
| `MODEL_NAME` | `google/gemma-3-27b-it` | Vision model to use |
| `MAX_CONCURRENT` | `8` | Max concurrent vLLM requests |
| `MAX_IMAGE_BYTES` | `10485760` | Max upload size in bytes |
| `MAX_IMAGE_PIXELS` | `30000000` | Max width × height pixels |
| `ALLOWED_IMAGE_FORMATS` | `PNG,JPEG,WEBP` | Allowed image formats |
| `REQUEST_TIMEOUT_S` | `120` | vLLM request timeout seconds |
| `MAX_TOKENS` | `4096` | vLLM completion token limit |
| `TEMPERATURE` | `0.0` | vLLM sampling temperature |
| `DEFAULT_PROMPT` | See `app/config.py` | Default OCR prompt |
| `LOG_LEVEL` | `INFO` | Logging level |

### vLLM Tuning

Edit `docker-compose.yml` vLLM command args. The file includes inline rationale for each setting.

| Arg | Default | Description |
|-----|---------|-------------|
| `--max-num-seqs` | `16` | Max concurrent sequences |
| `--gpu-memory-utilization` | `0.90` | GPU memory fraction |
| `--tensor-parallel-size` | `1` | GPUs for tensor parallelism |
| `--max-model-len` | `8192` | Max context length |

## Concurrency Model

- A shared `httpx.AsyncClient` is created at startup for connection reuse.
- Each vLLM call is guarded by a semaphore (`MAX_CONCURRENT`) to avoid GPU overload.
- Batch endpoints run one task per image; the semaphore provides backpressure.

## Image Validation

- Rejects empty files and oversized payloads.
- Validates that the bytes are a real image via PIL verify.
- Enforces format allow-list and pixel count limits.
- Encodes to a base64 data URI for the OpenAI-compatible API.

## Error Handling

- `400` for invalid uploads (empty files, unsupported format, size limits).
- `502` for upstream vLLM errors or unexpected response shapes.
- Batch requests return per-file errors instead of failing the entire request.

## Models

Swap the model by changing `MODEL_NAME` in both the vLLM command and app environment:

- `google/gemma-3-27b-it` — Best quality, needs ~60GB VRAM
- `google/gemma-3-12b-it` — Good balance
- `google/gemma-3-4b-it` — Lightweight, runs on smaller GPUs

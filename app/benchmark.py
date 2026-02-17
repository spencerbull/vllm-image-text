#!/usr/bin/env python3
"""
Concurrency Throughput Benchmark
================================
Tests the vLLM image-to-text API at various concurrency levels and prints
results in a formatted table.

Usage:
    # Against local docker compose setup
    python benchmark.py

    # Against remote server
    python benchmark.py --url http://10.0.0.5:8080

    # Custom concurrency levels and image
    python benchmark.py --levels 1,2,4,8,16,32 --image test.png --runs 3
"""

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx

# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_URL = "http://localhost:8080"
DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_LEVELS = [1, 2, 4, 8, 16]
DEFAULT_RUNS = 3  # runs per concurrency level (for stable averages)


# ─── GPU & vLLM info ────────────────────────────────────────────────────────

async def get_gpu_info() -> str:
    """Detect GPU name via nvidia-smi. Returns string like 'NVIDIA RTX Pro 6000'."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0 and stdout:
            lines = stdout.decode().strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    gpus.append(f"{parts[0]} ({parts[1]} MiB)")
                else:
                    gpus.append(line.strip())
            return " | ".join(gpus)
    except FileNotFoundError:
        pass
    return "Unknown GPU"


async def get_vllm_model_info(vllm_url: str) -> dict:
    """Query vLLM /v1/models endpoint for model name and details."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{vllm_url}/v1/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    model = data["data"][0]
                    return {"model": model.get("id", "unknown")}
    except Exception:
        pass
    return {"model": "unknown"}


# ─── Generate a test image if none provided ─────────────────────────────────

def generate_test_image(path: str = "/tmp/bench_test.png") -> str:
    """Create a simple test image with text for benchmarking."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (800, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        lines = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
        ]
        y = 20
        for line in lines:
            draw.text((20, y), line, fill=(0, 0, 0))
            y += 50
        img.save(path)
        return path
    except ImportError:
        raise RuntimeError("Pillow is required: pip install pillow")


# ─── Single request ─────────────────────────────────────────────────────────

async def send_request(
    client: httpx.AsyncClient,
    url: str,
    image_path: str,
) -> dict:
    """
    Send a single /extract-text request and measure latency.

    Returns:
        dict with 'latency_ms', 'status', 'text_lines' count, and 'error' if any.
    """
    start = time.perf_counter()
    try:
        with open(image_path, "rb") as f:
            resp = await client.post(
                f"{url}/extract-text",
                files={"file": ("test.png", f, "image/png")},
                timeout=120.0,
            )
        elapsed = (time.perf_counter() - start) * 1000  # ms

        if resp.status_code == 200:
            data = resp.json()
            # Extract token usage if vLLM returns it
            usage = data.get("usage", {})
            return {
                "latency_ms": elapsed,
                "status": 200,
                "text_lines": len(data.get("text", [])),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "error": None,
            }
        else:
            return {
                "latency_ms": elapsed,
                "status": resp.status_code,
                "text_lines": 0,
                "error": resp.text[:100],
            }
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "latency_ms": elapsed,
            "status": 0,
            "text_lines": 0,
            "error": str(e)[:100],
        }


async def send_request_streaming(
    client: httpx.AsyncClient,
    vllm_url: str,
    image_b64_uri: str,
    model_name: str,
    prompt: str = "Extract all text from this image. Return each line on a separate line.",
) -> dict:
    """
    Send request directly to vLLM with streaming to measure prefill vs decode.

    By streaming, we can measure:
      - time_to_first_token (TTFT): prefill latency
      - inter_token_latency (ITL): avg decode time per token
      - total generation time

    Returns dict with prefill_ms, decode_ms, ttft_ms, itl_ms, tokens, etc.
    """
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_token_time = None
    token_times = []
    completion_tokens = 0
    prompt_tokens = 0
    total_tokens = 0

    try:
        async with client.stream(
            "POST",
            f"{vllm_url}/v1/chat/completions",
            json=payload,
            timeout=120.0,
        ) as resp:
            if resp.status_code != 200:
                return {"status": resp.status_code, "error": "non-200"}

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = __import__("json").loads(data_str)
                except Exception:
                    continue

                # Check for usage in final chunk
                if "usage" in chunk and chunk["usage"]:
                    prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    completion_tokens = chunk["usage"].get("completion_tokens", 0)
                    total_tokens = chunk["usage"].get("total_tokens", 0)

                # Track token arrival times
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    token_times.append(now)

    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {"status": 0, "error": str(e)[:100], "latency_ms": elapsed}

    end = time.perf_counter()
    total_ms = (end - start) * 1000

    # Prefill = time to first token
    ttft_ms = ((first_token_time - start) * 1000) if first_token_time else total_ms

    # Decode = first token to last token
    if len(token_times) > 1:
        decode_ms = (token_times[-1] - token_times[0]) * 1000
        itl_ms = decode_ms / (len(token_times) - 1)  # avg inter-token latency
    else:
        decode_ms = 0
        itl_ms = 0

    # Tokens per second for decode phase only
    decode_tps = (len(token_times) / (decode_ms / 1000)) if decode_ms > 0 else 0

    return {
        "status": 200,
        "latency_ms": total_ms,
        "ttft_ms": ttft_ms,           # prefill time (time to first token)
        "decode_ms": decode_ms,        # total decode time
        "itl_ms": itl_ms,             # inter-token latency
        "decode_tps": decode_tps,      # decode tokens/sec
        "num_chunks": len(token_times),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "error": None,
    }


# ─── Run one concurrency level ──────────────────────────────────────────────

async def run_level(
    url: str,
    image_path: str,
    concurrency: int,
    num_requests: int | None = None,
) -> dict:
    """
    Fire `num_requests` concurrent requests (defaults to concurrency count).

    Returns:
        dict with aggregated stats: total_time_ms, avg_latency_ms, p50, p95,
        p99, throughput_rps, success_count, error_count.
    """
    num_requests = num_requests or concurrency

    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        tasks = [
            send_request(client, url, image_path)
            for _ in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_elapsed = (time.perf_counter() - wall_start) * 1000  # ms

    successes = [r for r in results if r["status"] == 200]
    latencies = [r["latency_ms"] for r in successes]
    total_tokens = sum(r.get("total_tokens", 0) for r in successes)
    completion_tokens = sum(r.get("completion_tokens", 0) for r in successes)
    errors = [r for r in results if r["status"] != 200]

    if not latencies:
        return {
            "concurrency": concurrency,
            "total_time_ms": wall_elapsed,
            "avg_latency_ms": 0,
            "p50_ms": 0,
            "p95_ms": 0,
            "p99_ms": 0,
            "throughput_rps": 0,
            "tokens_per_sec": 0,
            "completion_tok_per_sec": 0,
            "total_tokens": 0,
            "success": 0,
            "errors": len(errors),
        }

    latencies.sort()
    p50_idx = int(len(latencies) * 0.50)
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)

    return {
        "concurrency": concurrency,
        "total_time_ms": wall_elapsed,
        "avg_latency_ms": statistics.mean(latencies),
        "p50_ms": latencies[p50_idx],
        "p95_ms": latencies[p95_idx],
        "p99_ms": latencies[p99_idx],
        "throughput_rps": (len(latencies) / wall_elapsed) * 1000,
        "tokens_per_sec": (total_tokens / wall_elapsed) * 1000 if total_tokens else 0,
        "completion_tok_per_sec": (completion_tokens / wall_elapsed) * 1000 if completion_tokens else 0,
        "total_tokens": total_tokens,
        "success": len(latencies),
        "errors": len(errors),
    }


# ─── Pretty table printer ───────────────────────────────────────────────────

def print_results(all_results: list[dict], gpu_name: str = "", model_name: str = "") -> None:
    """Print benchmark results in a clean formatted table."""
    header = (
        f"{'Concurrency':>12} │ {'Avg (ms)':>10} │ {'P50 (ms)':>10} │ "
        f"{'P95 (ms)':>10} │ {'P99 (ms)':>10} │ {'RPS':>8} │ "
        f"{'tok/s':>8} │ {'gen tok/s':>10} │ "
        f"{'OK':>4} │ {'Err':>4} │ {'Wall (ms)':>10}"
    )
    sep = "─" * len(header)

    print()
    print("  ┌" + "─" * (len(header) + 2) + "┐")
    print("  │  vLLM Image-to-Text Concurrency Benchmark" + " " * (len(header) - 43) + "│")
    print("  └" + "─" * (len(header) + 2) + "┘")
    if gpu_name:
        print(f"  GPU: {gpu_name}")
    if model_name:
        print(f"  Model: {model_name}")
    print()
    print(f"  {header}")
    print(f"  {sep}")

    for r in all_results:
        row = (
            f"{r['concurrency']:>12} │ {r['avg_latency_ms']:>10.1f} │ "
            f"{r['p50_ms']:>10.1f} │ {r['p95_ms']:>10.1f} │ "
            f"{r['p99_ms']:>10.1f} │ {r['throughput_rps']:>8.2f} │ "
            f"{r['tokens_per_sec']:>8.1f} │ {r['completion_tok_per_sec']:>10.1f} │ "
            f"{r['success']:>4} │ {r['errors']:>4} │ {r['total_time_ms']:>10.1f}"
        )
        print(f"  {row}")

    print(f"  {sep}")
    print()


# ─── Prefill/Decode benchmark ────────────────────────────────────────────────

async def run_prefill_decode_level(
    vllm_url: str,
    image_b64_uri: str,
    model_name: str,
    concurrency: int,
) -> dict:
    """
    Run streaming requests at a given concurrency to measure prefill vs decode.

    Returns aggregated stats: avg TTFT, avg ITL, decode tok/s, prefill tok/s.
    """
    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        tasks = [
            send_request_streaming(client, vllm_url, image_b64_uri, model_name)
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall_ms = (time.perf_counter() - wall_start) * 1000

    successes = [r for r in results if r.get("status") == 200 and r.get("error") is None]
    errors = len(results) - len(successes)

    if not successes:
        return {
            "concurrency": concurrency,
            "avg_ttft_ms": 0, "avg_itl_ms": 0, "avg_decode_tps": 0,
            "total_prompt_tok": 0, "total_completion_tok": 0,
            "prefill_tok_per_sec": 0, "decode_tok_per_sec": 0,
            "wall_ms": wall_ms, "success": 0, "errors": errors,
        }

    avg_ttft = statistics.mean(r["ttft_ms"] for r in successes)
    avg_itl = statistics.mean(r["itl_ms"] for r in successes)
    avg_decode_tps = statistics.mean(r["decode_tps"] for r in successes)
    total_prompt = sum(r["prompt_tokens"] for r in successes)
    total_completion = sum(r["completion_tokens"] for r in successes)

    # Aggregate throughput: total tokens processed / wall clock
    prefill_tps = (total_prompt / wall_ms) * 1000 if total_prompt else 0
    decode_tps = (total_completion / wall_ms) * 1000 if total_completion else 0

    return {
        "concurrency": concurrency,
        "avg_ttft_ms": avg_ttft,
        "avg_itl_ms": avg_itl,
        "avg_decode_tps": avg_decode_tps,
        "total_prompt_tok": total_prompt,
        "total_completion_tok": total_completion,
        "prefill_tok_per_sec": prefill_tps,
        "decode_tok_per_sec": decode_tps,
        "wall_ms": wall_ms,
        "success": len(successes),
        "errors": errors,
    }


def print_prefill_decode_results(
    results: list[dict], gpu_name: str = "", model_name: str = ""
) -> None:
    """Print prefill/decode breakdown table."""
    header = (
        f"{'Concurrency':>12} │ {'TTFT (ms)':>10} │ {'ITL (ms)':>10} │ "
        f"{'Prefill t/s':>12} │ {'Decode t/s':>12} │ "
        f"{'Prompt tok':>10} │ {'Gen tok':>10} │ "
        f"{'OK':>4} │ {'Err':>4} │ {'Wall (ms)':>10}"
    )
    sep = "─" * len(header)

    print()
    print("  ┌" + "─" * (len(header) + 2) + "┐")
    print("  │  vLLM Prefill / Decode Breakdown" + " " * (len(header) - 33) + "│")
    print("  └" + "─" * (len(header) + 2) + "┘")
    if gpu_name:
        print(f"  GPU: {gpu_name}")
    if model_name:
        print(f"  Model: {model_name}")
    print()
    print(f"  TTFT = Time to First Token (prefill latency)")
    print(f"  ITL  = Inter-Token Latency (avg decode step)")
    print()
    print(f"  {header}")
    print(f"  {sep}")

    for r in results:
        row = (
            f"{r['concurrency']:>12} │ {r['avg_ttft_ms']:>10.1f} │ "
            f"{r['avg_itl_ms']:>10.1f} │ "
            f"{r['prefill_tok_per_sec']:>12.1f} │ {r['decode_tok_per_sec']:>12.1f} │ "
            f"{r['total_prompt_tok']:>10} │ {r['total_completion_tok']:>10} │ "
            f"{r['success']:>4} │ {r['errors']:>4} │ {r['wall_ms']:>10.1f}"
        )
        print(f"  {row}")

    print(f"  {sep}")
    print()


# ─── Chart generation ────────────────────────────────────────────────────────

def save_charts(all_results: list[dict], output_dir: str = ".") -> list[str]:
    """
    Generate and save benchmark charts as PNG files.

    Creates:
      - throughput_chart.png: RPS vs concurrency level
      - latency_chart.png: Avg/P50/P95/P99 latency vs concurrency

    Returns:
        List of saved file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping charts (pip install matplotlib)")
        return []

    levels = [r["concurrency"] for r in all_results]
    saved = []

    # ── Throughput chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    rps = [r["throughput_rps"] for r in all_results]
    ax.bar(range(len(levels)), rps, color="#4C9AFF", edgecolor="#2B6CB0", width=0.6)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("Concurrency Level", fontsize=12)
    ax.set_ylabel("Requests per Second (RPS)", fontsize=12)
    ax.set_title("vLLM Image-to-Text — Throughput vs Concurrency", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    # Add value labels on bars
    for i, v in enumerate(rps):
        ax.text(i, v + max(rps) * 0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = f"{output_dir}/throughput_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  📊 Saved: {path}")

    # ── Latency chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(levels))
    width = 0.2
    avg = [r["avg_latency_ms"] for r in all_results]
    p50 = [r["p50_ms"] for r in all_results]
    p95 = [r["p95_ms"] for r in all_results]
    p99 = [r["p99_ms"] for r in all_results]

    ax.bar([i - 1.5 * width for i in x], avg, width, label="Avg", color="#4C9AFF", edgecolor="#2B6CB0")
    ax.bar([i - 0.5 * width for i in x], p50, width, label="P50", color="#48BB78", edgecolor="#276749")
    ax.bar([i + 0.5 * width for i in x], p95, width, label="P95", color="#ECC94B", edgecolor="#975A16")
    ax.bar([i + 1.5 * width for i in x], p99, width, label="P99", color="#FC8181", edgecolor="#C53030")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("Concurrency Level", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("vLLM Image-to-Text — Latency Distribution vs Concurrency", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = f"{output_dir}/latency_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  📊 Saved: {path}")

    # ── Tokens/sec chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    tps = [r["tokens_per_sec"] for r in all_results]
    gen_tps = [r["completion_tok_per_sec"] for r in all_results]
    x = range(len(levels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], tps, width, label="Total tok/s", color="#4C9AFF", edgecolor="#2B6CB0")
    ax.bar([i + width / 2 for i in x], gen_tps, width, label="Generation tok/s", color="#48BB78", edgecolor="#276749")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("Concurrency Level", fontsize=12)
    ax.set_ylabel("Tokens per Second", fontsize=12)
    ax.set_title("vLLM Image-to-Text — Token Throughput vs Concurrency", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = f"{output_dir}/tokens_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  📊 Saved: {path}")

    return saved


def save_prefill_decode_charts(all_results: list[dict], output_dir: str = ".") -> list[str]:
    """Generate prefill/decode specific charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping charts")
        return []

    levels = [r["concurrency"] for r in all_results]
    saved = []

    # ── TTFT + ITL chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ttft = [r["avg_ttft_ms"] for r in all_results]
    ax1.bar(range(len(levels)), ttft, color="#FC8181", edgecolor="#C53030", width=0.6)
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels([str(l) for l in levels])
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Time to First Token (Prefill)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(ttft):
        ax1.text(i, v + max(ttft) * 0.02, f"{v:.0f}", ha="center", fontsize=9)

    itl = [r["avg_itl_ms"] for r in all_results]
    ax2.bar(range(len(levels)), itl, color="#4C9AFF", edgecolor="#2B6CB0", width=0.6)
    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels([str(l) for l in levels])
    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("ITL (ms)")
    ax2.set_title("Inter-Token Latency (Decode)", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(itl):
        ax2.text(i, v + max(itl) * 0.02 if max(itl) > 0 else 0, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("vLLM Prefill vs Decode Latency", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = f"{output_dir}/prefill_decode_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  📊 Saved: {path}")

    # ── Prefill vs Decode throughput chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(levels))
    width = 0.35
    prefill_tps = [r["prefill_tok_per_sec"] for r in all_results]
    decode_tps = [r["decode_tok_per_sec"] for r in all_results]

    ax.bar([i - width / 2 for i in x], prefill_tps, width, label="Prefill tok/s", color="#FC8181", edgecolor="#C53030")
    ax.bar([i + width / 2 for i in x], decode_tps, width, label="Decode tok/s", color="#4C9AFF", edgecolor="#2B6CB0")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("Concurrency Level", fontsize=12)
    ax.set_ylabel("Tokens per Second", fontsize=12)
    ax.set_title("vLLM Prefill vs Decode Throughput", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = f"{output_dir}/prefill_decode_throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  📊 Saved: {path}")

    return saved


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM image-to-text concurrency throughput"
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"API base URL (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--image", default=None,
        help="Path to test image (auto-generated if not provided)"
    )
    parser.add_argument(
        "--levels", default=",".join(str(l) for l in DEFAULT_LEVELS),
        help=f"Comma-separated concurrency levels (default: {','.join(str(l) for l in DEFAULT_LEVELS)})"
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Runs per level, results averaged (default: {DEFAULT_RUNS})"
    )
    parser.add_argument(
        "--charts", default=".",
        help="Directory to save chart PNGs (default: current dir)"
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation"
    )
    parser.add_argument(
        "--vllm-url", default=DEFAULT_VLLM_URL,
        help=f"vLLM direct URL for model info (default: {DEFAULT_VLLM_URL})"
    )
    args = parser.parse_args()

    # Resolve test image
    if args.image:
        image_path = args.image
        if not Path(image_path).exists():
            print(f"Error: image not found: {image_path}")
            return
    else:
        print("Generating test image...")
        image_path = generate_test_image()

    levels = [int(x) for x in args.levels.split(",")]

    # Health check
    print(f"Checking {args.url}/health ...")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{args.url}/health", timeout=5.0)
            if r.status_code != 200:
                print(f"Health check failed: {r.status_code}")
                return
        except Exception as e:
            print(f"Cannot reach API: {e}")
            return

    # Detect GPU and model
    gpu_name = await get_gpu_info()
    model_info = await get_vllm_model_info(args.vllm_url)
    print(f"GPU: {gpu_name}")
    print(f"Model: {model_info['model']}")
    print(f"Running benchmark: levels={levels}, runs={args.runs}")
    print()

    all_results = []
    for level in levels:
        level_runs = []
        for run_idx in range(args.runs):
            print(f"  Concurrency {level:>3} — run {run_idx + 1}/{args.runs} ...", end="", flush=True)
            result = await run_level(args.url, image_path, level)
            level_runs.append(result)
            print(f" {result['throughput_rps']:.2f} rps")

        # Average across runs
        avg_result = {
            "concurrency": level,
            "total_time_ms": statistics.mean(r["total_time_ms"] for r in level_runs),
            "avg_latency_ms": statistics.mean(r["avg_latency_ms"] for r in level_runs),
            "p50_ms": statistics.mean(r["p50_ms"] for r in level_runs),
            "p95_ms": statistics.mean(r["p95_ms"] for r in level_runs),
            "p99_ms": statistics.mean(r["p99_ms"] for r in level_runs),
            "throughput_rps": statistics.mean(r["throughput_rps"] for r in level_runs),
            "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in level_runs),
            "completion_tok_per_sec": statistics.mean(r["completion_tok_per_sec"] for r in level_runs),
            "total_tokens": sum(r["total_tokens"] for r in level_runs),
            "success": sum(r["success"] for r in level_runs),
            "errors": sum(r["errors"] for r in level_runs),
        }
        all_results.append(avg_result)

    print_results(all_results, gpu_name=gpu_name, model_name=model_info["model"])

    # ── Phase 2: Prefill/Decode breakdown (streaming directly to vLLM) ──
    print("=" * 60)
    print("Phase 2: Prefill / Decode Breakdown (streaming to vLLM)")
    print("=" * 60)

    # Prepare base64 image for direct vLLM calls
    import base64 as b64mod
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = b64mod.b64encode(img_bytes).decode()
    img_uri = f"data:image/png;base64,{img_b64}"

    pd_results = []
    for level in levels:
        pd_level_runs = []
        for run_idx in range(args.runs):
            print(f"  Concurrency {level:>3} — run {run_idx + 1}/{args.runs} ...", end="", flush=True)
            result = await run_prefill_decode_level(
                args.vllm_url, img_uri, model_info["model"], level
            )
            pd_level_runs.append(result)
            print(f" TTFT={result['avg_ttft_ms']:.0f}ms, decode={result['decode_tok_per_sec']:.1f} t/s")

        avg_pd = {
            "concurrency": level,
            "avg_ttft_ms": statistics.mean(r["avg_ttft_ms"] for r in pd_level_runs),
            "avg_itl_ms": statistics.mean(r["avg_itl_ms"] for r in pd_level_runs),
            "avg_decode_tps": statistics.mean(r["avg_decode_tps"] for r in pd_level_runs),
            "total_prompt_tok": sum(r["total_prompt_tok"] for r in pd_level_runs),
            "total_completion_tok": sum(r["total_completion_tok"] for r in pd_level_runs),
            "prefill_tok_per_sec": statistics.mean(r["prefill_tok_per_sec"] for r in pd_level_runs),
            "decode_tok_per_sec": statistics.mean(r["decode_tok_per_sec"] for r in pd_level_runs),
            "wall_ms": statistics.mean(r["wall_ms"] for r in pd_level_runs),
            "success": sum(r["success"] for r in pd_level_runs),
            "errors": sum(r["errors"] for r in pd_level_runs),
        }
        pd_results.append(avg_pd)

    print_prefill_decode_results(pd_results, gpu_name=gpu_name, model_name=model_info["model"])

    # Generate charts
    if not args.no_charts:
        save_charts(all_results, output_dir=args.charts)
        save_prefill_decode_charts(pd_results, output_dir=args.charts)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Concurrent OpenAI-compatible chat-completions benchmark for llama.cpp."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

try:
    import aiohttp
except ModuleNotFoundError:  # Allow --help to work before dependencies are installed.
    aiohttp = None


DEFAULT_PROMPT = (
    "Explain how a CPU and ALU work, then implement a small ALU and "
    "Von Neumann architecture CPU in synthesizable Verilog."
)


@dataclass(slots=True)
class RequestResult:
    success: bool
    status: int
    prompt_tokens: int
    completion_tokens: int
    tokens_estimated: bool
    elapsed_seconds: float
    ttft_seconds: float | None
    error: str | None = None


class BenchmarkError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a llama.cpp OpenAI-compatible chat endpoint. Both "
            "HOST PORT CONCURRENCY and HOST:PORT CONCURRENCY forms are supported."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("endpoint", help="host, host:port, or http(s) URL")
    parser.add_argument(
        "legacy",
        nargs="*",
        metavar="N",
        help="legacy positional port, concurrency, and max-token values",
    )
    parser.add_argument("--port", type=int, help="port when endpoint is a bare host")
    parser.add_argument("-c", "--concurrency", type=int, help="simultaneous requests")
    parser.add_argument("-n", "--max-tokens", type=int, help="maximum generated tokens per request")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-r", "--requests", type=int, help="total measured requests")
    mode.add_argument("-d", "--duration", type=float, help="load duration in seconds")
    parser.add_argument("-w", "--warmup", type=int, default=1, help="unmeasured warm-up requests")
    parser.add_argument("--model", help="model ID; defaults to the first ID from /v1/models")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLAMA_API_KEY"),
        help="Bearer token; prefer the LLAMA_API_KEY environment variable",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=DEFAULT_PROMPT, help="benchmark prompt")
    prompt_group.add_argument("--prompt-file", type=Path, help="UTF-8 file containing the prompt")
    parser.add_argument(
        "--temperature",
        type=float,
        help="override the server's configured sampling temperature",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="continue generating through EOS until max-tokens",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream responses to measure time to first token",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="total timeout per request")
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of the text report")
    return parser


def positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {value}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1")
    return parsed


def resolve_arguments(args: argparse.Namespace) -> argparse.Namespace:
    legacy = list(args.legacy)
    endpoint = args.endpoint.strip()
    if not endpoint:
        raise ValueError("endpoint cannot be empty")

    has_scheme = "://" in endpoint
    if args.port is not None:
        if has_scheme or ":" in endpoint:
            raise ValueError("--port can only be used with a bare host")
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        endpoint = f"{endpoint}:{args.port}"
    elif not has_scheme and ":" not in endpoint:
        # Backward compatibility with: llmload.py HOST PORT CONCURRENCY MAX_TOKENS
        if legacy:
            port = positive_int(legacy.pop(0), "port")
            if port > 65535:
                raise ValueError("port must be between 1 and 65535")
        else:
            port = 8080
        endpoint = f"{endpoint}:{port}"

    if args.concurrency is None and legacy:
        args.concurrency = positive_int(legacy.pop(0), "concurrency")
    if args.max_tokens is None and legacy:
        args.max_tokens = positive_int(legacy.pop(0), "max tokens")
    if legacy:
        raise ValueError(f"unexpected positional arguments: {' '.join(legacy)}")

    if args.concurrency is None:
        args.concurrency = 1
    if args.max_tokens is None:
        args.max_tokens = 256
    if args.concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("max tokens must be at least 1")
    if args.requests is not None and args.requests < 1:
        raise ValueError("requests must be at least 1")
    if args.duration is not None and args.duration <= 0:
        raise ValueError("duration must be greater than zero")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")
    if args.timeout <= 0 or args.connect_timeout <= 0:
        raise ValueError("timeouts must be greater than zero")
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("temperature cannot be negative")
    if args.requests is None and args.duration is None:
        args.requests = max(10, args.concurrency * 4)

    if args.prompt_file:
        try:
            args.prompt = args.prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read prompt file: {exc}") from exc
    if not args.prompt:
        raise ValueError("prompt cannot be empty")

    if not has_scheme:
        endpoint = f"http://{endpoint}"
    endpoint = endpoint.rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid endpoint: {args.endpoint}")
    if parsed.path.endswith("/v1/chat/completions"):
        completion_url = endpoint
    else:
        completion_url = f"{endpoint}/v1/chat/completions"
    args.completion_url = completion_url
    parsed = urlsplit(completion_url)
    args.models_url = urlunsplit((parsed.scheme, parsed.netloc, "/v1/models", "", ""))
    return args


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def discover_model(session: Any, models_url: str) -> str:
    try:
        async with session.get(models_url) as response:
            body = await response.text()
            if response.status != 200:
                raise BenchmarkError(f"GET {models_url} returned HTTP {response.status}: {body[:500]}")
            data = json.loads(body)
    except BenchmarkError:
        raise
    except Exception as exc:
        raise BenchmarkError(f"could not discover a model from {models_url}: {exc}") from exc
    models = data.get("data", [])
    if not models or not models[0].get("id"):
        raise BenchmarkError(f"no model IDs returned by {models_url}; pass --model explicitly")
    return str(models[0]["id"])


def streamed_text(delta: dict[str, Any]) -> str:
    parts = []
    for key in ("content", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    return "".join(parts)


async def execute_request(
    session: Any,
    url: str,
    payload: dict[str, Any],
    stream: bool,
) -> RequestResult:
    started = time.perf_counter()
    first_token_at: float | None = None
    token_events = 0
    prompt_tokens = 0
    completion_tokens: int | None = None

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                body = await response.text()
                return RequestResult(
                    False,
                    response.status,
                    0,
                    0,
                    False,
                    time.perf_counter() - started,
                    None,
                    f"HTTP {response.status}: {body[:500]}",
                )

            if not stream:
                data = await response.json(content_type=None)
                usage = data.get("usage", {})
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                value = usage.get("completion_tokens")
                completion_tokens = int(value) if value is not None else 0
                return RequestResult(
                    True,
                    response.status,
                    prompt_tokens,
                    completion_tokens,
                    value is None,
                    time.perf_counter() - started,
                    None,
                )

            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                encoded = line[5:].strip()
                if not encoded or encoded == "[DONE]":
                    continue
                event = json.loads(encoded)
                stream_error = event.get("error")
                if stream_error is not None:
                    if isinstance(stream_error, dict):
                        message = str(stream_error.get("message") or stream_error)
                        code = stream_error.get("code")
                        error = (
                            f"stream error {code}: {message}"
                            if code is not None
                            else f"stream error: {message}"
                        )
                    else:
                        error = f"stream error: {stream_error}"
                    elapsed = time.perf_counter() - started
                    estimated = completion_tokens is None
                    if completion_tokens is None:
                        completion_tokens = token_events
                    ttft = first_token_at - started if first_token_at is not None else None
                    return RequestResult(
                        False,
                        response.status,
                        prompt_tokens,
                        completion_tokens,
                        estimated,
                        elapsed,
                        ttft,
                        error,
                    )
                usage = event.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or 0)
                    value = usage.get("completion_tokens")
                    if value is not None:
                        completion_tokens = int(value)
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    if streamed_text(delta):
                        token_events += 1
                        if first_token_at is None:
                            first_token_at = time.perf_counter()

            elapsed = time.perf_counter() - started
            estimated = completion_tokens is None
            if completion_tokens is None:
                completion_tokens = token_events
            ttft = first_token_at - started if first_token_at is not None else None
            return RequestResult(
                True,
                response.status,
                prompt_tokens,
                completion_tokens,
                estimated,
                elapsed,
                ttft,
            )
    except asyncio.TimeoutError:
        error = "request timed out"
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        error = f"invalid JSON/SSE response: {exc}"
    except Exception as exc:  # aiohttp exceptions are optional until runtime.
        error = f"{type(exc).__name__}: {exc}"
    return RequestResult(
        False,
        0,
        0,
        0,
        False,
        time.perf_counter() - started,
        None,
        error,
    )


async def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], list[RequestResult]]:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    timeout = aiohttp.ClientTimeout(total=args.timeout, connect=args.connect_timeout)
    connector = aiohttp.TCPConnector(
        limit=args.concurrency,
        limit_per_host=args.concurrency,
        ttl_dns_cache=300,
    )

    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
        model = args.model or await discover_model(session, args.models_url)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "stream": args.stream,
        }
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.ignore_eos:
            payload["ignore_eos"] = True
        if args.stream:
            payload["stream_options"] = {"include_usage": True}

        for index in range(args.warmup):
            result = await execute_request(session, args.completion_url, payload, args.stream)
            if not result.success:
                raise BenchmarkError(f"warm-up request {index + 1} failed: {result.error}")

        results: list[RequestResult] = []
        next_request = 0
        measured_started = time.perf_counter()
        deadline = measured_started + args.duration if args.duration is not None else None

        async def worker() -> None:
            nonlocal next_request
            while True:
                if deadline is not None:
                    if time.perf_counter() >= deadline:
                        return
                else:
                    assert args.requests is not None
                    if next_request >= args.requests:
                        return
                    next_request += 1
                results.append(
                    await execute_request(session, args.completion_url, payload, args.stream)
                )

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        wall_seconds = time.perf_counter() - measured_started

    successes = [result for result in results if result.success]
    failures = [result for result in results if not result.success]
    prompt_tokens = sum(result.prompt_tokens for result in successes)
    completion_tokens = sum(result.completion_tokens for result in successes)
    total_tokens = prompt_tokens + completion_tokens
    latencies = [result.elapsed_seconds for result in successes]
    ttfts = [result.ttft_seconds for result in successes if result.ttft_seconds is not None]
    generation_rates = [
        (result.completion_tokens - 1) / (result.elapsed_seconds - result.ttft_seconds)
        for result in successes
        if result.ttft_seconds is not None
        and result.completion_tokens > 1
        and result.elapsed_seconds > result.ttft_seconds
    ]
    errors = Counter(result.error or "unknown error" for result in failures)

    report: dict[str, Any] = {
        "endpoint": args.completion_url,
        "model": model,
        "concurrency": args.concurrency,
        "max_tokens_per_request": args.max_tokens,
        "temperature_override": args.temperature,
        "streaming": args.stream,
        "ignore_eos": args.ignore_eos,
        "warmup_requests": args.warmup,
        "requested_requests": args.requests,
        "requested_duration_seconds": args.duration,
        "attempted_requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "wall_seconds": wall_seconds,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_estimated": any(result.tokens_estimated for result in successes),
        "total_tokens_per_second": total_tokens / wall_seconds if wall_seconds else 0.0,
        "completion_tokens_per_second": completion_tokens / wall_seconds if wall_seconds else 0.0,
        "successful_requests_per_second": len(successes) / wall_seconds if wall_seconds else 0.0,
        "latency_seconds": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "ttft_seconds": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "per_request_generation_tokens_per_second": {
            "p50": percentile(generation_rates, 0.50),
            "p95": percentile(generation_rates, 0.95),
        },
        "errors": dict(errors),
    }
    return report, results


def format_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def print_report(report: dict[str, Any]) -> None:
    latency = report["latency_seconds"]
    ttft = report["ttft_seconds"]
    generation = report["per_request_generation_tokens_per_second"]
    estimated = " (estimated from stream events)" if report["tokens_estimated"] else ""
    mode = (
        f"{report['requested_requests']} requests"
        if report["requested_requests"] is not None
        else f"{report['requested_duration_seconds']} seconds"
    )
    temperature = (
        "server default"
        if report["temperature_override"] is None
        else str(report["temperature_override"])
    )
    print("\n=== llama.cpp Load Benchmark ===")
    print(f"Endpoint: {report['endpoint']}")
    print(f"Model: {report['model']}")
    print(f"Mode: {mode}; concurrency={report['concurrency']}; warmup={report['warmup_requests']}")
    print(
        f"Request sampling: temperature={temperature}; ignore_eos={report['ignore_eos']}"
    )
    print(
        "Requests: "
        f"{report['attempted_requests']} attempted, "
        f"{report['successful_requests']} successful, "
        f"{report['failed_requests']} failed"
    )
    print(f"Wall time: {report['wall_seconds']:.3f}s")
    print(f"Prompt tokens: {report['prompt_tokens']}")
    print(f"Completion tokens: {report['completion_tokens']}{estimated}")
    print(f"Total tokens: {report['total_tokens']}{estimated}")
    print(
        "Aggregate total token throughput: "
        f"{report['total_tokens_per_second']:.2f} tokens/sec"
    )
    print(
        "Aggregate completion throughput: "
        f"{report['completion_tokens_per_second']:.2f} tokens/sec"
    )
    print(
        "Successful request throughput: "
        f"{report['successful_requests_per_second']:.2f} requests/sec"
    )
    print(
        "Latency p50/p95/p99/max: "
        f"{format_seconds(latency['p50'])} / {format_seconds(latency['p95'])} / "
        f"{format_seconds(latency['p99'])} / {format_seconds(latency['max'])}"
    )
    if report["streaming"]:
        print(
            "TTFT p50/p95/p99: "
            f"{format_seconds(ttft['p50'])} / {format_seconds(ttft['p95'])} / "
            f"{format_seconds(ttft['p99'])}"
        )
        if generation["p50"] is not None:
            print(
                "Per-request generation rate p50/p95: "
                f"{generation['p50']:.2f} / {generation['p95']:.2f} tokens/sec"
            )
    if report["errors"]:
        print("Errors:")
        for message, count in report["errors"].items():
            print(f"  {count}x {message}")
    print()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args = resolve_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))
    if aiohttp is None:
        parser.error(
            "aiohttp is required; install Ubuntu's python3-aiohttp package or "
            "run: python3 -m pip install -r requirements.txt"
        )

    try:
        report, _ = asyncio.run(run_benchmark(args))
    except BenchmarkError as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    return 2 if report["failed_requests"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

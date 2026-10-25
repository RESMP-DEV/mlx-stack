#!/usr/bin/env python3
"""Record local MLX serving state without interfering with the server."""

from __future__ import annotations

import argparse
import fcntl
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8081/metrics"
DEFAULT_OUTPUT = Path(
    "~/.local/state/local27-mxfp4/watch/metrics-watch.jsonl"
).expanduser()


def fetch_metrics(url: str, timeout: float) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def compact_state(metrics: dict[str, Any]) -> dict[str, Any]:
    current = metrics.get("current_request") or {}
    last = metrics.get("last_request") or {}
    cache = metrics.get("prefix_cache") or {}
    memory = metrics.get("memory") or {}
    return {
        "current": {
            "request_id": current.get("request_id"),
            "state": current.get("state"),
            "elapsed_s": current.get("elapsed_s"),
            "prompt_tokens": current.get("prompt_tokens"),
            "prefill_tokens_processed": current.get("prefill_tokens_processed"),
            "prefill_tokens_total": current.get("prefill_tokens_total"),
            "generated_tokens": current.get("generated_tokens"),
            "cache_status": current.get("cache_status"),
            "cache_hit_tokens": current.get("cache_hit_tokens"),
        },
        "last": {
            "request_id": last.get("request_id"),
            "finish_reason": last.get("finish_reason"),
            "prefill_s": last.get("prefill_s"),
            "ttft_s": last.get("ttft_s"),
            "generated_tokens": last.get("generated_tokens"),
            "cache_hit_tokens": last.get("cache_hit_tokens"),
        },
        "cache": {
            "entries": cache.get("entries"),
            "hits": cache.get("hits"),
            "misses": cache.get("misses"),
            "prefill_tokens_saved": cache.get("prefill_tokens_saved"),
        },
        "memory": {
            "mlx_active_gb": memory.get("mlx_active_gb"),
            "mlx_cache_gb": memory.get("mlx_cache_gb"),
            "mlx_peak_gb": memory.get("mlx_peak_gb"),
        },
    }


def state_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    current = state["current"]
    return (
        current.get("request_id"),
        current.get("state"),
        current.get("prefill_tokens_processed"),
        current.get("generated_tokens"),
    )


def write_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 32 * 1024 * 1024:
        rotated = path.with_suffix(path.suffix + ".1")
        if rotated.exists():
            rotated.unlink()
        path.replace(rotated)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(event, handle, separators=(",", ":"))
        handle.write("\n")
        handle.flush()


def acquire_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--heartbeat", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    lock = acquire_lock(args.output)
    last_signature: tuple[Any, ...] | None = None
    last_heartbeat = 0.0
    failures = 0

    while True:
        now = time.time()
        metrics = fetch_metrics(args.url, args.timeout)
        if metrics is None:
            failures += 1
            if failures == 1 or failures % 30 == 0:
                write_event(
                    args.output,
                    {"ts": now, "event": "metrics_unavailable", "failures": failures},
                )
        else:
            failures = 0
            state = compact_state(metrics)
            signature = state_signature(state)
            heartbeat = now - last_heartbeat >= args.heartbeat
            if signature != last_signature or heartbeat:
                write_event(args.output, {"ts": now, "event": "state", **state})
            if signature != last_signature:
                last_signature = signature
            if heartbeat:
                last_heartbeat = now

        if args.once:
            break
        time.sleep(args.interval)

    lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

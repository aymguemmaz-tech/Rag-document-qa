"""Logging and lightweight tracing (per-stage timings, request ids, in-process stats)."""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger("ragqa")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False


class StageTimer:
    """Collects wall-clock milliseconds per named stage."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - start) * 1e3

    def total(self) -> float:
        return (time.perf_counter() - self._t0) * 1e3


class Stats:
    """Thread-safe in-process service statistics exposed at ``/v1/stats``."""

    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.abstentions = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self._latencies: deque[float] = deque(maxlen=window)

    def record(self, latency_ms: float, abstained: bool, prompt_tokens: int, completion: int, cost: float) -> None:
        with self._lock:
            self.requests += 1
            self.abstentions += int(abstained)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion
            self.cost_usd += cost
            self._latencies.append(latency_ms)

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = sorted(self._latencies)

            def pct(p: float) -> float | None:
                if not lat:
                    return None
                return round(lat[min(len(lat) - 1, round(p * (len(lat) - 1)))], 2)

            return {
                "requests": self.requests,
                "errors": self.errors,
                "abstention_rate": round(self.abstentions / self.requests, 4) if self.requests else None,
                "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)},
                "tokens": {"prompt": self.prompt_tokens, "completion": self.completion_tokens},
                "cost_usd": round(self.cost_usd, 6),
            }

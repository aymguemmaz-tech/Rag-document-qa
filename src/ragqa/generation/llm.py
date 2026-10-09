"""LLM client for any OpenAI-compatible Chat Completions endpoint.

Works with OpenAI (default ``gpt-6-luna``) and with OpenAI-compatible servers such as Ollama,
vLLM or LM Studio via ``base_url``. Chat Completions is used deliberately: it is the API those
servers implement. The SDK already retries 429/5xx/connection errors with exponential backoff.

*Parameter negotiation*: models differ in what they accept (reasoning models may reject
``temperature``; local servers may reject ``reasoning_effort``). When a request fails with a
400 naming an unsupported parameter, the client drops it, remembers that, and retries - so a
single config works across providers.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ragqa.pricing import cost_usd
from ragqa.types import Usage

log = logging.getLogger(__name__)

Message = dict[str, str]
_NEGOTIABLE = ("temperature", "seed", "reasoning_effort", "top_p", "stream_options", "max_completion_tokens")


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str | None = None


class LLM(ABC):
    model: str = "llm"

    @abstractmethod
    def complete(self, messages: list[Message], **overrides: Any) -> LLMResponse: ...

    def stream(self, messages: list[Message], **overrides: Any) -> Iterator[str | LLMResponse]:
        """Yield text deltas, then the final :class:`LLMResponse` (full text + usage)."""
        response = self.complete(messages, **overrides)
        yield response.text
        yield response


def unsupported_params(exc: Exception, sent: dict[str, Any]) -> list[str]:
    """Return negotiable parameters an API error says are unsupported."""
    if type(exc).__name__ not in {"BadRequestError", "UnprocessableEntityError"}:
        return []
    message = str(exc).lower()
    if not any(w in message for w in ("unsupported", "not supported", "unrecognized", "unknown", "invalid")):
        return []
    return [p for p in _NEGOTIABLE if p in sent and p in message]


class OpenAIChatLLM(LLM):
    def __init__(
        self,
        model: str = "gpt-6-luna",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int = 500,
        timeout: float = 60.0,
        max_retries: int = 4,
        client: Any = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key or "not-needed", base_url=base_url, timeout=timeout, max_retries=max_retries
            )
        self._client = client
        self.model = model
        self.defaults: dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_tokens,
        }
        self._dropped: set[str] = set()
        self._lock = threading.Lock()

    def _params(self, overrides: dict[str, Any]) -> dict[str, Any]:
        params = {**self.defaults, **overrides}
        params = {k: v for k, v in params.items() if v is not None and k not in self._dropped}
        if "max_completion_tokens" in self._dropped:
            params["max_tokens"] = overrides.get("max_completion_tokens", self.defaults["max_completion_tokens"])
        return params

    def _create(self, **kwargs: Any) -> Any:
        for _ in range(len(_NEGOTIABLE) + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                bad = unsupported_params(exc, kwargs)
                if not bad:
                    raise
                log.warning("Model %s rejected %s; retrying without them", self.model, bad)
                with self._lock:
                    self._dropped.update(bad)
                for param in bad:
                    value = kwargs.pop(param)
                    if param == "max_completion_tokens":
                        kwargs["max_tokens"] = value
        raise RuntimeError("parameter negotiation did not converge")  # pragma: no cover

    def _usage(self, raw: Any) -> Usage:
        prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
        completion = int(getattr(raw, "completion_tokens", 0) or 0)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            llm_calls=1,
            cost_usd=cost_usd(self.model, prompt, completion),
        )

    def complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        start = time.perf_counter()
        response = self._create(model=self.model, messages=messages, **self._params(overrides))
        choice = response.choices[0]
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            usage=self._usage(getattr(response, "usage", None)),
            latency_ms=(time.perf_counter() - start) * 1e3,
            model=getattr(response, "model", self.model) or self.model,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def stream(self, messages: list[Message], **overrides: Any) -> Iterator[str | LLMResponse]:
        start = time.perf_counter()
        params = self._params(overrides)
        params["stream"] = True
        if "stream_options" not in self._dropped:
            params["stream_options"] = {"include_usage": True}
        events = self._create(model=self.model, messages=messages, **params)
        parts: list[str] = []
        usage = Usage(llm_calls=1)
        finish: str | None = None
        for event in events:
            if getattr(event, "usage", None):
                usage = self._usage(event.usage)
            for choice in getattr(event, "choices", None) or []:
                delta = getattr(choice.delta, "content", None)
                if delta:
                    parts.append(delta)
                    yield delta
                finish = getattr(choice, "finish_reason", None) or finish
        yield LLMResponse(
            text="".join(parts).strip(),
            usage=usage,
            latency_ms=(time.perf_counter() - start) * 1e3,
            model=self.model,
            finish_reason=finish,
        )

"""Test doubles for the LLM layer: a scripted LLM and a fake OpenAI client."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from ragqa.generation.llm import LLM, LLMResponse
from ragqa.types import Usage


class FakeLLM(LLM):
    """Scripted LLM for tests: a list of responses (last one repeats) or a function of messages."""

    model = "fake-llm"

    def __init__(self, responses: list[str] | Callable[[list[dict[str, str]]], str]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, str]], **overrides: Any) -> LLMResponse:
        self.calls.append((messages, overrides))
        if callable(self.responses):
            text = self.responses(messages)
        else:
            text = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        usage = Usage(prompt_tokens=100, completion_tokens=20, llm_calls=1, cost_usd=0.0001)
        return LLMResponse(text=text, usage=usage, latency_ms=1.0, model=self.model)


class BadRequestError(Exception):
    """Same class name as the OpenAI SDK error, which is what parameter negotiation inspects."""


def fake_openai_client(text: str = "Packs are stored at 15 °C to 25 °C [S1].", reject: tuple[str, ...] = ()) -> Any:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        for param in reject:
            if param in kwargs:
                raise BadRequestError(f"Unsupported parameter: '{param}' is not supported with this model.")
        if kwargs.get("stream"):
            words = text.split(" ")
            chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=w + (" " if i < len(words) - 1 else "")), finish_reason=None
                        )
                    ],
                    usage=None,
                )
                for i, w in enumerate(words)
            ]
            chunks.append(SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=60, completion_tokens=12)))
            return iter(chunks)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=60, completion_tokens=12),
            model=kwargs["model"],
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), calls=calls)

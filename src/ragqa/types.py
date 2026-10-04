"""Core data types shared by every stage of the pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Section:
    """A contiguous region of a document under one heading path (or one PDF page)."""

    heading_path: tuple[str, ...]
    start: int
    end: int
    page: int | None = None


@dataclass
class Document:
    """A loaded source document. ``text`` is the canonical text all offsets refer to."""

    doc_id: str
    title: str
    source: str
    text: str
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.title.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(self.text.encode("utf-8"))
        return digest.hexdigest()


@dataclass
class Chunk:
    """A retrievable unit of text. ``start``/``end`` are character offsets into the document."""

    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    index: int
    start: int
    end: int
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    context_header: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text used for embedding and lexical indexing (contextual header + body)."""
        if self.context_header:
            return f"{self.context_header}\n\n{self.text}"
        return self.text

    @property
    def location(self) -> str:
        parts = [self.doc_title, *self.heading_path]
        loc = " › ".join(dict.fromkeys(p for p in parts if p))
        if self.page is not None:
            loc += f" (p. {self.page})"
        return loc

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "text": self.text,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "heading_path": list(self.heading_path),
            "page": self.page,
            "context_header": self.context_header,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            doc_title=data["doc_title"],
            text=data["text"],
            index=int(data["index"]),
            start=int(data["start"]),
            end=int(data["end"]),
            heading_path=tuple(data.get("heading_path") or ()),
            page=data.get("page"),
            context_header=data.get("context_header") or "",
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ScoredChunk:
    """A chunk with a retrieval score. ``rank`` is 1-based within its list."""

    chunk: Chunk
    score: float
    rank: int = 0
    source: str = ""
    embedding: np.ndarray | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class Usage:
    """Token and cost accounting for one request (or an aggregate of requests)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "llm_calls": self.llm_calls,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class Citation:
    label: str
    chunk_id: str
    doc_id: str
    location: str


@dataclass
class Answer:
    """The full result of one question: answer text, evidence, quality signals and trace."""

    question: str
    text: str
    sources: list[ScoredChunk]
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    grounded_score: float | None = None
    unsupported_sentences: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    timings_ms: dict[str, float] = field(default_factory=dict)
    config_id: str = ""
    generator: str = ""

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "citations": [c.__dict__ for c in self.citations],
            "invalid_citations": self.invalid_citations,
            "grounded_score": self.grounded_score,
            "unsupported_sentences": self.unsupported_sentences,
            "flags": self.flags,
            "sources": [
                {
                    "label": f"S{i}",
                    "chunk_id": s.chunk.chunk_id,
                    "doc_id": s.chunk.doc_id,
                    "location": s.chunk.location,
                    "score": round(float(s.score), 6),
                    "flags": s.flags,
                    **({"text": s.chunk.text} if include_text else {}),
                }
                for i, s in enumerate(self.sources, start=1)
            ],
            "usage": self.usage.to_dict(),
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "config_id": self.config_id,
            "generator": self.generator,
        }

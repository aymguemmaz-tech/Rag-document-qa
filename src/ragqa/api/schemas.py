"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["How long is GPS trip data kept?"])
    k: int | None = Field(None, ge=1, le=50, description="Override the number of retrieved chunks")


class SourceOut(BaseModel):
    label: str
    chunk_id: str
    doc_id: str
    location: str
    score: float
    flags: list[str] = []
    text: str | None = None


class CitationOut(BaseModel):
    label: str
    chunk_id: str
    doc_id: str
    location: str


class AnswerOut(BaseModel):
    question: str
    answer: str
    abstained: bool
    citations: list[CitationOut]
    invalid_citations: list[str]
    grounded_score: float | None
    unsupported_sentences: list[str]
    flags: list[str]
    sources: list[SourceOut]
    usage: dict[str, Any]
    timings_ms: dict[str, float]
    config_id: str
    generator: str


class IngestOut(BaseModel):
    collection: str
    added: list[str]
    updated: list[str]
    skipped: list[str]
    chunks: int
    seconds: float


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    source: str
    content_hash: str
    num_chunks: int
    metadata: dict[str, Any] = {}


class HealthOut(BaseModel):
    status: str
    version: str
    collection: str
    documents: int
    chunks: int
    embedder: str
    generator: str
    config_id: str

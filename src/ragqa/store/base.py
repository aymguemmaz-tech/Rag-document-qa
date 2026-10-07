"""Vector store interface. Implementations: in-memory (NumPy + BM25) and PostgreSQL/pgvector."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ragqa.text import analyze
from ragqa.types import Chunk, Document, ScoredChunk


@dataclass(frozen=True)
class TermStats:
    """Corpus-level document frequencies of analyzed terms (one "document" = one chunk)."""

    n: int
    df: dict[str, int]

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> TermStats:
        df: Counter[str] = Counter()
        n = 0
        for text in texts:
            n += 1
            df.update(set(analyze(text)))
        return cls(n=n, df=dict(df))


class CollectionMismatchError(RuntimeError):
    """Raised when a collection was built with a different embedder or dimension."""


@dataclass
class CollectionInfo:
    name: str
    embedder: str
    dim: int
    chunking: dict[str, Any] = field(default_factory=dict)
    num_documents: int = 0
    num_chunks: int = 0


@dataclass
class DocumentInfo:
    doc_id: str
    title: str
    source: str
    content_hash: str
    num_chunks: int
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_collection_name(name: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,47}", name):
        raise ValueError("Collection names must match [a-z0-9][a-z0-9_]{0,47}")
    return name


class VectorStore(ABC):
    def ensure_collection(
        self, name: str, embedder: str, dim: int, chunking: dict[str, Any] | None = None
    ) -> CollectionInfo:
        """Create the collection, or verify an existing one matches the embedder."""
        validate_collection_name(name)
        existing = self.get_collection(name)
        if existing is None:
            return self._create_collection(name, embedder, dim, chunking or {})
        if existing.embedder != embedder or existing.dim != dim:
            raise CollectionMismatchError(
                f"Collection '{name}' was built with {existing.embedder} ({existing.dim}-d) but the "
                f"current embedder is {embedder} ({dim}-d). Re-ingest into a new collection."
            )
        return existing

    @abstractmethod
    def _create_collection(self, name: str, embedder: str, dim: int, chunking: dict[str, Any]) -> CollectionInfo: ...

    @abstractmethod
    def get_collection(self, name: str) -> CollectionInfo | None: ...

    @abstractmethod
    def drop_collection(self, name: str) -> None: ...

    @abstractmethod
    def upsert_document(self, collection: str, doc: Document, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Atomically replace all chunks of ``doc`` in the collection."""

    @abstractmethod
    def delete_document(self, collection: str, doc_id: str) -> bool: ...

    @abstractmethod
    def document_hash(self, collection: str, doc_id: str) -> str | None: ...

    @abstractmethod
    def list_documents(self, collection: str) -> list[DocumentInfo]: ...

    @abstractmethod
    def dense_search(self, collection: str, vector: np.ndarray, k: int) -> list[ScoredChunk]: ...

    @abstractmethod
    def lexical_search(self, collection: str, query: str, k: int) -> list[ScoredChunk]: ...

    def term_stats(self, collection: str) -> TermStats | None:
        """Corpus term statistics used to weight query terms (None if unavailable)."""
        return None

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (connections, files)."""

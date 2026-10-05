"""Embedding backends behind one interface, plus a persistent embedding cache.

* ``openai``    - text-embedding-3-small/large (production quality, paid)
* ``wordllama`` - a 256-d static embedding model bundled in its pip wheel: semantic, free, runs
                  offline in milliseconds on CPU (the default when no API key is configured)
* ``hashing``   - signed feature hashing of terms/bigrams: deterministic, zero dependencies;
                  used by unit tests and as a purely lexical "dense" baseline

All embedders return L2-normalised float32 matrices, so a dot product is cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ragqa.pricing import cost_usd
from ragqa.text import analyze

log = logging.getLogger(__name__)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix / norm if norm > 0 else matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Embedder(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.usage_tokens = 0
        self.cost_usd = 0.0

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def _embed(self, texts: list[str]) -> np.ndarray: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if idx:
            vectors = np.asarray(self._embed([texts[i] for i in idx]), dtype=np.float32)
            out[idx] = vectors
        return l2_normalize(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Signed feature hashing over stemmed terms and bigrams (sublinear tf)."""

    def __init__(self, dim: int = 1024, bigrams: bool = True) -> None:
        super().__init__()
        self._dim = dim
        self.bigrams = bigrams
        self.name = f"hashing:{dim}{'+bi' if bigrams else ''}"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            terms = analyze(text)
            feats = terms + ([f"{a}_{b}" for a, b in zip(terms, terms[1:])] if self.bigrams else [])
            for feat in feats:
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[row, h % self._dim] += 1.0 if (h >> 63) & 1 else -1.0
        return np.sign(out) * np.log1p(np.abs(out))


def _load_wordllama(dim: int) -> Any:
    import wordllama
    from wordllama import WordLlama

    try:  # use the weights/tokenizer bundled in the wheel - no network needed
        return WordLlama.load(dim=dim, cache_dir=Path(wordllama.__file__).parent, disable_download=True)
    except Exception:  # pragma: no cover - depends on the installed wheel layout
        return WordLlama.load(dim=dim)


class WordLlamaEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self._wl = _load_wordllama(dim)
        self._dim = dim
        self._word_cache: dict[str, np.ndarray] = {}
        self.name = f"wordllama:l2_supercat-{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._wl.embed(texts, norm=True), dtype=np.float32)

    def word_vectors(self, terms: Sequence[str]) -> np.ndarray:
        """Normalised vectors for single words (cached); used by the MaxSim reranker."""
        missing = [t for t in dict.fromkeys(terms) if t not in self._word_cache]
        if missing:
            for term, vec in zip(missing, self.embed(missing)):
                self._word_cache[term] = vec
        if not terms:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._word_cache[t] for t in terms])


_OPENAI_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 128,
        max_retries: int = 4,
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        super().__init__()
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout)
        self._client = client
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._dim: int | None = dimensions or _OPENAI_DIMS.get(model)
        self.name = f"openai:{model}" + (f"-{dimensions}" if dimensions else "")

    @property
    def dim(self) -> int:
        if self._dim is None:  # unknown model: probe once
            self._dim = int(self._call(["dimension probe"]).shape[1])
        return self._dim

    def _call(self, batch: list[str]) -> np.ndarray:
        kwargs: dict[str, Any] = {"model": self.model, "input": batch}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        data = sorted(response.data, key=lambda d: d.index)
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0)
        self.usage_tokens += tokens
        self.cost_usd += cost_usd(self.model, tokens)
        return np.asarray([d.embedding for d in data], dtype=np.float32)

    def _embed(self, texts: list[str]) -> np.ndarray:
        parts = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ")[:24000] for t in texts[i : i + self.batch_size]]
            parts.append(self._call(batch))
        return np.vstack(parts)


class CachedEmbedder(Embedder):
    """Persists vectors in SQLite keyed by (model, sha256(text)); re-ingestion costs nothing."""

    def __init__(self, inner: Embedder, path: str | Path) -> None:
        super().__init__()
        self.inner = inner
        self.name = inner.name
        self.hits = 0
        self.misses = 0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        self._db.commit()

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def usage_tokens(self) -> int:
        return self.inner.usage_tokens

    @usage_tokens.setter
    def usage_tokens(self, value: int) -> None:  # set by the base __init__
        pass

    @property
    def cost_usd(self) -> float:
        return self.inner.cost_usd

    @cost_usd.setter
    def cost_usd(self, value: float) -> None:
        pass

    def __getattr__(self, item: str) -> Any:  # expose backend extras such as word_vectors()
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.inner.name}\x00{text}".encode()).hexdigest()

    def _embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        with self._lock:
            unique = list(dict.fromkeys(keys))
            for i in range(0, len(unique), 500):
                batch = unique[i : i + 500]
                marks = ",".join("?" * len(batch))
                for key, blob in self._db.execute(f"SELECT key, vec FROM emb WHERE key IN ({marks})", batch):
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        missing = [i for i, k in enumerate(keys) if k not in found]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            todo = list(dict.fromkeys(keys[i] for i in missing))
            first_text = {keys[i]: texts[i] for i in missing}
            vectors = self.inner.embed([first_text[k] for k in todo])
            with self._lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
                    [(k, v.astype(np.float32).tobytes()) for k, v in zip(todo, vectors)],
                )
                self._db.commit()
            found.update(zip(todo, vectors))
        return np.stack([found[k] for k in keys])

    def close(self) -> None:
        with self._lock:
            self._db.close()


def build_embedder(
    backend: str,
    *,
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
    cache_path: str | Path | None = None,
) -> Embedder:
    inner: Embedder
    if backend == "openai":
        inner = OpenAIEmbedder(model=model, api_key=api_key, base_url=base_url, dimensions=dimensions)
    elif backend == "wordllama":
        inner = WordLlamaEmbedder()
    elif backend == "hashing":
        inner = HashingEmbedder()
    else:
        raise ValueError(f"Unknown embedder backend: {backend}")
    if cache_path is not None and backend != "hashing":
        return CachedEmbedder(inner, cache_path)
    return inner

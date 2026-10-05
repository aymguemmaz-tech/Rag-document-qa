from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ragqa.embeddings import CachedEmbedder, HashingEmbedder, OpenAIEmbedder, WordLlamaEmbedder


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dim=256)
    a = emb.embed(["battery storage temperature", "battery storage temperature", ""])
    assert np.allclose(a[0], a[1])
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert np.allclose(a[2], 0.0)  # empty text -> zero vector, never NaN
    assert emb.embed([]).shape == (0, 256)


def test_wordllama_captures_paraphrases() -> None:
    emb = WordLlamaEmbedder()
    v = emb.embed(["price of the monthly pass", "how much does the subscription cost", "battery fire extinguisher"])
    assert v.shape == (3, 256)
    assert v[0] @ v[1] > v[0] @ v[2]
    words = emb.word_vectors(["price", "cost", "battery"])
    assert words[0] @ words[1] > words[0] @ words[2]


def test_cached_embedder_persists_vectors(tmp_path: Path) -> None:
    calls: list[int] = []

    class Counting(HashingEmbedder):
        def _embed(self, texts: list[str]) -> np.ndarray:
            calls.append(len(texts))
            return super()._embed(texts)

    path = tmp_path / "cache.sqlite"
    first = CachedEmbedder(Counting(dim=64), path)
    v1 = first.embed(["a b c", "d e f", "a b c"])
    assert calls == [2] and first.misses == 3
    first.close()
    second = CachedEmbedder(Counting(dim=64), path)
    v2 = second.embed(["a b c", "d e f"])
    assert calls == [2]  # served from disk
    assert second.hits == 2
    assert np.allclose(v1[:2], v2)
    second.close()


def test_openai_embedder_batches_sorts_and_counts_usage() -> None:
    requests: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        requests.append(kwargs)
        n = len(kwargs["input"])
        data = [SimpleNamespace(index=i, embedding=[float(i + 1), 1.0, 0.0]) for i in range(n)][::-1]
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=10 * n))

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    emb = OpenAIEmbedder(model="text-embedding-3-small", dimensions=3, batch_size=2, client=client)
    vectors = emb.embed(["one", "two", "three"])
    assert [len(r["input"]) for r in requests] == [2, 1]
    assert all(r["dimensions"] == 3 for r in requests)
    assert vectors[1] @ np.array([2.0, 1.0, 0.0]) / np.sqrt(5) == pytest.approx(1.0, abs=1e-6)  # order restored
    assert emb.usage_tokens == 30 and emb.cost_usd == pytest.approx(30 * 0.02 / 1e6)

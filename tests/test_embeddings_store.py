from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ragqa.config import ChunkingConfig
from ragqa.embeddings import CachedEmbedder, HashingEmbedder, OpenAIEmbedder, WordLlamaEmbedder
from ragqa.ingest.chunking import build_chunker
from ragqa.store import CollectionMismatchError, MemoryStore
from ragqa.text import analyze
from ragqa.types import Document


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


def _chunks(doc: Document, emb: HashingEmbedder) -> tuple[list[Any], np.ndarray]:
    chunks = build_chunker(ChunkingConfig(size=60)).chunk(doc)
    return chunks, emb.embed([c.embed_text for c in chunks])


def test_memory_store_crud_search_and_persistence(
    tmp_path: Path, sample_doc: Document, embedder: HashingEmbedder
) -> None:
    store = MemoryStore(persist_dir=tmp_path)
    store.ensure_collection("docs", embedder.name, embedder.dim)
    chunks, vectors = _chunks(sample_doc, embedder)
    store.upsert_document("docs", sample_doc, chunks, vectors)
    assert store.document_hash("docs", "battery-sop") == sample_doc.content_hash
    assert store.get_collection("docs").num_chunks == len(chunks)

    dense = store.dense_search("docs", embedder.embed_query("AVD extinguisher fire"), 3)
    assert "AVD" in dense[0].chunk.text and [d.rank for d in dense] == [1, 2, 3]
    lexical = store.lexical_search("docs", "quarantine bin swollen", 2)
    assert "quarantine" in lexical[0].chunk.text

    store.upsert_document("docs", sample_doc, chunks[:2], vectors[:2])  # replace, not append
    assert store.get_collection("docs").num_chunks == 2

    reloaded = MemoryStore(persist_dir=tmp_path)
    assert reloaded.get_collection("docs").num_chunks == 2
    assert reloaded.delete_document("docs", "battery-sop") is True
    assert reloaded.delete_document("docs", "battery-sop") is False
    assert reloaded.list_documents("docs") == []


def test_collection_refuses_a_different_embedder(embedder: HashingEmbedder) -> None:
    store = MemoryStore()
    store.ensure_collection("docs", embedder.name, embedder.dim)
    with pytest.raises(CollectionMismatchError):
        store.ensure_collection("docs", "openai:text-embedding-3-small", 1536)
    with pytest.raises(ValueError):
        store.ensure_collection("Bad Name!", embedder.name, embedder.dim)


def test_term_stats_reflect_corpus_frequencies(sample_doc: Document, embedder: HashingEmbedder) -> None:
    store = MemoryStore()
    assert store.term_stats("missing") is None
    store.ensure_collection("docs", embedder.name, embedder.dim)
    chunks, vectors = _chunks(sample_doc, embedder)
    store.upsert_document("docs", sample_doc, chunks, vectors)
    stats = store.term_stats("docs")
    assert stats is not None and stats.n == len(chunks)
    common, rare, unseen = (analyze(w)[0] for w in ("battery", "quarantine", "ceo"))  # same analyzer as BM25
    assert stats.idf(common) < stats.idf(rare) < stats.idf(unseen)

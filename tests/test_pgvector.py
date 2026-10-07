"""Integration tests against PostgreSQL + pgvector (set RAGQA_TEST_DATABASE_URL)."""

from collections.abc import Iterator

import numpy as np
import pytest

from ragqa.config import ChunkingConfig
from ragqa.embeddings import HashingEmbedder
from ragqa.ingest.chunking import build_chunker
from ragqa.store import CollectionMismatchError, MemoryStore
from ragqa.types import Document

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_store(pg_url: str, collection_name: str) -> Iterator[object]:
    from ragqa.store.pgvector import PgVectorStore

    store = PgVectorStore(pg_url)
    yield store
    store.drop_collection(collection_name)
    store.close()


def test_pgvector_crud_and_dense_search(pg_store: object, collection_name: str, sample_doc: Document) -> None:
    from ragqa.store.pgvector import PgVectorStore

    assert isinstance(pg_store, PgVectorStore)
    emb = HashingEmbedder(dim=256)
    pg_store.ensure_collection(collection_name, emb.name, emb.dim, {"strategy": "structure"})
    with pytest.raises(CollectionMismatchError):
        pg_store.ensure_collection(collection_name, "other", 3)
    chunks = build_chunker(ChunkingConfig(size=60)).chunk(sample_doc)
    vectors = emb.embed([c.embed_text for c in chunks])
    pg_store.upsert_document(collection_name, sample_doc, chunks, vectors)
    info = pg_store.get_collection(collection_name)
    assert info is not None and info.num_documents == 1 and info.num_chunks == len(chunks)
    assert pg_store.document_hash(collection_name, sample_doc.doc_id) == sample_doc.content_hash

    hits = pg_store.dense_search(collection_name, emb.embed_query("AVD extinguisher fire"), 3)
    assert "AVD" in hits[0].chunk.text
    assert hits[0].chunk.heading_path[-1] == "Fire" and hits[0].embedding is not None
    assert hits[0].chunk.metadata["status"] == "current"

    pg_store.upsert_document(collection_name, sample_doc, chunks[:1], vectors[:1])  # atomic replace
    assert pg_store.get_collection(collection_name).num_chunks == 1
    assert pg_store.delete_document(collection_name, sample_doc.doc_id)
    assert pg_store.list_documents(collection_name) == []


def test_sql_bm25_agrees_with_python_bm25(pg_store: object, collection_name: str, corpus_docs: list[Document]) -> None:
    emb = HashingEmbedder(dim=128)
    memory = MemoryStore()
    chunker = build_chunker(ChunkingConfig())
    for store in (pg_store, memory):
        store.ensure_collection(collection_name, emb.name, emb.dim)  # type: ignore[attr-defined]
        for doc in corpus_docs:
            chunks = chunker.chunk(doc)
            store.upsert_document(collection_name, doc, chunks, emb.embed([c.embed_text for c in chunks]))  # type: ignore[attr-defined]
    queries = [
        "certificate expiry monitoring",
        "pedestrian zones speed limit",
        "Kestrel Ridge wind farm",
        "lost items kept",
        "MQTT telemetry message",
    ]
    agree = 0
    for query in queries:
        sql_top = pg_store.lexical_search(collection_name, query, 3)  # type: ignore[attr-defined]
        py_top = memory.lexical_search(collection_name, query, 3)
        assert sql_top and np.all(np.diff([s.score for s in sql_top]) <= 1e-9)
        agree += sql_top[0].chunk_id == py_top[0].chunk_id
    assert agree >= 4  # different stemmers (Snowball vs ours) may disagree on a rare tie


def test_pgvector_term_stats_match_memory(pg_store: object, collection_name: str, sample_doc: Document) -> None:
    emb = HashingEmbedder(dim=64)
    memory = MemoryStore()
    chunks = build_chunker(ChunkingConfig(size=60)).chunk(sample_doc)
    vectors = emb.embed([c.embed_text for c in chunks])
    for store in (pg_store, memory):
        store.ensure_collection(collection_name, emb.name, emb.dim)  # type: ignore[attr-defined]
        store.upsert_document(collection_name, sample_doc, chunks, vectors)  # type: ignore[attr-defined]
    assert pg_store.term_stats(collection_name) == memory.term_stats(collection_name)  # type: ignore[attr-defined]

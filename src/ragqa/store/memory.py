"""In-memory vector store (NumPy brute force + BM25) with optional on-disk persistence.

Used for tests, the offline evaluation grid (dozens of throwaway indexes) and "lite mode" when
no PostgreSQL is available. Exact search, so it doubles as a recall reference for HNSW.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ragqa.retrieval.bm25 import BM25Index
from ragqa.store.base import CollectionInfo, DocumentInfo, TermStats, VectorStore
from ragqa.types import Chunk, Document, ScoredChunk


@dataclass
class _Collection:
    info: CollectionInfo
    documents: dict[str, DocumentInfo] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    vectors: np.ndarray | None = None
    bm25: BM25Index | None = None

    def matrix(self) -> np.ndarray:
        if self.vectors is None:
            return np.zeros((0, self.info.dim), dtype=np.float32)
        return self.vectors


class MemoryStore(VectorStore):
    def __init__(self, persist_dir: str | Path | None = None) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._collections: dict[str, _Collection] = {}
        self._lock = threading.RLock()
        if self.persist_dir and self.persist_dir.exists():
            for meta in sorted(self.persist_dir.glob("*/meta.json")):
                self._load(meta.parent)

    # -- persistence ----------------------------------------------------------------------
    def _load(self, folder: Path) -> None:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        info = CollectionInfo(**meta["info"])
        coll = _Collection(info=info)
        coll.documents = {d["doc_id"]: DocumentInfo(**d) for d in meta["documents"]}
        chunks_file = folder / "chunks.jsonl"
        if chunks_file.exists():
            with chunks_file.open(encoding="utf-8") as fh:
                coll.chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
        vec_file = folder / "vectors.npy"
        coll.vectors = np.load(vec_file) if vec_file.exists() and coll.chunks else None
        self._collections[info.name] = coll

    def _save(self, name: str) -> None:
        if not self.persist_dir:
            return
        coll = self._collections[name]
        folder = self.persist_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        meta = {
            "info": coll.info.__dict__,
            "documents": [d.__dict__ for d in coll.documents.values()],
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in coll.chunks:
                fh.write(json.dumps(chunk.to_dict()) + "\n")
        np.save(folder / "vectors.npy", coll.matrix())

    # -- collections ----------------------------------------------------------------------
    def _create_collection(self, name: str, embedder: str, dim: int, chunking: dict[str, Any]) -> CollectionInfo:
        with self._lock:
            info = CollectionInfo(name=name, embedder=embedder, dim=dim, chunking=chunking)
            self._collections[name] = _Collection(info=info)
            self._save(name)
            return info

    def get_collection(self, name: str) -> CollectionInfo | None:
        coll = self._collections.get(name)
        if coll is None:
            return None
        coll.info.num_documents = len(coll.documents)
        coll.info.num_chunks = len(coll.chunks)
        return coll.info

    def drop_collection(self, name: str) -> None:
        with self._lock:
            self._collections.pop(name, None)
            if self.persist_dir and (self.persist_dir / name).exists():
                shutil.rmtree(self.persist_dir / name)

    def _get(self, name: str) -> _Collection:
        try:
            return self._collections[name]
        except KeyError:
            raise KeyError(f"Unknown collection '{name}'. Ingest documents first.") from None

    # -- documents ------------------------------------------------------------------------
    def upsert_document(self, collection: str, doc: Document, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        with self._lock:
            coll = self._get(collection)
            keep = [i for i, c in enumerate(coll.chunks) if c.doc_id != doc.doc_id]
            old = coll.matrix()
            coll.chunks = [coll.chunks[i] for i in keep] + list(chunks)
            parts = [old[keep]] if keep else []
            if len(chunks):
                parts.append(np.asarray(embeddings, dtype=np.float32))
            coll.vectors = np.vstack(parts) if parts else None
            coll.documents[doc.doc_id] = DocumentInfo(
                doc_id=doc.doc_id,
                title=doc.title,
                source=doc.source,
                content_hash=doc.content_hash,
                num_chunks=len(chunks),
                metadata=doc.metadata,
            )
            coll.bm25 = None
            self._save(collection)

    def delete_document(self, collection: str, doc_id: str) -> bool:
        with self._lock:
            coll = self._get(collection)
            if doc_id not in coll.documents:
                return False
            keep = [i for i, c in enumerate(coll.chunks) if c.doc_id != doc_id]
            coll.vectors = coll.matrix()[keep] if keep else None
            coll.chunks = [coll.chunks[i] for i in keep]
            del coll.documents[doc_id]
            coll.bm25 = None
            self._save(collection)
            return True

    def document_hash(self, collection: str, doc_id: str) -> str | None:
        coll = self._collections.get(collection)
        if coll is None or doc_id not in coll.documents:
            return None
        return coll.documents[doc_id].content_hash

    def list_documents(self, collection: str) -> list[DocumentInfo]:
        coll = self._collections.get(collection)
        return sorted(coll.documents.values(), key=lambda d: d.doc_id) if coll else []

    # -- search ---------------------------------------------------------------------------
    def dense_search(self, collection: str, vector: np.ndarray, k: int) -> list[ScoredChunk]:
        coll = self._get(collection)
        matrix = coll.matrix()
        if not len(matrix):
            return []
        scores = matrix @ np.asarray(vector, dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]
        return [
            ScoredChunk(chunk=coll.chunks[i], score=float(scores[i]), rank=r, source="dense", embedding=matrix[i])
            for r, i in enumerate(top, start=1)
        ]

    def lexical_search(self, collection: str, query: str, k: int) -> list[ScoredChunk]:
        coll = self._get(collection)
        with self._lock:
            if coll.bm25 is None:
                coll.bm25 = BM25Index([c.embed_text for c in coll.chunks])
            index = coll.bm25
        matrix = coll.matrix()
        return [
            ScoredChunk(
                chunk=coll.chunks[i], score=s, rank=r, source="lexical", embedding=matrix[i] if len(matrix) else None
            )
            for r, (i, s) in enumerate(index.search(query, k), start=1)
        ]

    def term_stats(self, collection: str) -> TermStats | None:
        coll = self._collections.get(collection)
        if coll is None or not coll.chunks:
            return None
        with self._lock:
            if coll.bm25 is None:
                coll.bm25 = BM25Index([c.embed_text for c in coll.chunks])
            index = coll.bm25
        return TermStats(n=index.n, df={term: len(postings) for term, postings in index.postings.items()})

    def all_chunks(self, collection: str) -> list[Chunk]:
        return list(self._get(collection).chunks)

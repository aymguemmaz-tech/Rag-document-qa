from ragqa.store.base import (
    CollectionInfo,
    CollectionMismatchError,
    DocumentInfo,
    VectorStore,
)
from ragqa.store.memory import MemoryStore

__all__ = [
    "CollectionInfo",
    "CollectionMismatchError",
    "DocumentInfo",
    "MemoryStore",
    "VectorStore",
    "build_store",
]


def build_store(kind: str, *, database_url: str = "", persist_dir: str | None = None) -> VectorStore:
    if kind == "memory":
        return MemoryStore(persist_dir=persist_dir)
    if kind == "pgvector":
        from ragqa.store.pgvector import PgVectorStore

        return PgVectorStore(database_url)
    raise ValueError(f"Unknown store: {kind}")

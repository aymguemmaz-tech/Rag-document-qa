"""PostgreSQL + pgvector store.

* Dense search: HNSW index with cosine distance (``vector_cosine_ops``); ``hnsw.ef_search`` is
  raised per query to trade a little latency for recall.
* Lexical search: Okapi BM25 implemented in SQL. Postgres's built-in ``ts_rank`` has no inverse
  document frequency, so rare, discriminative terms (part numbers, error codes, names) are
  under-weighted. Here the GIN index finds candidates and BM25 is computed from ``tsvector``
  term frequencies plus per-term document frequencies.
* One chunk table per collection: every table has a fixed vector dimension and its own HNSW
  index, so filtering by collection never truncates the approximate-nearest-neighbour results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from ragqa.store.base import CollectionInfo, DocumentInfo, TermStats, VectorStore, validate_collection_name
from ragqa.types import Chunk, Document, ScoredChunk

log = logging.getLogger(__name__)

_BASE_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_collections (
    name        TEXT PRIMARY KEY,
    embedder    TEXT NOT NULL,
    dim         INT NOT NULL,
    chunking    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS rag_documents (
    collection   TEXT NOT NULL REFERENCES rag_collections(name) ON DELETE CASCADE,
    doc_id       TEXT NOT NULL,
    title        TEXT NOT NULL,
    source       TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    num_chunks   INT NOT NULL DEFAULT 0,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, doc_id)
);
"""

_CHUNK_TABLE = """
CREATE TABLE IF NOT EXISTS {t} (
    chunk_id       TEXT PRIMARY KEY,
    collection     TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    doc_title      TEXT NOT NULL,
    idx            INT NOT NULL,
    text           TEXT NOT NULL,
    context_header TEXT NOT NULL DEFAULT '',
    heading_path   TEXT[] NOT NULL DEFAULT '{{}}',
    page           INT,
    start_char     INT NOT NULL,
    end_char       INT NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding      vector({dim}) NOT NULL,
    tsv            tsvector NOT NULL,
    dl             INT NOT NULL,
    FOREIGN KEY (collection, doc_id) REFERENCES rag_documents(collection, doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS {hnsw} ON {t} USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS {gin} ON {t} USING gin (tsv);
CREATE INDEX IF NOT EXISTS {doc} ON {t} (doc_id);
"""

_COLS = (
    "c.chunk_id, c.doc_id, c.doc_title, c.idx, c.text, c.context_header, c.heading_path, "
    "c.page, c.start_char, c.end_char, c.metadata, c.embedding"
)

_INSERT = """
INSERT INTO {t} (chunk_id, collection, doc_id, doc_title, idx, text, context_header,
                 heading_path, page, start_char, end_char, metadata, embedding, tsv, dl)
SELECT %(chunk_id)s::text, %(collection)s::text, %(doc_id)s::text, %(doc_title)s::text,
       %(idx)s::int, %(text)s::text, %(context_header)s::text, %(heading_path)s::text[],
       %(page)s::int, %(start_char)s::int, %(end_char)s::int, %(metadata)s, %(embedding)s, v.tsv,
       (SELECT coalesce(sum(coalesce(array_length(u.positions, 1), 1)), 0)
          FROM unnest(v.tsv) AS u)
FROM (SELECT to_tsvector('english', %(embed_text)s) AS tsv) AS v
"""

_DENSE = """
SELECT {cols}, 1 - (c.embedding <=> %(v)s) AS score
FROM {t} AS c
ORDER BY c.embedding <=> %(v)s
LIMIT %(k)s
"""

_BM25 = """
WITH q AS (
    SELECT DISTINCT u.lexeme AS term FROM unnest(to_tsvector('english', %(query)s)) AS u
),
qtsq AS (
    SELECT string_agg(quote_literal(term), ' | ')::tsquery AS tsq FROM q
),
stats AS (
    SELECT count(*)::float8 AS n, greatest(avg(dl), 1)::float8 AS avgdl FROM {t}
),
df AS (
    SELECT q.term, count(c.chunk_id)::float8 AS df
    FROM q JOIN {t} AS c ON c.tsv @@ quote_literal(q.term)::tsquery
    GROUP BY q.term
),
tf AS (
    SELECT c.chunk_id, c.dl, u.lexeme AS term,
           coalesce(array_length(u.positions, 1), 1)::float8 AS tf
    FROM {t} AS c
    CROSS JOIN qtsq
    CROSS JOIN LATERAL unnest(c.tsv) AS u
    WHERE c.tsv @@ qtsq.tsq AND u.lexeme IN (SELECT term FROM q)
),
scored AS (
    SELECT tf.chunk_id,
           sum(ln(1 + (stats.n - df.df + 0.5) / (df.df + 0.5))
               * tf.tf * (%(k1)s + 1)
               / (tf.tf + %(k1)s * (1 - %(b)s + %(b)s * tf.dl / stats.avgdl))) AS score
    FROM tf JOIN df ON df.term = tf.term CROSS JOIN stats
    GROUP BY tf.chunk_id
)
SELECT {cols}, s.score
FROM scored AS s JOIN {t} AS c ON c.chunk_id = s.chunk_id
ORDER BY s.score DESC, c.chunk_id
LIMIT %(k)s
"""


class PgVectorStore(VectorStore):
    def __init__(
        self,
        dsn: str,
        *,
        max_connections: int = 8,
        ef_search: int = 100,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        self._term_stats: dict[str, TermStats] = {}
        self.ef_search = ef_search
        self.k1 = k1
        self.b = b
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(_BASE_SCHEMA)
        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=max_connections,
            kwargs={"autocommit": True},
            configure=register_vector,
            open=True,
        )

    @staticmethod
    def _table(collection: str) -> str:
        return f"rag_chunks_{validate_collection_name(collection)}"

    def _sql(self, template: str, collection: str, **extra: Any) -> Any:
        from psycopg import sql

        table = self._table(collection)
        idents = {
            "t": sql.Identifier(table),
            "hnsw": sql.Identifier(f"{table}_hnsw"),
            "gin": sql.Identifier(f"{table}_tsv"),
            "doc": sql.Identifier(f"{table}_doc"),
            "cols": sql.SQL(_COLS),
        }
        idents.update({k: sql.SQL(str(v)) for k, v in extra.items()})
        return sql.SQL(template).format(**{k: v for k, v in idents.items() if "{" + k + "}" in template})

    # -- collections ----------------------------------------------------------------------
    def _create_collection(self, name: str, embedder: str, dim: int, chunking: dict[str, Any]) -> CollectionInfo:
        from psycopg.types.json import Jsonb

        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO rag_collections (name, embedder, dim, chunking) VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (name) DO NOTHING",
                (name, embedder, dim, Jsonb(chunking)),
            )
            conn.execute(self._sql(_CHUNK_TABLE, name, dim=int(dim)))
        log.info("Created collection %s (%s, %d-d)", name, embedder, dim)
        return CollectionInfo(name=name, embedder=embedder, dim=dim, chunking=chunking)

    def get_collection(self, name: str) -> CollectionInfo | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT c.name, c.embedder, c.dim, c.chunking,"
                " (SELECT count(*) FROM rag_documents d WHERE d.collection = c.name),"
                " (SELECT coalesce(sum(num_chunks), 0) FROM rag_documents d WHERE d.collection = c.name)"
                " FROM rag_collections c WHERE c.name = %s",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return CollectionInfo(
            name=row[0],
            embedder=row[1],
            dim=row[2],
            chunking=row[3] or {},
            num_documents=int(row[4]),
            num_chunks=int(row[5]),
        )

    def drop_collection(self, name: str) -> None:
        self._term_stats.pop(name, None)
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(self._sql("DROP TABLE IF EXISTS {t}", name))
            conn.execute("DELETE FROM rag_collections WHERE name = %s", (name,))

    # -- documents ------------------------------------------------------------------------
    def upsert_document(self, collection: str, doc: Document, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        from psycopg.types.json import Jsonb

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._term_stats.pop(collection, None)
        rows = [
            {
                "chunk_id": c.chunk_id,
                "collection": collection,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "idx": c.index,
                "text": c.text,
                "context_header": c.context_header,
                "heading_path": list(c.heading_path),
                "page": c.page,
                "start_char": c.start,
                "end_char": c.end,
                "metadata": Jsonb(c.metadata),
                "embedding": np.asarray(e, dtype=np.float32),
                "embed_text": c.embed_text,
            }
            for c, e in zip(chunks, embeddings)
        ]
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                "DELETE FROM rag_documents WHERE collection = %s AND doc_id = %s",
                (collection, doc.doc_id),
            )
            conn.execute(
                "INSERT INTO rag_documents (collection, doc_id, title, source, content_hash,"
                " num_chunks, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    collection,
                    doc.doc_id,
                    doc.title,
                    doc.source,
                    doc.content_hash,
                    len(chunks),
                    Jsonb(json.loads(json.dumps(doc.metadata, default=str))),
                ),
            )
            if rows:
                with conn.cursor() as cur:
                    cur.executemany(self._sql(_INSERT, collection), rows)

    def delete_document(self, collection: str, doc_id: str) -> bool:
        self._term_stats.pop(collection, None)
        with self._pool.connection() as conn:
            cur = conn.execute(
                "DELETE FROM rag_documents WHERE collection = %s AND doc_id = %s",
                (collection, doc_id),
            )
            return cur.rowcount > 0

    def document_hash(self, collection: str, doc_id: str) -> str | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT content_hash FROM rag_documents WHERE collection = %s AND doc_id = %s",
                (collection, doc_id),
            ).fetchone()
        return row[0] if row else None

    def list_documents(self, collection: str) -> list[DocumentInfo]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT doc_id, title, source, content_hash, num_chunks, metadata"
                " FROM rag_documents WHERE collection = %s ORDER BY doc_id",
                (collection,),
            ).fetchall()
        return [DocumentInfo(r[0], r[1], r[2], r[3], int(r[4]), r[5] or {}) for r in rows]

    # -- search ---------------------------------------------------------------------------
    @staticmethod
    def _to_scored(row: tuple[Any, ...], rank: int, source: str) -> ScoredChunk:
        chunk = Chunk(
            chunk_id=row[0],
            doc_id=row[1],
            doc_title=row[2],
            index=row[3],
            text=row[4],
            context_header=row[5] or "",
            heading_path=tuple(row[6] or ()),
            page=row[7],
            start=row[8],
            end=row[9],
            metadata=row[10] or {},
        )
        raw = row[11]
        if raw is not None and hasattr(raw, "to_numpy"):  # pgvector-python returns Vector objects
            raw = raw.to_numpy()
        embedding = np.asarray(raw, dtype=np.float32) if raw is not None else None
        return ScoredChunk(chunk=chunk, score=float(row[12]), rank=rank, source=source, embedding=embedding)

    def dense_search(self, collection: str, vector: np.ndarray, k: int) -> list[ScoredChunk]:
        with self._pool.connection() as conn, conn.transaction():
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(self.ef_search, k)),))
            rows = conn.execute(
                self._sql(_DENSE, collection),
                {"v": np.asarray(vector, dtype=np.float32), "k": k},
            ).fetchall()
        return [self._to_scored(r, i, "dense") for i, r in enumerate(rows, start=1)]

    def lexical_search(self, collection: str, query: str, k: int) -> list[ScoredChunk]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                self._sql(_BM25, collection),
                {"query": query, "k": k, "k1": self.k1, "b": self.b},
            ).fetchall()
        return [self._to_scored(r, i, "lexical") for i, r in enumerate(rows, start=1)]

    def term_stats(self, collection: str) -> TermStats | None:
        if collection not in self._term_stats:
            if self.get_collection(collection) is None:
                return None
            with self._pool.connection() as conn:
                rows = conn.execute(self._sql("SELECT context_header, text FROM {t}", collection)).fetchall()
            self._term_stats[collection] = TermStats.from_texts(f"{h}\n\n{t}" if h else t for h, t in rows)
        return self._term_stats[collection]

    def close(self) -> None:
        self._pool.close()

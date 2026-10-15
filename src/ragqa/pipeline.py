"""The RAG pipeline: ingestion and question answering, assembled from configuration."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragqa.config import PipelineConfig, Settings
from ragqa.embeddings import Embedder, WordLlamaEmbedder, build_embedder
from ragqa.generation.citations import ABSTAIN_TEXT
from ragqa.generation.generators import ExtractiveGenerator, GenerationResult, Generator, LLMGenerator
from ragqa.generation.grounding import GroundingVerifier, InjectionGuard
from ragqa.generation.llm import LLM, OpenAIChatLLM
from ragqa.generation.prompts import load_templates
from ragqa.ingest.chunking import build_chunker
from ragqa.ingest.loaders import load_path
from ragqa.observability import StageTimer
from ragqa.retrieval.query import HyDEExpander, MultiQueryExpander, QueryExpander
from ragqa.retrieval.rerank import (
    CrossEncoderReranker,
    LLMReranker,
    MaxSimReranker,
    MMRReranker,
    Reranker,
)
from ragqa.retrieval.retriever import RetrievalResult, Retriever
from ragqa.store import VectorStore, build_store
from ragqa.store.base import TermStats
from ragqa.types import Answer, Citation, Document

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    collection: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    chunks: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "seconds": round(self.seconds, 3)}


def build_llm(settings: Settings) -> LLM | None:
    if not (settings.has_openai or settings.llm_base_url):
        return None
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return OpenAIChatLLM(
        settings.llm_model,
        api_key=key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        reasoning_effort=settings.llm_reasoning_effort,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def build_reranker(config: PipelineConfig, embedder: Embedder, llm: LLM | None) -> Reranker | None:
    kind = config.retrieval.rerank
    if kind == "none":
        return None
    if kind == "mmr":
        return MMRReranker(embedder, config.retrieval.mmr_lambda)
    if kind == "maxsim":
        word_embedder = embedder if hasattr(embedder, "word_vectors") else WordLlamaEmbedder()
        return MaxSimReranker(word_embedder, fusion_k=config.retrieval.rrf_k)
    if kind == "llm":
        if llm is None:
            raise ValueError("retrieval.rerank=llm needs an LLM (set OPENAI_API_KEY or RAGQA_LLM_BASE_URL)")
        return LLMReranker(llm)
    return CrossEncoderReranker(config.retrieval.cross_encoder_model)


def build_expander(config: PipelineConfig, llm: LLM | None) -> QueryExpander | None:
    kind = config.retrieval.query_expansion
    if kind == "none":
        return None
    if llm is None:
        raise ValueError(f"query_expansion={kind} needs an LLM (set OPENAI_API_KEY or RAGQA_LLM_BASE_URL)")
    if kind == "multi_query":
        return MultiQueryExpander(llm, config.retrieval.num_query_variants)
    return HyDEExpander(llm)


def build_generator(
    kind: str, config: PipelineConfig, embedder: Embedder, llm: LLM | None, seed: int | None = None
) -> Generator:
    if kind == "extractive":
        return ExtractiveGenerator(embedder, config.generation, seed=seed)
    if llm is None:
        raise ValueError("generator=llm needs an LLM (set OPENAI_API_KEY or RAGQA_LLM_BASE_URL)")
    templates = load_templates()
    if config.generation.prompt not in templates:
        raise ValueError(f"Unknown prompt template '{config.generation.prompt}'. Options: {sorted(templates)}")
    return LLMGenerator(llm, templates[config.generation.prompt], config.generation)


class RAGPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        store: VectorStore,
        embedder: Embedder,
        generator: Generator,
        collection: str = "default",
        llm: LLM | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.collection = collection
        self.llm = llm
        self.chunker = build_chunker(config.chunking, embedder)
        self.guard = InjectionGuard(config.guard.injection)
        self.retriever = Retriever(
            store,
            embedder,
            collection,
            config.retrieval,
            reranker=build_reranker(config, embedder, llm),
            expander=build_expander(config, llm),
            guard=self.guard if config.guard.injection != "off" else None,
        )
        self.verifier = (
            GroundingVerifier(embedder, config.guard.grounding_threshold) if config.guard.grounding != "off" else None
        )
        self._stats: TermStats | None = None
        self._stats_dirty = True
        if isinstance(generator, ExtractiveGenerator) and generator.term_stats is None:
            generator.term_stats = self.term_stats

    @classmethod
    def from_settings(cls, settings: Settings | None = None, config: PipelineConfig | None = None) -> RAGPipeline:
        settings = settings or Settings()
        config = config or settings.load_pipeline_config()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        embedder = build_embedder(
            settings.resolved_embedder,
            model=settings.embedding_model,
            api_key=key,
            dimensions=settings.embedding_dimensions,
            cache_path=settings.data_dir / "embeddings.sqlite" if settings.embedding_cache else None,
        )
        store = build_store(
            settings.store,
            database_url=settings.database_url,
            persist_dir=str(settings.data_dir / "index"),
        )
        llm = build_llm(settings)
        generator = build_generator(settings.resolved_generator, config, embedder, llm)
        return cls(config, store=store, embedder=embedder, generator=generator, collection=settings.collection, llm=llm)

    def term_stats(self) -> TermStats | None:
        """Corpus term statistics of the collection (cached; refreshed after writes)."""
        if self._stats_dirty:
            self._stats = self.store.term_stats(self.collection)
            self._stats_dirty = False
        return self._stats

    # -- ingestion ------------------------------------------------------------------------
    def ingest(self, documents: Iterable[Document], force: bool = False) -> IngestReport:
        start = time.perf_counter()
        report = IngestReport(collection=self.collection)
        chunking = self.config.chunking.model_dump()
        info = self.store.ensure_collection(self.collection, self.embedder.name, self.embedder.dim, chunking)
        if info.chunking and info.chunking != chunking:
            log.warning("Collection %s was built with different chunking settings", self.collection)
        for doc in documents:
            previous = self.store.document_hash(self.collection, doc.doc_id)
            if previous == doc.content_hash and not force:
                report.skipped.append(doc.doc_id)
                continue
            chunks = self.chunker.chunk(doc)
            vectors = self.embedder.embed([c.embed_text for c in chunks])
            self.store.upsert_document(self.collection, doc, chunks, vectors)
            (report.updated if previous else report.added).append(doc.doc_id)
            self._stats_dirty = True
            report.chunks += len(chunks)
            log.debug("Indexed %s (%d chunks)", doc.doc_id, len(chunks))
        report.seconds = time.perf_counter() - start
        return report

    def ingest_path(self, path: str | Path, force: bool = False) -> IngestReport:
        return self.ingest(load_path(path), force=force)

    # -- question answering ---------------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        return self.retriever.retrieve(question, k)

    def _finalize(
        self, question: str, retrieval: RetrievalResult, result: GenerationResult, timer: StageTimer
    ) -> Answer:
        by_label = {f"S{i}": s for i, s in enumerate(result.sources, start=1)}
        citations: list[Citation] = []
        invalid: list[str] = []
        for label in result.labels:
            source = by_label.get(label)
            if source is None:
                invalid.append(label)
            else:
                citations.append(Citation(label, source.chunk.chunk_id, source.chunk.doc_id, source.chunk.location))
        text, abstained = result.text, result.abstained
        flags: list[str] = []
        if retrieval.flagged:
            flags.append(f"injection_flagged:{len(retrieval.flagged)}")
        if invalid:
            flags.append("invalid_citations")
        if not abstained and not citations:
            flags.append("uncited_answer")

        grounded: float | None = None
        unsupported: list[str] = []
        if self.verifier is not None and not abstained:
            with timer.stage("verify"):
                check = self.verifier.verify(text, result.sources)
            grounded, unsupported = check.score, check.unsupported
            if grounded is not None and grounded < 0.5:
                flags.append("low_grounding")
                if self.config.guard.grounding == "abstain":
                    text, abstained, citations = ABSTAIN_TEXT, True, []
                    flags.append("abstained_low_grounding")

        timings = {**retrieval.timings_ms, **timer.timings}
        timings["total"] = timer.total()
        return Answer(
            question=question,
            text=text,
            sources=result.sources,
            citations=citations,
            abstained=abstained,
            grounded_score=grounded,
            unsupported_sentences=unsupported,
            invalid_citations=invalid,
            flags=flags,
            usage=retrieval.usage + result.usage,
            timings_ms=timings,
            config_id=self.config.config_id,
            generator=self.generator.name,
        )

    def answer_from_retrieval(
        self, question: str, retrieval: RetrievalResult, timer: StageTimer | None = None
    ) -> Answer:
        timer = timer or StageTimer()
        with timer.stage("generate"):
            result = self.generator.generate(question, retrieval.chunks)
        return self._finalize(question, retrieval, result, timer)

    def ask(self, question: str, k: int | None = None) -> Answer:
        timer = StageTimer()
        with timer.stage("retrieve"):
            retrieval = self.retrieve(question, k)
        return self.answer_from_retrieval(question, retrieval, timer)

    def stream(self, question: str, k: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield ``sources``, then ``token`` deltas, then ``done`` with the full answer."""
        timer = StageTimer()
        with timer.stage("retrieve"):
            retrieval = self.retrieve(question, k)
        result: GenerationResult | None = None
        announced = False
        with timer.stage("generate"):
            for item in self.generator.stream(question, retrieval.chunks):
                if isinstance(item, GenerationResult):
                    result = item
                    continue
                if not announced:
                    announced = True
                    yield {"event": "sources", "data": self._preview_sources(retrieval)}
                yield {"event": "token", "data": item}
        if result is None:  # pragma: no cover - generators always yield a result
            raise RuntimeError("generator produced no result")
        answer = self._finalize(question, retrieval, result, timer)
        if not announced:
            yield {"event": "sources", "data": self._preview_sources(retrieval)}
        yield {"event": "done", "data": answer.to_dict()}

    @staticmethod
    def _preview_sources(retrieval: RetrievalResult) -> list[dict[str, Any]]:
        return [
            {"chunk_id": s.chunk.chunk_id, "location": s.chunk.location, "score": round(float(s.score), 5)}
            for s in retrieval.chunks
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        return [d.__dict__ for d in self.store.list_documents(self.collection)]

    def delete_document(self, doc_id: str) -> bool:
        self._stats_dirty = True
        return self.store.delete_document(self.collection, doc_id)

    def close(self) -> None:
        self.store.close()
        close = getattr(self.embedder, "close", None)
        if callable(close):
            close()

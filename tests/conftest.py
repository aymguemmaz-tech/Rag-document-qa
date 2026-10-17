from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from ragqa.config import PipelineConfig
from ragqa.embeddings import HashingEmbedder
from ragqa.generation.generators import ExtractiveGenerator
from ragqa.ingest.loaders import load_path, parse_markdown
from ragqa.pipeline import RAGPipeline
from ragqa.store import MemoryStore
from ragqa.types import Document

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
DATASET = ROOT / "data" / "eval" / "qa.jsonl"

SAMPLE_MD = """---
title: Battery Handling SOP
status: current
---

# Battery Handling SOP

Intro paragraph about batteries.

## Storage

Lithium-ion packs must be stored between 15 °C and 25 °C. Packs are kept at 40–60% charge.

## Damaged packs

A swollen pack is placed in the quarantine bin within 10 minutes.

### Fire

Use the AVD extinguisher. Never use water on a charging cabinet.

## Transport

| Item | Limit |
| --- | --- |
| Packs per van | 40 |
| Pack weight | 3.1 kg |
"""


@pytest.fixture
def sample_doc() -> Document:
    return parse_markdown(SAMPLE_MD, "battery-sop", "battery-sop.md")


@pytest.fixture(scope="session")
def corpus_docs() -> list[Document]:
    return load_path(CORPUS)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=512)


@pytest.fixture
def pg_url() -> Iterator[str]:
    url = os.environ.get("RAGQA_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set RAGQA_TEST_DATABASE_URL to run PostgreSQL/pgvector integration tests")
    yield url


@pytest.fixture
def collection_name() -> str:
    return "t_" + uuid.uuid4().hex[:10]


@pytest.fixture
def pipeline(embedder: HashingEmbedder, sample_doc: Document) -> RAGPipeline:
    config = PipelineConfig().with_overrides({"generation.abstain_threshold": 0.3})
    pipe = RAGPipeline(
        config,
        store=MemoryStore(),
        embedder=embedder,
        generator=ExtractiveGenerator(embedder, config.generation),
        collection="test",
    )
    pipe.ingest([sample_doc])
    return pipe

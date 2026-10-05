import pytest

from ragqa.config import ChunkingConfig
from ragqa.embeddings import HashingEmbedder
from ragqa.ingest.chunking import build_chunker
from ragqa.ingest.loaders import load_bytes, parse_html, slugify
from ragqa.text import count_tokens
from ragqa.types import Document
from tests.conftest import CORPUS


def test_markdown_sections_heading_paths_and_front_matter(sample_doc: Document) -> None:
    assert sample_doc.title == "Battery Handling SOP"
    assert sample_doc.metadata["status"] == "current"
    paths = [s.heading_path for s in sample_doc.sections]
    assert ("Battery Handling SOP", "Damaged packs", "Fire") in paths
    for section in sample_doc.sections:
        assert 0 <= section.start < section.end <= len(sample_doc.text)


def test_html_is_converted_to_sections() -> None:
    doc = parse_html(
        "<html><body><h1>Guide</h1><p>Hello <b>world</b>.</p><h2>Part</h2><ul><li>One</li></ul>"
        "<script>evil()</script></body></html>",
        "guide",
        "guide.html",
    )
    assert doc.title == "Guide"
    assert "evil" not in doc.text
    assert ("Guide", "Part") in [s.heading_path for s in doc.sections]


def test_pdf_pages_become_sections_with_page_numbers() -> None:
    pdf = CORPUS / "sustainability-report-2025.pdf"
    doc = load_bytes(pdf.name, pdf.read_bytes())
    assert doc.title == "Sustainability Report 2025"
    assert [s.page for s in doc.sections] == [1, 2]
    assert "ReCell Nordic" in doc.text


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_bytes("notes.docx", b"...")
    assert slugify("Folder/My Notes.md") == "folder-my-notes"


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic", "structure"])
def test_chunks_are_exact_substrings_within_size(strategy: str, sample_doc: Document) -> None:
    cfg = ChunkingConfig(strategy=strategy, size=24, overlap=6, min_size=0)
    chunks = build_chunker(cfg, HashingEmbedder(dim=256)).chunk(sample_doc)
    assert chunks
    for i, chunk in enumerate(chunks):
        assert chunk.text == sample_doc.text[chunk.start : chunk.end]
        assert chunk.chunk_id == f"battery-sop#{i:04d}"
        assert chunk.metadata["strategy"] == strategy
        assert chunk.metadata["status"] == "current"  # lifecycle metadata travels with chunks
        if strategy != "semantic":
            assert count_tokens(chunk.text) <= 24 + 1


def test_structure_chunker_never_crosses_headings(sample_doc: Document) -> None:
    chunks = build_chunker(ChunkingConfig(strategy="structure", size=400)).chunk(sample_doc)
    for chunk in chunks:
        assert chunk.text.count("\n#") == 0  # a heading may only start a chunk
    fire = next(c for c in chunks if "AVD" in c.text)
    assert fire.heading_path[-1] == "Fire"
    assert fire.context_header == "Battery Handling SOP › Damaged packs › Fire"
    assert fire.embed_text.startswith("Battery Handling SOP › Damaged packs › Fire\n\n")


def test_contextual_headers_can_be_disabled(sample_doc: Document) -> None:
    chunks = build_chunker(ChunkingConfig(contextual_headers=False)).chunk(sample_doc)
    assert all(c.context_header == "" and c.embed_text == c.text for c in chunks)


def test_fixed_chunker_overlaps(sample_doc: Document) -> None:
    chunks = build_chunker(ChunkingConfig(strategy="fixed", size=20, overlap=5)).chunk(sample_doc)
    assert all(b.start < a.end for a, b in zip(chunks, chunks[1:]))


def test_semantic_chunker_requires_embedder() -> None:
    with pytest.raises(ValueError):
        build_chunker(ChunkingConfig(strategy="semantic"))


def test_front_matter_dates_are_json_serialisable() -> None:
    import json

    doc = load_bytes("p.md", b"---\neffective: 2026-03-01\nstatus: superseded\n---\n# Prices\n\nText here.\n")
    assert doc.metadata == {"effective": "2026-03-01", "status": "superseded"}
    json.dumps(doc.metadata)

"""Chunking strategies. All of them work on character spans so offsets stay exact.

* ``fixed``     - sliding window of N tokens with overlap (baseline; ignores structure)
* ``recursive`` - split by paragraph -> line -> sentence -> token until pieces fit, then pack
* ``semantic``  - break where embedding similarity between neighbouring sentences drops
* ``structure`` - never cross a Markdown heading; oversize sections are split recursively

Independently of the strategy, ``contextual_headers`` prepends "Doc title › Section" to the text
that is embedded and indexed (not to the text shown to the LLM), which helps short chunks that
do not repeat their own subject.
"""

from __future__ import annotations

import bisect
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ragqa.config import ChunkingConfig
from ragqa.text import count_tokens, split_sentences, token_spans
from ragqa.types import Chunk, Document

if TYPE_CHECKING:
    from ragqa.embeddings import Embedder

Span = tuple[int, int]
_LIFECYCLE_KEYS = ("status", "effective", "version", "superseded_by")
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+.*$", re.MULTILINE)


def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_on_regex(text: str, start: int, end: int, pattern: str) -> list[Span]:
    pieces: list[Span] = []
    cursor = start
    for match in re.finditer(pattern, text[start:end]):
        s, e = _trim(text, cursor, start + match.start())
        if e > s:
            pieces.append((s, e))
        cursor = start + match.end()
    s, e = _trim(text, cursor, end)
    if e > s:
        pieces.append((s, e))
    return pieces


def _split_tokens(text: str, start: int, end: int, size: int) -> list[Span]:
    spans = [(start + s, start + e) for s, e in token_spans(text[start:end])]
    return [(w[0][0], w[-1][1]) for w in (spans[i : i + size] for i in range(0, len(spans), size)) if w]


_SEPARATORS = ("paragraph", "line", "sentence", "token")


def split_recursive(text: str, start: int, end: int, size: int, level: int = 0) -> list[Span]:
    """Split ``text[start:end]`` into atoms of at most ``size`` tokens, coarsest separator first."""
    if count_tokens(text[start:end]) <= size:
        s, e = _trim(text, start, end)
        return [(s, e)] if e > s else []
    if level >= len(_SEPARATORS) - 1:
        return _split_tokens(text, start, end, size)
    sep = _SEPARATORS[level]
    if sep == "paragraph":
        pieces = _split_on_regex(text, start, end, r"\n[ \t]*\n")
    elif sep == "line":
        pieces = _split_on_regex(text, start, end, r"\n")
    else:
        pieces = [(start + s, start + e) for s, e in split_sentences(text[start:end])]
    if len(pieces) <= 1:
        return split_recursive(text, start, end, size, level + 1)
    atoms: list[Span] = []
    for s, e in pieces:
        if count_tokens(text[s:e]) > size:
            atoms.extend(split_recursive(text, s, e, size, level + 1))
        else:
            atoms.append((s, e))
    return atoms


def pack(text: str, atoms: list[Span], size: int, overlap: int) -> list[Span]:
    """Greedily pack consecutive atoms into spans of at most ``size`` tokens, with overlap."""
    chunks: list[Span] = []
    current: list[tuple[Span, int]] = []
    total = 0
    for atom in atoms:
        n = count_tokens(text[atom[0] : atom[1]])
        if current and total + n > size:
            chunks.append((current[0][0][0], current[-1][0][1]))
            kept: list[tuple[Span, int]] = []
            kept_total = 0
            for item in reversed(current[1:]):  # never carry the whole previous chunk
                if kept_total + item[1] > overlap:
                    break
                kept.insert(0, item)
                kept_total += item[1]
            if kept_total + n > size:
                kept, kept_total = [], 0
            current, total = kept, kept_total
        current.append((atom, n))
        total += n
    if current:
        chunks.append((current[0][0][0], current[-1][0][1]))
    return chunks


class Chunker(ABC):
    name: str = "base"

    def __init__(self, cfg: ChunkingConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def spans(self, doc: Document) -> list[Span]: ...

    def chunk(self, doc: Document) -> list[Chunk]:
        starts = [s.start for s in doc.sections]
        lifecycle = {k: str(doc.metadata[k]) for k in _LIFECYCLE_KEYS if k in doc.metadata}
        chunks: list[Chunk] = []
        seen: set[Span] = set()
        for raw_start, raw_end in self.spans(doc):
            start, end = _trim(doc.text, raw_start, raw_end)
            if end <= start or (start, end) in seen:
                continue
            body = doc.text[start:end]
            if count_tokens(_HEADING_LINE_RE.sub("", body)) < 3:
                continue  # heading-only fragments carry no answerable content
            seen.add((start, end))
            section = doc.sections[max(0, bisect.bisect_right(starts, start) - 1)]
            header = ""
            if self.cfg.contextual_headers:
                header = " › ".join(dict.fromkeys(p for p in (doc.title, *section.heading_path) if p))
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):04d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=body,
                    index=len(chunks),
                    start=start,
                    end=end,
                    heading_path=section.heading_path,
                    page=section.page,
                    context_header=header,
                    metadata={"strategy": self.name, **lifecycle},
                )
            )
        return chunks


class FixedTokenChunker(Chunker):
    name = "fixed"

    def spans(self, doc: Document) -> list[Span]:
        tokens = token_spans(doc.text)
        if not tokens:
            return []
        size, step = self.cfg.size, max(1, self.cfg.size - self.cfg.overlap)
        out: list[Span] = []
        for i in range(0, len(tokens), step):
            window = tokens[i : i + size]
            out.append((window[0][0], window[-1][1]))
            if i + size >= len(tokens):
                break
        return out


class RecursiveChunker(Chunker):
    name = "recursive"

    def spans(self, doc: Document) -> list[Span]:
        atoms = split_recursive(doc.text, 0, len(doc.text), self.cfg.size)
        return pack(doc.text, atoms, self.cfg.size, self.cfg.overlap)


class StructureChunker(Chunker):
    name = "structure"

    def spans(self, doc: Document) -> list[Span]:
        out: list[Span] = []
        for section in doc.sections:
            if count_tokens(doc.text[section.start : section.end]) <= self.cfg.size:
                out.append((section.start, section.end))
            else:
                atoms = split_recursive(doc.text, section.start, section.end, self.cfg.size)
                out.extend(pack(doc.text, atoms, self.cfg.size, self.cfg.overlap))
        return out


class SemanticChunker(Chunker):
    """Greg Kamradt-style semantic chunking with a size cap and a minimum size."""

    name = "semantic"

    def __init__(self, cfg: ChunkingConfig, embedder: Embedder) -> None:
        super().__init__(cfg)
        self.embedder = embedder

    def spans(self, doc: Document) -> list[Span]:
        text = doc.text
        sents: list[Span] = []
        for s, e in split_sentences(text):  # sentences longer than the cap are pre-split
            if count_tokens(text[s:e]) > self.cfg.size:
                sents.extend(_split_tokens(text, s, e, self.cfg.size))
            else:
                sents.append((s, e))
        if len(sents) <= 2:
            return [(sents[0][0], sents[-1][1])] if sents else []

        windows = [" ".join(text[a:b] for a, b in sents[max(0, i - 1) : i + 2]) for i in range(len(sents))]
        emb = self.embedder.embed(windows)
        distances = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)
        threshold = float(np.percentile(distances, self.cfg.semantic_percentile))

        groups: list[list[Span]] = [[sents[0]]]
        sizes = [count_tokens(text[sents[0][0] : sents[0][1]])]
        for i in range(1, len(sents)):
            n = count_tokens(text[sents[i][0] : sents[i][1]])
            if distances[i - 1] > threshold or sizes[-1] + n > self.cfg.size:
                groups.append([sents[i]])
                sizes.append(n)
            else:
                groups[-1].append(sents[i])
                sizes[-1] += n

        merged: list[tuple[Span, int]] = []
        for group, n in zip(groups, sizes):
            span = (group[0][0], group[-1][1])
            if merged and (n < self.cfg.min_size or merged[-1][1] < self.cfg.min_size):
                prev_span, prev_n = merged[-1]
                if prev_n + n <= self.cfg.size:
                    merged[-1] = ((prev_span[0], span[1]), prev_n + n)
                    continue
            merged.append((span, n))
        return [span for span, _ in merged]


def build_chunker(cfg: ChunkingConfig, embedder: Embedder | None = None) -> Chunker:
    if cfg.strategy == "fixed":
        return FixedTokenChunker(cfg)
    if cfg.strategy == "recursive":
        return RecursiveChunker(cfg)
    if cfg.strategy == "structure":
        return StructureChunker(cfg)
    if cfg.strategy == "semantic":
        if embedder is None:
            raise ValueError("The semantic chunker needs an embedder")
        return SemanticChunker(cfg, embedder)
    raise ValueError(f"Unknown chunking strategy: {cfg.strategy}")

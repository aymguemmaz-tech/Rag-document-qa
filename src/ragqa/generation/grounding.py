"""Safety and faithfulness guards.

``InjectionGuard`` scans retrieved chunks for prompt-injection patterns *before* they reach the
generator (indirect prompt injection is the main RAG-specific attack: the attacker plants
instructions in a document rather than in the question).

``GroundingVerifier`` checks each answer sentence against the sources it cites, combining
sentence-level semantic similarity, chunk-level term coverage and a strict numeric
consistency check (a number in the answer that does not appear in the cited source is a
classic hallucination). It is a fast heuristic for runtime flagging; the evaluation layer can
additionally use an LLM judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ragqa.embeddings import Embedder
from ragqa.generation.citations import is_abstention, parse_citations, strip_citations
from ragqa.text import analyze, clean_inline_markdown, split_sentences, words
from ragqa.types import ScoredChunk

INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "ignore_instructions": (
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|"
            r"any|your|the)\b[^.\n]{0,25}\b(instructions?|rules|prompts?|directions|guidelines)\b"
        ),
        "role_override": r"\byou are now\b|\bact as (an?|the) \w+|\bpretend (to be|you are)\b",
        "system_prompt": r"\b(system prompt|developer message|hidden instructions?)\b",
        "ai_directive": (
            r"\b(note|message|instructions?)\s+(to|for)\s+(any\s+)?(ai|llms?|language models?|"
            r"(ai\s+)?assistants?|chatbots?)\b|\b(ai|llm)\s+(assistants?|models?|systems?)\s+"
            r"(reading|processing|summari[sz]ing)\s+this\b"
        ),
        "special_tokens": r"<\|(im_start|im_end|system|endoftext)\|>|\[/?INST\]|<<SYS>>",
        "exfiltration": (
            r"\b(reveal|print|output|leak|send)\b[^.\n]{0,30}\b(api key|password|secret|"
            r"credentials|system prompt)\b"
        ),
    }.items()
}


class InjectionGuard:
    def __init__(self, mode: str = "drop") -> None:
        if mode not in {"off", "flag", "drop"}:
            raise ValueError(f"Unknown injection guard mode: {mode}")
        self.mode = mode

    @staticmethod
    def scan(text: str) -> list[str]:
        return [name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(text)]

    def apply(self, candidates: list[ScoredChunk]) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
        """Return (kept, flagged). In ``flag`` mode flagged chunks are kept but annotated."""
        if self.mode == "off":
            return candidates, []
        kept: list[ScoredChunk] = []
        flagged: list[ScoredChunk] = []
        for item in candidates:
            hits = self.scan(item.chunk.text)
            if hits:
                item.flags = sorted({*item.flags, *(f"injection:{h}" for h in hits)})
                flagged.append(item)
                if self.mode == "drop":
                    continue
            kept.append(item)
        return kept, flagged


def _numbers(text: str) -> set[str]:
    return {re.sub(r"(?<=\d)[,](?=\d)", "", w) for w in words(text) if any(ch.isdigit() for ch in w)}


@dataclass
class SentenceCheck:
    sentence: str
    support: float
    supported: bool
    cited: list[str] = field(default_factory=list)


@dataclass
class GroundingResult:
    score: float | None
    checks: list[SentenceCheck]

    @property
    def unsupported(self) -> list[str]:
        return [c.sentence for c in self.checks if not c.supported]


class GroundingVerifier:
    def __init__(self, embedder: Embedder, threshold: float = 0.5) -> None:
        self.embedder = embedder
        self.threshold = threshold

    def verify(self, answer: str, sources: list[ScoredChunk]) -> GroundingResult:
        if is_abstention(answer) or not sources:
            return GroundingResult(score=None, checks=[])
        label_to_source = {f"S{i}": s for i, s in enumerate(sources, start=1)}
        checks: list[SentenceCheck] = []
        pending: list[tuple[str, list[str], list[ScoredChunk]]] = []
        for start, end in split_sentences(answer):
            raw = answer[start:end]
            labels = parse_citations(raw)
            text = clean_inline_markdown(strip_citations(raw))
            if len(analyze(text)) < 2:
                continue
            cited = [label_to_source[label] for label in labels if label in label_to_source]
            pending.append((text, labels, cited or sources))
        if not pending:
            return GroundingResult(score=None, checks=[])

        answer_vecs = self.embedder.embed([p[0] for p in pending])
        chunk_cache: dict[str, tuple[np.ndarray, set[str], set[str]]] = {}
        for (text, labels, candidates), vec in zip(pending, answer_vecs):
            terms = set(analyze(text))
            numbers = _numbers(text)
            best = 0.0
            for item in candidates:
                cid = item.chunk.chunk_id
                if cid not in chunk_cache:
                    sents = [item.chunk.text[s:e] for s, e in split_sentences(item.chunk.text)]
                    chunk_cache[cid] = (
                        self.embedder.embed(sents or [item.chunk.text]),
                        set(analyze(item.chunk.text)),
                        _numbers(item.chunk.text),
                    )
                sent_vecs, chunk_terms, chunk_numbers = chunk_cache[cid]
                semantic = float(np.max(sent_vecs @ vec)) if len(sent_vecs) else 0.0
                coverage = len(terms & chunk_terms) / max(1, len(terms))
                support = 0.5 * max(0.0, semantic) + 0.5 * coverage
                if numbers - chunk_numbers:  # an unsupported number caps support below threshold
                    support = min(support, self.threshold - 1e-6)
                best = max(best, support)
            checks.append(SentenceCheck(text, round(best, 4), best >= self.threshold, labels))
        score = sum(c.supported for c in checks) / len(checks)
        return GroundingResult(score=round(score, 4), checks=checks)

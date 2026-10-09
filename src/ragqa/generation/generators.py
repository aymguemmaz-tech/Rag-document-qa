"""Answer generators.

* :class:`LLMGenerator` - prompt template + any OpenAI-compatible chat model (GPT by default).
* :class:`ExtractiveGenerator` - a deterministic, zero-cost baseline that selects the source
  sentences best supported by the question (semantic similarity + IDF-weighted term coverage)
  and abstains below a calibrated threshold. It makes the whole system - and its evaluation -
  runnable offline and in CI, and gives a floor that an LLM must beat to justify its cost.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ragqa.config import GenerationConfig
from ragqa.embeddings import Embedder
from ragqa.generation.citations import ABSTAIN_TEXT, is_abstention, parse_citations
from ragqa.generation.llm import LLM, LLMResponse
from ragqa.generation.prompts import PromptTemplate, build_context
from ragqa.text import (
    analyze,
    clean_inline_markdown,
    content_words,
    render_table_row,
    split_sentences,
    table_row_headers,
)
from ragqa.types import ScoredChunk, Usage

if TYPE_CHECKING:
    from ragqa.store.base import TermStats


@dataclass
class GenerationResult:
    text: str
    labels: list[str]
    abstained: bool
    sources: list[ScoredChunk]
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    prompt: str | None = None


class Generator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, question: str, sources: list[ScoredChunk]) -> GenerationResult: ...

    def stream(self, question: str, sources: list[ScoredChunk]) -> Iterator[str | GenerationResult]:
        result = self.generate(question, sources)
        yield result.text
        yield result


class LLMGenerator(Generator):
    def __init__(self, llm: LLM, template: PromptTemplate, cfg: GenerationConfig) -> None:
        self.llm = llm
        self.template = template
        self.cfg = cfg
        self.name = f"llm:{llm.model}:{template.name}"

    def _messages(self, question: str, sources: list[ScoredChunk]) -> tuple[list[dict[str, str]], list[ScoredChunk]]:
        context, used = build_context(sources, self.cfg.max_context_tokens, self.cfg.context_order)
        return self.template.render(question, context), used

    def _result(self, response: LLMResponse, used: list[ScoredChunk]) -> GenerationResult:
        text = response.text.strip() or ABSTAIN_TEXT
        return GenerationResult(
            text=text,
            labels=parse_citations(text),
            abstained=is_abstention(text),
            sources=used,
            usage=response.usage,
            model=response.model,
            prompt=self.template.name,
        )

    def generate(self, question: str, sources: list[ScoredChunk]) -> GenerationResult:
        if not sources:
            return GenerationResult(ABSTAIN_TEXT, [], True, [], prompt=self.template.name)
        messages, used = self._messages(question, sources)
        return self._result(self.llm.complete(messages), used)

    def stream(self, question: str, sources: list[ScoredChunk]) -> Iterator[str | GenerationResult]:
        if not sources:
            yield ABSTAIN_TEXT
            yield GenerationResult(ABSTAIN_TEXT, [], True, [], prompt=self.template.name)
            return
        messages, used = self._messages(question, sources)
        for item in self.llm.stream(messages):
            if isinstance(item, LLMResponse):
                yield self._result(item, used)
            else:
                yield item


def _with_citation(sentence: str, label: int) -> str:
    body = sentence.rstrip()
    end = body[-1] if body and body[-1] in ".!?" else "."
    core = body[:-1].rstrip() if body and body[-1] in ".!?" else body
    return f"{core} [S{label}]{end}"


class ExtractiveGenerator(Generator):
    name = "extractive"

    def __init__(
        self,
        embedder: Embedder,
        cfg: GenerationConfig,
        seed: int | None = None,
        margin: float = 0.08,
        term_stats: Callable[[], TermStats | None] | None = None,
    ) -> None:
        self.embedder = embedder
        self.cfg = cfg
        self.margin = margin
        self.rng = np.random.default_rng(seed)
        self.term_stats = term_stats

    @staticmethod
    def candidates(used: list[ScoredChunk]) -> list[tuple[int, str, str]]:
        """(source label, display sentence, scoring text) for every candidate sentence.

        Table rows are rendered with their column headers, and each sentence is scored together
        with its document title and section heading - the same idea as contextual chunk headers,
        applied at sentence level - while only the sentence itself is shown in the answer.
        """
        out: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        for label, item in enumerate(used, start=1):
            text = item.chunk.text
            headers = table_row_headers(text)
            section = item.chunk.heading_path[-1] if item.chunk.heading_path else ""
            for start, end in split_sentences(text):
                raw = text[start:end]
                if raw.lstrip().startswith("#"):
                    section = clean_inline_markdown(raw)
                    continue
                clean = render_table_row(raw, headers[start]) if start in headers else clean_inline_markdown(raw)
                key = clean.lower()
                if len(content_words(clean)) < 3 or key in seen:
                    continue
                seen.add(key)
                context = " › ".join(dict.fromkeys(p for p in (item.chunk.doc_title, section) if p))
                out.append((label, clean, f"{context}: {clean}" if context else clean))
        return out

    def score(self, question: str, cands: list[tuple[int, str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, sentence embeddings) for candidate sentences."""
        vecs = self.embedder.embed([scoring for _, _, scoring in cands])
        cosine = vecs @ self.embedder.embed_query(question)
        q_terms = set(analyze(question))
        sent_terms = [set(analyze(scoring)) for _, _, scoring in cands]
        stats = self.term_stats() if self.term_stats is not None else None
        if stats is not None:  # corpus-level importance: words found everywhere ("Veloria") weigh little
            idf = {t: stats.idf(t) for t in q_terms}
        else:  # fallback: rarity among the candidate sentences
            df = Counter(t for terms in sent_terms for t in terms & q_terms)
            idf = {t: math.log(1.0 + (len(cands) + 1) / (df[t] + 0.5)) for t in q_terms}
        denom = sum(idf.values()) or 1.0
        coverage = np.array([sum(idf[t] for t in terms & q_terms) / denom for terms in sent_terms])
        prior = np.array([1.0 / label for label, _, _ in cands])
        return 0.5 * np.clip(cosine, 0, None) + 0.4 * coverage + 0.1 * prior, vecs

    def generate(self, question: str, sources: list[ScoredChunk]) -> GenerationResult:
        _, used = build_context(sources, self.cfg.max_context_tokens, "relevance")
        cands = self.candidates(used)
        if not cands:
            return GenerationResult(ABSTAIN_TEXT, [], True, used, model=self.name)
        scores, vecs = self.score(question, cands)
        order = list(np.argsort(-scores, kind="stable"))
        best = float(scores[order[0]])
        if best < self.cfg.abstain_threshold:
            return GenerationResult(ABSTAIN_TEXT, [], True, used, model=self.name)

        if self.cfg.temperature > 0:
            eligible = [i for i in order if scores[i] >= self.cfg.abstain_threshold]
            logits = np.array([scores[i] for i in eligible]) / self.cfg.temperature
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            first = int(eligible[int(self.rng.choice(len(eligible), p=probs))])
        else:
            first = int(order[0])
        selected = [first]
        anchor = float(scores[first])
        for j in order:
            if len(selected) >= self.cfg.max_sentences:
                break
            if j in selected or scores[j] < max(self.cfg.abstain_threshold, anchor - self.margin):
                continue
            if max(float(vecs[j] @ vecs[k]) for k in selected) > 0.9:
                continue
            selected.append(int(j))
        text = " ".join(_with_citation(cands[i][1], cands[i][0]) for i in selected)
        return GenerationResult(text=text, labels=parse_citations(text), abstained=False, sources=used, model=self.name)

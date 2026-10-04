"""Deterministic text utilities: tokenization, sentence splitting and answer normalization.

Chunk sizes are measured with a regex tokenizer (words + punctuation) rather than a model
tokenizer, so chunk boundaries - and therefore every ablation result - are reproducible on any
machine regardless of which LLM provider is configured.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Iterator

_TOKEN_RE = re.compile(r"\w+(?:[-'’.]\w+)*|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "s",
        "t",
        "don",
        "shall",
        "may",
        "might",
        "must",
        "also",
        "per",
        "via",
    ]
)


def tokenize(text: str) -> list[str]:
    """Split text into word and punctuation tokens (used for chunk sizing)."""
    return _TOKEN_RE.findall(text)


def token_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of every token in ``text``."""
    return [m.span() for m in _TOKEN_RE.finditer(text)]


def count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN_RE.finditer(text))


def fold(text: str) -> str:
    """Unicode-normalize and lowercase; maps typographic dashes/quotes to ASCII."""
    text = unicodedata.normalize("NFKC", text)
    table = str.maketrans({"–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "€": " eur "})
    return text.translate(table).lower()


def words(text: str) -> list[str]:
    """Lowercase alphanumeric words; keeps decimals such as ``2.5`` and ``1,200`` intact."""
    return _WORD_RE.findall(fold(text))


def stem(word: str) -> str:
    """A light, deterministic suffix stripper (plurals and common verb endings)."""
    if len(word) <= 3 or word[0].isdigit():
        return word
    for suffix, repl, min_len in (
        ("ies", "y", 5),
        ("sses", "ss", 6),
        ("ing", "", 6),
        ("edly", "", 7),
        ("ed", "", 5),
        ("es", "", 6),
        ("s", "", 4),
    ):
        if word.endswith(suffix) and len(word) >= min_len:
            if suffix == "s" and word.endswith(("ss", "us", "is")):
                return word
            word = word[: -len(suffix)] + repl
            break
    if len(word) >= 5 and word.endswith("e"):  # store/stored, acknowledge/acknowledged
        word = word[:-1]
    return word


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS]


def analyze(text: str) -> list[str]:
    """Normalized terms for lexical retrieval and overlap metrics."""
    return [stem(w) for w in content_words(text)]


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: fold case, drop punctuation and articles, squash spaces."""
    text = fold(text)
    text = re.sub(r"(?<=\d)[.,](?=\d)", "", text)  # 1,200 -> 1200 ; 1.20 -> 120 consistently
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------------------------
# Sentence splitting (span-preserving, markdown-aware)
# --------------------------------------------------------------------------------------------

_ABBREVIATIONS = frozenset(
    [
        "e.g",
        "i.e",
        "vs",
        "no",
        "nos",
        "approx",
        "dr",
        "mr",
        "mrs",
        "ms",
        "fig",
        "incl",
        "dept",
        "st",
        "ref",
        "vol",
        "pp",
        "ca",
        "cf",
    ]
)
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|\||#{1,6}\s+|>\s*)")
_BOUNDARY_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s)")


def _lines(text: str) -> Iterator[tuple[int, int]]:
    pos = 0
    for line in text.splitlines(keepends=True):
        yield pos, pos + len(line.rstrip("\r\n"))
        pos += len(line)


def _blocks(text: str) -> Iterator[tuple[int, int]]:
    """Paragraphs, plus one block per list item, table row, heading or quote line."""
    block_start: int | None = None
    block_end = 0
    for start, end in _lines(text):
        line = text[start:end]
        if not line.strip():
            if block_start is not None:
                yield block_start, block_end
                block_start = None
            continue
        if _LIST_LINE_RE.match(line):
            if block_start is not None:
                yield block_start, block_end
                block_start = None
            yield start, end
            continue
        if block_start is None:
            block_start = start
        block_end = end
    if block_start is not None:
        yield block_start, block_end


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_sentences(text: str, offset: int = 0, max_item_tokens: int = 60) -> list[tuple[int, int]]:
    """Return character spans of sentences. Spans are relative to ``text`` plus ``offset``.

    Short list items, table rows and headings are kept whole (one unit each), because their
    parts ("Owner: ... Due: ...") only make sense together.
    """
    spans: list[tuple[int, int]] = []
    for b_start, b_end in _blocks(text):
        block = text[b_start:b_end]
        if _LIST_LINE_RE.match(block) and count_tokens(block) <= max_item_tokens:
            s, e = _trim(block, 0, len(block))
            if e > s:
                spans.append((b_start + s + offset, b_start + e + offset))
            continue
        start = 0
        for match in _BOUNDARY_RE.finditer(block):
            end = match.end()
            prefix = block[start : match.start()]
            last_word = re.search(r"([A-Za-z][A-Za-z.]*)$", prefix)
            if match.group(0).startswith(".") and last_word:
                candidate = last_word.group(1).lower().rstrip(".")
                before = prefix[: last_word.start()]
                is_initial = len(candidate) == 1 and (not before or before[-1].isspace())
                if candidate in _ABBREVIATIONS or is_initial:
                    continue
            rest = block[end:].lstrip()
            if rest and rest[0].islower():
                continue
            s, e = _trim(block, start, end)
            if e > s:
                spans.append((b_start + s + offset, b_start + e + offset))
            start = end
        s, e = _trim(block, start, len(block))
        if e > s:
            spans.append((b_start + s + offset, b_start + e + offset))
    return spans


def sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentences(text)]


_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_row_headers(text: str) -> dict[int, list[str]]:
    """Map the start offset of every Markdown table data row to its table's header cells."""
    lines = list(_lines(text))
    headers: dict[int, list[str]] = {}
    current: list[str] | None = None
    for i, (start, end) in enumerate(lines):
        line = text[start:end]
        if not line.strip().startswith("|"):
            current = None
            continue
        if i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(text[lines[i + 1][0] : lines[i + 1][1]]):
            current = _cells(line)
            continue
        if _TABLE_SEPARATOR_RE.match(line):
            continue
        if current is not None:
            stripped = len(line) - len(line.lstrip())
            headers[start + stripped] = current
    return headers


def render_table_row(row: str, headers: list[str]) -> str:
    """'| SEV1 | ... | 5 minutes |' -> 'Severity: SEV1; ...; Acknowledge within: 5 minutes'."""
    cells = _cells(row)
    if len(cells) != len(headers):
        return clean_inline_markdown(row)
    return "; ".join(f"{h}: {c}" for h, c in zip(headers, cells) if c)


_MD_INLINE_RE = re.compile(r"(\*\*|__|`)")


def clean_inline_markdown(text: str) -> str:
    """Remove emphasis/backtick markers and leading list/heading syntax for display."""
    text = _MD_INLINE_RE.sub("", text)
    text = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s*)", "", text)
    if text.strip().startswith("|"):
        cells = [c.strip() for c in text.strip().strip("|").split("|")]
        text = " — ".join(c for c in cells if c and not set(c) <= set("-: "))
    return " ".join(text.split())

"""Citation parsing and abstention detection for ``[S#]``-style grounded answers."""

from __future__ import annotations

import re

from ragqa.text import fold

ABSTAIN_TEXT = "I don't know based on the provided documents."

_CITE_GROUP_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S?\d+)*)\s*\]", re.IGNORECASE)
_ABSTAIN_PATTERNS = (
    "i don't know",
    "i do not know",
    "not enough information",
    "do not contain the answer",
    "does not contain the answer",
    "don't contain the answer",
    "no information about",
    "not mentioned in the provided",
    "cannot be determined from the provided",
)


def parse_citations(text: str) -> list[str]:
    """Return unique citation labels (``S1``, ``S3``...) in order of first appearance."""
    labels: list[str] = []
    for group in _CITE_GROUP_RE.findall(text):
        for part in re.split(r"\s*[,;]\s*", group):
            label = "S" + part.upper().lstrip("S")
            if label not in labels:
                labels.append(label)
    return labels


def strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", _CITE_GROUP_RE.sub("", text)).strip()


def is_abstention(text: str) -> bool:
    folded = fold(text)
    if not folded.strip():
        return True
    head = folded[:200]
    return any(p in head for p in _ABSTAIN_PATTERNS)

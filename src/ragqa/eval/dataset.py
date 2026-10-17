"""Gold evaluation set: questions, accepted answers and *evidence spans*.

Evidence is stored as exact text from the source document, then located to character offsets.
Retrieval is scored by whether a retrieved chunk overlaps the evidence span - not by chunk ids -
so the same gold set fairly compares chunking strategies that produce completely different chunks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ragqa.types import Document


@dataclass
class Evidence:
    doc: str
    text: str
    start: int = -1
    end: int = -1


@dataclass
class EvalItem:
    id: str
    question: str
    answers: list[str]
    evidence: list[Evidence]
    category: str
    answerable: bool = True
    split: str = "test"
    injected_claim: str | None = None
    notes: str = ""
    evidence_any: bool = False  # True: any one evidence span is sufficient (fact stated in several docs)


def load_dataset(path: str | Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                items.append(
                    EvalItem(
                        id=raw["id"],
                        question=raw["question"],
                        answers=list(raw.get("answers", [])),
                        evidence=[Evidence(e["doc"], e["text"]) for e in raw.get("evidence", [])],
                        category=raw.get("category", "factoid"),
                        answerable=bool(raw.get("answerable", True)),
                        split=raw.get("split", "test"),
                        injected_claim=raw.get("injected_claim"),
                        notes=raw.get("notes", ""),
                        evidence_any=bool(raw.get("evidence_any", False)),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no}: missing field {exc}") from None
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids in dataset")
    return items


def dataset_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _whitespace_find(text: str, needle: str) -> tuple[int, int] | None:
    chars: list[str] = []
    mapping: list[int] = []
    previous_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if previous_space:
                continue
            chars.append(" ")
            previous_space = True
        else:
            chars.append(ch)
            previous_space = False
        mapping.append(i)
    normalized = "".join(chars)
    target = " ".join(needle.split())
    j = normalized.find(target)
    if j < 0 or not target:
        return None
    return mapping[j], mapping[j + len(target) - 1] + 1


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def locate_evidence(items: list[EvalItem], documents: dict[str, Document]) -> ValidationReport:
    """Resolve evidence spans to offsets in place and report problems."""
    report = ValidationReport()
    for item in items:
        if item.answerable and not item.evidence:
            report.errors.append(f"{item.id}: answerable question without evidence")
        if item.answerable and not item.answers:
            report.errors.append(f"{item.id}: answerable question without accepted answers")
        if not item.answerable and item.evidence:
            report.warnings.append(f"{item.id}: unanswerable question lists evidence")
        for ev in item.evidence:
            doc = documents.get(ev.doc)
            if doc is None:
                report.errors.append(f"{item.id}: unknown document '{ev.doc}'")
                continue
            start = doc.text.find(ev.text)
            if start >= 0:
                ev.start, ev.end = start, start + len(ev.text)
                if doc.text.find(ev.text, start + 1) >= 0:
                    report.warnings.append(f"{item.id}: evidence occurs more than once in {ev.doc}")
                continue
            span = _whitespace_find(doc.text, ev.text)
            if span is None:
                report.errors.append(f"{item.id}: evidence not found in {ev.doc}: {ev.text[:60]!r}")
            else:
                ev.start, ev.end = span
    return report


def filter_split(items: list[EvalItem], split: str) -> list[EvalItem]:
    return items if split in ("all", "") else [i for i in items if i.split == split]

"""Document loaders for Markdown, plain text, HTML and PDF.

Every loader produces a :class:`~ragqa.types.Document` whose ``sections`` carry heading paths
(or page numbers) with character offsets, so chunks can always be traced back to an exact
location in the source - the basis for citations and for evidence-span evaluation.
"""

from __future__ import annotations

import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar

import yaml

from ragqa.types import Document, Section

SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".pdf", ".html", ".htm"})

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def slugify(value: str) -> str:
    value = re.sub(r"\.[A-Za-z0-9]+$", "", value)
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value or "document"


def _title_from_name(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[-_]+", " ", stem).strip().title() or "Untitled"


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    after = text[end + 4 :]
    after = after[1:] if after.startswith("\n") else after
    try:
        meta = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, after
    # YAML turns `effective: 2026-03-01` into a date; keep metadata JSON-serialisable for every store.
    return json.loads(json.dumps(meta, default=str)), after


def parse_markdown(text: str, doc_id: str, source: str, name: str = "") -> Document:
    text = text.replace("\r\n", "\n").lstrip("﻿")
    metadata, text = _split_front_matter(text)

    headings: list[tuple[int, int, str]] = []  # (char offset, level, title)
    in_fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING_RE.match(stripped)
            if match:
                headings.append((pos, len(match.group(1)), match.group(2).strip()))
        pos += len(line)

    title = str(metadata.get("title") or "")
    if not title:
        h1 = next((h for h in headings if h[1] == 1), None)
        title = h1[2] if h1 else _title_from_name(name or doc_id)

    sections: list[Section] = []
    first = headings[0][0] if headings else len(text)
    if text[:first].strip():
        sections.append(Section(heading_path=(), start=0, end=first))
    stack: list[tuple[int, str]] = []
    for i, (offset, level, heading) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        sections.append(Section(heading_path=tuple(h for _, h in stack), start=offset, end=end))
    if not sections:
        sections.append(Section(heading_path=(), start=0, end=len(text)))
    return Document(doc_id=doc_id, title=title, source=source, text=text, sections=sections, metadata=metadata)


def parse_text(text: str, doc_id: str, source: str, name: str = "") -> Document:
    text = text.replace("\r\n", "\n").lstrip("﻿")
    return Document(
        doc_id=doc_id,
        title=_title_from_name(name or doc_id),
        source=source,
        text=text,
        sections=[Section(heading_path=(), start=0, end=len(text))],
    )


def _clean_pdf_page(raw: str) -> str:
    text = raw.replace("\r\n", "\n")
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", " ", text)  # keep paragraph breaks
    text = re.sub(r"\s*\n\s*", " ", text)  # join wrapped lines
    text = text.replace(" ", "\n\n")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def parse_pdf(data: bytes, doc_id: str, source: str, name: str = "") -> Document:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    sections: list[Section] = []
    pos = 0
    for page_no, page in enumerate(reader.pages, start=1):
        page_text = _clean_pdf_page(page.extract_text() or "")
        if not page_text:
            continue
        if parts:
            pos += 2  # the "\n\n" joiner
        sections.append(Section(heading_path=(), start=pos, end=pos + len(page_text), page=page_no))
        parts.append(page_text)
        pos += len(page_text)
    text = "\n\n".join(parts)
    meta_title = None
    if reader.metadata is not None:
        meta_title = reader.metadata.title
    title = str(meta_title).strip() if meta_title else _title_from_name(name or doc_id)
    return Document(
        doc_id=doc_id,
        title=title,
        source=source,
        text=text,
        sections=sections or [Section(heading_path=(), start=0, end=len(text))],
        metadata={"pages": len(reader.pages)},
    )


class _HTMLToMarkdown(HTMLParser):
    _BLOCK: ClassVar[frozenset[str]] = frozenset(
        {"p", "div", "section", "article", "li", "tr", "br", "table", "ul", "ol", "pre"}
    )
    _SKIP: ClassVar[frozenset[str]] = frozenset({"script", "style", "noscript", "head", "nav", "footer"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip_depth = 0
        self._heading: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self._heading = int(tag[1])
            self.out.append("\n\n" + "#" * self._heading + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in self._BLOCK:
            self.out.append("\n\n" if tag != "br" else "\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif re.fullmatch(r"h[1-6]", tag):
            self._heading = None
            self.out.append("\n\n")
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.out.append(re.sub(r"\s+", " ", data) if self._heading is None else data.strip())

    def markdown(self) -> str:
        text = "".join(self.out)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def parse_html(text: str, doc_id: str, source: str, name: str = "") -> Document:
    parser = _HTMLToMarkdown()
    parser.feed(text)
    return parse_markdown(parser.markdown(), doc_id, source, name)


def load_bytes(name: str, data: bytes, doc_id: str | None = None, source: str | None = None) -> Document:
    """Load a document from raw bytes (used by the upload API)."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    doc_id = doc_id or slugify(Path(name).name)
    source = source or name
    if suffix == ".pdf":
        return parse_pdf(data, doc_id, source, name)
    text = data.decode("utf-8", errors="replace")
    if suffix in (".md", ".markdown"):
        return parse_markdown(text, doc_id, source, name)
    if suffix in (".html", ".htm"):
        return parse_html(text, doc_id, source, name)
    return parse_text(text, doc_id, source, name)


def load_path(path: str | Path) -> list[Document]:
    """Load one file or every supported file under a directory (sorted, recursive)."""
    path = Path(path)
    if path.is_file():
        files = [path]
        root = path.parent
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        root = path
    else:
        raise FileNotFoundError(path)
    docs = []
    for file in files:
        rel = file.relative_to(root).as_posix()
        docs.append(load_bytes(file.name, file.read_bytes(), doc_id=slugify(rel), source=rel))
    return docs

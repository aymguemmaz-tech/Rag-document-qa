from ragqa.text import (
    analyze,
    clean_inline_markdown,
    count_tokens,
    normalize_answer,
    render_table_row,
    sentences,
    split_sentences,
    stem,
    table_row_headers,
    tokenize,
)


def test_tokenize_keeps_decimals_and_hyphenated_words() -> None:
    assert tokenize("Packs weigh 4.2 kg, e-bikes too.") == ["Packs", "weigh", "4.2", "kg", ",", "e-bikes", "too", "."]
    assert count_tokens("one two, three") == 4


def test_sentence_splitter_handles_abbreviations_units_and_decimals() -> None:
    text = "Store at 10 °C. Keep 40.5% charge, e.g. for storage. Dr. Smith agrees! Next one?"
    assert sentences(text) == [
        "Store at 10 °C.",
        "Keep 40.5% charge, e.g. for storage.",
        "Dr. Smith agrees!",
        "Next one?",
    ]


def test_sentence_spans_are_exact_offsets() -> None:
    text = "First sentence here.  Second one follows.\n\n- A list item. With two parts.\n| a | b |"
    for start, end in split_sentences(text):
        assert text[start:end] == text[start:end].strip()
    # short list items and table rows stay whole
    assert "- A list item. With two parts." in sentences(text)
    assert "| a | b |" in sentences(text)


def test_stemmer_is_consistent_across_forms() -> None:
    assert stem("batteries") == stem("battery")
    assert stem("acknowledged") == stem("acknowledge")
    assert stem("stored") == stem("store")
    assert stem("charging") == stem("charge")
    assert analyze("The batteries were stored") == ["battery", "stor"]


def test_normalize_answer() -> None:
    assert normalize_answer("The fee is €1.00!") == normalize_answer("fee is EUR 1.00")
    assert normalize_answer("6,214 attempts") == "6214 attempts"
    assert normalize_answer("40–60%") == normalize_answer("40-60 %")


def test_table_rows_are_rendered_with_headers() -> None:
    text = "| Severity | Acknowledge within |\n| --- | --- |\n| SEV1 | 5 minutes |\n"
    headers = table_row_headers(text)
    row_start = text.index("| SEV1")
    assert headers[row_start] == ["Severity", "Acknowledge within"]
    assert (
        render_table_row("| SEV1 | 5 minutes |", headers[row_start]) == "Severity: SEV1; Acknowledge within: 5 minutes"
    )


def test_clean_inline_markdown() -> None:
    assert clean_inline_markdown("- **Bold** `code` item") == "Bold code item"
    assert clean_inline_markdown("| Day Pass | €9.50 |") == "Day Pass — €9.50"

"""Generate the sample PDF document of the corpus (exercises the PDF loader and page citations).

Usage: python scripts/make_sample_pdf.py  (requires reportlab: pip install "ragqa[docs]")
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "corpus" / "sustainability-report-2025.pdf"

PAGES = [
    [
        ("h1", "Sustainability Report 2025"),
        (
            "p",
            "This report summarises the environmental performance of Veloria Mobility in Arvenna "
            "for the calendar year 2025.",
        ),
        ("h2", "Lifecycle emissions"),
        (
            "p",
            "Lifecycle emissions fell to 38 g CO2e per passenger-km in 2025, down from 52 g CO2e "
            "per passenger-km in 2023. The figure covers vehicle manufacturing, charging, "
            "operations and end-of-life treatment.",
        ),
        ("p", "Our target for 2030 is 25 g CO2e per passenger-km."),
        ("h2", "Energy"),
        (
            "p",
            "Since July 2024 all electricity used for charging comes from renewable sources "
            "through a power purchase agreement with the Kestrel Ridge wind farm.",
        ),
    ],
    [
        ("h2", "Vehicle and battery lifetime"),
        ("p", "Veloria S3 scooters now last 4.5 years on average, and Veloria B2 e-bikes last 6 years on average."),
        (
            "p",
            "Retired battery packs are recycled by our partner ReCell Nordic. In 2025, 96% of "
            "retired packs were recycled.",
        ),
        ("h2", "Service fleet"),
        (
            "p",
            "At the end of 2025, 62% of our service vans were electric. We aim for a fully electric van fleet by 2027.",
        ),
        ("h2", "Mode shift"),
        ("p", "In our 2025 rider survey, 27% of respondents said their last Veloria trip replaced a car trip."),
    ],
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, page in enumerate(PAGES):
        if i:
            story.append(PageBreak())
        for kind, text in page:
            style = {"h1": styles["Title"], "h2": styles["Heading2"], "p": styles["BodyText"]}[kind]
            story.append(Paragraph(text, style))
            story.append(Spacer(1, 6))
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, title="Sustainability Report 2025", author="Veloria Mobility (fictional)"
    )
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

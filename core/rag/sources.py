"""The FIA regulation documents we index: final issues for 2025, latest issues for 2026."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DocType = Literal["sporting", "technical", "financial"]

DOC_TITLES: dict[DocType, str] = {
    "sporting": "Sporting Regulations",
    "technical": "Technical Regulations",
    "financial": "Financial Regulations",
}


@dataclass(frozen=True)
class RegSource:
    doc_type: DocType
    year: int
    issue: str
    url: str

    @property
    def doc_id(self) -> str:
        return f"{self.year}_{self.doc_type}"

    @property
    def filename(self) -> str:
        return f"{self.doc_id}.pdf"

    @property
    def section_letter(self) -> str:
        """2026 regulations prefix every article with their section letter."""
        if self.year < 2026:
            return ""
        return {"sporting": "B", "technical": "C", "financial": "D"}[self.doc_type]

    @property
    def title(self) -> str:
        return f"{self.year} F1 {DOC_TITLES[self.doc_type]} (Issue {self.issue})"


_FIA = "https://www.fia.com/system/files/documents/"
REG_SOURCES: tuple[RegSource, ...] = (
    RegSource(
        "sporting",
        2025,
        "5",
        _FIA + "fia_2025_formula_1_sporting_regulations_-_issue_5_-_2025-04-30.pdf",
    ),
    RegSource(
        "technical",
        2025,
        "3",
        "https://www.fia.com/sites/default/files/documents/"
        "fia_2025_formula_1_technical_regulations_-_issue_03_-_2025-04-07.pdf",
    ),
    RegSource(
        "financial",
        2025,
        "25",
        _FIA + "2025_fia_formula_1_financial_regulations_-_issue_25_-_2025-07-31.pdf",
    ),
    RegSource(
        "sporting",
        2026,
        "9",
        _FIA + "fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf",
    ),
    RegSource(
        "technical",
        2026,
        "20",
        _FIA + "fia_2026_f1_regulations_-_section_c_technical_-_iss_20_-_2026-08-05.pdf",
    ),
    RegSource(
        "financial",
        2026,
        "8",
        _FIA + "fia_2026_f1_regulations_-_section_d_financial_-_f1_teams_-_iss_08_-_2026-10-01.pdf",
    ),
)

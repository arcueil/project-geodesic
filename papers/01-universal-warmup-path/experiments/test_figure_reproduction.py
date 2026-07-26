from __future__ import annotations

from pathlib import Path

import pytest

from verify_figure_reproduction import (
    FigureVerificationError,
    normalized_figure_bytes,
)


def test_pdf_comparison_normalizes_only_creation_date(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pdf"
    generated = tmp_path / "generated.pdf"
    reference.write_bytes(
        b"prefix /CreationDate (D:20260726120519+02'00') >> suffix"
    )
    generated.write_bytes(
        b"prefix /CreationDate (D:20260726133000+02'00') >> suffix"
    )

    assert normalized_figure_bytes(
        reference,
        "pdf_creation_date_normalized",
    ) == normalized_figure_bytes(
        generated,
        "pdf_creation_date_normalized",
    )


def test_pdf_comparison_rejects_missing_creation_date(tmp_path: Path) -> None:
    path = tmp_path / "figure.pdf"
    path.write_bytes(b"no metadata")

    with pytest.raises(FigureVerificationError):
        normalized_figure_bytes(path, "pdf_creation_date_normalized")

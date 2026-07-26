"""Analytic and tiny-render checks for the BBP calibration figure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from make_figure_bbp import bbp_lambda, bbp_overlap, render


def test_bbp_threshold_formulas() -> None:
    gamma = np.array([0.1, 1.0])

    eigenvalue = bbp_lambda(2.0, gamma)
    overlap = bbp_overlap(2.0, gamma)

    assert eigenvalue[0] > (1 + np.sqrt(gamma[0])) ** 2
    assert eigenvalue[1] == pytest.approx(4.0)
    assert overlap[0] > 0
    assert overlap[1] == pytest.approx(0.0)


def test_tiny_bbp_render_is_nonoverwriting(tmp_path: Path) -> None:
    output = tmp_path / "bbp.pdf"

    render(output, d=5, trials=1)

    assert output.stat().st_size > 0
    with pytest.raises(FileExistsError):
        render(output, d=5, trials=1)

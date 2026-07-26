"""Focused analytic checks for the configurable correlated-axis GMM."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).with_name("gmm_boundary.py")
_SPEC = importlib.util.spec_from_file_location("gmm_boundary", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gmm_boundary = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gmm_boundary
_SPEC.loader.exec_module(gmm_boundary)


def test_default_two_axis_construction_is_backward_compatible() -> None:
    spec = gmm_boundary.build_gmm(4.0)
    direction = np.array([1.0, 1.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)
    delta_sq = 4.0**2 * 22.0 / (1.0 + 0.3 * 0.7 * 4.0**2)
    expected_beta = 21.0 - 0.3 * 0.7 * delta_sq

    np.testing.assert_allclose(spec.mu1, -0.7 * np.sqrt(delta_sq) * direction)
    np.testing.assert_allclose(spec.mu2, 0.3 * np.sqrt(delta_sq) * direction)
    np.testing.assert_allclose(
        spec.Sigma_marginal, np.eye(5) + 21.0 * np.outer(direction, direction)
    )
    np.testing.assert_allclose(
        spec.Sigma_within, np.eye(5) + expected_beta * np.outer(direction, direction)
    )
    assert spec.correlated_axes == 2


@pytest.mark.parametrize(
    ("correlated_axes", "expected_spike", "expected_pairs"),
    [(2, 1.9130434782608696, 1), (3, 2.75, 3), (4, 3.52, 6), (5, 4.230769230769231, 10)],
)
def test_invariance_and_marginal_diagnostic_are_frozen(
    correlated_axes: int, expected_spike: float, expected_pairs: int
) -> None:
    max_err, spike, _ = gmm_boundary.verify_invariance(
        np.linspace(0.0, 10.0, 21), correlated_axes
    )
    diagnostic = gmm_boundary.analytic_diagnostic(correlated_axes)

    assert max_err < 1e-12
    assert spike == pytest.approx(expected_spike, abs=1e-12)
    assert diagnostic["marginal_whitened_spike"] == pytest.approx(
        expected_spike, abs=1e-12
    )
    assert diagnostic["off_diagonal_correlations"] == expected_pairs


@pytest.mark.parametrize("correlated_axes", [0, 1, 2.5, 6])
def test_invalid_correlated_axes_are_rejected(correlated_axes: int | float) -> None:
    with pytest.raises(ValueError, match="correlated_axes must be between 2 and 5"):
        gmm_boundary.build_gmm(1.0, correlated_axes)


def test_k3_within_spike_crosses_controller_threshold() -> None:
    sr_grid = np.linspace(0.0, 10.0, 21)
    _, marginal_spike, within_spikes = gmm_boundary.verify_invariance(sr_grid, 3)

    assert marginal_spike == pytest.approx(2.75, abs=1e-12)
    crossings = [
        (left, right)
        for left, right, left_spike, right_spike in zip(
            sr_grid[:-1], sr_grid[1:], within_spikes[:-1], within_spikes[1:]
        )
        if left_spike >= 2.0 > right_spike
    ]
    assert crossings == [(4.5, 5.0)]


def test_result_record_carries_correlated_axes_without_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    import blackjax.adaptation.staged_adaptation as staged_module

    def no_warmup(*args: object, **kwargs: object) -> object:
        raise RuntimeError("warmup deliberately disabled for record test")

    monkeypatch.setattr(staged_module, "staged_adaptation", no_warmup)
    record = gmm_boundary.run_point(
        1.5, seed=42, init_kind="broad", budget=10, correlated_axes=3
    )

    assert record["correlated_axes"] == 3
    assert {
        "metric_route_status",
        "metric_route_basis",
        "metric_scope",
        "observed_ensemble_evidence",
        "global_exploration",
        "handoff",
        "confidence_scope",
    } <= record.keys()
    assert record["error"] is not None
    assert "deliberately disabled" in record["error"]


def test_correlated_projection_uses_requested_axes() -> None:
    positions = np.array(
        [[np.sqrt(3.0), np.sqrt(3.0), np.sqrt(3.0), 10.0, 10.0]]
    )

    projection = gmm_boundary.correlated_projection(positions, 3)

    np.testing.assert_allclose(projection, [3.0])


def test_projected_transcript_decomposition_is_exact() -> None:
    spec = gmm_boundary.build_gmm(3.0, correlated_axes=3)
    positions = np.arange(4 * 3 * 5, dtype=float).reshape(4, 3, 5) / 10.0

    diagnostic = gmm_boundary.projected_transcript_decomposition(positions, spec)

    assert diagnostic["projected_partition_error"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostic["projected_target_variance"] == pytest.approx(22.0, abs=1e-12)
    assert diagnostic["projected_total_reference_error"] == pytest.approx(
        diagnostic["projected_total_variance"] - 22.0, abs=1e-12
    )


def test_append_jsonl_row_flushes_each_completed_row(tmp_path: Path) -> None:
    out_path = tmp_path / "incremental.jsonl"
    out_path.write_text("stale row\n")
    first = {
        "SR": 1.5,
        "value": np.nan,
        "flag": False,
        "nested": {"flag": True},
        "error": None,
    }
    failed = {"SR": 9.0, "value": None, "error": "synthetic failure"}

    with out_path.open("w") as fh:
        assert out_path.read_text() == ""

        gmm_boundary._append_jsonl_row(fh, first)
        assert [json.loads(line) for line in out_path.read_text().splitlines()] == [
            {
                "SR": 1.5,
                "value": None,
                "flag": False,
                "nested": {"flag": True},
                "error": None,
            }
        ]

        gmm_boundary._append_jsonl_row(fh, failed)
        assert [json.loads(line) for line in out_path.read_text().splitlines()] == [
            {
                "SR": 1.5,
                "value": None,
                "flag": False,
                "nested": {"flag": True},
                "error": None,
            },
            failed,
        ]


@pytest.mark.parametrize("mode", ["--smoke", "--sweep"])
def test_main_persists_smoke_and_sweep_rows_incrementally(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_path = tmp_path / f"{mode[2:]}.jsonl"
    calls: list[float] = []
    returned: list[dict] = []

    monkeypatch.setattr(gmm_boundary, "_SMOKE_SR", [1.5, 9.0])
    monkeypatch.setattr(gmm_boundary, "_SMOKE_SEEDS", [42])
    monkeypatch.setattr(gmm_boundary, "_SWEEP_SR", [1.5, 9.0])
    monkeypatch.setattr(gmm_boundary, "_SWEEP_SEEDS", [42])

    def fake_run_point(
        sr: float,
        seed: int,
        init_kind: str,
        budget: int,
        **kwargs: object,
    ) -> dict:
        if not calls:
            assert out_path.read_text() == ""
        else:
            persisted = [
                json.loads(line) for line in out_path.read_text().splitlines()
            ]
            assert len(persisted) == 1
            assert persisted[0]["SR"] == 1.5
        calls.append(sr)

        is_error = sr == 9.0
        record = {
            "SR": sr,
            "seed": seed,
            "init_kind": init_kind,
            "budget": budget,
            "route": None if is_error else "low_rank",
            "effective_rank": None if is_error else 1,
            "deferred_to_ensemble": None if is_error else False,
            "detection_branch": None if is_error else "pooled_within",
            "within_lam1": None if is_error else 2.1,
            "chain_collinearity": None if is_error else 0.8,
            "unimodality_gate": None if is_error else "pass",
            "mode_coverage": None if is_error else "multi_chain_uncertified",
            "split_rhat": None if is_error else 1.0,
            "mode_weight_est": None if is_error else 0.7,
            "min_ess_per_grad": None if is_error else 0.1,
            "both_modes_visited_frac": None if is_error else 1.0,
            "error": "synthetic failure" if is_error else None,
        }
        returned.append(record)
        return record

    monkeypatch.setattr(gmm_boundary, "run_point", fake_run_point)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gmm_boundary.py", mode, "--out", str(out_path)],
    )

    gmm_boundary.main()

    persisted = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert persisted == returned
    assert persisted[1]["error"] == "synthetic failure"
    output = capsys.readouterr().out
    assert "persisted_rows=1" in output
    assert "persisted_rows=2" in output


def test_main_refuses_to_replace_raw_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_path = tmp_path / "existing.jsonl"
    out_path.write_text("preserve me\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gmm_boundary.py", "--smoke", "--out", str(out_path)],
    )

    with pytest.raises(FileExistsError):
        gmm_boundary.main()

    assert out_path.read_text() == "preserve me\n"

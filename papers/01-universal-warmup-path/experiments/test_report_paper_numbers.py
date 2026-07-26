from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from report_paper_numbers import (
    ReportError,
    _fixed_efficiency,
    _k3_cluster_report,
    _ratio_contrasts,
    build_report,
)


HERE = Path(__file__).resolve().parent
REFERENCE_SUMMARY = (
    HERE
    / "results"
    / "paper1-full-29d246-20260726-v1-validation-v2"
    / "frozen_summary.json"
)
REFERENCE_RESULTS = (
    HERE / "results" / "paper1-full-29d246-20260726-v1"
)
REFERENCE_ATTESTATION = (
    HERE
    / "results"
    / "paper1-full-29d246-20260726-v1-validation-v2"
    / "validation_attestation.json"
)


class FixedEfficiencyReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))

    def test_recomputes_all_three_estimand_ratios(self) -> None:
        report = _fixed_efficiency(self.summary)
        first = report["cells"][0]
        comparator = first["comparators"]["fisher_low_rank"]
        ratios = comparator["paper_ratio_components"]
        automatic = first["automatic"]
        population = comparator["equal_split_population"]

        expected_marginal = [
            numerator / denominator
            for numerator, denominator in zip(
                automatic["marginal_per_chain"],
                population["marginal_per_chain"],
                strict=True,
            )
        ]
        self.assertEqual(
            ratios["marginal_per_chain_over_equal_split_population"],
            expected_marginal,
        )
        self.assertAlmostEqual(
            ratios[
                "pooled_population_total_over_equal_split_population"
            ],
            automatic["pooled_population_total"]
            / population["pooled_population_total"],
        )
        self.assertIn(
            "welford_diag",
            report["paper_ratios_by_model"]["german_credit"],
        )

    def test_paper_geometric_means_are_frozen(self) -> None:
        report = _fixed_efficiency(self.summary)
        expected = {
            "ill_cond_50": {
                "fisher_low_rank": (2.451, 2.324, 1.409),
                "welford_diag": (22.572, 21.124, 14.196),
            },
            "german_credit": {
                "fisher_low_rank": (1.951, 1.882, 1.131),
                "welford_diag": (6.264, 5.887, 6.009),
            },
        }
        for model, comparators in expected.items():
            for comparator, values in comparators.items():
                observed = report["paper_ratios_by_model"][model][comparator][
                    "geometric_mean"
                ]
                triple = (
                    observed[
                        "pooled_population_total_over_equal_split_population"
                    ],
                    observed[
                        "marginal_per_chain_over_equal_split_population"
                    ],
                    observed[
                        "one_output_total_over_historical_single_chain"
                    ],
                )
                for actual, target in zip(triple, values, strict=True):
                    self.assertAlmostEqual(actual, target, places=3)

    def test_rejects_an_inconsistent_stored_pooled_ratio(self) -> None:
        broken = copy.deepcopy(self.summary)
        broken["efficiency"]["cells"][0]["comparators"]["fisher_low_rank"][
            "automatic_to_manual_pooled_ratio"
        ] += 0.1
        with self.assertRaises(ReportError):
            _fixed_efficiency(broken)


class GmmAndLogRatioReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))

    def test_k3_clustered_report_reproduces_frozen_ablation_numbers(self) -> None:
        arm = next(
            arm
            for arm in self.summary["controlled_gmm"]["arms"]
            if arm["arm_id"] == "gmm_k3_matched_diagonal_60k"
        )
        report = _k3_cluster_report(arm)
        self.assertAlmostEqual(
            report["overall_geometric_mean"],
            1.036784308744642,
        )
        self.assertAlmostEqual(
            report["seed_clustered_log_t_interval_95"][0],
            0.9983244616478046,
        )
        self.assertAlmostEqual(
            report["seed_clustered_log_t_interval_95"][1],
            1.0767257982288356,
        )
        self.assertEqual(report["cluster_count"], 16)

    def test_log_ratio_transform_reports_ratio_and_percent(self) -> None:
        rows = _ratio_contrasts(
            [{"log_ratio": math.log(1.25)}],
            log_key="log_ratio",
            ratio_name="ratio",
            percent_name="percent",
        )
        self.assertAlmostEqual(rows[0]["ratio"], 1.25)
        self.assertAlmostEqual(rows[0]["percent"], 25.0)


class FrozenManuscriptHeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report(REFERENCE_SUMMARY, REFERENCE_RESULTS)
        cls.headlines = cls.report[
            "manuscript_headlines"
        ]

    def test_authenticates_packaged_summary_and_attestation(self) -> None:
        trusted = self.report["verification"]["trusted_reference"]
        self.assertEqual(
            trusted["frozen_summary"]["sha256"],
            (
                "94136b0e528b883148540efdbe43cd24ebd39bcaa905b86bd73a8c8d4763f24a"
            ),
        )
        self.assertEqual(
            trusted["validation_attestation"]["sha256"],
            (
                "85b60b2958cfc3395828c0236a77d1fb4ee7ded16a4249e499fff339cee60b02"
            ),
        )

    def test_authentication_is_content_based_for_copied_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied_summary = root / "frozen_summary.json"
            copied_attestation = root / "validation_attestation.json"
            copied_summary.write_bytes(REFERENCE_SUMMARY.read_bytes())
            copied_attestation.write_bytes(REFERENCE_ATTESTATION.read_bytes())
            copied = build_report(
                copied_summary,
                REFERENCE_RESULTS,
                copied_attestation,
            )
        self.assertEqual(
            copied["manuscript_headlines"],
            self.report["manuscript_headlines"],
        )
        self.assertEqual(
            copied["verification"]["trusted_reference"],
            self.report["verification"]["trusted_reference"],
        )

    def test_rejects_a_mutated_summary_before_reporting(self) -> None:
        mutated = bytearray(REFERENCE_SUMMARY.read_bytes())
        marker = b'"status": "valid"'
        self.assertIn(marker, mutated)
        start = mutated.index(marker)
        mutated[start : start + len(marker)] = b'"status": "bogus"'
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "frozen_summary.json"
            summary_path.write_bytes(mutated)
            with self.assertRaisesRegex(
                ReportError,
                "trusted release SHA-256",
            ):
                build_report(
                    summary_path,
                    REFERENCE_RESULTS,
                    REFERENCE_ATTESTATION,
                )

    def test_rejects_a_mutated_attestation_before_reporting(self) -> None:
        mutated = bytearray(REFERENCE_ATTESTATION.read_bytes())
        marker = b'"status": "valid"'
        self.assertIn(marker, mutated)
        start = mutated.index(marker)
        mutated[start : start + len(marker)] = b'"status": "bogus"'
        with tempfile.TemporaryDirectory() as directory:
            attestation_path = Path(directory) / "validation_attestation.json"
            attestation_path.write_bytes(mutated)
            with self.assertRaisesRegex(
                ReportError,
                "trusted release SHA-256",
            ):
                build_report(
                    REFERENCE_SUMMARY,
                    REFERENCE_RESULTS,
                    attestation_path,
                )

    def test_fixed_and_shared_step_headlines(self) -> None:
        fixed = self.headlines["fixed_efficiency"]
        self.assertEqual(
            fixed["automatic_low_rank_routes"],
            {"count": 12, "total": 12},
        )
        self.assertEqual(
            fixed["declared_population_quality_gates"],
            {"passed": 36, "total": 36},
        )
        self.assertEqual(fixed["warmup_divergences"], 4996)
        self.assertEqual(fixed["post_warmup_divergences"], 0)

        shared = self.headlines["shared_step_size"][
            "warmup_gradient_ratio_historical_over_current_by_model"
        ]
        self.assertEqual(
            shared["ill_cond_50"],
            [19.381019074296386, 19.8931011681728],
        )
        self.assertEqual(
            shared["german_credit"],
            [31.817934532042415, 35.604386295180724],
        )

    def test_gmm_headlines(self) -> None:
        gmm = self.headlines["controlled_gmm"]
        self.assertEqual(
            gmm["k2"]["majority_population_handoff_onset_SR_by_budget"],
            {"20k": 5.0, "60k": 6.0, "120k": 7.0},
        )
        regime = {
            row["SR"]: row for row in gmm["k2"]["primary_regime_table"]
        }
        self.assertEqual(regime[7.0]["persistent_disagreement_count"], 3)
        self.assertEqual(regime[7.0]["population_handoff_count"], 2)
        self.assertEqual(
            [cell["route"] for cell in gmm["k2"]["single_chain"]],
            ["low_rank", "diagonal", "diagonal"],
        )
        self.assertEqual(gmm["k2"]["single_chain"][-1]["mode_weight_est"], 1.0)
        self.assertEqual(
            [
                cell["ratio"]
                for cell in gmm["k2"][
                    "matched_diagonal_low_rank_over_diagonal_ratios"
                ]
            ],
            [
                0.8692303407998037,
                0.9977211099369621,
                1.1702882661506633,
                0.997752395362499,
            ],
        )

        persistent = gmm["k3"]["primary_persistent_suffix"]
        self.assertEqual(
            persistent["first_SR_with_all_seeds_persistent"],
            6.0,
        )
        self.assertTrue(
            all(
                row["population_handoff_count"] == 2
                and row["total"] == 3
                for row in persistent[
                    "population_handoff_counts_from_that_SR"
                ]
            )
        )
        completion = gmm["k3"]["matched_diagonal"]["completion_report"]
        self.assertEqual((completion["completed"], completion["total"]), (64, 64))
        self.assertEqual(
            (
                completion["paired_ratio_above_one"],
                completion["paired_ratio_below_one"],
            ),
            (39, 25),
        )
        self.assertEqual(completion["low_rank_divergences_total"], 0)
        self.assertEqual(completion["matched_diagonal_divergences_total"], 0)
        self.assertAlmostEqual(
            completion["maximum_projection_split_rhat"],
            1.0056683151303065,
        )
        self.assertEqual(
            set(completion["valid_low_rank_profiles_by_SR"].values()),
            {16},
        )

    def test_restart_kernel_and_schedule_headlines(self) -> None:
        restart = self.headlines["restart_ablation"]
        self.assertEqual(
            (restart["continuous_wins"], restart["total_pairs"]),
            (3, 6),
        )
        self.assertEqual(
            restart["continuous_over_reseed_ratio_range"],
            [0.9464399539971725, 1.1385261705232919],
        )
        self.assertAlmostEqual(
            restart["continuous_over_reseed_geometric_mean_by_model"][
                "ill_cond_50"
            ],
            1.0138503781529107,
        )
        self.assertAlmostEqual(
            restart["continuous_over_reseed_geometric_mean_by_model"][
                "german_credit"
            ],
            1.024371205388064,
        )

        kernel = self.headlines["kernel_family"]
        self.assertEqual(
            (kernel["low_rank_routes"], kernel["total_cells"]),
            (12, 12),
        )
        self.assertEqual(kernel["automatic_post_warmup_divergences"], 0)
        self.assertEqual(
            (
                kernel["multinomial_losses_to_comparator"],
                kernel["multinomial_cells"],
            ),
            (2, 6),
        )

        failing = self.headlines["schedule_configuration"][
            "radon_failing_configuration"
        ]
        self.assertEqual(failing["schedule_family"], "proportional_growing")
        self.assertEqual(failing["buffer_policy"], "accumulating")
        self.assertEqual(failing["recompute_every"], 1)
        self.assertEqual(
            [cell["seed"] for cell in failing["cells"]],
            [0, 1, 2],
        )
        self.assertTrue(
            all(
                "run_id" not in cell and "cell_id" not in cell
                for cell in failing["cells"]
            )
        )
        self.assertEqual(len(failing["predeclared_contrasts"]), 6)


if __name__ == "__main__":
    unittest.main()

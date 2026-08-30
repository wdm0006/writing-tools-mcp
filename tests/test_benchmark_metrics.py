"""Exact-value tests for the benchmark metric calculations.

Every expected number here is hand-computed from the fixture rather than read
off a run, so a change in the metric code fails rather than silently
re-baselining the report.
"""

import math

import pytest

from benchmarks import metrics, report

HUMAN = metrics.HUMAN
MACHINE = metrics.MACHINE


def make_record(doc_id, label, stylometry=None, perplexity=None):
    record = {"doc_id": doc_id, "label": label, "source": {}}
    if stylometry is not None:
        record["stylometry"] = stylometry
    if perplexity is not None:
        record["perplexity"] = perplexity
    return record


def ok_stylometry(flagged, confidence, features=None, z_scores=None):
    return {
        "ok": True,
        "high_ai_probability": flagged,
        "confidence_score": confidence,
        "ai_indicators": [],
        "features": features or {},
        "z_scores": z_scores or {},
    }


class TestConfusionMatrix:
    def test_counts_each_cell_exactly(self):
        pairs = [
            (MACHINE, True),
            (MACHINE, True),
            (MACHINE, False),
            (HUMAN, True),
            (HUMAN, False),
            (HUMAN, False),
            (HUMAN, False),
        ]
        assert metrics.confusion_matrix(pairs) == {"tp": 2, "fn": 1, "fp": 1, "tn": 3}

    def test_empty_input_is_all_zeros(self):
        assert metrics.confusion_matrix([]) == {"tp": 0, "fn": 0, "fp": 0, "tn": 0}

    def test_unknown_label_is_rejected(self):
        with pytest.raises(ValueError):
            metrics.confusion_matrix([("robot", True)])


class TestClassificationMetrics:
    def test_exact_values(self):
        values = metrics.classification_metrics({"tp": 2, "fn": 1, "fp": 1, "tn": 3})
        assert values["accuracy"] == pytest.approx(5 / 7)
        assert values["precision"] == pytest.approx(2 / 3)
        assert values["recall"] == pytest.approx(2 / 3)
        assert values["f1"] == pytest.approx(2 / 3)
        assert values["fpr"] == pytest.approx(1 / 4)
        assert values["tpr"] == pytest.approx(2 / 3)
        assert values["specificity"] == pytest.approx(3 / 4)

    def test_asymmetric_case_distinguishes_precision_from_recall(self):
        values = metrics.classification_metrics({"tp": 3, "fn": 7, "fp": 1, "tn": 9})
        assert values["precision"] == pytest.approx(3 / 4)
        assert values["recall"] == pytest.approx(3 / 10)
        assert values["f1"] == pytest.approx(2 * 0.75 * 0.3 / (0.75 + 0.3))
        assert values["fpr"] == pytest.approx(1 / 10)
        assert values["accuracy"] == pytest.approx(12 / 20)

    def test_undefined_rates_are_none_not_zero(self):
        values = metrics.classification_metrics({"tp": 0, "fn": 0, "fp": 0, "tn": 5})
        assert values["recall"] is None
        assert values["precision"] is None
        assert values["f1"] is None
        assert values["fpr"] == pytest.approx(0.0)

    def test_no_predictions_gives_zero_f1_not_none(self):
        values = metrics.classification_metrics({"tp": 0, "fn": 4, "fp": 0, "tn": 4})
        assert values["precision"] is None
        assert values["recall"] == pytest.approx(0.0)
        assert values["f1"] is None


class TestFailureHandling:
    """Failed analyses must be excluded from the matrix, never counted as negatives."""

    @staticmethod
    def records():
        return [
            make_record("m1", MACHINE, stylometry=ok_stylometry(True, 0.8)),
            make_record("m2", MACHINE, stylometry=ok_stylometry(False, 0.2)),
            make_record("m3", MACHINE, stylometry={"ok": False, "error": "Analysis failed: boom"}),
            make_record("h1", HUMAN, stylometry=ok_stylometry(True, 0.9)),
            make_record("h2", HUMAN, stylometry=ok_stylometry(False, 0.1)),
            make_record("h3", HUMAN, stylometry={"ok": False, "error": "Empty text provided"}),
            make_record("h4", HUMAN),  # analysis never ran at all
        ]

    def test_partition_splits_scored_from_failed(self):
        scored, failed = metrics.partition_outcomes(self.records(), "stylometry")
        assert [record["doc_id"] for record in scored] == ["m1", "m2", "h1", "h2"]
        assert [record["doc_id"] for record in failed] == ["m3", "h3", "h4"]

    def test_failures_are_excluded_from_the_matrix(self):
        evaluation = metrics.evaluate_boolean_decision(self.records(), "stylometry")
        assert evaluation["n_total"] == 7
        assert evaluation["n_scored"] == 4
        assert evaluation["n_failed"] == 3
        assert evaluation["scored_label_counts"] == {HUMAN: 2, MACHINE: 2}
        assert evaluation["failed_label_counts"] == {HUMAN: 2, MACHINE: 1}
        # Folding the three failures in as "predicted human" would give
        # {"tp": 1, "fn": 2, "fp": 1, "tn": 3} and halve the reported FPR.
        assert evaluation["confusion_matrix"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
        assert evaluation["metrics"]["fpr"] == pytest.approx(0.5)
        assert evaluation["metrics"]["accuracy"] == pytest.approx(0.5)

    def test_feature_summaries_ignore_failed_documents(self):
        records = self.records()
        records[0]["stylometry"]["features"] = {"ttr": 0.4}
        records[1]["stylometry"]["features"] = {"ttr": 0.6}
        records[3]["stylometry"]["features"] = {"ttr": 0.8}
        records[4]["stylometry"]["features"] = {"ttr": 1.0}
        records[2]["stylometry"]["features"] = {"ttr": 99.0}  # failed: must not be counted
        summaries = metrics.feature_separation(records, "stylometry", "features")
        assert len(summaries) == 1
        assert summaries[0]["feature"] == "ttr"
        assert summaries[0]["machine"]["n"] == 2
        assert summaries[0]["machine"]["mean"] == pytest.approx(0.5)
        assert summaries[0]["human"]["mean"] == pytest.approx(0.9)


class TestThresholdSweep:
    @staticmethod
    def pairs():
        return [
            (MACHINE, 0.9),
            (MACHINE, 0.7),
            (MACHINE, 0.7),
            (MACHINE, 0.2),
            (HUMAN, 0.7),
            (HUMAN, 0.2),
            (HUMAN, 0.0),
            (HUMAN, 0.0),
        ]

    def test_sweep_uses_only_observed_scores(self):
        sweep = metrics.threshold_sweep(self.pairs())
        assert [point["threshold"] for point in sweep] == [0.0, 0.2, 0.7, 0.9]

    def test_sweep_values_are_exact(self):
        sweep = {point["threshold"]: point for point in metrics.threshold_sweep(self.pairs())}
        assert sweep[0.9]["confusion_matrix"] == {"tp": 1, "fn": 3, "fp": 0, "tn": 4}
        assert sweep[0.7]["confusion_matrix"] == {"tp": 3, "fn": 1, "fp": 1, "tn": 3}
        assert sweep[0.2]["confusion_matrix"] == {"tp": 4, "fn": 0, "fp": 2, "tn": 2}
        assert sweep[0.0]["confusion_matrix"] == {"tp": 4, "fn": 0, "fp": 4, "tn": 0}

    def test_fpr_at_tpr_targets_reports_achieved_not_requested(self):
        rows = metrics.fpr_at_tpr_targets(self.pairs(), [0.50, 0.75, 0.90])
        by_target = {row["target_tpr"]: row for row in rows}

        # 0.75 is exactly reachable at threshold 0.7 (3 of 4 machine documents).
        assert by_target[0.75]["threshold"] == pytest.approx(0.7)
        assert by_target[0.75]["achieved_tpr"] == pytest.approx(0.75)
        assert by_target[0.75]["fpr"] == pytest.approx(0.25)

        # 0.50 is met by the same threshold - the highest one still at or above target.
        assert by_target[0.50]["threshold"] == pytest.approx(0.7)
        assert by_target[0.50]["achieved_tpr"] == pytest.approx(0.75)

        # 0.90 overshoots to 1.00: the scores are discrete, so it is unreachable exactly.
        assert by_target[0.90]["threshold"] == pytest.approx(0.2)
        assert by_target[0.90]["achieved_tpr"] == pytest.approx(1.0)
        assert by_target[0.90]["fpr"] == pytest.approx(0.5)

    def test_unreachable_target_reports_none(self):
        rows = metrics.fpr_at_tpr_targets([(MACHINE, 0.1), (HUMAN, 0.9)], [0.9])
        assert rows[0]["threshold"] == pytest.approx(0.1)

        rows = metrics.fpr_at_tpr_targets([(HUMAN, 0.5)], [0.5])
        assert rows[0] == {"target_tpr": 0.5, "threshold": None, "achieved_tpr": None, "fpr": None}


class TestDescriptiveStatistics:
    def test_describe_exact_values(self):
        stats = metrics.describe([1.0, 2.0, 3.0, 4.0])
        assert stats["n"] == 4
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["std"] == pytest.approx(math.sqrt(5 / 3))
        assert stats["median"] == pytest.approx(2.5)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(4.0)

    def test_odd_length_median_is_the_middle_value(self):
        assert metrics.describe([5.0, 1.0, 3.0])["median"] == pytest.approx(3.0)

    def test_single_value_has_no_standard_deviation(self):
        stats = metrics.describe([7.0])
        assert stats["n"] == 1
        assert stats["std"] is None

    def test_empty_input_reports_none_not_zero(self):
        assert metrics.describe([]) == {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}

    def test_cohens_d_exact_value(self):
        human = metrics.describe([1.0, 2.0, 3.0])
        machine = metrics.describe([3.0, 4.0, 5.0])
        # Both sides have sample std 1.0, so pooled std is 1.0 and d is the raw gap.
        assert metrics.cohens_d(human, machine) == pytest.approx(2.0)

    def test_cohens_d_is_none_without_spread(self):
        constant = metrics.describe([2.0, 2.0])
        assert metrics.cohens_d(constant, constant) is None
        assert metrics.cohens_d(metrics.describe([1.0]), metrics.describe([1.0, 2.0])) is None


class TestFeatureSeparation:
    def test_direction_and_effect_size(self):
        records = [
            make_record("h1", HUMAN, stylometry=ok_stylometry(False, 0.0, features={"ttr": 0.5, "fog": 12.0})),
            make_record("h2", HUMAN, stylometry=ok_stylometry(False, 0.0, features={"ttr": 0.7, "fog": 14.0})),
            make_record("m1", MACHINE, stylometry=ok_stylometry(True, 1.0, features={"ttr": 0.3, "fog": 12.0})),
            make_record("m2", MACHINE, stylometry=ok_stylometry(True, 1.0, features={"ttr": 0.5, "fog": 14.0})),
        ]
        summaries = {
            summary["feature"]: summary for summary in metrics.feature_separation(records, "stylometry", "features")
        }
        assert summaries["ttr"]["direction"] == "lower in machine"
        assert summaries["ttr"]["mean_difference"] == pytest.approx(-0.2)
        assert summaries["fog"]["direction"] == "no difference"
        assert summaries["fog"]["mean_difference"] == pytest.approx(0.0)
        assert summaries["fog"]["cohens_d"] == pytest.approx(0.0)

    def test_missing_and_non_numeric_values_are_skipped_per_feature(self):
        records = [
            make_record("h1", HUMAN, stylometry=ok_stylometry(False, 0.0, features={"a": 1.0, "b": None})),
            make_record("h2", HUMAN, stylometry=ok_stylometry(False, 0.0, features={"a": 3.0, "b": 2.0, "c": "text"})),
            make_record("m1", MACHINE, stylometry=ok_stylometry(True, 1.0, features={"a": 5.0, "b": 4.0})),
            make_record("m2", MACHINE, stylometry=ok_stylometry(True, 1.0, features={"a": 7.0, "b": float("nan")})),
        ]
        summaries = {
            summary["feature"]: summary for summary in metrics.feature_separation(records, "stylometry", "features")
        }
        assert set(summaries) == {"a", "b"}
        assert summaries["a"]["human"]["n"] == 2
        assert summaries["b"]["human"]["n"] == 1
        assert summaries["b"]["machine"]["n"] == 1
        assert summaries["b"]["human"]["std"] is None

    def test_booleans_are_not_treated_as_numbers(self):
        records = [
            make_record("h1", HUMAN, stylometry=ok_stylometry(False, 0.0, features={"flagged": False})),
            make_record("m1", MACHINE, stylometry=ok_stylometry(True, 1.0, features={"flagged": True})),
        ]
        assert metrics.feature_separation(records, "stylometry", "features") == []


class TestReportRendering:
    @staticmethod
    def records():
        return [
            make_record(
                "h1",
                HUMAN,
                stylometry=ok_stylometry(False, 0.1, features={"ttr": 0.5}),
                perplexity={
                    "ok": True,
                    "high_ai_probability": False,
                    "measurements": {"doc_ppl": 40.0, "doc_burstiness": 12.0},
                },
            ),
            make_record(
                "m1",
                MACHINE,
                stylometry=ok_stylometry(True, 0.9, features={"ttr": 0.3}),
                perplexity={
                    "ok": True,
                    "high_ai_probability": True,
                    "measurements": {"doc_ppl": 12.0, "doc_burstiness": None},
                },
            ),
            make_record(
                "m2",
                MACHINE,
                stylometry={"ok": False, "error": "Analysis failed: boom"},
                perplexity={"ok": False, "error": "Analysis failed: boom"},
            ),
        ]

    def build(self):
        return report.build_report(
            self.records(),
            corpus_info={"name": "fixture", "description": "3 documents", "manifest": "n/a"},
            thresholds={"stylometry.thresholds.ai_confidence_threshold": 0.7},
            methods=["stylometry", "perplexity"],
        )

    def test_report_states_the_positive_class_and_the_disclaimer(self):
        markdown = self.build()
        assert "Positive class is **machine-generated text**" in markdown
        assert report.DISCLAIMER in markdown

    def test_report_carries_counts_matrices_and_failures(self):
        markdown = self.build()
        assert "1 failed" in markdown
        assert "1x `Analysis failed: boom`" in markdown
        assert "| **actual machine** | 1 (TP) | 0 (FN) |" in markdown
        assert "| **actual human** | 0 (FP) | 1 (TN) |" in markdown
        assert "`doc_burstiness` null on 1 of 2 scored documents." in markdown

    def test_report_includes_feature_direction_and_sweep(self):
        markdown = self.build()
        assert "lower in machine" in markdown
        assert "## Stylometry FPR at declared TPR targets" in markdown
        assert "at least two named AI indicators" in markdown

    def test_report_is_deterministic(self):
        assert self.build() == self.build()


class TestNullValueCounts:
    def test_counts_only_nulls_on_successful_analyses(self):
        records = [
            make_record("h1", HUMAN, stylometry=ok_stylometry(False, 0.0, features={"a": 1.0, "b": None})),
            make_record("h2", HUMAN, stylometry=ok_stylometry(False, 0.0, features={"a": 2.0, "b": None})),
            make_record("m1", MACHINE, stylometry=ok_stylometry(True, 1.0, features={"a": None, "b": None})),
            make_record("m2", MACHINE, stylometry={"ok": False, "error": "boom", "features": {"a": None}}),
        ]
        # `a` is null once and `b` on every scored document; the failed record contributes nothing.
        assert metrics.null_value_counts(records, "stylometry", "features") == {"a": 1, "b": 3}

    def test_fully_measured_block_reports_nothing(self):
        records = [make_record("h1", HUMAN, stylometry=ok_stylometry(False, 0.0, features={"a": 1.0}))]
        assert metrics.null_value_counts(records, "stylometry", "features") == {}

    def test_report_names_a_never_measurable_key(self):
        records = [
            make_record(
                "h1", HUMAN, stylometry={**ok_stylometry(False, 0.1), "measurements": {"char_ngram_similarity": None}}
            ),
            make_record(
                "m1", MACHINE, stylometry={**ok_stylometry(True, 0.9), "measurements": {"char_ngram_similarity": None}}
            ),
        ]
        markdown = report.build_report(
            records,
            corpus_info={"name": "fixture", "description": "2 documents", "manifest": "n/a"},
            thresholds={},
            methods=["stylometry"],
        )
        # Without this note the key would simply be absent from the table, which
        # reads identically to it never having been computed.
        assert "`char_ngram_similarity` (2 of 2)" in markdown

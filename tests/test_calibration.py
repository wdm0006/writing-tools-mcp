"""Hand-computed acceptance tests for the published calibration generator.

Every expected number below is computed by hand from the seven-document
fixture, not read off a run, so a change in the metric or rendering code fails
loudly instead of silently re-baselining the published table. The fixture
mirrors the record shape ``benchmarks/runner.py`` commits: per-method blocks
with ``ok``, ``measurements``, and the recorded boolean decision.
"""

import json

import pytest

from benchmarks import calibrate_detection
from benchmarks.calibrate_detection import (
    load_scores,
    main,
    measurement_pairs,
    render_calibration,
)

HUMAN = "human"
MACHINE = "machine"


def stylometry_block(confidence, flagged=None, ok=True):
    """A stylometry result in the shape the runner commits to scores.jsonl."""
    if not ok:
        return {"ok": False, "error": "Analysis failed: boom"}
    return {
        "ok": True,
        "high_ai_probability": bool(flagged),
        "measurements": {"confidence_score": confidence, "char_ngram_similarity": None},
    }


def perplexity_block(doc_ppl, flagged=None, ok=True):
    """A perplexity result in the shape the runner commits to scores.jsonl."""
    if not ok:
        return {"ok": False, "error": "Analysis failed: boom"}
    return {
        "ok": True,
        "high_ai_probability": bool(flagged),
        "measurements": {"doc_ppl": doc_ppl, "doc_burstiness": 5.0},
    }


def make_record(
    doc_id, label, confidence=None, flagged_styl=None, doc_ppl=None, flagged_ppl=None, styl_ok=True, ppl_ok=True
):
    record = {"doc_id": doc_id, "label": label, "source": {}}
    record["stylometry"] = stylometry_block(confidence, flagged_styl, ok=styl_ok)
    record["perplexity"] = perplexity_block(doc_ppl, flagged_ppl, ok=ppl_ok)
    return record


@pytest.fixture()
def records():
    """Seven documents whose metrics are worked out by hand in this module.

    Stylometry sweep pairs (r6's null measurement and r7's failure drop out):
        (h, 0.1), (h, 0.3), (h, 0.5), (m, 0.7), (m, 0.9)
    Perplexity sweep pairs, machine-oriented (negated doc_ppl):
        (h, -30), (h, -40), (m, -10), (m, -20), (m, -15)
    """
    return [
        make_record("r1", HUMAN, confidence=0.1, flagged_styl=False, doc_ppl=30.0, flagged_ppl=False),
        make_record("r2", HUMAN, confidence=0.3, flagged_styl=False, doc_ppl=40.0, flagged_ppl=False),
        # r3: analyzed fine, but doc_ppl unmeasurable - excluded from the ppl sweep only.
        make_record("r3", HUMAN, confidence=0.5, flagged_styl=False, doc_ppl=None, flagged_ppl=False),
        make_record("r4", MACHINE, confidence=0.7, flagged_styl=True, doc_ppl=10.0, flagged_ppl=True),
        make_record("r5", MACHINE, confidence=0.9, flagged_styl=True, doc_ppl=20.0, flagged_ppl=True),
        # r6: analyzed fine, but confidence_score null - excluded from the stylometry sweep only.
        make_record("r6", MACHINE, confidence=None, flagged_styl=False, doc_ppl=15.0, flagged_ppl=True),
        # r7: both analyses failed - excluded everywhere, never "predicted human".
        make_record("r7", HUMAN, styl_ok=False, ppl_ok=False),
    ]


class TestMeasurementPairs:
    def test_drops_failures_and_nulls_and_orients_machine_higher(self, records):
        stylometry = measurement_pairs(records, "stylometry")
        assert sorted(stylometry) == [(HUMAN, 0.1), (HUMAN, 0.3), (HUMAN, 0.5), (MACHINE, 0.7), (MACHINE, 0.9)]

        perplexity = measurement_pairs(records, "perplexity")
        # doc_ppl is negated so the machine class scores higher, the helpers' convention.
        assert sorted(perplexity) == [
            (HUMAN, -40.0),
            (HUMAN, -30.0),
            (MACHINE, -20.0),
            (MACHINE, -15.0),
            (MACHINE, -10.0),
        ]

    def test_unknown_label_is_rejected_by_the_sweep_math(self):
        """load_scores guards labels on ingest; the metrics math defends too."""
        bad = [("uncertain", 0.5)]
        with pytest.raises(ValueError, match="Unknown label"):
            calibrate_detection.metrics.threshold_sweep(bad)


class TestSweepMetrics:
    def test_stylometry_sweep_matches_hand_computed_values(self, records):
        pairs = measurement_pairs(records, "stylometry")
        sweep = calibrate_detection.metrics.threshold_sweep(pairs)
        # (cut, precision, recall, fpr, tp, fn, fp, tn), hand-computed.
        expected = [
            (0.1, 2 / 5, 1.0, 1.0, 2, 0, 3, 0),
            (0.3, 2 / 4, 1.0, 2 / 3, 2, 0, 2, 1),
            (0.5, 2 / 3, 1.0, 1 / 3, 2, 0, 1, 2),
            (0.7, 2 / 2, 1.0, 0.0, 2, 0, 0, 3),
            (0.9, 1 / 1, 0.5, 0.0, 1, 1, 0, 3),
        ]
        for (threshold, precision, recall, fpr, tp, fn, fp, tn), point in zip(expected, sweep, strict=False):
            assert point["threshold"] == threshold
            assert point["confusion_matrix"] == {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
            assert point["metrics"]["precision"] == pytest.approx(precision)
            assert point["metrics"]["recall"] == pytest.approx(recall)
            assert point["metrics"]["fpr"] == pytest.approx(fpr)

    def test_perplexity_targets_render_doc_ppl_cuts(self, records):
        """The published perplexity cut is doc_ppl <= t, reached from negated scores."""
        pairs = measurement_pairs(records, "perplexity")
        rows = calibrate_detection.metrics.fpr_at_tpr_targets(pairs, calibrate_detection.TPR_TARGETS)

        # TPR 0.50 is cheapest at doc_ppl <= 15 (recall 2/3); 0.75 and 0.90 at doc_ppl <= 20 (recall 1.0).
        assert [
            (row["threshold"], calibrate_detection._display_cut("perplexity", row["threshold"])) for row in rows
        ] == [
            (-15.0, "<= 15.00"),
            (-20.0, "<= 20.00"),
            (-20.0, "<= 20.00"),
        ]
        assert rows[0]["achieved_tpr"] == pytest.approx(2 / 3)
        assert rows[0]["fpr"] == 0.0
        assert rows[0]["confusion_matrix"] == {"tp": 2, "fn": 1, "fp": 0, "tn": 2}

    def test_stylometry_targets_are_cheapest_reaching_cuts(self, records):
        pairs = measurement_pairs(records, "stylometry")
        rows = calibrate_detection.metrics.fpr_at_tpr_targets(pairs, calibrate_detection.TPR_TARGETS)
        # 0.50 reaches at the 0.9 cut (recall exactly 0.5); 0.75 and 0.90 at the 0.7 cut (recall 1.0).
        assert [row["threshold"] for row in rows] == [0.9, 0.7, 0.7]
        assert all(row["fpr"] == 0.0 for row in rows)


class TestShippedDecisions:
    def test_stylometry_shipped_decision_excludes_failures_not_as_negatives(self, records):
        evaluation = calibrate_detection.metrics.evaluate_boolean_decision(records, "stylometry")
        # r7 (failed) is excluded; r6 (null measurement, flag false) stays and counts as a FN.
        assert evaluation["confusion_matrix"] == {"tp": 2, "fn": 1, "fp": 0, "tn": 3}
        assert evaluation["metrics"]["precision"] == pytest.approx(1.0)
        assert evaluation["metrics"]["recall"] == pytest.approx(2 / 3)
        assert evaluation["metrics"]["fpr"] == 0.0
        assert evaluation["n_failed"] == 1

    def test_perplexity_shipped_decision(self, records):
        evaluation = calibrate_detection.metrics.evaluate_boolean_decision(records, "perplexity")
        # Scored set is r1-r6 (r7 failed): all three machines flagged, no human is.
        assert evaluation["confusion_matrix"] == {"tp": 3, "fn": 0, "fp": 0, "tn": 3}
        assert evaluation["metrics"]["recall"] == pytest.approx(1.0)


class TestRendering:
    def test_render_is_deterministic(self, records):
        first = render_calibration(
            records, calibrate_detection.DEFAULT_SCORES, provenance={"corpus": "c", "baseline": "b"}
        )
        second = render_calibration(
            records, calibrate_detection.DEFAULT_SCORES, provenance={"corpus": "c", "baseline": "b"}
        )
        assert first == second

    def test_render_contains_hand_computed_rows(self, records):
        text = render_calibration(records, calibrate_detection.DEFAULT_SCORES)
        # Stylometry full sweep rows, from the hand-computed table above.
        assert "| >= 0.700 | 1.0000 | 1.0000 | 0.0000 | 2 | 0 | 0 | 3 |" in text
        assert "| >= 0.100 | 0.4000 | 1.0000 | 1.0000 | 2 | 0 | 3 | 0 |" in text
        # Perplexity target rows render as doc_ppl <= cuts.
        assert "<= 15.00" in text
        assert "<= 20.00" in text
        # The doc names its positive class and its drift rule.
        assert "machine-generated text" in text
        assert "calibrate_detection.py" in text


class TestLoadScores:
    def test_loads_and_skips_blank_lines(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        path.write_text(
            json.dumps({"doc_id": "a", "label": "human"})
            + "\n\n"
            + json.dumps({"doc_id": "b", "label": "machine"})
            + "\n",
            encoding="utf-8",
        )
        records = load_scores(path)
        assert [r["doc_id"] for r in records] == ["a", "b"]

    def test_rejects_missing_and_unknown_fields(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        path.write_text(json.dumps({"label": "human"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="doc_id"):
            load_scores(path)

        path.write_text(json.dumps({"doc_id": "x", "label": "synthetic"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown label"):
            load_scores(path)


class TestMain:
    def test_write_then_check_round_trip(self, records, tmp_path):
        scores = tmp_path / "scores.jsonl"
        scores.write_text("\n".join(calibrate_detection.json.dumps(r) for r in records) + "\n", encoding="utf-8")
        out = tmp_path / "docs" / "calibration.md"

        assert main(["--scores", str(scores), "--metadata", str(tmp_path / "missing.json"), "--out", str(out)]) == 0
        first = out.read_bytes()

        # Regeneration is byte-identical, and --check passes while the file is current.
        assert main(["--scores", str(scores), "--metadata", str(tmp_path / "missing.json"), "--out", str(out)]) == 0
        assert out.read_bytes() == first
        assert (
            main(["--scores", str(scores), "--metadata", str(tmp_path / "missing.json"), "--out", str(out), "--check"])
            == 0
        )

        # A hand edit (or a stale table after a threshold change) is drift.
        out.write_bytes(first + b"stale\n")
        assert main(["--scores", str(scores), "--out", str(out), "--check"]) == 1

    def test_check_fails_when_published_file_missing(self, records, tmp_path):
        scores = tmp_path / "scores.jsonl"
        scores.write_text(calibrate_detection.json.dumps(records[0]) + "\n", encoding="utf-8")
        assert main(["--scores", str(scores), "--out", str(tmp_path / "never-written.md"), "--check"]) == 1


@pytest.mark.skipif(
    not calibrate_detection.DEFAULT_SCORES.exists() or not calibrate_detection.DEFAULT_OUT.exists(),
    reason="committed benchmark scores or published table not present (e.g. mutation shadow tree)",
)
class TestCommittedTable:
    def test_published_table_reproduces_byte_identically(self, tmp_path):
        """The acceptance gate: docs/calibration.md is a pure function of the committed scores."""
        fresh = tmp_path / "calibration.md"
        assert main(["--out", str(fresh)]) == 0
        assert fresh.read_bytes() == calibrate_detection.DEFAULT_OUT.read_bytes()

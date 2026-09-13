"""W9 preference-evaluation tests: statistics, aggregation, deterministic pipeline.

The full-pipeline test reruns the evaluation on the committed corpus and
compares the result to the committed ``results.json`` byte for byte (after
canonical serialization) — that equivalence IS the determinism guarantee.
No GPU, no network: the GPT-2 manager stays lazily unloaded because the
pipeline never touches perplexity.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from evals.preference.corpus import Pair, load_corpus
from evals.preference.evaluate import (
    ALPHA,
    PREFERENCE_EDITED,
    PREFERENCE_SPLIT,
    PREFERENCE_UNEDITED,
    RESULTS_PATH,
    VERDICT_IMPROVED,
    VERDICT_REGRESSED,
    VERDICT_UNCHANGED,
    aggregate,
    binom_cdf,
    clopper_pearson,
    evaluate_pair,
    preference_from_verdicts,
    run_evaluation,
)


class TestPreferenceReduction:
    def test_majority_edited(self) -> None:
        assert preference_from_verdicts([VERDICT_IMPROVED] * 3 + [VERDICT_REGRESSED] * 2) == PREFERENCE_EDITED

    def test_majority_unedited(self) -> None:
        assert preference_from_verdicts([VERDICT_IMPROVED] * 2 + [VERDICT_REGRESSED] * 3) == PREFERENCE_UNEDITED

    def test_tie_is_split(self) -> None:
        assert preference_from_verdicts([VERDICT_IMPROVED, VERDICT_REGRESSED]) == PREFERENCE_SPLIT

    def test_no_movement_is_split(self) -> None:
        assert preference_from_verdicts([VERDICT_UNCHANGED] * 4) == PREFERENCE_SPLIT

    def test_empty_is_split(self) -> None:
        assert preference_from_verdicts([]) == PREFERENCE_SPLIT

    def test_unchanged_does_not_vote(self) -> None:
        # 2 improved + 2 regressed + 5 unchanged: unchanged must not break the tie.
        verdicts = [VERDICT_IMPROVED, VERDICT_IMPROVED, VERDICT_REGRESSED, VERDICT_REGRESSED] + [VERDICT_UNCHANGED] * 5
        assert preference_from_verdicts(verdicts) == PREFERENCE_SPLIT


class TestBinomialCDF:
    def test_known_values(self) -> None:
        assert binom_cdf(5, 12, 0.5) == pytest.approx(0.38720703125)
        assert binom_cdf(0, 5, 0.5) == pytest.approx(0.03125)
        assert binom_cdf(3, 3, 0.5) == pytest.approx(1.0)

    def test_edges(self) -> None:
        assert binom_cdf(3, 5, 0.0) == 1.0
        assert binom_cdf(2, 5, 1.0) == 0.0
        assert binom_cdf(5, 5, 1.0) == 1.0

    def test_monotone_decreasing_in_p(self) -> None:
        values = [binom_cdf(5, 12, p) for p in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]]
        assert values == sorted(values, reverse=True)


class TestClopperPearson:
    def test_closed_form_cases(self) -> None:
        # These have exact closed forms via (1-p)^n equations.
        lo, hi = clopper_pearson(1, 2)
        assert lo == pytest.approx(1 - (1 - ALPHA / 2) ** 0.5, abs=1e-12)
        assert hi == pytest.approx((1 - ALPHA / 2) ** 0.5, abs=1e-12)

        lo, hi = clopper_pearson(0, 5)
        assert lo == 0.0
        assert hi == pytest.approx(1 - (ALPHA / 2) ** 0.2, abs=1e-12)

        lo, hi = clopper_pearson(5, 5)
        assert hi == 1.0
        assert lo == pytest.approx((ALPHA / 2) ** 0.2, abs=1e-12)

    def test_regression_pins(self) -> None:
        # Cross-verified against scipy.stats.beta.ppf to 2e-16 on this working
        # tree; pinned here as a regression guard (scipy is not a project dep).
        lo, hi = clopper_pearson(5, 12)
        assert lo == pytest.approx(0.151652, abs=1e-5)
        assert hi == pytest.approx(0.723330, abs=1e-5)

        lo, hi = clopper_pearson(98, 206)
        assert lo == pytest.approx(0.405890, abs=1e-5)
        assert hi == pytest.approx(0.546275, abs=1e-5)

    def test_brackets_the_rate(self) -> None:
        for k, n in [(0, 13), (3, 13), (5, 12), (98, 206), (13, 13)]:
            lo, hi = clopper_pearson(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            clopper_pearson(-1, 5)
        with pytest.raises(ValueError):
            clopper_pearson(6, 5)
        with pytest.raises(ValueError):
            clopper_pearson(1, 0)


def _pair_result(pair_id: str, verdicts: list[str]) -> dict[str, Any]:
    """Synthetic per-pair result with one per_statistic entry per verdict."""
    return {
        "pair_id": pair_id,
        "work": "w",
        "stratum": "edition_revision",
        "n_statistics": len(verdicts),
        "n_improved": verdicts.count(VERDICT_IMPROVED),
        "n_regressed": verdicts.count(VERDICT_REGRESSED),
        "n_unchanged": verdicts.count(VERDICT_UNCHANGED),
        "preference": preference_from_verdicts(verdicts),
        "per_statistic": [
            {"statistic": f"stat_{i}", "z_unedited": 1.0, "z_edited": 0.5, "delta": -0.5, "verdict": v}
            for i, v in enumerate(verdicts)
        ],
    }


class TestAggregate:
    def test_counts_and_intervals(self) -> None:
        results = [
            _pair_result("a", [VERDICT_IMPROVED, VERDICT_IMPROVED, VERDICT_REGRESSED]),  # edited
            _pair_result("b", [VERDICT_REGRESSED, VERDICT_REGRESSED]),  # unedited
            _pair_result("c", [VERDICT_UNCHANGED]),  # split
            _pair_result("d", [VERDICT_IMPROVED]),  # edited
        ]
        agg = aggregate(results)
        assert agg["pair_level"]["n_pairs"] == 4
        assert agg["pair_level"]["decided_pairs"] == 3
        assert agg["pair_level"]["agreeing_pairs"] == 2
        assert agg["pair_level"]["agreement_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert agg["pair_level"]["ci95"] == [round(b, 4) for b in clopper_pearson(2, 3)]

        assert agg["verdict_level"]["decided_verdicts"] == 6
        assert agg["verdict_level"]["improved"] == 3
        assert agg["verdict_level"]["regressed"] == 3
        assert agg["verdict_level"]["unchanged"] == 1
        assert agg["verdict_level"]["agreement_rate"] == pytest.approx(0.5, abs=1e-4)

    def test_statistic_level_rows(self) -> None:
        results = [
            _pair_result("a", [VERDICT_IMPROVED, VERDICT_REGRESSED]),
            _pair_result("b", [VERDICT_IMPROVED, VERDICT_UNCHANGED]),
        ]
        # give statistic 0 a known mean delta
        for result in results:
            for entry in result["per_statistic"]:
                entry["delta"] = -0.25
        agg = aggregate(results)
        row = agg["statistic_level"]["stat_0"]
        assert row["improved"] == 2
        assert row["regressed"] == 0
        assert row["unchanged"] == 0
        assert row["agreement"] == 1.0
        assert row["mean_delta"] == pytest.approx(-0.25)
        assert agg["statistic_level"]["stat_1"]["agreement"] == 0.0

    def test_no_decided_pairs(self) -> None:
        results = [_pair_result("a", [VERDICT_UNCHANGED]), _pair_result("b", [VERDICT_UNCHANGED, VERDICT_UNCHANGED])]
        agg = aggregate(results)
        assert agg["pair_level"]["decided_pairs"] == 0
        assert agg["pair_level"]["agreement_rate"] == 0.0
        assert agg["pair_level"]["ci95"] is None
        assert agg["verdict_level"]["decided_verdicts"] == 0
        assert agg["verdict_level"]["ci95"] is None


class StubDeltaAnalyzer:
    """Returns a canned stylometric_delta response; records the call args."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str | None]] = []

    def stylometric_delta(self, text_a: str, text_b: str, baseline: str | None = None) -> dict[str, Any]:
        self.calls.append((text_a, text_b, baseline))
        return self.response


def _delta_response(entries: list[tuple[str, float, float]]) -> dict[str, Any]:
    """Shape mirrors the W4 response: deltas and per-statistic verdicts."""
    deltas = [
        {"statistic": stat, "z_a": round(z_a, 2), "z_b": round(z_b, 2), "delta": round(z_b - z_a, 2)}
        for stat, z_a, z_b in entries
    ]
    verdict = [
        {
            "statistic": entry["statistic"],
            "delta": entry["delta"],
            "verdict": "improved" if abs(entry["z_b"]) < abs(entry["z_a"]) else "regressed",
        }
        for entry in deltas
    ]
    return {"baseline": "brown_corpus", "deltas": deltas, "verdict": verdict}


class TestEvaluatePair:
    def test_maps_delta_response_and_argument_order(self) -> None:
        pair = load_corpus()[0]
        response = _delta_response([("ttr", 1.5, 0.5), ("pos_adv", 0.5, 2.0)])
        stub = StubDeltaAnalyzer(response)

        result = evaluate_pair(stub, pair)
        # text_a must be the UNEDITED text, text_b the EDITED one.
        assert stub.calls == [(pair["unedited"]["text"], pair["edited"]["text"], "brown_corpus")]

        assert result["pair_id"] == pair["pair_id"]
        assert result["n_statistics"] == 2
        assert result["n_improved"] == 1
        assert result["n_regressed"] == 1
        assert result["preference"] == PREFERENCE_SPLIT
        assert result["per_statistic"][0] == {
            "statistic": "ttr",
            "z_unedited": 1.5,
            "z_edited": 0.5,
            "delta": -1.0,
            "verdict": "improved",
        }

    def test_pipeline_error_is_raised_not_swallowed(self) -> None:
        pair = load_corpus()[0]
        stub = StubDeltaAnalyzer({"error": "baseline boom"})
        with pytest.raises(RuntimeError, match="baseline boom"):
            evaluate_pair(stub, pair)


@pytest.mark.usefixtures("nlp")
class TestDeterministicPipeline:
    def test_run_matches_committed_results(self) -> None:
        # The committed results.json must be exactly what the current code and
        # corpus produce: any drift in corpus, features, or statistics fails here.
        committed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        assert run_evaluation() == committed

    def test_corpus_digest_recorded(self) -> None:
        committed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        assert len(committed["pairs"]) == 13
        assert len(committed["corpus_sha256"]) == 64

    def test_main_writes_canonical_results(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import evals.preference.evaluate as evaluate_module

        monkeypatch.setattr(evaluate_module, "RESULTS_PATH", tmp_path / "results.json")
        assert evaluate_module.main([]) == 0
        written = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert written["pair_level"]["n_pairs"] == 13

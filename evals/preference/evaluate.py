"""Pairwise preference evaluation (W9): delta pipeline and agreement statistics.

For every committed pair, the W4 delta machinery
(``AIDetectionAnalyzer.stylometric_delta``) profiles both excerpts against the
same baseline and reports, per statistic, whether the edited (later) text
moved closer to the baseline's range (``improved``), further away
(``regressed``), or not at all (``unchanged``).

The question this evaluation answers: does the delta DIRECTION agree with the
edited variant? Two views are reported:

- **Pair level** — each pair's per-statistic verdicts are reduced to a single
  preference (majority of decided statistics; ties are ``split``). The
  agreement rate is the fraction of decided pairs preferring the edited text.
- **Verdict level** — all decided (pair, statistic) verdicts pooled: what
  fraction of individual statistic movements were improvements? This shows
  WHICH statistics carry the signal, which the pair-level vote hides.

Both rates get exact Clopper-Pearson confidence intervals: with 13 pairs the
interval is wide, and stating that width is part of the result.

The pipeline is deterministic (no GPU, no network): the same corpus and the
same spaCy model produce byte-identical results — that property is what the
tests pin. Run as a module to regenerate ``results.json``:

    uv run python -m evals.preference.evaluate
"""

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from evals.preference.corpus import CORPUS_PATH, Pair, load_corpus

__all__ = [
    "ALPHA",
    "PREFERENCE_EDITED",
    "PREFERENCE_SPLIT",
    "PREFERENCE_UNEDITED",
    "aggregate",
    "binom_cdf",
    "clopper_pearson",
    "evaluate_pair",
    "preference_from_verdicts",
    "run_evaluation",
]

RESULTS_PATH = Path(__file__).parent / "results.json"

#: Significance level for the exact confidence intervals.
ALPHA = 0.05
#: Baseline every pair is scored against (committed, offline — see corpus.py).
EVAL_BASELINE = "brown_corpus"

PREFERENCE_EDITED = "edited"
PREFERENCE_UNEDITED = "unedited"
PREFERENCE_SPLIT = "split"

VERDICT_IMPROVED = "improved"
VERDICT_REGRESSED = "regressed"
VERDICT_UNCHANGED = "unchanged"


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), via exact sums (n is tiny here)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * p**i * (1.0 - p) ** (n - i)
    return min(1.0, max(0.0, total))


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for k successes in n trials.

    Pure ``math``: each bound is the p value where the relevant binomial tail
    probability hits ``alpha / 2``, found by bisection (the CDF is monotone
    in p). Deterministic, no dependencies.

    Returns:
        ``(lower, upper)`` with 0.0 <= lower <= upper <= 1.0. ``k = 0`` pins
        the lower bound at 0.0 and ``k = n`` pins the upper bound at 1.0.
    """
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"clopper_pearson needs 0 <= k ({k}) <= n ({n}) > 0")

    # Lower bound: smallest p with P(X <= k | p) <= alpha/2.
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if binom_cdf(k, n, mid) > alpha / 2.0:
            hi = mid
        else:
            lo = mid
    lower = (lo + hi) / 2.0

    # Upper bound: largest p with P(X <= k-1 | p) < 1 - alpha/2.
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if binom_cdf(k - 1, n, mid) < 1.0 - alpha / 2.0:
            lo = mid
        else:
            hi = mid
    upper = (lo + hi) / 2.0

    return lower, upper


def preference_from_verdicts(verdicts: list[str]) -> str:
    """Reduce per-statistic verdicts to one pair-level preference.

    Majority among decided (improved/regressed) statistics: ``edited`` when
    improvements dominate, ``unedited`` when regressions dominate, ``split``
    on a tie or when nothing moved.
    """
    improved = verdicts.count(VERDICT_IMPROVED)
    regressed = verdicts.count(VERDICT_REGRESSED)
    if improved > regressed:
        return PREFERENCE_EDITED
    if regressed > improved:
        return PREFERENCE_UNEDITED
    return PREFERENCE_SPLIT


def evaluate_pair(analyzer: Any, pair: Pair) -> dict[str, Any]:
    """Run the W4 delta pipeline on one pair and reduce it to a preference.

    ``text_a`` is the unedited (earlier) excerpt and ``text_b`` the edited
    (later) one, so an ``improved`` verdict means the edit moved that
    statistic toward the baseline's range.
    """
    result = analyzer.stylometric_delta(pair["unedited"]["text"], pair["edited"]["text"], EVAL_BASELINE)
    if "error" in result:
        raise RuntimeError(f"pair {pair['pair_id']!r}: stylometric_delta failed: {result['error']}")

    deltas = result["deltas"]
    verdict_by_statistic = {entry["statistic"]: entry["verdict"] for entry in result["verdict"]}
    verdicts = [verdict_by_statistic[entry["statistic"]] for entry in deltas]

    return {
        "pair_id": pair["pair_id"],
        "work": pair["work"],
        "stratum": pair["stratum"],
        "n_statistics": len(deltas),
        "n_improved": verdicts.count(VERDICT_IMPROVED),
        "n_regressed": verdicts.count(VERDICT_REGRESSED),
        "n_unchanged": verdicts.count(VERDICT_UNCHANGED),
        "preference": preference_from_verdicts(verdicts),
        "per_statistic": [
            {
                "statistic": entry["statistic"],
                "z_unedited": entry["z_a"],
                "z_edited": entry["z_b"],
                "delta": entry["delta"],
                "verdict": verdict_by_statistic[entry["statistic"]],
            }
            for entry in deltas
        ],
    }


def aggregate(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-pair results to the pair-level and verdict-level agreement views."""
    decided_pairs = [r for r in pair_results if r["preference"] in (PREFERENCE_EDITED, PREFERENCE_UNEDITED)]
    agreeing_pairs = [r for r in decided_pairs if r["preference"] == PREFERENCE_EDITED]

    k_pairs, n_pairs = len(agreeing_pairs), len(decided_pairs)
    pair_rate = k_pairs / n_pairs if n_pairs else 0.0

    verdict_totals: dict[str, int] = {VERDICT_IMPROVED: 0, VERDICT_REGRESSED: 0, VERDICT_UNCHANGED: 0}
    statistic_rows: dict[str, dict[str, Any]] = {}
    for result in pair_results:
        for entry in result["per_statistic"]:
            verdict_totals[entry["verdict"]] += 1
            row = statistic_rows.setdefault(
                entry["statistic"],
                {"improved": 0, "regressed": 0, "unchanged": 0, "delta_sum": 0.0, "n": 0},
            )
            row[entry["verdict"]] += 1
            row["delta_sum"] += entry["delta"]
            row["n"] += 1

    k_verdicts = verdict_totals[VERDICT_IMPROVED]
    decided_verdicts = k_verdicts + verdict_totals[VERDICT_REGRESSED]
    verdict_rate = k_verdicts / decided_verdicts if decided_verdicts else 0.0

    statistic_level = {
        statistic: {
            "improved": row["improved"],
            "regressed": row["regressed"],
            "unchanged": row["unchanged"],
            "mean_delta": round(row["delta_sum"] / row["n"], 3),
            "agreement": round(row["improved"] / (row["improved"] + row["regressed"]), 3)
            if (row["improved"] + row["regressed"])
            else None,
        }
        for statistic, row in sorted(statistic_rows.items())
    }

    return {
        "pair_level": {
            "n_pairs": len(pair_results),
            "decided_pairs": n_pairs,
            "agreeing_pairs": k_pairs,
            "agreement_rate": round(pair_rate, 4),
            "ci95": [round(bound, 4) for bound in clopper_pearson(k_pairs, n_pairs)] if n_pairs else None,
        },
        "verdict_level": {
            "decided_verdicts": decided_verdicts,
            "improved": k_verdicts,
            "regressed": verdict_totals[VERDICT_REGRESSED],
            "unchanged": verdict_totals[VERDICT_UNCHANGED],
            "agreement_rate": round(verdict_rate, 4),
            "ci95": [round(bound, 4) for bound in clopper_pearson(k_verdicts, decided_verdicts)]
            if decided_verdicts
            else None,
        },
        "statistic_level": statistic_level,
    }


def _corpus_sha256() -> str:
    return hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def run_evaluation() -> dict[str, Any]:
    """Evaluate the committed corpus end to end and return the results document."""
    from server.analyzers import AIDetectionAnalyzer
    from server.config import load_config
    from server.models import initialize_models

    config = load_config()
    models = initialize_models(config)
    analyzer = AIDetectionAnalyzer(models["spacy"].get_model(), models["gpt2"], config)

    pairs = load_corpus()
    pair_results = [evaluate_pair(analyzer, pair) for pair in pairs]
    result = {
        "parameters": {
            "baseline": EVAL_BASELINE,
            "alpha": ALPHA,
            "text_a": "unedited (earlier) excerpt",
            "text_b": "edited (later) excerpt",
            "preference_rule": "majority of decided per-statistic verdicts; split on tie or no movement",
            "delta_machinery": "AIDetectionAnalyzer.stylometric_delta (W4); z-scores rounded to 2 decimals",
        },
        "corpus_sha256": _corpus_sha256(),
        "pairs": pair_results,
    }
    result.update(aggregate(pair_results))
    return result


def main(argv: list[str] | None = None) -> int:
    result = run_evaluation()
    RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pair_level = result["pair_level"]
    verdict_level = result["verdict_level"]
    print(
        f"wrote {RESULTS_PATH}\n"
        f"pair level   : {pair_level['agreeing_pairs']}/{pair_level['decided_pairs']} pairs prefer the edited text"
        f" ({pair_level['agreement_rate']:.0%}), 95% CI"
        f" [{pair_level['ci95'][0]:.2f}, {pair_level['ci95'][1]:.2f}]\n"
        f"verdict level: {verdict_level['improved']}/{verdict_level['decided_verdicts']} statistic movements"
        f" improved ({verdict_level['agreement_rate']:.0%}), 95% CI"
        f" [{verdict_level['ci95'][0]:.2f}, {verdict_level['ci95'][1]:.2f}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

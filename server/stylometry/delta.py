"""Draft-vs-revision delta math over baseline z-scores.

Pure functions behind ``AIDetectionAnalyzer.stylometric_delta``: given the
z-score dictionaries two texts produce against the SAME baseline, they report
what the revision moved, per statistic, and whether each movement helped.

Improvement is defined against the baseline the way the whole detector family
defines it: a z-score of 0 sits exactly in the baseline's range, and the flag
logic (``generate_flags``) fires indicators on large ``|z|``. So a statistic
improves when the revision moves its z-score closer to zero and regresses when
the movement pushes it further out — never mind the sign.
"""

from typing import Any

__all__ = ["compute_statistic_deltas", "compute_verdicts"]

# The tool family reports z-scores rounded to 2 decimals (stylometric_analysis
# rounds before returning). Deltas are computed and reported at the same
# precision so the reported delta is always exactly the difference of the two
# reported z-scores.
Z_SCORE_DECIMALS = 2

DIRECTION_INCREASED = "increased"
DIRECTION_DECREASED = "decreased"
DIRECTION_NONE = "none"

VERDICT_IMPROVED = "improved"
VERDICT_REGRESSED = "regressed"
VERDICT_UNCHANGED = "unchanged"


def compute_statistic_deltas(z_a: dict[str, float], z_b: dict[str, float]) -> list[dict[str, Any]]:
    """Compute per-statistic movement between two z-score dictionaries.

    Only statistics present in BOTH dictionaries are compared: a z-score is
    omitted when a statistic cannot be measured (or the baseline lacks it), and
    a one-sided number supports no comparison. Output is sorted by statistic
    name so the response is deterministic.

    Args:
        z_a: Draft's z-scores, as ``stylometric_analysis`` returns them
            (rounded to :data:`Z_SCORE_DECIMALS`).
        z_b: Revised text's z-scores, same shape.

    Returns:
        A list of ``{"statistic", "z_a", "z_b", "delta", "direction"}`` dicts,
        where ``delta = z_b - z_a`` and ``direction`` is ``"increased"`` when
        the raw z-score went up, ``"decreased"`` when it went down, and
        ``"none"`` when it did not move. Direction is the raw movement — it is
        independent of whether the movement helped (see
        :func:`compute_verdicts`).
    """
    deltas: list[dict[str, Any]] = []

    for statistic in sorted(set(z_a) & set(z_b)):
        z_a_value = _rounded(z_a[statistic])
        z_b_value = _rounded(z_b[statistic])
        delta = _rounded(z_b_value - z_a_value)
        deltas.append(
            {
                "statistic": statistic,
                "z_a": z_a_value,
                "z_b": z_b_value,
                "delta": delta,
                "direction": _direction(delta),
            }
        )

    return deltas


def compute_verdicts(deltas: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Judge each delta: did the revision move that statistic toward the baseline?

    Takes :func:`compute_statistic_deltas` output and labels each entry
    ``"improved"`` when ``|z_b| < |z_a|`` (closer to the baseline's range),
    ``"regressed"`` when ``|z_b| > |z_a|``, and ``"unchanged"`` otherwise. The
    verdict list keeps the input order, so it runs parallel to ``deltas``.

    Args:
        deltas: The per-statistic delta entries to judge.

    Returns:
        A list of ``{"statistic", "verdict"}`` dicts.
    """
    verdicts: list[dict[str, str]] = []

    for entry in deltas:
        z_a = entry["z_a"]
        z_b = entry["z_b"]
        if abs(z_b) < abs(z_a):
            verdict = VERDICT_IMPROVED
        elif abs(z_b) > abs(z_a):
            verdict = VERDICT_REGRESSED
        else:
            verdict = VERDICT_UNCHANGED
        verdicts.append({"statistic": entry["statistic"], "verdict": verdict})

    return verdicts


def _rounded(value: float) -> float:
    return round(value, Z_SCORE_DECIMALS)


def _direction(delta: float) -> str:
    if delta > 0:
        return DIRECTION_INCREASED
    if delta < 0:
        return DIRECTION_DECREASED
    return DIRECTION_NONE

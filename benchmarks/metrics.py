"""Metric calculations for the labeled detector benchmark.

Pure functions over already-scored records, with no analyzer or model
dependency, so every number in a report can be checked against hand-computed
values in a unit test.

Throughout this module the **positive class is machine-generated text**: a
"true positive" is a machine-written document the detector flagged, and a
"false positive" is a human-written document it flagged.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: The two labels a corpus record may carry.
HUMAN = "human"
MACHINE = "machine"
LABELS = (HUMAN, MACHINE)


def partition_outcomes(records: Iterable[Dict[str, Any]], method: str) -> Tuple[List[dict], List[dict]]:
    """Split records into those a method scored and those it failed on.

    A record is *scored* when ``record[method]`` exists and reports ``ok``.
    Anything else - a missing block, or one carrying ``ok: false`` - is a
    failure. Failures are never folded into the confusion matrix: an
    error-shaped analyzer result reports ``high_ai_probability: false``, and
    counting that as a "predicted human" would silently deflate the false
    positive rate.
    """
    scored, failed = [], []
    for record in records:
        outcome = record.get(method)
        if isinstance(outcome, dict) and outcome.get("ok"):
            scored.append(record)
        else:
            failed.append(record)
    return scored, failed


def label_counts(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Count records per label, always reporting both labels."""
    counts = dict.fromkeys(LABELS, 0)
    for record in records:
        label = record.get("label")
        counts[label] = counts.get(label, 0) + 1
    return counts


def confusion_matrix(pairs: Sequence[Tuple[str, bool]]) -> Dict[str, int]:
    """Exact confusion matrix for ``(label, predicted_machine)`` pairs."""
    matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for label, predicted_machine in pairs:
        if label == MACHINE:
            matrix["tp" if predicted_machine else "fn"] += 1
        elif label == HUMAN:
            matrix["fp" if predicted_machine else "tn"] += 1
        else:
            raise ValueError(f"Unknown label: {label!r}")
    return matrix


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """A rate, or None when its denominator is empty.

    None means "not measurable from this sample" and is reported as such;
    substituting 0.0 would be indistinguishable from a real measurement.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def classification_metrics(matrix: Dict[str, int]) -> Dict[str, Optional[float]]:
    """Accuracy, precision, recall, F1, TPR, FPR and specificity for a matrix."""
    tp, fp, tn, fn = matrix["tp"], matrix["fp"], matrix["tn"], matrix["fn"]

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": recall,
        "fpr": _ratio(fp, fp + tn),
        "specificity": _ratio(tn, fp + tn),
    }


def evaluate_boolean_decision(
    records: Sequence[Dict[str, Any]], method: str, field: str = "high_ai_probability"
) -> Dict[str, Any]:
    """Score one of the detector's shipped boolean decisions.

    Returns the sample counts (including failures, which are excluded from the
    matrix), the confusion matrix, and the derived rates.
    """
    scored, failed = partition_outcomes(records, method)
    pairs = [(record["label"], bool(record[method].get(field))) for record in scored]
    matrix = confusion_matrix(pairs)
    return {
        "method": method,
        "field": field,
        "n_total": len(records),
        "n_scored": len(scored),
        "n_failed": len(failed),
        "scored_label_counts": label_counts(scored),
        "failed_label_counts": label_counts(failed),
        "confusion_matrix": matrix,
        "metrics": classification_metrics(matrix),
    }


def threshold_sweep(scored_pairs: Sequence[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """Sweep every distinct score as a ``score >= threshold`` cut point.

    Only thresholds that actually occur in the sample are candidates, which is
    what makes the sweep exactly reproducible: it never interpolates between
    observed scores.
    """
    thresholds = sorted({score for _, score in scored_pairs})
    sweep = []
    for threshold in thresholds:
        matrix = confusion_matrix([(label, score >= threshold) for label, score in scored_pairs])
        sweep.append({"threshold": threshold, "confusion_matrix": matrix, "metrics": classification_metrics(matrix)})
    return sweep


def fpr_at_tpr_targets(scored_pairs: Sequence[Tuple[str, float]], targets: Sequence[float]) -> List[Dict[str, Any]]:
    """False positive rate at the cheapest threshold reaching each TPR target.

    "Cheapest" is the highest threshold whose true positive rate is still at or
    above the target, since a higher cut point can only lower the FPR. The
    achieved TPR is reported alongside the target: the scores here are coarse
    and discrete, so an exact 0.90 is generally unreachable and quoting the
    target alone would overstate what was measured. ``threshold`` is None when
    no cut point reaches the target at all.
    """
    sweep = threshold_sweep(scored_pairs)
    results = []
    for target in targets:
        reaching = [
            point for point in sweep if point["metrics"]["tpr"] is not None and point["metrics"]["tpr"] >= target
        ]
        if not reaching:
            results.append({"target_tpr": target, "threshold": None, "achieved_tpr": None, "fpr": None})
            continue
        best = max(reaching, key=lambda point: point["threshold"])
        results.append(
            {
                "target_tpr": target,
                "threshold": best["threshold"],
                "achieved_tpr": best["metrics"]["tpr"],
                "fpr": best["metrics"]["fpr"],
                "confusion_matrix": best["confusion_matrix"],
            }
        )
    return results


def describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Count, mean, sample standard deviation, median, min and max."""
    numbers = list(values)
    if not numbers:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}

    mean = sum(numbers) / len(numbers)
    if len(numbers) < 2:
        std = None
    else:
        std = math.sqrt(sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1))

    ordered = sorted(numbers)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    return {
        "n": len(numbers),
        "mean": mean,
        "std": std,
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
    }


def cohens_d(human_stats: Dict[str, Optional[float]], machine_stats: Dict[str, Optional[float]]) -> Optional[float]:
    """Pooled-standard-deviation effect size for machine minus human.

    None when either side has too few values for a standard deviation, or when
    both sides are constant (no spread to standardize by).
    """
    if human_stats["n"] < 2 or machine_stats["n"] < 2:
        return None
    if human_stats["std"] is None or machine_stats["std"] is None:
        return None

    n_h, n_m = human_stats["n"], machine_stats["n"]
    pooled_variance = ((n_h - 1) * human_stats["std"] ** 2 + (n_m - 1) * machine_stats["std"] ** 2) / (n_h + n_m - 2)
    if pooled_variance <= 0:
        return None
    return (machine_stats["mean"] - human_stats["mean"]) / math.sqrt(pooled_variance)


def separation_direction(human_stats: Dict[str, Optional[float]], machine_stats: Dict[str, Optional[float]]) -> str:
    """Which way the feature moves for machine text relative to human text."""
    if human_stats["mean"] is None or machine_stats["mean"] is None:
        return "unmeasured"
    if machine_stats["mean"] > human_stats["mean"]:
        return "higher in machine"
    if machine_stats["mean"] < human_stats["mean"]:
        return "lower in machine"
    return "no difference"


def null_value_counts(records: Sequence[Dict[str, Any]], method: str, block: str) -> Dict[str, int]:
    """How often each key in a block came back null on a *successful* analysis.

    A feature the analyzer could not measure is reported as null rather than
    zero, and a null is skipped by :func:`feature_separation`. Without this
    count such a key would simply be absent from a summary table, which is
    indistinguishable from it never having existed.
    """
    scored, _ = partition_outcomes(records, method)
    counts: Dict[str, int] = {}
    for record in scored:
        for key, value in (record[method].get(block) or {}).items():
            counts.setdefault(key, 0)
            if value is None:
                counts[key] += 1
    return {key: count for key, count in sorted(counts.items()) if count}


def feature_separation(records: Sequence[Dict[str, Any]], method: str, block: str) -> List[Dict[str, Any]]:
    """Per-feature human/machine distributions and direction of separation.

    ``block`` names the sub-dictionary of a scored method's output to summarize
    (``"features"`` or ``"z_scores"``). Non-numeric and missing values are
    skipped per document, so each feature reports its own sample counts and a
    feature measurable on only some documents cannot silently borrow another's.
    """
    scored, _ = partition_outcomes(records, method)

    values: Dict[str, Dict[str, List[float]]] = {}
    for record in scored:
        label = record["label"]
        for feature, value in (record[method].get(block) or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                continue
            values.setdefault(feature, {HUMAN: [], MACHINE: []})[label].append(float(value))

    summaries = []
    for feature in sorted(values):
        human_stats = describe(values[feature][HUMAN])
        machine_stats = describe(values[feature][MACHINE])
        summaries.append(
            {
                "feature": feature,
                "human": human_stats,
                "machine": machine_stats,
                "mean_difference": (
                    None
                    if human_stats["mean"] is None or machine_stats["mean"] is None
                    else machine_stats["mean"] - human_stats["mean"]
                ),
                "cohens_d": cohens_d(human_stats, machine_stats),
                "direction": separation_direction(human_stats, machine_stats),
            }
        )
    return summaries

"""Deterministic Markdown rendering of a benchmark run.

Nothing in this module reads the clock, the environment, or a library version:
given the same scored records it always produces byte-identical Markdown.
Anything that does vary between runs belongs in ``run_metadata.json``, which
the runner writes separately.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

from benchmarks import metrics

#: True positive rates the stylometry threshold sweep is reported at.
DEFAULT_TPR_TARGETS = (0.50, 0.75, 0.90)

DISCLAIMER = (
    "These are statistical measurements of writing style, not evidence of authorship. "
    "A flagged document is a document whose measured features differ from a baseline; "
    "it is not a document proven to be machine-written."
)


def _num(value: Optional[float], places: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{places}f}"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _confusion_lines(matrix: Dict[str, int]) -> List[str]:
    return [
        "| | predicted machine | predicted human |",
        "| --- | ---: | ---: |",
        f"| **actual machine** | {matrix['tp']} (TP) | {matrix['fn']} (FN) |",
        f"| **actual human** | {matrix['fp']} (FP) | {matrix['tn']} (TN) |",
    ]


def _decision_section(title: str, evaluation: Dict[str, Any], note: Optional[str] = None) -> List[str]:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    scored, failed = evaluation["scored_label_counts"], evaluation["failed_label_counts"]
    lines += [
        f"Documents: {evaluation['n_total']} total, {evaluation['n_scored']} scored "
        f"({scored[metrics.HUMAN]} human / {scored[metrics.MACHINE]} machine), "
        f"{evaluation['n_failed']} failed "
        f"({failed[metrics.HUMAN]} human / {failed[metrics.MACHINE]} machine). "
        "Failed analyses are excluded from the matrix below rather than counted as negatives.",
        "",
    ]
    lines += _confusion_lines(evaluation["confusion_matrix"])
    lines.append("")
    values = evaluation["metrics"]
    lines += [
        "| metric | value |",
        "| --- | ---: |",
        f"| accuracy | {_num(values['accuracy'])} |",
        f"| precision | {_num(values['precision'])} |",
        f"| recall (TPR) | {_num(values['recall'])} |",
        f"| F1 | {_num(values['f1'])} |",
        f"| false positive rate | {_num(values['fpr'])} |",
        f"| true positive rate | {_num(values['tpr'])} |",
        f"| specificity | {_num(values['specificity'])} |",
        "",
    ]
    return lines


def _failure_section(records: Sequence[Dict[str, Any]], methods: Sequence[str]) -> List[str]:
    lines = ["## Failed analyses", ""]
    any_failure = False
    for method in methods:
        _, failed = metrics.partition_outcomes(records, method)
        if not failed:
            lines.append(f"- `{method}`: 0 failures.")
            continue
        any_failure = True
        reasons: Dict[str, int] = {}
        for record in failed:
            outcome = record.get(method) or {}
            reason = outcome.get("error", "no result recorded")
            reasons[reason] = reasons.get(reason, 0) + 1
        lines.append(f"- `{method}`: {len(failed)} failures.")
        for reason in sorted(reasons):
            lines.append(f"  - {reasons[reason]}x `{reason}`")
    if not any_failure:
        lines.append("")
        lines.append("Every document produced an analyzer result, so no scores are missing.")
    lines.append("")
    return lines


def _feature_table(summaries: Sequence[Dict[str, Any]], value_places: int = 4) -> List[str]:
    lines = [
        "| feature | human n | human mean | human median | human std "
        "| machine n | machine mean | machine median | machine std "
        "| mean diff | median diff | Cohen's d | direction (mean) | direction (median) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for summary in summaries:
        human, machine = summary["human"], summary["machine"]
        lines.append(
            f"| `{summary['feature']}` | {human['n']} | {_num(human['mean'], value_places)} "
            f"| {_num(human['median'], value_places)} | {_num(human['std'], value_places)} "
            f"| {machine['n']} | {_num(machine['mean'], value_places)} "
            f"| {_num(machine['median'], value_places)} | {_num(machine['std'], value_places)} "
            f"| {_num(summary['mean_difference'], value_places)} "
            f"| {_num(summary['median_difference'], value_places)} "
            f"| {_num(summary['cohens_d'])} | {summary['direction']} | {summary['direction_median']} |"
        )
    return lines


def _null_note(records: Sequence[Dict[str, Any]], method: str, block: str) -> List[str]:
    counts = metrics.null_value_counts(records, method, block)
    if not counts:
        return []
    scored, _ = metrics.partition_outcomes(records, method)
    items = ", ".join(f"`{key}` ({count} of {len(scored)})" for key, count in counts.items())
    return [
        "",
        f"Reported as null on some scored documents, and excluded from those rows rather than "
        f"counted as zero: {items}.",
    ]


def build_report(
    records: Sequence[Dict[str, Any]],
    corpus_info: Dict[str, Any],
    thresholds: Dict[str, Any],
    methods: Sequence[str],
    tpr_targets: Sequence[float] = DEFAULT_TPR_TARGETS,
) -> str:
    """Render the full Markdown report for a set of scored records."""
    counts = metrics.label_counts(records)
    lines = [
        f"# Detector benchmark - {corpus_info['name']}",
        "",
        "Positive class is **machine-generated text**: a true positive is a machine-written "
        "document the detector flagged, a false positive is a human-written document it flagged.",
        "",
        DISCLAIMER,
        "",
        "This report is generated; do not edit it by hand. Regenerate with the command in "
        "`benchmarks/README.md`. Run-varying values (timestamps, library versions) are recorded "
        "in `run_metadata.json` beside this file, not here.",
        "",
        "## Corpus",
        "",
        f"- Corpus: `{corpus_info['name']}` ({corpus_info.get('description', '')})",
        f"- Documents: {len(records)} ({counts[metrics.HUMAN]} human, {counts[metrics.MACHINE]} machine)",
        f"- Manifest: `{corpus_info.get('manifest', 'n/a')}`",
        f"- Analyses run: {', '.join(f'`{method}`' for method in methods)}",
        "",
        "## Configuration under test",
        "",
        "Thresholds are read from the server's own configuration and are reported here only so a "
        "reader can tell which constants produced these numbers. This benchmark changes none of them.",
        "",
        "| setting | value |",
        "| --- | ---: |",
    ]
    for key in sorted(thresholds):
        lines.append(f"| `{key}` | {thresholds[key]} |")
    lines.append("")

    lines += ["## Sample counts", ""]
    lines += [
        "| method | scored | scored human | scored machine | failed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        evaluation = metrics.evaluate_boolean_decision(records, method)
        scored = evaluation["scored_label_counts"]
        lines.append(
            f"| `{method}` | {evaluation['n_scored']} | {scored[metrics.HUMAN]} "
            f"| {scored[metrics.MACHINE]} | {evaluation['n_failed']} |"
        )
    lines.append("")
    lines += _failure_section(records, methods)

    lines += ["## Shipped boolean decisions", ""]
    if "stylometry" in methods:
        lines += _decision_section(
            "`stylometric_analysis` -> `flags.high_ai_probability`",
            metrics.evaluate_boolean_decision(records, "stylometry"),
        )
    if "perplexity" in methods:
        lines += _decision_section(
            "`perplexity_analysis` -> `flags.high_ai_probability`",
            metrics.evaluate_boolean_decision(records, "perplexity"),
        )

    if "perplexity" in methods:
        unmeasured = _unmeasured_counts(records)
        lines += [
            "### Unmeasured perplexity statistics",
            "",
            "A successful analysis can still leave a statistic undefined - the analyzer reports "
            "`null` rather than substituting a value, and a `null` can never set the flag.",
            "",
            f"- `doc_ppl` null on {unmeasured['doc_ppl']} of {unmeasured['scored']} scored documents.",
            f"- `doc_burstiness` null on {unmeasured['doc_burstiness']} of {unmeasured['scored']} scored documents.",
            "",
        ]
        lines += _perplexity_sweep_section(records, tpr_targets)

    if "stylometry" in methods:
        lines += _sweep_section(records, tpr_targets)
        lines += _fire_rate_section(records)
        lines += _indicator_section(records)

    lines += ["## Per-feature class summaries", ""]
    if "stylometry" in methods:
        lines += [
            "### Stylometric features (raw values)",
            "",
            "Direction is the sign of the machine-minus-human difference, reported separately for "
            "the mean and the median. Several of these features are heavy-tailed, so the two can "
            "disagree - where they do, the median describes the typical document and the mean is "
            "being carried by a few extreme ones. Cohen's d is the pooled-standard-deviation effect "
            "size and inherits the mean's sensitivity to those outliers; values near zero mean the "
            "feature does not separate the two classes in this corpus.",
            "",
        ]
        lines += _feature_table(metrics.feature_separation(records, "stylometry", "features"))
        lines += _null_note(records, "stylometry", "features")
        lines += [
            "",
            "### Stylometric z-scores against the baseline",
            "",
        ]
        lines += _feature_table(metrics.feature_separation(records, "stylometry", "z_scores"))
        lines += [
            "",
            "### Stylometric per-document measurements",
            "",
            "`confidence_score` is the number the shipped boolean thresholds on; "
            "`char_ngram_similarity` is the whole-profile cosine similarity to the baseline.",
            "",
        ]
        lines += _feature_table(metrics.feature_separation(records, "stylometry", "measurements"))
        lines += _null_note(records, "stylometry", "measurements")
        lines.append("")
    if "perplexity" in methods:
        lines += ["### Perplexity measurements", ""]
        lines += _feature_table(metrics.feature_separation(records, "perplexity", "measurements"))
        lines += _null_note(records, "perplexity", "measurements")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def _unmeasured_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    scored, _ = metrics.partition_outcomes(records, "perplexity")
    counts = {"scored": len(scored), "doc_ppl": 0, "doc_burstiness": 0}
    for record in scored:
        measurements = record["perplexity"].get("measurements") or {}
        for key in ("doc_ppl", "doc_burstiness"):
            if measurements.get(key) is None:
                counts[key] += 1
    return counts


def _sweep_section(records: Sequence[Dict[str, Any]], tpr_targets: Sequence[float]) -> List[str]:
    scored, _ = metrics.partition_outcomes(records, "stylometry")
    pairs = [
        (record["label"], float(record["stylometry"]["confidence_score"]))
        for record in scored
        if record["stylometry"].get("confidence_score") is not None
    ]
    lines = [
        "## Stylometry FPR at declared TPR targets",
        "",
        "This sweeps `flags.confidence_score` alone with a `score >= threshold` rule. That is a "
        "*different* classifier from the shipped boolean above, which additionally requires at "
        "least two named AI indicators - so the two sections are expected to disagree.",
        "",
        "`confidence_score` takes a small number of discrete values, so an exact target TPR is "
        "generally unreachable. Each row reports the achieved TPR next to the target; the "
        "threshold chosen is the highest one still reaching the target.",
        "",
        "| target TPR | threshold | achieved TPR | FPR | TP | FN | FP | TN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.fpr_at_tpr_targets(pairs, tpr_targets):
        matrix = row.get("confusion_matrix")
        if matrix is None:
            lines.append(
                f"| {_pct(row['target_tpr'])} | n/a | n/a | n/a | - | - | - | - | (no threshold reaches this target)"
            )
            continue
        lines.append(
            f"| {_pct(row['target_tpr'])} | {_num(row['threshold'], 3)} | {_num(row['achieved_tpr'])} "
            f"| {_num(row['fpr'])} | {matrix['tp']} | {matrix['fn']} | {matrix['fp']} | {matrix['tn']} |"
        )
    lines.append("")

    lines += [
        "### Full confidence-score sweep",
        "",
        "| threshold | TPR | FPR | TP | FN | FP | TN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in metrics.threshold_sweep(pairs):
        matrix = point["confusion_matrix"]
        values = point["metrics"]
        lines.append(
            f"| {_num(point['threshold'], 3)} | {_num(values['tpr'])} | {_num(values['fpr'])} "
            f"| {matrix['tp']} | {matrix['fn']} | {matrix['fp']} | {matrix['tn']} |"
        )
    lines.append("")
    return lines


#: Perplexity statistics swept independently, each with the direction the
#: shipped rule applies to it: both flag a document when the value is *low*.
PERPLEXITY_SWEEPS = (
    ("doc_ppl", "ppl_max"),
    ("doc_burstiness", "burstiness_min"),
)

#: List-valued stylometry outputs counted in the fire-rate table.
FIRE_RATE_KINDS = ("warnings", "errors", "ai_indicators")


def _perplexity_pairs(records: Sequence[Dict[str, Any]], statistic: str):
    """``(label, value)`` pairs for one statistic, and how many were excluded per label."""
    scored, _ = metrics.partition_outcomes(records, "perplexity")
    pairs = []
    excluded = dict.fromkeys(metrics.LABELS, 0)
    for record in scored:
        value = (record["perplexity"].get("measurements") or {}).get(statistic)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            excluded[record["label"]] += 1
            continue
        pairs.append((record["label"], float(value)))
    return pairs, excluded, len(scored)


def _sweep_table(pairs, direction: str, places: int) -> List[str]:
    lines = [
        "| threshold | TPR | FPR | TP | FN | FP | TN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in metrics.threshold_sweep(pairs, direction):
        matrix = point["confusion_matrix"]
        values = point["metrics"]
        lines.append(
            f"| {_num(point['threshold'], places)} | {_num(values['tpr'])} | {_num(values['fpr'])} "
            f"| {matrix['tp']} | {matrix['fn']} | {matrix['fp']} | {matrix['tn']} |"
        )
    return lines


def _perplexity_sweep_section(records: Sequence[Dict[str, Any]], tpr_targets: Sequence[float]) -> List[str]:
    lines = [
        "## Perplexity statistics swept independently",
        "",
        "Each statistic below is swept alone with a `value <= threshold` rule - low perplexity and "
        "low burstiness are the machine-like direction the shipped thresholds encode. A "
        "single-statistic cut is a *different* classifier from the shipped boolean above, which "
        "flags a document only when perplexity is below `ppl_max` **and** burstiness is below "
        "`burstiness_min` - so these sweeps are expected to disagree with it.",
        "",
        "Documents whose statistic is `null` (or non-finite) on a successful analysis are excluded "
        "from that statistic's sweep, never counted as negatives. Failed analyses are excluded too.",
        "",
    ]
    for statistic, setting in PERPLEXITY_SWEEPS:
        pairs, excluded, n_scored = _perplexity_pairs(records, statistic)
        n_excluded = excluded[metrics.HUMAN] + excluded[metrics.MACHINE]
        lines += [
            f"### `{statistic}` (shipped setting: `perplexity.thresholds.{setting}`)",
            "",
            f"Swept over {len(pairs)} of {n_scored} scored documents; {n_excluded} excluded for a "
            f"`null` value ({excluded[metrics.HUMAN]} human / {excluded[metrics.MACHINE]} machine).",
            "",
            "| target TPR | threshold | achieved TPR | FPR | TP | FN | FP | TN |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in metrics.fpr_at_tpr_targets(pairs, tpr_targets, metrics.AT_OR_BELOW):
            matrix = row.get("confusion_matrix")
            if matrix is None:
                lines.append(
                    f"| {_pct(row['target_tpr'])} | n/a | n/a | n/a | - | - | - | - "
                    "| (no threshold reaches this target)"
                )
                continue
            lines.append(
                f"| {_pct(row['target_tpr'])} | {_num(row['threshold'], 2)} | {_num(row['achieved_tpr'])} "
                f"| {_num(row['fpr'])} | {matrix['tp']} | {matrix['fn']} | {matrix['fp']} | {matrix['tn']} |"
            )
        lines += ["", f"Full `{statistic}` sweep, one row per observed value:", ""]
        lines += _sweep_table(pairs, metrics.AT_OR_BELOW, 2)
        lines.append("")
    return lines


def _fire_rate_section(records: Sequence[Dict[str, Any]]) -> List[str]:
    denominators, rows = metrics.flag_fire_rates(records, "stylometry", FIRE_RATE_KINDS)
    lines = [
        "## Stylometry flag fire rates by class",
        "",
        "How many scored documents of each class each individual `warnings`, `errors` and "
        "`ai_indicators` entry fired on, sorted by total fire count. A flag that fires at the same "
        "rate on both classes carries no information about either; one that fires on nearly every "
        "document of both points at a miscalibrated baseline rather than at the text.",
        "",
        f"Denominators: {denominators[metrics.HUMAN]} scored human, "
        f"{denominators[metrics.MACHINE]} scored machine documents.",
        "",
        "| flag | human | human rate | machine | machine rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['kind']}: {row['flag']}` | {row['human']}/{denominators[metrics.HUMAN]} "
            f"| {_pct(row['human_rate'])} | {row['machine']}/{denominators[metrics.MACHINE]} "
            f"| {_pct(row['machine_rate'])} |"
        )
    lines.append("")
    return lines


def _indicator_section(records: Sequence[Dict[str, Any]]) -> List[str]:
    denominators, rows = metrics.indicator_separation(records)
    lines = [
        "## Stylometry AI indicators: per-indicator separation",
        "",
        "For each `ai_indicators` entry seen: machine TPR (share of scored machine documents it "
        "fired on), human FPR (share of scored human documents), and TPR minus FPR. An indicator "
        "with a difference near zero or negative does not separate the classes in this corpus.",
        "",
        f"Denominators: {denominators[metrics.HUMAN]} scored human, "
        f"{denominators[metrics.MACHINE]} scored machine documents.",
        "",
        "| indicator | machine | machine TPR | human | human FPR | TPR - FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['indicator']}` | {row['machine']}/{denominators[metrics.MACHINE]} | {_pct(row['tpr'])} "
            f"| {row['human']}/{denominators[metrics.HUMAN]} | {_pct(row['fpr'])} | {_num(row['difference'])} |"
        )
    lines += [
        "",
        "## Stylometry confidence score by class",
        "",
        "Distribution of `confidence_score` per class. The median is shown beside the mean because "
        "the two can point in different directions.",
        "",
        "| class | n | mean | median | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stats in metrics.confidence_by_class(records).items():
        lines.append(
            f"| {label} | {stats['n']} | {_num(stats['mean'])} | {_num(stats['median'])} "
            f"| {_num(stats['min'])} | {_num(stats['max'])} |"
        )
    lines.append("")
    return lines

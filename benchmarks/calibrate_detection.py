"""Publish the AI-detection operating characteristics as a Markdown table.

Reads the committed benchmark scores (``benchmarks/reports/raid_wiki_v1/scores.jsonl``)
and renders ``docs/calibration.md``: the shipped boolean decisions measured on
labeled text, plus precision/recall/FPR at target true-positive rates for the
two continuous per-document measurements underneath them. Pure post-processing
- no analyzer, no model, no network - so the published table regenerates
anywhere in milliseconds and CI fails when it drifts from the committed scores.

All math comes from :mod:`benchmarks.metrics` (``threshold_sweep``,
``fpr_at_tpr_targets``, ``classification_metrics``); nothing here reimplements a
rate. Rendering is deterministic: no clock, no environment, no version strings
- the same scores always produce byte-identical Markdown.

Throughout, the **positive class is machine-generated text**, matching
``benchmarks/metrics.py``: a true positive is a machine-written document the
detector flagged, a false positive is a human-written document it flagged.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import metrics  # noqa: E402
from benchmarks import report as report_module  # noqa: E402

DEFAULT_SCORES = REPO_ROOT / "benchmarks" / "reports" / "raid_wiki_v1" / "scores.jsonl"
DEFAULT_METADATA = REPO_ROOT / "benchmarks" / "reports" / "raid_wiki_v1" / "run_metadata.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "calibration.md"

#: True positive rates the published target tables are reported at. Kept
#: identical to the benchmark report's targets so the two documents compare.
TPR_TARGETS = report_module.DEFAULT_TPR_TARGETS

#: The single per-document measurement each method's sweep is published for.
#: Every sweep runs on the machine-oriented score, i.e. oriented so the machine
#: class scores HIGHER - the ``score >= threshold`` convention ``threshold_sweep``
#: and ``fpr_at_tpr_targets`` are built on. ``doc_ppl`` runs the other way
#: (machine text measures *lower* perplexity under GPT-2), so its values are
#: negated before sweeping and rendered back as a ``doc_ppl <= cut`` rule.
SWEEP_MEASUREMENTS: Dict[str, Dict[str, Any]] = {
    "stylometry": {"measurement": "confidence_score", "machine_higher": True, "places": 3},
    "perplexity": {"measurement": "doc_ppl", "machine_higher": False, "places": 2},
}

#: A full sweep is one row per distinct observed value. Published only when it
#: stays this small; beyond it the target table above carries the decision
#: content and the doc says so instead of printing hundreds of rows.
MAX_FULL_SWEEP_ROWS = 25


def load_scores(path: Path) -> List[Dict[str, Any]]:
    """Read a committed scores JSONL file, validating the fields the table needs."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for field in ("doc_id", "label"):
                if field not in record:
                    raise ValueError(f"{path}:{line_number} is missing required field {field!r}")
            if record["label"] not in metrics.LABELS:
                raise ValueError(f"{path}:{line_number} has unknown label {record['label']!r}")
            records.append(record)
    return records


def load_provenance(path: Path) -> Dict[str, str]:
    """Corpus and baseline recorded beside the scores, for the header lines."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: str(data[key]) for key in ("corpus", "baseline") if key in data}


def measurement_pairs(records: Sequence[Dict[str, Any]], method: str) -> List[Tuple[str, float]]:
    """(label, machine-oriented score) pairs for the method's sweep measurement.

    Sweeps must run over the recorded *measurement*, never the recorded
    ``high_ai_probability`` flag: the flags were decided at the thresholds in
    effect when the harness ran, so a sweep over the flags would reproduce those
    thresholds and nothing else. Documents whose analysis failed are excluded -
    an error-shaped result reports ``high_ai_probability: false``, and counting
    that as "predicted human" would deflate the false positive rate - as are
    documents where the measurement itself is null.
    """
    spec = SWEEP_MEASUREMENTS[method]
    scored, _ = metrics.partition_outcomes(records, method)
    pairs = []
    for record in scored:
        value = (record[method].get("measurements") or {}).get(spec["measurement"])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        score = float(value)
        pairs.append((record["label"], score if spec["machine_higher"] else -score))
    return pairs


def _num(value: Optional[float], places: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{places}f}"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _display_cut(method: str, threshold: float) -> str:
    """Render an internal (machine-oriented) sweep threshold in measurement units."""
    spec = SWEEP_MEASUREMENTS[method]
    value = threshold if spec["machine_higher"] else -threshold
    relation = ">=" if spec["machine_higher"] else "<="
    return f"{relation} {value:.{spec['places']}f}"


def _target_table(method: str, pairs: Sequence[Tuple[str, float]]) -> List[str]:
    """FPR (with precision and recall) at each declared TPR target."""
    lines = [
        "| target TPR | cut | achieved TPR | precision | recall (TPR) | FPR | TP | FN | FP | TN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.fpr_at_tpr_targets(pairs, TPR_TARGETS):
        matrix = row.get("confusion_matrix")
        if matrix is None:
            # No cut point in this sample reaches the target; threshold is None and
            # the row renders as unmeasurable rather than inventing a cut.
            values: Dict[str, Optional[float]] = {"precision": None, "recall": None, "fpr": None}
            cut = "n/a"
        else:
            values = metrics.classification_metrics(matrix)
            cut = _display_cut(method, row["threshold"])
        lines.append(
            f"| {_pct(row['target_tpr'])} | {cut} | {_num(row['achieved_tpr'])} "
            f"| {_num(values['precision'])} | {_num(values['recall'])} | {_num(values['fpr'])} "
            f"| {_matrix_counts(matrix)} |"
        )
    return lines


def _matrix_counts(matrix: Optional[Dict[str, int]]) -> str:
    if matrix is None:
        return "- | - | - | -"
    return " | ".join(str(matrix[key]) for key in ("tp", "fn", "fp", "tn"))


def _sweep_section(records: Sequence[Dict[str, Any]], method: str, title: str, note: str) -> List[str]:
    """Target-TPR table plus (when small) the full sweep for one measurement."""
    spec = SWEEP_MEASUREMENTS[method]
    pairs = measurement_pairs(records, method)
    cut_name = spec["measurement"]

    lines = [f"### {title}", "", note, ""]
    lines += [
        "Each row is the highest cut point whose measured recall still reaches the target - the "
        "cheapest threshold for that recall, since a higher cut can only lower the FPR. Scores are "
        "coarse and discrete, so the achieved TPR is reported next to the target rather than "
        "assumed equal to it.",
        "",
        "Cut points that reach no target are omitted; `-` means no threshold in this sample reaches the target at all.",
        "",
    ]
    lines += _target_table(method, pairs)
    lines.append("")

    sweep = metrics.threshold_sweep(pairs)
    if len(sweep) > MAX_FULL_SWEEP_ROWS:
        lines += [
            f"`{cut_name}` takes {len(sweep)} distinct values across this corpus, more than the "
            f"{MAX_FULL_SWEEP_ROWS}-row limit for a published full sweep, so only the target rows "
            f"above are reproduced here. The full sweep is computed by the same "
            f"`threshold_sweep` helper and available from the generator.",
            "",
        ]
        return lines

    lines += [
        f"#### Full `{cut_name}` sweep",
        "",
        f"Every distinct `{cut_name}` value as a cut point, ascending.",
        "",
        "| cut | precision | recall (TPR) | FPR | TP | FN | FP | TN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in sweep:
        values = point["metrics"]
        lines.append(
            f"| {_display_cut(method, point['threshold'])} | {_num(values['precision'])} "
            f"| {_num(values['recall'])} | {_num(values['fpr'])} | {_matrix_counts(point['confusion_matrix'])} |"
        )
    lines.append("")
    return lines


def _shipped_section(records: Sequence[Dict[str, Any]], method: str, title: str) -> List[str]:
    """The recorded boolean decision, measured as-is on the labeled corpus."""
    evaluation = metrics.evaluate_boolean_decision(records, method)
    scored, failed = evaluation["scored_label_counts"], evaluation["failed_label_counts"]
    values = evaluation["metrics"]
    lines = [
        f"### {title}",
        "",
        f"Documents: {evaluation['n_total']} total, {evaluation['n_scored']} scored "
        f"({scored[metrics.HUMAN]} human / {scored[metrics.MACHINE]} machine), "
        f"{evaluation['n_failed']} failed "
        f"({failed[metrics.HUMAN]} human / {failed[metrics.MACHINE]} machine). "
        "Failed analyses are excluded from the matrix below rather than counted as negatives.",
        "",
    ]
    lines += report_module._confusion_lines(evaluation["confusion_matrix"])
    lines += [
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| accuracy | {_num(values['accuracy'])} |",
        f"| precision | {_num(values['precision'])} |",
        f"| recall (TPR) | {_num(values['recall'])} |",
        f"| F1 | {_num(values['f1'])} |",
        f"| false positive rate | {_num(values['fpr'])} |",
        f"| specificity | {_num(values['specificity'])} |",
        "",
    ]
    return lines


def render_calibration(
    records: Sequence[Dict[str, Any]], scores_path: Path, provenance: Optional[Dict[str, str]] = None
) -> str:
    """Render the published calibration Markdown. Deterministic in its inputs."""
    provenance = provenance or {}
    counts = metrics.label_counts(records)
    try:
        scores_label = str(Path(scores_path).relative_to(REPO_ROOT))
    except ValueError:
        scores_label = str(scores_path)

    lines = [
        "# AI-detection calibration",
        "",
        "Published operating characteristics for the two AI-detection signals, `stylometric_analysis` "
        "and `perplexity_analysis`, derived entirely from the committed benchmark scores - no new "
        "data collection, no model inference. Positive class is **machine-generated text**: a true "
        "positive is a machine-written document the detector flagged, a false positive is a "
        "human-written document it flagged.",
        "",
        report_module.DISCLAIMER,
        "",
        "This file is generated; do not edit it by hand. Regenerate with "
        "`uv run benchmarks/calibrate_detection.py` and commit the result - CI fails when this "
        "file drifts from the committed scores, so a threshold change must re-derive the published "
        "numbers.",
        "",
        "## Provenance",
        "",
        f"- Scores: `{scores_label}` - {len(records)} documents ({counts[metrics.HUMAN]} human / "
        f"{counts[metrics.MACHINE]} machine)",
    ]
    if provenance.get("corpus") or provenance.get("baseline"):
        lines.append(
            f"- Corpus: `{provenance.get('corpus', 'n/a')}`, baseline `{provenance.get('baseline', 'n/a')}` "
            "(from `run_metadata.json` beside the scores)"
        )
    lines += [
        "- Scores were produced by `benchmarks/run_benchmark.py` running the repository's own "
        "detector code; `benchmarks/reports/raid_wiki_v1/report.md` records the thresholds in "
        "effect for the boolean flags below.",
        "",
        "## Shipped boolean decisions, measured",
        "",
        "The recorded `flags.high_ai_probability` decisions, scored as-is on the labeled corpus.",
        "",
    ]
    lines += _shipped_section(records, "stylometry", "`stylometric_analysis` → `flags.high_ai_probability`")
    lines += _shipped_section(records, "perplexity", "`perplexity_analysis` → `flags.high_ai_probability`")

    lines += [
        "## Single-measurement sweeps at target true-positive rates",
        "",
        "The shipped booleans each combine several conditions, so no single knob maps onto them. "
        "The tables below are deliberately simpler classifiers - one measurement, one cut - built "
        "from the same recorded scores. They answer the question a threshold change actually asks "
        "(what recall costs at a given false-positive rate) and are expected to disagree with the "
        "shipped decisions above. Sweeps run over the recorded measurements, never over the "
        "recorded boolean flags: the flags were decided at the thresholds in effect when the "
        "harness ran, so sweeping them would reproduce those thresholds and nothing else.",
        "",
    ]
    lines += _sweep_section(
        records,
        "stylometry",
        "Stylometry: `confidence_score` against a cut",
        "`confidence_score` is the 0-1 number the stylometry flag thresholds on - the sweep cuts it "
        "with `confidence_score >= t`, the direction the machine class scores higher.",
    )
    lines += _sweep_section(
        records,
        "perplexity",
        "Perplexity: `doc_ppl` against a cut",
        "Machine text measures *lower* GPT-2 perplexity, so the cut renders as `doc_ppl <= t`; "
        "internally the measurement is negated and the same `score >= threshold` helpers run "
        "unchanged.",
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Regenerate the published table, or (--check) verify it is still current."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scores", type=Path, default=DEFAULT_SCORES, help="Committed scores JSONL to derive the table from"
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="run_metadata.json beside the scores")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Markdown file to write (or check)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 instead of writing when the published table differs from a regeneration",
    )
    args = parser.parse_args(argv)

    records = load_scores(args.scores)
    text = render_calibration(records, args.scores, provenance=load_provenance(args.metadata))

    if args.check:
        published = args.out.read_text(encoding="utf-8") if args.out.exists() else None
        if published == text:
            print(f"ok: {args.out} matches a regeneration from {args.scores}", file=sys.stderr)
            return 0
        print(
            f"drift: {args.out} does not match a regeneration from {args.scores}\n"
            "fix: uv run benchmarks/calibrate_detection.py  # then commit the updated table",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} from {args.scores} ({len(records)} documents)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

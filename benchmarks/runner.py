"""Score a labeled corpus with the repository's own detector code.

Every number this module produces comes out of ``AIDetectionAnalyzer`` - the
same object ``server/app.py`` hands the ``stylometric_analysis`` and
``perplexity_analysis`` MCP tools. Nothing here reimplements a feature, a
z-score, or a flag rule; it only records what the analyzer returned and,
when the analyzer failed, that it failed.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from server.analyzers.ai_detection import AIDetectionAnalyzer
from server.config import load_config
from server.models import initialize_models
from server.text_processing import initialize_preprocessor

# The re-export in ``server.text_processing`` is a no-op placeholder; the real
# initializer lives on the submodule, which is what conftest and the server use.
from server.text_processing.sentence_splitter import initialize_sentence_splitter

logger = logging.getLogger(__name__)

#: Analyses the runner knows how to drive, in report order.
ANALYSES = ("stylometry", "perplexity")

#: Feature keys dropped from the per-document record. Each is either an internal
#: intermediate or a several-hundred-entry mapping that would dominate the
#: output file without telling a reader anything a summary statistic doesn't.
_BULK_FEATURE_KEYS = ("sentence_positions", "char_ngram_profile", "function_word_freqs", "pos_bigram_ratios")


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    """Read a corpus JSONL file, validating the fields the runner depends on."""
    records = []
    seen_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for field in ("doc_id", "label", "text"):
                if field not in record:
                    raise ValueError(f"{path}:{line_number} is missing required field {field!r}")
            if record["label"] not in ("human", "machine"):
                raise ValueError(f"{path}:{line_number} has unknown label {record['label']!r}")
            if record["doc_id"] in seen_ids:
                raise ValueError(f"{path}:{line_number} repeats doc_id {record['doc_id']!r}")
            seen_ids.add(record["doc_id"])
            records.append(record)
    return records


def build_analyzer(config: Optional[Dict[str, Any]] = None) -> AIDetectionAnalyzer:
    """Construct the analyzer exactly as the server does, models included.

    The shared text-processing globals are initialized here rather than left to
    a caller: the runner has to work outside pytest, where nothing else sets
    them up and the sentence splitter would otherwise have no model.
    """
    config = config if config is not None else load_config()
    managers = initialize_models(config)
    nlp = managers["spacy"].get_model()
    initialize_preprocessor(nlp)
    initialize_sentence_splitter(nlp)
    return AIDetectionAnalyzer(nlp, managers["gpt2"], config)


def _flatten_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce the analyzer's feature dict to flat, per-document scalars."""
    flattened: Dict[str, Any] = {}
    for key, value in features.items():
        if key in _BULK_FEATURE_KEYS:
            continue
        if key == "pos_ratios" and isinstance(value, dict):
            for tag in sorted(value):
                flattened[f"pos_ratios.{tag}"] = value[tag]
        elif isinstance(value, (int, float)) or value is None:
            flattened[key] = value
    return dict(sorted(flattened.items()))


def score_stylometry(analyzer: AIDetectionAnalyzer, text: str, baseline: str) -> Dict[str, Any]:
    """Run the real stylometric analysis and record its result or its failure."""
    result = analyzer.stylometric_analysis(text, baseline=baseline)
    if "error" in result:
        return {"ok": False, "error": result["error"]}

    flags = result.get("flags", {})
    return {
        "ok": True,
        "high_ai_probability": bool(flags.get("high_ai_probability")),
        "confidence_score": flags.get("confidence_score"),
        "ai_detection_confidence": flags.get("ai_detection_confidence"),
        "ai_indicators": flags.get("ai_indicators", []),
        "warnings": flags.get("warnings", []),
        "errors": flags.get("errors", []),
        # The two continuous per-document numbers the stylometry path produces
        # that are not features: the score the shipped flag thresholds on, and
        # the whole-profile n-gram similarity. Grouped so the report can
        # summarize their distributions the same way it does features.
        "measurements": {
            "confidence_score": flags.get("confidence_score"),
            "char_ngram_similarity": result.get("char_ngram_similarity"),
        },
        "features": _flatten_features(result.get("features", {})),
        "z_scores": dict(sorted((result.get("z_scores") or {}).items())),
    }


def score_perplexity(analyzer: AIDetectionAnalyzer, text: str) -> Dict[str, Any]:
    """Run the real perplexity analysis and record its result or its failure."""
    result = analyzer.perplexity_analysis(text)
    if "error" in result:
        return {"ok": False, "error": result["error"]}

    sentences = result.get("sentences", [])
    flags = result.get("flags", {})
    return {
        "ok": True,
        "high_ai_probability": bool(flags.get("high_ai_probability")),
        "reasons": flags.get("reasons", []),
        "measurements": {
            "doc_ppl": result.get("doc_ppl"),
            "doc_burstiness": result.get("doc_burstiness"),
        },
        "n_sentences": len(sentences),
        "n_scored_sentences": sum(1 for sentence in sentences if sentence.get("ppl") is not None),
    }


def score_records(
    records: Iterable[Dict[str, Any]],
    analyzer: AIDetectionAnalyzer,
    analyses: Sequence[str] = ANALYSES,
    baseline: str = "brown_corpus",
    progress: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Score every corpus record, preserving corpus order.

    An analyzer that raises rather than returning its error-shaped result is
    recorded as a failure too - a benchmark that aborted halfway would report
    nothing, and one that dropped the document would understate its own
    sample count.
    """
    scored = []
    for index, record in enumerate(records, start=1):
        text = record["text"]
        entry: Dict[str, Any] = {
            "doc_id": record["doc_id"],
            "label": record["label"],
            "source": record.get("source", {}),
        }
        for analysis in analyses:
            try:
                if analysis == "stylometry":
                    entry[analysis] = score_stylometry(analyzer, text, baseline)
                elif analysis == "perplexity":
                    entry[analysis] = score_perplexity(analyzer, text)
                else:
                    raise ValueError(f"Unknown analysis: {analysis!r}")
            except Exception as exc:  # noqa: BLE001 - a failed document must not end the run
                logger.warning("%s analysis raised on %s: %s", analysis, record["doc_id"], exc)
                entry[analysis] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        scored.append(entry)
        if progress is not None:
            progress(index, record["doc_id"])
    return scored


def effective_thresholds(config: Dict[str, Any]) -> Dict[str, Any]:
    """The threshold constants that produced a run's decisions, for the report."""
    perplexity = config.get("perplexity", {}).get("thresholds", {})
    stylometry = config.get("stylometry", {}).get("thresholds", {})
    thresholds = {f"perplexity.thresholds.{key}": value for key, value in perplexity.items()}
    thresholds.update({f"stylometry.thresholds.{key}": value for key, value in stylometry.items()})
    thresholds["perplexity.model_name"] = config.get("perplexity", {}).get("model_name")
    return thresholds


def write_scores(path: Path, scored: Sequence[Dict[str, Any]]) -> None:
    """Write per-document scores as JSONL with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in scored:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")

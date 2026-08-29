"""
Build a stylometric baseline from a corpus of an author's own writing.

The built-in ``brown_corpus`` baseline answers "does this read like typical
1961 published prose." That's often the wrong question: a technical blog is
never going to look like Brown Corpus fiction and news, AI-authored or not.
This module answers a narrower, more useful question instead: "does this read
like *this author's own* pre-existing writing," by building a baseline the
same shape as ``brown_corpus`` from a corpus of the author's own documents.

Not every feature ``StylemetricAnalyzer.extract_features`` computes is safe to
average across documents of different lengths. Type-token ratio and the hapax
legomena rate both fall monotonically as a document gets longer, for any
author - it is a property of counting distinct types out of a shrinking-relative
total, not a property of style. A baseline built from a corpus with mixed
document lengths, scored against drafts of yet another length, will flag length
differences as if they were style differences. ``DEFAULT_ROBUST_FEATURES`` and
``DEFAULT_ROBUST_POS_TAGS`` exclude those two features (and every POS tag other
than ADP/DET) for that reason; pass ``ALL_SIMPLE_FEATURES`` explicitly to opt
back into the full, length-confounded feature set.
"""

import statistics
from typing import Any, Dict, List, Optional

from server.stylometry.analyzer import StylemetricAnalyzer

#: Simple (non-POS) features safe to average across documents of varying length.
DEFAULT_ROBUST_FEATURES = [
    "avg_sentence_len",
    "sentence_len_std",
    "fog",
    "kincaid",
]

#: POS tags safe to average across documents of varying length.
DEFAULT_ROBUST_POS_TAGS = ["ADP", "DET"]

#: Every simple feature StylemetricAnalyzer.extract_features computes, including
#: the length-confounded ones. Pass this as `features=` to opt into all of them.
ALL_SIMPLE_FEATURES = [
    "avg_sentence_len",
    "sentence_len_std",
    "ttr",
    "hapax_legomena_rate",
    "avg_word_len",
    "punct_density",
    "comma_ratio",
    "function_word_ratio",
    "fog",
    "kincaid",
]


def build_baseline_from_texts(
    texts: List[str],
    nlp_model,
    corpus_info: Optional[Dict[str, Any]] = None,
    features: Optional[List[str]] = None,
    pos_tags: Optional[List[str]] = None,
    min_words: int = 50,
) -> Dict[str, Any]:
    """
    Build a stylometric baseline (per-feature mean/std) from a corpus of documents.

    Each text is analyzed independently with `StylemetricAnalyzer`; the baseline is
    the per-feature mean and standard deviation *across documents*, not a single
    analysis of the concatenated corpus. The result is a baseline dict in the same
    schema as the built-in Brown Corpus baseline, so it can be saved with
    `BaselineManager.save_baseline()` and immediately used as
    `stylometric_analysis(text, baseline=<name>)`.

    Args:
        texts: Raw text of each document in the corpus.
        nlp_model: A loaded spaCy model, as passed to `StylemetricAnalyzer`.
        corpus_info: Optional metadata to store on the baseline (name, description,
            etc). `sample_size` and `language` are filled in automatically if absent.
        features: Which simple (non-POS) features to include. Defaults to
            `DEFAULT_ROBUST_FEATURES`. Pass `ALL_SIMPLE_FEATURES` to include every
            feature `extract_features` computes, including the length-confounded
            `ttr` and `hapax_legomena_rate` (see module docstring).
        pos_tags: Which spaCy POS tags to include as `pos_ratios`. Defaults to
            `DEFAULT_ROBUST_POS_TAGS`.
        min_words: Documents shorter than this (by whitespace-split word count) are
            dropped before computing statistics.

    Returns:
        `{"corpus_info": {...}, "statistics": {...}}`.

    Raises:
        ValueError: If fewer than two documents remain after the `min_words` filter -
            a standard deviation is undefined for fewer than two samples.
    """
    features = list(DEFAULT_ROBUST_FEATURES) if features is None else list(features)
    pos_tags = list(DEFAULT_ROBUST_POS_TAGS) if pos_tags is None else list(pos_tags)

    kept_texts = [text for text in texts if len(text.split()) >= min_words]
    if len(kept_texts) < 2:
        raise ValueError(
            f"Need at least 2 documents with >= {min_words} words to compute a standard "
            f"deviation; only {len(kept_texts)} qualified out of {len(texts)} given"
        )

    analyzer = StylemetricAnalyzer(nlp_model)
    per_doc_features = [analyzer.extract_features(text) for text in kept_texts]

    statistics_out: Dict[str, Any] = {}
    for feature in features:
        values = [doc[feature] for doc in per_doc_features if doc.get(feature) is not None]
        if len(values) < 2:
            continue
        statistics_out[feature] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}

    pos_ratios_out: Dict[str, Any] = {}
    for tag in pos_tags:
        values = [doc["pos_ratios"][tag] for doc in per_doc_features if tag in doc.get("pos_ratios", {})]
        if len(values) < 2:
            continue
        pos_ratios_out[tag] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
    if pos_ratios_out:
        statistics_out["pos_ratios"] = pos_ratios_out

    info = dict(corpus_info or {})
    info.setdefault("sample_size", len(kept_texts))
    info.setdefault("language", "en")

    return {"corpus_info": info, "statistics": statistics_out}

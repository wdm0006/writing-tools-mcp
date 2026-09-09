"""AI detection analysis functionality."""

import logging
import statistics
from typing import Any

import numpy as np
import torch

from server.stylometry import (
    BaselineManager,
    StylemetricAnalyzer,
    calculate_char_ngram_similarity,
    calculate_sentence_z_scores,
    calculate_z_scores,
    compute_statistic_deltas,
    compute_verdicts,
    generate_flags,
    resolve_baseline_name,
)
from server.text_processing import split_into_sentences

logger = logging.getLogger(__name__)


def _delta_error(message: str, baseline_used: str) -> dict:
    """The delta tool's error envelope: the tool family's ``{"error": ...}``
    shape, with the delta-specific keys present but empty so callers can
    destructure the response without shape-switching."""
    return {
        "error": message,
        "baseline_used": baseline_used,
        "deltas": [],
        "verdict": [],
        "text_b_analysis": None,
    }


class AIDetectionAnalyzer:
    """Handles AI-generated content detection using perplexity and stylometric analysis."""

    def __init__(self, nlp_model, gpt2_manager, config):
        self.nlp = nlp_model
        self.gpt2_manager = gpt2_manager
        self.config = config

        # Initialize stylometry components
        if nlp_model:
            self.stylometry_analyzer = StylemetricAnalyzer(nlp_model)
        # Config-driven: stylometry.custom_baselines_dir selects the custom save/load root.
        self.baseline_manager = BaselineManager(config)

    def perplexity_analysis(self, text: str, language: str = "en") -> dict:
        """
        Analyze text for perplexity and burstiness to detect AI-generated content.
        """
        if language != "en":
            return {
                "error": "Only English language ('en') is currently supported",
                "doc_ppl": None,
                "doc_burstiness": None,
                "sentences": [],
                "config": {},
                "flags": {"high_ai_probability": False, "reasons": []},
            }

        if not text.strip():
            return {
                "error": "Empty text provided",
                "doc_ppl": None,
                "doc_burstiness": None,
                "sentences": [],
                "config": {},
                "flags": {"high_ai_probability": False, "reasons": []},
            }

        try:
            # Load model and configuration
            model, tokenizer, config = self.gpt2_manager.get_model_and_tokenizer()

            # Split text into sentences
            sentences = split_into_sentences(text)
            if not sentences:
                return {
                    "error": "No valid sentences found in text",
                    "doc_ppl": None,
                    "doc_burstiness": None,
                    "sentences": [],
                    "config": config,
                    "flags": {"high_ai_probability": False, "reasons": []},
                }

            # Calculate perplexity for each sentence
            sentence_results = []
            sentence_perplexities = []

            for sentence in sentences:
                if sentence.strip():
                    # For long sentences, chunk them and average the perplexity
                    chunks = self._chunk_text(sentence, tokenizer, config["max_length"], config["overlap"])
                    chunk_perplexities = []

                    for chunk in chunks:
                        chunk_ppl = self._calculate_perplexity(chunk, model, tokenizer)
                        if not np.isinf(chunk_ppl):
                            chunk_perplexities.append(chunk_ppl)

                    # Average perplexity across chunks for this sentence
                    if chunk_perplexities:
                        sentence_ppl = sum(chunk_perplexities) / len(chunk_perplexities)
                    else:
                        sentence_ppl = float("inf")

                    sentence_results.append(
                        {"text": sentence, "ppl": round(sentence_ppl, 2) if np.isfinite(sentence_ppl) else None}
                    )

                    if np.isfinite(sentence_ppl):
                        sentence_perplexities.append(sentence_ppl)

            # Calculate document-level perplexity
            if sentence_perplexities:
                doc_ppl = sum(sentence_perplexities) / len(sentence_perplexities)
            else:
                doc_ppl = float("inf")

            # Calculate burstiness
            doc_burstiness = self._calculate_burstiness(sentence_perplexities)

            # Check against thresholds for AI detection flags
            flags: dict[str, Any] = {"high_ai_probability": False, "reasons": []}
            thresholds = config["thresholds"]

            # An unmeasurable burstiness is not a low one, so it can never contribute
            # to the AI flag. A measured burstiness implies two or more finite
            # sentence perplexities, and therefore a finite doc_ppl.
            burstiness_measured = doc_burstiness is not None
            low_burstiness = burstiness_measured and doc_burstiness < thresholds["burstiness_min"]
            low_ppl = np.isfinite(doc_ppl) and doc_ppl < thresholds["ppl_max"]

            if low_ppl and low_burstiness:
                flags["high_ai_probability"] = True
                flags["reasons"].append(
                    f"Low perplexity ({doc_ppl:.2f} < {thresholds['ppl_max']}) and low burstiness ({doc_burstiness:.2f} < {thresholds['burstiness_min']})"
                )
            elif low_ppl and burstiness_measured:
                flags["reasons"].append(
                    f"Low perplexity ({doc_ppl:.2f} < {thresholds['ppl_max']}) but acceptable burstiness ({doc_burstiness:.2f})"
                )
            elif low_ppl:
                flags["reasons"].append(
                    f"Low perplexity ({doc_ppl:.2f} < {thresholds['ppl_max']}); "
                    "burstiness requires at least two scored sentences"
                )
            elif low_burstiness:
                flags["reasons"].append(
                    f"Low burstiness ({doc_burstiness:.2f} < {thresholds['burstiness_min']}) but acceptable perplexity"
                )
            elif not burstiness_measured:
                flags["reasons"].append("Burstiness requires at least two scored sentences")

            return {
                "doc_ppl": round(doc_ppl, 2) if np.isfinite(doc_ppl) else None,
                "doc_burstiness": round(doc_burstiness, 2) if burstiness_measured else None,
                "sentences": sentence_results,
                "config": {"model": config["model_name"], "thresholds": thresholds},
                "flags": flags,
            }

        except Exception as e:
            logger.error(f"Error in perplexity analysis: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "doc_ppl": None,
                "doc_burstiness": None,
                "sentences": [],
                "config": {},
                "flags": {"high_ai_probability": False, "reasons": []},
            }

    def stylometric_analysis(self, text: str, baseline: str | None = None, language: str = "en") -> dict:
        """
        Analyze text for stylometric features and detect AI-generated content.

        An omitted ``baseline`` resolves to the configured ``stylometry.default_baseline``,
        falling back to the built-in default; every response reports the baseline actually
        measured against in ``baseline_used``.
        """
        baseline_used = resolve_baseline_name(baseline, self.config)

        if language != "en":
            return {
                "error": "Only English language ('en') is currently supported",
                "baseline_used": baseline_used,
                "features": {},
                "z_scores": {},
                "flags": {"high_ai_probability": False, "reasons": []},
                "sentence_analysis": [],
                "char_ngram_similarity": None,
                "config": {},
            }

        if not text.strip():
            return {
                "error": "Empty text provided",
                "baseline_used": baseline_used,
                "features": {},
                "z_scores": {},
                "flags": {"high_ai_probability": False, "reasons": []},
                "sentence_analysis": [],
                "char_ngram_similarity": None,
                "config": {},
            }

        try:
            # Load configuration
            stylometry_config = self.config.get("stylometry", {})
            thresholds = stylometry_config.get(
                "thresholds", {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}
            )

            # Load baseline
            try:
                baseline_data = self.baseline_manager.load_baseline(baseline_used)
                baseline_stats = baseline_data.get("statistics", {})
            except ValueError as e:
                return {
                    "error": f"Failed to load baseline '{baseline_used}': {str(e)}",
                    "baseline_used": baseline_used,
                    "features": {},
                    "z_scores": {},
                    "flags": {"high_ai_probability": False, "reasons": []},
                    "sentence_analysis": [],
                    "char_ngram_similarity": None,
                    "config": {"baseline": baseline_used, "thresholds": thresholds},
                }

            # Extract stylometric features
            features = self.stylometry_analyzer.extract_features(text)

            # Calculate z-scores against baseline
            z_scores = calculate_z_scores(features, baseline_stats)

            # Generate AI detection flags
            flags = generate_flags(z_scores, features, thresholds)

            # Calculate sentence-level z-scores
            sentence_analysis = calculate_sentence_z_scores(
                features.get("sentence_positions", []), baseline_stats.get("avg_sentence_len", {})
            )

            # Character n-gram profile similarity: a whole-profile comparison, not a
            # per-key z-score, so it's computed separately and reported alongside
            # z_scores rather than folded into it.
            char_ngram_similarity = calculate_char_ngram_similarity(features, baseline_stats)

            # Round numerical values for cleaner output. char_ngram_profile is dropped
            # here rather than rounded: it's an internal, several-hundred-entry
            # intermediate for char_ngram_similarity above, not something a caller
            # needs to see key-by-key.
            rounded_features = {}
            for key, value in features.items():
                if key == "char_ngram_profile":
                    continue
                elif key == "sentence_positions":
                    rounded_features[key] = value  # Keep as-is, already processed
                elif key == "pos_ratios":
                    rounded_features[key] = {k: round(v, 3) for k, v in value.items()}
                elif isinstance(value, float):
                    rounded_features[key] = round(value, 3)
                else:
                    rounded_features[key] = value

            rounded_z_scores = {k: round(v, 2) for k, v in z_scores.items()}

            return {
                "features": rounded_features,
                "z_scores": rounded_z_scores,
                "flags": flags,
                "sentence_analysis": sentence_analysis,
                "char_ngram_similarity": round(char_ngram_similarity, 3) if char_ngram_similarity is not None else None,
                "baseline_used": baseline_used,
                "config": {
                    "baseline": baseline_used,
                    "baseline_info": baseline_data.get("corpus_info", {}),
                    "thresholds": thresholds,
                },
            }

        except Exception as e:
            logger.error(f"Error in stylometric analysis: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "baseline_used": baseline_used,
                "features": {},
                "z_scores": {},
                "flags": {"high_ai_probability": False, "reasons": []},
                "sentence_analysis": [],
                "char_ngram_similarity": None,
                "config": {"baseline": baseline_used, "thresholds": thresholds},
            }

    def stylometric_delta(self, text_a: str, text_b: str, baseline: str | None = None) -> dict:
        """
        Compare a draft (``text_a``) against its revision (``text_b``) against one baseline.

        Both texts are profiled by :meth:`stylometric_analysis` against the same
        baseline — the configured ``stylometry.default_baseline`` when ``baseline``
        is omitted — and the response reports what the revision actually moved:
        per-statistic z-score deltas, an improved/regressed/unchanged verdict per
        statistic, and the revised text's full stylometric profile, so the
        response carries the evidence for its own findings.

        A statistic improves when the revision moves its z-score closer to zero
        (into the baseline's range) and regresses when the movement pushes it
        further out; only statistics measurable in both texts are compared.
        """
        baseline_used = resolve_baseline_name(baseline, self.config)

        if not text_a.strip():
            return _delta_error("Empty text_a provided", baseline_used)
        if not text_b.strip():
            return _delta_error("Empty text_b provided", baseline_used)

        try:
            analysis_a = self.stylometric_analysis(text_a, baseline_used, "en")
            if "error" in analysis_a:
                return _delta_error(str(analysis_a["error"]), baseline_used)

            analysis_b = self.stylometric_analysis(text_b, baseline_used, "en")
            if "error" in analysis_b:
                return _delta_error(str(analysis_b["error"]), baseline_used)

            deltas = compute_statistic_deltas(analysis_a["z_scores"], analysis_b["z_scores"])

            return {
                "baseline_used": baseline_used,
                "deltas": deltas,
                "verdict": compute_verdicts(deltas),
                "text_b_analysis": analysis_b,
            }

        except Exception as e:
            logger.error(f"Error in stylometric delta analysis: {e}")
            return _delta_error(f"Analysis failed: {str(e)}", baseline_used)

    def _chunk_text(self, text, tokenizer, max_length=512, overlap=50):
        """Split text into overlapping chunks for processing long texts."""
        # Tokenize the full text
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= max_length:
            return [text]

        # Ensure overlap is not larger than max_length to prevent infinite loops
        overlap = min(overlap, max_length - 1)

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_length, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
            chunks.append(chunk_text)

            # Move to next chunk with overlap
            next_start = end - overlap

            # Ensure we always advance to prevent infinite loops
            if next_start <= start:
                next_start = start + 1

            start = next_start

            # Safety check to prevent infinite loops
            if len(chunks) > 100:  # Reasonable upper limit
                break

        return chunks

    def _calculate_perplexity(self, text, model, tokenizer):
        """Calculate perplexity for a given text using GPT-2."""
        if not text.strip():
            return float("inf")

        try:
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs.input_ids

            # Calculate loss
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()

            # Calculate perplexity from loss
            perplexity = torch.exp(torch.tensor(loss)).item()

            # A single-token input gives the model nothing to predict, so the loss
            # comes back NaN and so does exp(loss). NaN is not a usable perplexity
            # and it poisons every downstream aggregation, so report it the same
            # way as a failed calculation.
            if not np.isfinite(perplexity):
                return float("inf")

            return perplexity

        except Exception as e:
            logger.warning(f"Error calculating perplexity for text: {e}")
            return float("inf")

    def _calculate_burstiness(self, sentence_perplexities):
        """Calculate burstiness as the standard deviation of sentence perplexities.

        Returns None when fewer than two sentences were scored: a sample standard
        deviation is undefined there, and reporting 0.0 would be indistinguishable
        from a real reading of "perfectly uniform".
        """
        if len(sentence_perplexities) < 2:
            return None

        # Filter out non-finite values. `statistics.stdev` raises
        # "cannot convert NaN to integer ratio" on a NaN, which previously failed
        # the whole analysis, so this must reject NaN as well as infinity.
        valid_perplexities = [p for p in sentence_perplexities if np.isfinite(p)]

        if len(valid_perplexities) < 2:
            return None

        return statistics.stdev(valid_perplexities)

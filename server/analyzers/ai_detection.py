"""AI detection analysis functionality."""

import logging
import statistics

import numpy as np
import torch

from server.stylometry import (
    BaselineManager,
    StylemetricAnalyzer,
    calculate_sentence_z_scores,
    calculate_z_scores,
    generate_flags,
)
from server.text_processing import split_into_sentences

logger = logging.getLogger(__name__)


class AIDetectionAnalyzer:
    """Handles AI-generated content detection using perplexity and stylometric analysis."""

    def __init__(self, nlp_model, gpt2_manager, config):
        self.nlp = nlp_model
        self.gpt2_manager = gpt2_manager
        self.config = config

        # Initialize stylometry components
        if nlp_model:
            self.stylometry_analyzer = StylemetricAnalyzer(nlp_model)
        self.baseline_manager = BaselineManager()

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
                        {"text": sentence, "ppl": round(sentence_ppl, 2) if not np.isinf(sentence_ppl) else None}
                    )

                    if not np.isinf(sentence_ppl):
                        sentence_perplexities.append(sentence_ppl)

            # Calculate document-level perplexity
            if sentence_perplexities:
                doc_ppl = sum(sentence_perplexities) / len(sentence_perplexities)
            else:
                doc_ppl = float("inf")

            # Calculate burstiness
            doc_burstiness = self._calculate_burstiness(sentence_perplexities)

            # Check against thresholds for AI detection flags
            flags = {"high_ai_probability": False, "reasons": []}
            thresholds = config["thresholds"]

            if not np.isinf(doc_ppl) and doc_ppl < thresholds["ppl_max"]:
                if doc_burstiness < thresholds["burstiness_min"]:
                    flags["high_ai_probability"] = True
                    flags["reasons"].append(
                        f"Low perplexity ({doc_ppl:.2f} < {thresholds['ppl_max']}) and low burstiness ({doc_burstiness:.2f} < {thresholds['burstiness_min']})"
                    )
                else:
                    flags["reasons"].append(
                        f"Low perplexity ({doc_ppl:.2f} < {thresholds['ppl_max']}) but acceptable burstiness ({doc_burstiness:.2f})"
                    )
            elif doc_burstiness < thresholds["burstiness_min"]:
                flags["reasons"].append(
                    f"Low burstiness ({doc_burstiness:.2f} < {thresholds['burstiness_min']}) but acceptable perplexity"
                )

            return {
                "doc_ppl": round(doc_ppl, 2) if not np.isinf(doc_ppl) else None,
                "doc_burstiness": round(doc_burstiness, 2),
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

    def stylometric_analysis(self, text: str, baseline: str = "brown_corpus", language: str = "en") -> dict:
        """
        Analyze text for stylometric features and detect AI-generated content.
        """
        if language != "en":
            return {
                "error": "Only English language ('en') is currently supported",
                "features": {},
                "z_scores": {},
                "flags": {"high_ai_probability": False, "reasons": []},
                "sentence_analysis": [],
                "config": {},
            }

        if not text.strip():
            return {
                "error": "Empty text provided",
                "features": {},
                "z_scores": {},
                "flags": {"high_ai_probability": False, "reasons": []},
                "sentence_analysis": [],
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
                baseline_data = self.baseline_manager.load_baseline(baseline)
                baseline_stats = baseline_data.get("statistics", {})
            except ValueError as e:
                return {
                    "error": f"Failed to load baseline '{baseline}': {str(e)}",
                    "features": {},
                    "z_scores": {},
                    "flags": {"high_ai_probability": False, "reasons": []},
                    "sentence_analysis": [],
                    "config": {"baseline": baseline, "thresholds": thresholds},
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

            # Round numerical values for cleaner output
            rounded_features = {}
            for key, value in features.items():
                if key == "sentence_positions":
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
                "config": {
                    "baseline": baseline,
                    "baseline_info": baseline_data.get("corpus_info", {}),
                    "thresholds": thresholds,
                },
            }

        except Exception as e:
            logger.error(f"Error in stylometric analysis: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "features": {},
                "z_scores": {},
                "flags": {"high_ai_probability": False, "reasons": []},
                "sentence_analysis": [],
                "config": {"baseline": baseline, "thresholds": thresholds},
            }

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

            return perplexity

        except Exception as e:
            logger.warning(f"Error calculating perplexity for text: {e}")
            return float("inf")

    def _calculate_burstiness(self, sentence_perplexities):
        """Calculate burstiness as the standard deviation of sentence perplexities."""
        if len(sentence_perplexities) < 2:
            return 0.0

        # Filter out infinite values
        valid_perplexities = [p for p in sentence_perplexities if not np.isinf(p)]

        if len(valid_perplexities) < 2:
            return 0.0

        return statistics.stdev(valid_perplexities)

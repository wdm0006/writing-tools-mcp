"""
Statistical analysis module for stylometric features.

This module provides functions for calculating z-scores against baselines,
flagging outliers, and generating AI detection confidence scores.
"""

from typing import Any, Dict, List


def calculate_z_scores(features: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate z-scores for each feature against baseline statistics.

    Args:
        features: Dictionary of extracted stylometric features
        baseline: Dictionary of baseline statistics with mean/std for each feature

    Returns:
        Dictionary mapping feature names to their z-scores
    """
    z_scores = {}

    # Simple features that can be directly compared
    simple_features = [
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
        "mtld",
        "mattr",
        "smog",
        "coleman_liau",
        "ari",
        "dale_chall",
        "mean_dependency_distance",
        "subordinate_clause_ratio",
        "fourgram_repetition_rate",
        "zipf_slope",
    ]

    for feature in simple_features:
        if feature in features and feature in baseline:
            feature_value = features[feature]
            baseline_stats = baseline[feature]

            # A feature that could not be measured has no z-score at all; scoring
            # it as if it were 0.0 would misreport an absent value as an outlier.
            if feature_value is None:
                continue

            if isinstance(baseline_stats, dict) and "mean" in baseline_stats and "std" in baseline_stats:
                mean = baseline_stats["mean"]
                std = baseline_stats["std"]

                if std > 0:  # Avoid division by zero
                    z_score = (feature_value - mean) / std
                    z_scores[feature] = z_score
                else:
                    z_scores[feature] = 0.0
            else:
                z_scores[feature] = 0.0

    # Handle POS ratios separately
    if "pos_ratios" in features and "pos_ratios" in baseline:
        feature_pos = features["pos_ratios"]
        baseline_pos = baseline["pos_ratios"]

        for pos_tag, baseline_stats in baseline_pos.items():
            if pos_tag in feature_pos:
                feature_value = feature_pos[pos_tag]
                if isinstance(baseline_stats, dict) and "mean" in baseline_stats and "std" in baseline_stats:
                    mean = baseline_stats["mean"]
                    std = baseline_stats["std"]

                    if std > 0:
                        z_score = (feature_value - mean) / std
                        z_scores[f"pos_{pos_tag.lower()}"] = z_score
                    else:
                        z_scores[f"pos_{pos_tag.lower()}"] = 0.0

    # Handle per-function-word frequencies (Burrows' Delta): score each tracked word
    # individually against the baseline, then reduce all of them to one aggregate
    # distance - the mean absolute z-score across every word actually scored. This is
    # Burrows' Delta's own definition, just built on top of the same z-scoring already
    # done per word, rather than a separate statistic.
    if "function_word_freqs" in features and "function_word_freqs" in baseline:
        feature_freqs = features["function_word_freqs"]
        baseline_freqs = baseline["function_word_freqs"]
        word_z_scores = []

        for word, baseline_stats in baseline_freqs.items():
            if word not in feature_freqs:
                continue
            feature_value = feature_freqs[word]
            if not (isinstance(baseline_stats, dict) and "mean" in baseline_stats and "std" in baseline_stats):
                continue

            mean = baseline_stats["mean"]
            std = baseline_stats["std"]
            z_score = (feature_value - mean) / std if std > 0 else 0.0
            z_scores[f"fw_{word}"] = z_score
            word_z_scores.append(z_score)

        if word_z_scores:
            z_scores["burrows_delta"] = sum(abs(z) for z in word_z_scores) / len(word_z_scores)

    return z_scores


def flag_outliers(
    z_scores: Dict[str, float], warning_threshold: float = 2.0, error_threshold: float = 3.0
) -> Dict[str, List[str]]:
    """
    Flag features with abnormal z-scores as warnings or errors.

    Args:
        z_scores: Dictionary of feature z-scores
        warning_threshold: Absolute z-score threshold for warnings (default: 2.0)
        error_threshold: Absolute z-score threshold for errors (default: 3.0)

    Returns:
        Dictionary with 'warnings' and 'errors' lists of flagged features
    """
    warnings = []
    errors = []

    for feature, z_score in z_scores.items():
        abs_z = abs(z_score)

        if abs_z >= error_threshold:
            errors.append(feature)
        elif abs_z >= warning_threshold:
            warnings.append(feature)

    return {"warnings": warnings, "errors": errors}


def generate_flags(
    z_scores: Dict[str, float], features: Dict[str, Any], thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """
    Generate AI detection flags based on z-score patterns and feature combinations.

    Args:
        z_scores: Dictionary of feature z-scores
        features: Original feature values
        thresholds: Configuration thresholds

    Returns:
        Dictionary with AI detection flags, confidence, and explanations
    """
    warning_threshold = thresholds.get("warning_z", 2.0)
    error_threshold = thresholds.get("error_z", 3.0)
    confidence_threshold = thresholds.get("ai_confidence_threshold", 0.7)

    # Get outlier flags
    outlier_flags = flag_outliers(z_scores, warning_threshold, error_threshold)

    # Calculate AI detection confidence based on multiple factors
    ai_indicators = []
    confidence_score = 0.0
    reasons = []

    # Check for key AI indicators
    # 1. Low lexical diversity (TTR)
    if "ttr" in z_scores and z_scores["ttr"] < -warning_threshold:
        ai_indicators.append("low_ttr")
        confidence_score += 0.3
        reasons.append(f"Low lexical diversity (TTR z-score: {z_scores['ttr']:.2f})")

    # 2. Low hapax legomena rate (repetitive vocabulary)
    if "hapax_legomena_rate" in z_scores and z_scores["hapax_legomena_rate"] < -warning_threshold:
        ai_indicators.append("low_hapax")
        confidence_score += 0.25
        reasons.append(f"Low hapax legomena rate (z-score: {z_scores['hapax_legomena_rate']:.2f})")

    # 3. Uniform sentence lengths (low standard deviation)
    if "sentence_len_std" in z_scores and z_scores["sentence_len_std"] < -warning_threshold:
        ai_indicators.append("uniform_sentences")
        confidence_score += 0.2
        reasons.append(f"Uniform sentence lengths (std z-score: {z_scores['sentence_len_std']:.2f})")

    # 4. Unusual average sentence length
    if "avg_sentence_len" in z_scores and abs(z_scores["avg_sentence_len"]) > warning_threshold:
        ai_indicators.append("unusual_sentence_length")
        confidence_score += 0.15
        direction = "short" if z_scores["avg_sentence_len"] < 0 else "long"
        reasons.append(f"Unusually {direction} sentences (avg length z-score: {z_scores['avg_sentence_len']:.2f})")

    # 5. Check POS ratio anomalies
    pos_anomalies = []
    for feature, z_score in z_scores.items():
        if feature.startswith("pos_") and abs(z_score) > warning_threshold:
            pos_tag = feature.replace("pos_", "").upper()
            pos_anomalies.append(f"{pos_tag}: {z_score:.2f}")
            confidence_score += 0.1

    if pos_anomalies:
        ai_indicators.append("pos_anomalies")
        reasons.append(f"Unusual POS ratios ({', '.join(pos_anomalies)})")

    # 6. Function word ratio anomalies
    if "function_word_ratio" in z_scores and abs(z_scores["function_word_ratio"]) > warning_threshold:
        ai_indicators.append("function_word_anomaly")
        confidence_score += 0.1
        direction = "low" if z_scores["function_word_ratio"] < 0 else "high"
        reasons.append(f"Unusual function word usage ({direction}, z-score: {z_scores['function_word_ratio']:.2f})")

    # 7. Unusual reading grade level. Gunning Fog is used as the single representative
    # readability z-score rather than also checking Kincaid, SMOG, Coleman-Liau, ARI, or
    # Dale-Chall: all are correlated grade-level/complexity estimates, and scoring more
    # than one would double-count what is really a single underlying signal.
    if "fog" in z_scores and abs(z_scores["fog"]) > warning_threshold:
        ai_indicators.append("unusual_reading_level")
        confidence_score += 0.15
        direction = "simpler" if z_scores["fog"] < 0 else "more complex"
        reasons.append(f"Unusually {direction} reading level (Gunning Fog z-score: {z_scores['fog']:.2f})")

    # 8. Low MTLD: a length-robust lexical-diversity check, alongside (not replacing)
    # the raw-TTR check above - MTLD is the more reliable of the two at varying
    # document lengths, but TTR is cheap to keep scoring wherever a baseline has it.
    if "mtld" in z_scores and z_scores["mtld"] < -warning_threshold:
        ai_indicators.append("low_mtld")
        confidence_score += 0.25
        reasons.append(f"Low length-robust lexical diversity (MTLD z-score: {z_scores['mtld']:.2f})")

    # 9. Burrows' Delta: mean absolute z-score across individually-tracked function
    # words. Delta itself is already a magnitude (mean of absolute values), so it's
    # compared directly against warning_threshold rather than via abs().
    if "burrows_delta" in z_scores and z_scores["burrows_delta"] > warning_threshold:
        ai_indicators.append("distinct_function_word_profile")
        confidence_score += 0.2
        reasons.append(
            f"Function-word usage differs sharply from baseline (Burrows' Delta: {z_scores['burrows_delta']:.2f})"
        )

    # Cap confidence score at 1.0
    confidence_score = min(confidence_score, 1.0)

    # Determine confidence level
    if confidence_score >= confidence_threshold:
        confidence_level = "high"
    elif confidence_score >= 0.4:
        confidence_level = "medium"
    elif confidence_score >= 0.2:
        confidence_level = "low"
    else:
        confidence_level = "very_low"

    # Overall AI detection flag
    high_ai_probability = confidence_score >= confidence_threshold and len(ai_indicators) >= 2

    return {
        "warnings": outlier_flags["warnings"],
        "errors": outlier_flags["errors"],
        "ai_indicators": ai_indicators,
        "ai_detection_confidence": confidence_level,
        "confidence_score": round(confidence_score, 3),
        "high_ai_probability": high_ai_probability,
        "reasons": reasons,
    }


def calculate_sentence_z_scores(
    sentence_positions: List[Dict[str, Any]], baseline_sentence_stats: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Calculate z-scores for individual sentences relative to baseline.

    Args:
        sentence_positions: List of sentence data with lengths
        baseline_sentence_stats: Baseline statistics for sentence lengths

    Returns:
        Updated sentence data with z-scores
    """
    if not baseline_sentence_stats or "mean" not in baseline_sentence_stats:
        # If no baseline, just return original data
        for sentence in sentence_positions:
            sentence["z_score"] = 0.0
        return sentence_positions

    mean = baseline_sentence_stats["mean"]
    std = baseline_sentence_stats.get("std", 1.0)

    if std <= 0:
        std = 1.0  # Avoid division by zero

    for sentence in sentence_positions:
        length = sentence["length"]
        z_score = (length - mean) / std
        sentence["z_score"] = round(z_score, 2)

    return sentence_positions

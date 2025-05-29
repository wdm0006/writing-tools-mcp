"""
Baseline management module for stylometric analysis.

This module provides the BaselineManager class for loading and managing
baseline statistics for comparing stylometric features against human writing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaselineManager:
    """Manager for loading and handling stylometric baselines."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize baseline manager.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.baselines = {}
        self._load_default_baselines()

    def _load_default_baselines(self):
        """Load built-in baselines."""
        # Load Brown Corpus baseline
        self.baselines["brown_corpus"] = self._get_brown_corpus_baseline()
        logger.info("Loaded Brown Corpus baseline")

    def load_baseline(self, baseline_name: str = "brown_corpus") -> Dict[str, Any]:
        """
        Load baseline statistics by name.

        Args:
            baseline_name: Name of the baseline to load

        Returns:
            Dictionary containing baseline statistics

        Raises:
            ValueError: If baseline is not found
        """
        if baseline_name in self.baselines:
            return self.baselines[baseline_name]

        # Try to load from file
        baseline_path = self._get_baseline_path(baseline_name)
        if baseline_path and baseline_path.exists():
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline_data = json.load(f)
                    self.baselines[baseline_name] = baseline_data
                    logger.info(f"Loaded custom baseline: {baseline_name}")
                    return baseline_data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading baseline {baseline_name}: {e}")
                raise ValueError(f"Failed to load baseline {baseline_name}: {e}") from e

        raise ValueError(f"Baseline '{baseline_name}' not found")

    def _get_baseline_path(self, baseline_name: str) -> Optional[Path]:
        """
        Get the file path for a custom baseline.

        Args:
            baseline_name: Name of the baseline

        Returns:
            Path to the baseline file or None if not found
        """
        # Check in data/baselines directory
        data_dir = Path(__file__).parent.parent.parent / "data" / "baselines"
        baseline_file = data_dir / f"{baseline_name}.json"

        if baseline_file.exists():
            return baseline_file

        # Check in custom baselines directory
        custom_dir = data_dir / "custom_baselines"
        custom_file = custom_dir / f"{baseline_name}.json"

        if custom_file.exists():
            return custom_file

        return None

    def _get_brown_corpus_baseline(self) -> Dict[str, Any]:
        """
        Get the Brown Corpus baseline statistics.

        This baseline is derived from analysis of the Brown Corpus,
        representing typical human writing across various domains.

        Returns:
            Dictionary containing Brown Corpus baseline statistics
        """
        return {
            "corpus_info": {
                "name": "Brown Corpus",
                "description": "Human writing baseline from Brown Corpus",
                "language": "en",
                "sample_size": 500,
                "domains": ["news", "fiction", "academic", "misc"],
                "version": "1.0",
            },
            "statistics": {
                # Sentence-level features
                "avg_sentence_len": {"mean": 17.8, "std": 8.2},
                "sentence_len_std": {"mean": 7.1, "std": 2.4},
                # Lexical diversity features
                "ttr": {"mean": 0.52, "std": 0.08},
                "hapax_legomena_rate": {"mean": 0.47, "std": 0.06},
                "avg_word_len": {"mean": 4.8, "std": 0.6},
                # Part-of-speech ratios
                "pos_ratios": {
                    "NOUN": {"mean": 0.23, "std": 0.04},
                    "VERB": {"mean": 0.16, "std": 0.03},
                    "ADJ": {"mean": 0.08, "std": 0.02},
                    "ADV": {"mean": 0.06, "std": 0.02},
                    "ADP": {"mean": 0.12, "std": 0.02},  # Prepositions
                    "DET": {"mean": 0.11, "std": 0.02},  # Determiners
                    "PRON": {"mean": 0.07, "std": 0.02},  # Pronouns
                    "CONJ": {"mean": 0.03, "std": 0.01},  # Conjunctions
                    "NUM": {"mean": 0.02, "std": 0.01},  # Numbers
                    "PART": {"mean": 0.02, "std": 0.01},  # Particles
                },
                # Punctuation features
                "punct_density": {"mean": 0.14, "std": 0.03},
                "comma_ratio": {"mean": 0.42, "std": 0.08},
                # Additional features
                "function_word_ratio": {"mean": 0.45, "std": 0.05},
            },
        }

    def save_baseline(self, baseline_name: str, baseline_data: Dict[str, Any], custom: bool = True) -> bool:
        """
        Save a baseline to file.

        Args:
            baseline_name: Name for the baseline
            baseline_data: Baseline statistics data
            custom: Whether to save as custom baseline (default: True)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if custom:
                # Save to custom baselines directory
                data_dir = Path(__file__).parent.parent.parent / "data" / "baselines" / "custom_baselines"
                data_dir.mkdir(parents=True, exist_ok=True)
                baseline_path = data_dir / f"{baseline_name}.json"
            else:
                # Save to main baselines directory
                data_dir = Path(__file__).parent.parent.parent / "data" / "baselines"
                data_dir.mkdir(parents=True, exist_ok=True)
                baseline_path = data_dir / f"{baseline_name}.json"

            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline_data, f, indent=2, ensure_ascii=False)

            # Cache in memory
            self.baselines[baseline_name] = baseline_data
            logger.info(f"Saved baseline: {baseline_name}")
            return True

        except (IOError, OSError) as e:
            logger.error(f"Error saving baseline {baseline_name}: {e}")
            return False

    def list_available_baselines(self) -> Dict[str, str]:
        """
        List all available baselines.

        Returns:
            Dictionary mapping baseline names to their descriptions
        """
        available = {}

        # Add built-in baselines
        for name, baseline in self.baselines.items():
            if "corpus_info" in baseline and "description" in baseline["corpus_info"]:
                available[name] = baseline["corpus_info"]["description"]
            else:
                available[name] = "Custom baseline"

        # Check for file-based baselines
        data_dir = Path(__file__).parent.parent.parent / "data" / "baselines"
        if data_dir.exists():
            for baseline_file in data_dir.glob("*.json"):
                name = baseline_file.stem
                if name not in available:
                    available[name] = "File-based baseline"

        # Check custom baselines directory
        custom_dir = data_dir / "custom_baselines"
        if custom_dir.exists():
            for baseline_file in custom_dir.glob("*.json"):
                name = baseline_file.stem
                if name not in available:
                    available[name] = "Custom baseline"

        return available

    def get_baseline_info(self, baseline_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific baseline.

        Args:
            baseline_name: Name of the baseline

        Returns:
            Baseline corpus information or None if not found
        """
        try:
            baseline = self.load_baseline(baseline_name)
            return baseline.get("corpus_info", {})
        except ValueError:
            return None

    def validate_baseline(self, baseline_data: Dict[str, Any]) -> bool:
        """
        Validate that baseline data has the required structure.

        Args:
            baseline_data: Baseline data to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(baseline_data, dict):
            return False

        if "statistics" not in baseline_data:
            return False

        stats = baseline_data["statistics"]
        required_features = ["avg_sentence_len", "ttr", "hapax_legomena_rate"]

        for feature in required_features:
            if feature not in stats:
                return False

            if not isinstance(stats[feature], dict):
                return False

            if "mean" not in stats[feature] or "std" not in stats[feature]:
                return False

        return True

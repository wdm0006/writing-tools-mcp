"""
Baseline management module for stylometric analysis.

This module provides the BaselineManager class for loading and managing
baseline statistics for comparing stylometric features against human writing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

BASELINES_DIR = Path(__file__).parent.parent / "data" / "baselines"
CUSTOM_BASELINES_SUBDIR = "custom_baselines"

#: Baseline used when a call omits one and no ``stylometry.default_baseline`` is configured.
DEFAULT_BASELINE_NAME = "brown_corpus"

_SAFE_BASELINE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_baseline_name(baseline_name: str) -> str:
    """
    Validate that a baseline name is a plain identifier, not a path.

    Args:
        baseline_name: Name of the baseline

    Returns:
        The validated baseline name

    Raises:
        ValueError: If the name is empty, contains path separators, parent
            references, or any other character outside the allowed set
    """
    if not isinstance(baseline_name, str) or not _SAFE_BASELINE_NAME.match(baseline_name) or ".." in baseline_name:
        raise ValueError(
            f"Invalid baseline name {baseline_name!r}: baseline names must contain only letters, digits, "
            "'.', '_' or '-', and may not contain path separators or parent directory references"
        )
    return baseline_name


def _resolve_baseline_file(directory: Path, baseline_name: str) -> Optional[Path]:
    """
    Resolve ``<directory>/<baseline_name>.json`` and confirm it stays inside ``directory``.

    Args:
        directory: Approved baseline directory
        baseline_name: Already-validated baseline name

    Returns:
        The resolved path, or None if it would escape the approved directory
    """
    approved_dir = directory.resolve()
    candidate = (approved_dir / f"{baseline_name}.json").resolve()

    if not candidate.is_relative_to(approved_dir) or candidate == approved_dir:
        logger.error(f"Rejected baseline path outside approved directory: {candidate}")
        return None

    return candidate


def custom_baselines_dir_from_config(config: Optional[Dict[str, Any]]) -> Optional[Path]:
    """
    Resolve ``stylometry.custom_baselines_dir`` to an absolute directory.

    Relative paths resolve against the server's working directory — the same anchor
    ``load_config`` uses for ``.mcp-config.yaml`` — so a path written in the config
    stays relative to where the server runs. Returns None when unset (or set to a
    non-string/empty value in a hand-built config), which keeps the built-in package
    directory as the save/load root.

    Args:
        config: Configuration dictionary (or None)

    Returns:
        The resolved directory, or None when no usable directory is configured
    """
    stylometry_config = (config or {}).get("stylometry", {})
    configured = stylometry_config.get("custom_baselines_dir") if isinstance(stylometry_config, dict) else None
    if not isinstance(configured, str) or not configured.strip():
        return None
    return Path(configured).expanduser().resolve()


def resolve_baseline_name(baseline: Optional[str], config: Optional[Dict[str, Any]]) -> str:
    """
    Pick the baseline a stylometry call measures against.

    An explicit per-call argument wins; otherwise the configured
    ``stylometry.default_baseline``; otherwise the built-in default. Empty or
    whitespace-only values count as omitted at each step, so the response's
    ``baseline_used`` always names a concrete baseline.

    Args:
        baseline: Baseline name passed by the caller, or None when omitted
        config: Configuration dictionary (or None)

    Returns:
        The baseline name to measure against
    """
    if baseline and baseline.strip():
        return baseline

    stylometry_config = (config or {}).get("stylometry", {})
    configured = stylometry_config.get("default_baseline") if isinstance(stylometry_config, dict) else None
    if isinstance(configured, str) and configured.strip():
        return configured

    return DEFAULT_BASELINE_NAME


class BaselineManager:
    """Manager for loading and handling stylometric baselines."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize baseline manager.

        Args:
            config: Optional configuration dictionary. ``stylometry.custom_baselines_dir``
                selects where custom baselines are saved and loaded from; when unset,
                the built-in package directory keeps that role.
        """
        self.config = config or {}
        self.custom_baselines_dir = custom_baselines_dir_from_config(self.config)
        if self.custom_baselines_dir is not None:
            logger.info(f"Custom baselines directory: {self.custom_baselines_dir}")
        self.baselines: dict[str, dict[str, Any]] = {}
        self._load_default_baselines()

    def _search_directories(self) -> tuple[Path, ...]:
        """Directories searched for file-based baselines, most specific first.

        With a configured custom directory it wins over the package locations, so a
        user-built baseline shadows a same-named shipped file; the package custom
        subdir and built-in root stay reachable so baselines saved before a custom
        directory was configured (and shipped examples) still load.
        """
        if self.custom_baselines_dir is not None:
            return (self.custom_baselines_dir, BASELINES_DIR / CUSTOM_BASELINES_SUBDIR, BASELINES_DIR)
        return (BASELINES_DIR, BASELINES_DIR / CUSTOM_BASELINES_SUBDIR)

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
            ValueError: If the name is not a valid baseline identifier, or the baseline is not found
        """
        _validate_baseline_name(baseline_name)

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

        Raises:
            ValueError: If the name is not a valid baseline identifier
        """
        _validate_baseline_name(baseline_name)

        for directory in self._search_directories():
            baseline_file = _resolve_baseline_file(directory, baseline_name)
            if baseline_file is not None and baseline_file.is_file():
                return baseline_file

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

        Raises:
            ValueError: If the name is not a valid baseline identifier
        """
        _validate_baseline_name(baseline_name)

        if custom:
            # A configured custom_baselines_dir replaces the package custom subdir
            # as the save root; without one, behavior is unchanged.
            data_dir = (
                self.custom_baselines_dir
                if self.custom_baselines_dir is not None
                else BASELINES_DIR / CUSTOM_BASELINES_SUBDIR
            )
        else:
            data_dir = BASELINES_DIR

        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            baseline_path = _resolve_baseline_file(data_dir, baseline_name)
            if baseline_path is None:
                raise ValueError(f"Refusing to write baseline outside of {data_dir}")

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

        # Check file-based baselines, most specific source first. The built-in root
        # holds shipped file-backed baselines; everything else is user (or example) content.
        for directory in self._search_directories():
            if not directory.exists():
                continue
            label = "File-based baseline" if directory == BASELINES_DIR else "Custom baseline"
            for baseline_file in directory.glob("*.json"):
                name = baseline_file.stem
                if name not in available:
                    available[name] = label

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

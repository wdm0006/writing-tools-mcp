"""Default configuration values."""

from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "perplexity": {
        "model_name": "gpt2",
        "max_length": 512,
        "overlap": 50,
        "thresholds": {"ppl_max": 25.0, "burstiness_min": 2.5},
        "device": "cpu",
        "language": "en",
    },
    "stylometry": {"thresholds": {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}},
    "logging": {"level": "INFO", "format": "%(asctime)s - %(levelname)s - %(message)s"},
}

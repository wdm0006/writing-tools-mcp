"""Model management for spaCy and GPT-2 models."""

from server.models.gpt2_manager import GPT2Manager
from server.models.spacy_manager import SpacyManager

__all__ = [
    "SpacyManager",
    "GPT2Manager",
]


def initialize_models(config):
    """Initialize model managers with configuration (lazy loading)."""
    spacy_config = config.get("spacy", {})
    gpt2_config = config.get("gpt2", {})

    spacy_manager = SpacyManager(model_name=spacy_config.get("model_name", "en_core_web_sm"))
    gpt2_manager = GPT2Manager(gpt2_config)

    # Return managers instead of loaded models for lazy loading
    return {"spacy": spacy_manager, "gpt2": gpt2_manager}

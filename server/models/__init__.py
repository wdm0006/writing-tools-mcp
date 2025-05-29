"""Model management for spaCy and GPT-2 models."""

from .gpt2_manager import GPT2Manager
from .spacy_manager import SpacyManager

__all__ = [
    "SpacyManager",
    "GPT2Manager",
]


def initialize_models(config):
    """Initialize model managers with configuration."""
    spacy_config = config.get("spacy", {})
    gpt2_config = config.get("gpt2", {})

    spacy_manager = SpacyManager(model_name=spacy_config.get("model_name", "en_core_web_sm"))
    gpt2_manager = GPT2Manager(gpt2_config)

    return {"spacy": spacy_manager.get_model(), "gpt2": gpt2_manager}

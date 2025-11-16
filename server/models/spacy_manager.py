"""spaCy model management."""

import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

import spacy

logger = logging.getLogger(__name__)


class SpacyManager:
    """Manages spaCy model loading and initialization with lazy loading."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._model = None
        logger.info(f"SpacyManager initialized (model will be loaded on first use)")

    def get_model(self):
        """Get or load the spaCy model (lazy loading)."""
        if self._model is None:
            logger.info("Loading spaCy model on first use...")
            self._model = self._load_model()
            logger.info("spaCy model loaded successfully")
        return self._model

    def _load_model(self):
        """Load the spaCy model with fallback strategies."""
        try:
            return spacy.load(self.model_name)
        except TypeError:
            # Try loading from local path
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / self.model_name
            return spacy.load(str(model_path))
        except OSError:
            logger.info("Downloading spaCy English model, this may take a while...")
            from spacy.cli import download

            # Redirect stdout to stderr to avoid breaking MCP JSON-RPC protocol
            with redirect_stdout(sys.stderr):
                download(self.model_name)
            return spacy.load(self.model_name)

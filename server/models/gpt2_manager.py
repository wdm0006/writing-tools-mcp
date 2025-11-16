"""GPT-2 model management for perplexity analysis."""

import logging

from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


class GPT2Manager:
    """Manages GPT-2 model loading and caching with lazy loading."""

    def __init__(self, config: dict):
        self.config = config
        self._model = None
        self._tokenizer = None
        logger.info("GPT2Manager initialized (model will be loaded on first use)")

    def get_model_and_tokenizer(self):
        """Get or load the GPT-2 model and tokenizer (lazy loading)."""
        if self._model is None or self._tokenizer is None:
            self._load_model()
        return self._model, self._tokenizer, self.config
    
    def unload_model(self):
        """Unload the GPT-2 model to free memory."""
        if self._model is not None or self._tokenizer is not None:
            logger.info("Releasing GPT-2 model memory...")
            self._model = None
            self._tokenizer = None
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            import gc
            gc.collect()
            logger.info("GPT-2 model memory released")

    def _load_model(self):
        """Load the GPT-2 model and tokenizer."""
        try:
            logger.info("Loading GPT-2 model for perplexity analysis...")
            model_name = self.config.get("model_name", "gpt2")

            self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self._model = GPT2LMHeadModel.from_pretrained(model_name)

            # Configure model
            self._model.eval()
            if self.config.get("device") == "cpu":
                self._model = self._model.to("cpu")

            # Add padding token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            logger.info("GPT-2 model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading GPT-2 model: {e}")
            raise

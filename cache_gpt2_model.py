#!/usr/bin/env python3
"""
Script to pre-download and cache the GPT-2 model for faster subsequent loading.
Run this once to cache the model locally.
"""

import logging
import os
import sys
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Set up verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Enable transformers logging for download progress
logging.getLogger("transformers").setLevel(logging.INFO)
logging.getLogger("transformers.utils.hub").setLevel(logging.INFO)

def check_cache_status():
    """Check if model is already cached"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
    logger.info(f"Checking cache directory: {cache_dir}")
    
    if cache_dir.exists():
        cached_models = list(cache_dir.glob("*gpt2*"))
        if cached_models:
            logger.info(f"Found {len(cached_models)} cached GPT-2 related files:")
            for model in cached_models[:5]:  # Show first 5
                logger.info(f"  - {model.name}")
            if len(cached_models) > 5:
                logger.info(f"  ... and {len(cached_models) - 5} more")
            return True
    
    logger.info("No cached GPT-2 model found")
    return False

def get_model_size():
    """Estimate model download size"""
    logger.info("GPT-2 small model info:")
    logger.info("  - Parameters: ~124M")
    logger.info("  - Download size: ~500MB")
    logger.info("  - Disk space needed: ~1GB")

def cache_gpt2_model():
    """Download and cache the GPT-2 model and tokenizer"""
    model_name = "gpt2"
    
    logger.info("=" * 60)
    logger.info(f"Starting GPT-2 model caching process")
    logger.info("=" * 60)
    
    # Check system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check cache status
    already_cached = check_cache_status()
    if already_cached:
        logger.info("⚠️  GPT-2 model appears to already be cached")
        logger.info("Continuing anyway to ensure complete download...")
    
    # Show model info
    get_model_size()
    
    logger.info("=" * 40)
    logger.info("Starting download process...")
    logger.info("=" * 40)
    
    try:
        # Download and cache tokenizer
        logger.info("🔽 STEP 1: Downloading tokenizer...")
        logger.info("This should be quick (few MB)...")
        
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_name,
            cache_dir=None,  # Use default cache
            local_files_only=False,
            force_download=False  # Don't re-download if exists
        )
        logger.info("✅ Tokenizer downloaded and cached successfully")
        
        # Show tokenizer info
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        logger.info(f"Tokenizer type: {type(tokenizer).__name__}")
        
        # Download and cache model
        logger.info("\n🔽 STEP 2: Downloading model...")
        logger.info("This will take longer (~500MB download)...")
        logger.info("You should see progress bars below...")
        
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=None,  # Use default cache
            local_files_only=False,
            force_download=False,  # Don't re-download if exists
            torch_dtype=torch.float32  # Explicit dtype
        )
        logger.info("✅ Model downloaded and cached successfully")
        
        # Show model info
        logger.info(f"Model parameters: {model.num_parameters():,}")
        logger.info(f"Model config: {model.config.name_or_path}")
        
        # Test that everything works
        logger.info("\n🧪 STEP 3: Testing model functionality...")
        test_text = "The quick brown fox jumps over the lazy dog."
        logger.info(f"Test text: '{test_text}'")
        
        # Set model to evaluation mode
        model.eval()
        
        # Test tokenization
        inputs = tokenizer(test_text, return_tensors="pt")
        logger.info(f"Tokenized length: {inputs['input_ids'].shape[1]} tokens")
        
        # Test model inference
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
            perplexity = torch.exp(outputs.loss).item()
        
        logger.info(f"Test loss: {loss:.4f}")
        logger.info(f"Test perplexity: {perplexity:.2f}")
        logger.info("✅ Model test passed!")
        
        # Final status
        logger.info("\n" + "=" * 60)
        logger.info("🎉 CACHING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Model and tokenizer are now cached locally")
        logger.info(f"Cache location: ~/.cache/huggingface/transformers/")
        logger.info(f"Future loads will be much faster!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error caching model: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("\n" + "🤖 " + "=" * 58)
    print("🤖 GPT-2 Model Download & Caching Script")
    print("🤖 " + "=" * 58)
    print("This script will download the GPT-2 model (~500MB)")
    print("and cache it locally for faster future access.")
    print("=" * 60)
    
    try:
        success = cache_gpt2_model()
        
        if success:
            print("\n" + "✅ " + "=" * 58)
            print("✅ SUCCESS: GPT-2 model cached successfully!")
            print("✅ You can now run perplexity analysis and tests.")
            print("✅ " + "=" * 58)
        else:
            print("\n" + "❌ " + "=" * 58)
            print("❌ FAILED: Could not cache GPT-2 model.")
            print("❌ Check your internet connection and try again.")
            print("❌ " + "=" * 58)
            
    except KeyboardInterrupt:
        print("\n" + "⚠️ " + "=" * 58)
        print("⚠️ Download interrupted by user.")
        print("⚠️ You can run this script again to resume.")
        print("⚠️ " + "=" * 58)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 
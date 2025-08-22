#!/usr/bin/env python3
"""
Test script for the LocalLLM class.
"""
import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env file in the project root directory
    project_root = Path(__file__).parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning("No .env file found in project root.")
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env file loading")

# Test LocalLLM class
try:
    from local_llm import LocalLLM
    logger.info("Testing LocalLLM class...")
    llm = LocalLLM()
    
    # Test translation
    test_text = "Hello, how are you today?"
    logger.info(f"Original text: {test_text}")
    
    translated_text = llm.translate(test_text, src="en", dest="zh")
    logger.info(f"Translated text: {translated_text}")
    
    # Test with a longer text
    longer_text = "The quick brown fox jumps over the lazy dog. This is a sample sentence used for testing."
    logger.info(f"Original text: {longer_text}")
    
    translated_longer_text = llm.translate(longer_text, src="en", dest="zh")
    logger.info(f"Translated longer text: {translated_longer_text}")

except Exception as e:
    logger.error(f"Error testing LocalLLM: {e}", exc_info=True)
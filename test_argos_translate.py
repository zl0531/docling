#!/usr/bin/env python3
"""
Test script for the ArgosTranslate class.
"""
import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test ArgosTranslate class
try:
    from argos_translate import ArgosTranslate
    logger.info("Testing ArgosTranslate class...")
    
    # Create an instance of ArgosTranslate
    translator = ArgosTranslate()
    
    # Test translation
    test_text = "Hello, how are you today?"
    logger.info(f"Original text: {test_text}")
    
    translated_text = translator.translate(test_text, src="en", dest="zh")
    logger.info(f"Translated text: {translated_text}")
    
    # Test with a longer text
    longer_text = "The quick brown fox jumps over the lazy dog. This is a sample sentence used for testing."
    logger.info(f"Original text: {longer_text}")
    
    translated_longer_text = translator.translate(longer_text, src="en", dest="zh")
    logger.info(f"Translated longer text: {translated_longer_text}")
    
except Exception as e:
    logger.error(f"Error testing ArgosTranslate: {e}", exc_info=True)
# Document Translation Examples

This directory contains examples of how to translate documents using different translation backends.

## Quick Start

### 1. Using Local LLM (requires network and authentication)

```bash
# Convert and translate using local LLM
python convert_to_interactive_html.py path/to/document.pdf --translate --translation-backend local_llm
```

### 2. Using Argos Translate (offline translation)

```bash
# Convert and translate using Argos Translate
python convert_to_interactive_html.py path/to/document.pdf --translate --translation-backend argos
```

### 3. Convert without translation

```bash
# Convert without translation
python convert_to_interactive_html.py path/to/document.pdf
```

## Testing Translation Backends

### Test Local LLM

```bash
python test_local_llm.py
```

### Test Argos Translate

```bash
python test_argos_translate.py
```

## File Structure

- `convert_to_interactive_html.py`: Main conversion script with translation support
- `local_llm.py`: Local LLM translation implementation
- `argos_translate.py`: Argos Translate offline translation implementation
- `test_local_llm.py`: Test script for Local LLM
- `test_argos_translate.py`: Test script for Argos Translate

## Requirements

- For Local LLM: Valid credentials in `.env` file
- For Argos Translate: `pip install argostranslate`
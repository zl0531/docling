# Document Translation with Local LLM and Argos Translate

This example demonstrates how to use translation services to translate document text and generate HTML files.

## Files

All Python files have been moved to the project root directory for easier imports:

- `convert_to_interactive_html.py`: Main script to convert documents and optionally translate them
- `local_llm.py`: Implementation of the LocalLLM class for calling local LLM services
- `argos_translate.py`: Implementation of the ArgosTranslate class for offline translation
- `test_local_llm.py`: Test script for the LocalLLM class
- `test_argos_translate.py`: Test script for the ArgosTranslate class
- `.env.example`: Template for environment variables (located in the project root)
- `LOCAL_LLM_USAGE.md`: Documentation for using local LLM models (in docs/examples/)

## Setup

1. Copy the `.env.example` file from the project root to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Fill in your actual values in the `.env` file:
   ```bash
   LOCAL_LLM_APP_KEY=your_app_key_here
   LOCAL_LLM_SECRET_KEY=your_secret_key_here
   LOCAL_LLM_APP_CODE=your_app_code_here
   ```

3. For Argos Translate, install the package:
   ```bash
   pip install argostranslate
   ```

## Usage

Convert a document and translate it using local LLM:
```bash
python convert_to_interactive_html.py path/to/document.pdf --translate --translation-backend local_llm
```

Convert a document and translate it using Argos Translate (offline):
```bash
python convert_to_interactive_html.py path/to/document.pdf --translate --translation-backend argos
```

Convert a document without translation:
```bash
python convert_to_interactive_html.py path/to/document.pdf
```

Test the LocalLLM class:
```bash
python test_local_llm.py
```

Test the ArgosTranslate class:
```bash
python test_argos_translate.py
```

## How it works

1. The `convert_to_interactive_html.py` script converts documents to various formats using Docling
2. When the `--translate` flag is used, it calls the specified translation backend to translate text
3. Both original and translated versions are saved

## Translation Backends

### Local LLM
- Requires access to a local LLM service
- Provides high-quality translations
- Needs network connectivity
- Requires authentication credentials in `.env` file

### Argos Translate
- Offline translation using machine learning models
- No network connectivity required after initial setup
- Automatically downloads and installs translation models
- Good quality for common language pairs

## Local LLM Implementation

The `LocalLLM` class in `local_llm.py` handles:
- Authentication with the local LLM service
- Token management and caching
- Translation requests to the LLM service
- Error handling

See `docs/examples/LOCAL_LLM_USAGE.md` for more detailed information about using local LLM models.

## Argos Translate Implementation

The `ArgosTranslate` class in `argos_translate.py` handles:
- Automatic downloading and installation of translation models
- Offline translation using machine learning
- Support for multiple language pairs

## Troubleshooting

If you encounter issues:

1. **Environment Variables Not Loaded**: Make sure the `.env` file is in the project root directory and contains the correct values.

2. **Token Acquisition Failed**: Check that your `LOCAL_LLM_APP_KEY` and `LOCAL_LLM_SECRET_KEY` are correct.

3. **Network Issues**: Verify that you can access the local LLM service:
   - Office network: `https://ea-ai-gateway.corp.kuaishou.com/ea-ai-gateway/open/v1`
   - IDC: `http://ea-ai-gateway.common.ee-prod.internal:18080/ea-ai-gateway/open/v1`

4. **Argos Translate Issues**: 
   - Make sure you have enough disk space for downloading translation models.
   - If you get "No module named 'argostranslate'", ensure you've installed it with `pip install argostranslate`
   - Initial translation may take some time as it needs to download models

5. **Debugging**: You can run the test scripts with increased logging:
   ```bash
   python test_local_llm.py
   python test_argos_translate.py
   ```

6. **TableCell Attribute Error**: If you encounter errors about `TableCell` not having an `orig` attribute, this is expected as we cannot modify the structure of `TableCell` objects. The current implementation translates table cell text without storing the original.
"""
Converts a document (PDF or DOCX) to multiple output formats.

This example demonstrates how to convert a document to various formats
supported by Docling, including interactive HTML, JSON, Markdown, etc.
It also shows how to use translation services to translate the document text.
"""
import argparse
import logging
import os
from pathlib import Path

from docling_core.types.doc import ImageRefMode, TextItem, TableItem

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, OutputFormat
from docling.document_converter import PdfFormatOption

# Import translation modules
from local_llm import LocalLLM
# Try to import ArgosTranslate, but make it optional
try:
    from argos_translate import ArgosTranslate
    ARGOS_TRANSLATE_AVAILABLE = True
except ImportError:
    ARGOS_TRANSLATE_AVAILABLE = False
    ArgosTranslate = None

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env file in the project root directory
    # The project structure is: /Users/zhoulu/Documents/LLM/docling/docs/examples/convert_to_interactive_html.py
    # So we need to go up three levels to reach the project root
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loaded environment variables from {env_path}")
    else:
        # Also try the parent directory in case the structure is different
        project_root_alt = Path(__file__).parent.parent
        env_path_alt = project_root_alt / ".env"
        if env_path_alt.exists():
            load_dotenv(env_path_alt)
            logging.info(f"Loaded environment variables from {env_path_alt}")
        else:
            logging.warning(f"No .env file found at {env_path} or {env_path_alt}")
except ImportError:
    pass  # dotenv not installed, ignore

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def translate_document(document, translator, translation_backend: str = "local_llm"):
    """Translate document text using the specified translation backend.
    
    Args:
        document: The DoclingDocument to translate
        translator: The translator instance (LocalLLM or ArgosTranslate)
        translation_backend: The translation backend to use ("local_llm" or "argos")
    """
    _log.info(f"Translating document text using {translation_backend}...")
    
    # Iterate through all items in the document
    for item, _level in document.iterate_items():
        if isinstance(item, TextItem):
            # Store original text
            item.orig = item.text
            # Translate text
            item.text = translator.translate(item.text, src="en", dest="zh")
            
        elif isinstance(item, TableItem):
            # Translate table cell text
            if hasattr(item, 'data') and hasattr(item.data, 'table_cells'):
                for cell in item.data.table_cells:
                    if hasattr(cell, 'text'):
                        # Create a temporary attribute to store original text
                        # We can't directly add 'orig' to TableCell as it's not a defined field
                        # Instead, we'll just translate the text without storing the original
                        cell.text = translator.translate(cell.text, src="en", dest="zh")
    
    _log.info("Document translation completed.")


def main():
    """
    Converts a sample PDF document to multiple formats.
    """
    parser = argparse.ArgumentParser(description="Convert a document to multiple formats.")
    parser.add_argument("source_file", help="Path to the source document (PDF or DOCX)")
    parser.add_argument("-o", "--output-dir", default="scratch", help="Output directory for the files")
    parser.add_argument("--translate", action="store_true", help="Translate document text")
    parser.add_argument("--translation-backend", choices=["local_llm", "argos"], default="local_llm", 
                        help="Translation backend to use (default: local_llm)")
    
    args = parser.parse_args()
    source_file = args.source_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    _log.info(f"Converting '{source_file}' to multiple formats.")

    # To get images in the output, we need to enable them in the pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = 2.0 # Higher scale for better image quality

    # Initialize the DocumentConverter with the pipeline options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert the document
    result = converter.convert(source_file)

    if result.document:
        stem = Path(source_file).stem
        
        # Translate the document if requested
        if args.translate:
            # Initialize the appropriate translator
            if args.translation_backend == "local_llm":
                translator = LocalLLM()
            elif args.translation_backend == "argos":
                if not ARGOS_TRANSLATE_AVAILABLE:
                    _log.error("Argos Translate is not available. Please install it with 'pip install argostranslate'")
                    return
                translator = ArgosTranslate()
            else:
                _log.error(f"Unknown translation backend: {args.translation_backend}")
                return
                
            translate_document(result.document, translator, args.translation_backend)
        
        # Save as interactive HTML
        interactive_html_filename = output_dir / f"{stem}_interactive.html"
        result.document.save_as_html(
            interactive_html_filename,
            image_mode=ImageRefMode.EMBEDDED,
            split_page_view=True,
        )
        _log.info(f"Successfully created interactive HTML: {interactive_html_filename}")
        
        # Save as interactive HTML with translation
        if args.translate:
            interactive_html_translated_filename = output_dir / f"{stem}_interactive_{args.translation_backend}.html"
            result.document.save_as_html(
                interactive_html_translated_filename,
                image_mode=ImageRefMode.EMBEDDED,
                split_page_view=True,
            )
            _log.info(f"Successfully created interactive HTML with {args.translation_backend} translation: {interactive_html_translated_filename}")
        
        # Save as JSON
        json_filename = output_dir / f"{stem}.json"
        result.document.save_as_json(json_filename)
        _log.info(f"Successfully created JSON: {json_filename}")
        
        # Save as Markdown
        md_filename = output_dir / f"{stem}.md"
        result.document.save_as_markdown(md_filename)
        _log.info(f"Successfully created Markdown: {md_filename}")
        
        # Save as Markdown with translation
        if args.translate:
            md_translated_filename = output_dir / f"{stem}_{args.translation_backend}.md"
            result.document.save_as_markdown(md_translated_filename)
            _log.info(f"Successfully created Markdown with {args.translation_backend} translation: {md_translated_filename}")
        
        # Save as plain text
        txt_filename = output_dir / f"{stem}.txt"
        with open(txt_filename, "w") as f:
            f.write(result.document.export_to_text())
        _log.info(f"Successfully created plain text: {txt_filename}")
        
        # Save as plain text with translation
        if args.translate:
            txt_translated_filename = output_dir / f"{stem}_{args.translation_backend}.txt"
            with open(txt_translated_filename, "w") as f:
                f.write(result.document.export_to_text())
            _log.info(f"Successfully created plain text with {args.translation_backend} translation: {txt_translated_filename}")
        
        # Save as HTML (single file)
        html_filename = output_dir / f"{stem}.html"
        result.document.save_as_html(html_filename, image_mode=ImageRefMode.EMBEDDED)
        _log.info(f"Successfully created HTML: {html_filename}")
        
        # Save as HTML (single file) with translation
        if args.translate:
            html_translated_filename = output_dir / f"{stem}_{args.translation_backend}.html"
            result.document.save_as_html(html_translated_filename, image_mode=ImageRefMode.EMBEDDED)
            _log.info(f"Successfully created HTML with {args.translation_backend} translation: {html_translated_filename}")
        
        # Save as DocTags
        doctags_filename = output_dir / f"{stem}.doctags"
        result.document.save_as_document_tokens(doctags_filename)
        _log.info(f"Successfully created DocTags: {doctags_filename}")
        
    else:
        _log.error("Failed to convert the document.")


if __name__ == "__main__":
    main()

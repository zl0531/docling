"""
Reconstructs HTML and PDF files from a Docling JSON document.

This example shows how to use the DoclingDocument class to load a JSON file 
and then save it as HTML or PDF, ensuring that images are properly handled.
"""
import argparse
import logging
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def main():
    """
    Reconstructs HTML and PDF files from a Docling JSON document.
    """
    parser = argparse.ArgumentParser(description="Reconstruct files from Docling JSON.")
    parser.add_argument("json_file", help="Path to the Docling JSON file")
    parser.add_argument("-o", "--output-dir", default="scratch", help="Output directory for the files")
    
    args = parser.parse_args()
    json_file = args.json_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    _log.info(f"Loading JSON document from '{json_file}'...")
    
    # Load the JSON document
    doc = DoclingDocument.load_from_json(Path(json_file))
    
    if doc:
        stem = Path(json_file).stem.replace('.json', '')
        
        # Save as HTML with embedded images
        html_filename = output_dir / f"{stem}_from_json.html"
        doc.save_as_html(html_filename, image_mode="embedded")
        _log.info(f"Successfully created HTML with embedded images: {html_filename}")
        
        # Save as Markdown (images will be embedded as base64)
        md_filename = output_dir / f"{stem}_from_json.md"
        doc.save_as_markdown(md_filename)
        _log.info(f"Successfully created Markdown: {md_filename}")
        
        # Note: PDF reconstruction is not directly supported by DoclingDocument
        # You would need to use other libraries or tools to convert HTML/Markdown to PDF
        _log.info("Note: Direct PDF reconstruction from JSON is not supported by Docling.")
        _log.info("You can convert the generated HTML or Markdown to PDF using other tools.")
    else:
        _log.error("Failed to load the JSON document.")


if __name__ == "__main__":
    main()
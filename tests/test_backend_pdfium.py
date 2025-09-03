from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox

from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
    PyPdfiumPageBackend,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@pytest.fixture
def test_doc_path():
    return Path("./tests/data/pdf/2206.01062.pdf")


def _get_backend(pdf_doc):
    in_doc = InputDocument(
        path_or_stream=pdf_doc,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )

    doc_backend = in_doc._backend
    return doc_backend


def test_get_text_from_rect_rotated():
    pdf_doc = Path("./tests/data_scanned/sample_with_rotation_mismatch.pdf")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
        }
    )
    conv_res = doc_converter.convert(pdf_doc)

    assert "1972" in conv_res.document.export_to_markdown()


def test_text_cell_counts():
    pdf_doc = Path("./tests/data/pdf/redp5110_sampled.pdf")

    doc_backend = _get_backend(pdf_doc)

    for page_index in range(doc_backend.page_count()):
        last_cell_count = None
        for i in range(10):
            page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)
            cells = list(page_backend.get_text_cells())

            if last_cell_count is None:
                last_cell_count = len(cells)

            if len(cells) != last_cell_count:
                assert False, (
                    "Loading page multiple times yielded non-identical text cell counts"
                )
            last_cell_count = len(cells)


def test_get_text_from_rect(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)

    # Get the title text of the DocLayNet paper
    textpiece = page_backend.get_text_in_rect(
        bbox=BoundingBox(l=102, t=77, r=511, b=124)
    )
    ref = "DocLayNet: A Large Human-Annotated Dataset for\r\nDocument-Layout Analysis"

    assert textpiece.strip() == ref


def test_crop_page_image(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)

    # Crop out "Figure 1" from the DocLayNet paper
    page_backend.get_page_image(
        scale=2, cropbox=BoundingBox(l=317, t=246, r=574, b=527)
    )
    # im.show()


def test_num_pages(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    doc_backend.page_count() == 9


def test_merge_row():
    pdf_doc = Path("./tests/data/pdf/multi_page.pdf")

    doc_backend = _get_backend(pdf_doc)
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(4)
    cell = page_backend.get_text_cells()[0]

    assert (
        cell.text
        == "The journey of the word processor—from clunky typewriters to AI-powered platforms—"
    )

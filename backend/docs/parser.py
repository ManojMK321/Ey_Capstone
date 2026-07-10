"""
parser.py
---------
Load a PDF from disk and convert it into LangChain Documents.

Pipeline:
    PDF File -> pypdf.PdfReader -> List[Document]

Uses pypdf directly instead of langchain_community.document_loaders.PyPDFLoader
to avoid the import chain:
  langchain_community -> langchain_core.document_loaders ->
  langchain_text_splitters -> sentence_transformers -> torch -> c10.dll (crash)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pypdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Loads a PDF using pypdf.
    Returns one Document per page with metadata: page, source.
    Identical output format to the previous PyPDFLoader-based implementation.
    """

    def parse(self, pdf_path: str | Path) -> list[Document]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Loading PDF: %s", pdf_path.name)
        pages: list[Document] = []
        reader = pypdf.PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": page_num, "source": str(pdf_path)},
                )
            )
        logger.info("Parsed '%s' into %d page(s).", pdf_path.name, len(pages))
        return pages

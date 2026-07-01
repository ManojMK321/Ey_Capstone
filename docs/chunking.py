"""
chunking.py
-----------
Split parsed PDF pages into overlapping chunks and attach metadata.

Pipeline:
    List[Document] -> _RecursiveCharacterSplitter -> List[Document]

Uses a pure-Python recursive character splitter instead of
langchain_text_splitters.RecursiveCharacterTextSplitter to avoid the chain:
  langchain_text_splitters -> sentence_transformers -> torch -> c10.dll (crash)
The output (Document list + metadata schema) is identical to the previous
implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", ", ", " ", ""])


# ---------------------------------------------------------------------------
# Pure-Python recursive character splitter
# ---------------------------------------------------------------------------

class _RecursiveCharacterSplitter:
    """
    Drop-in replacement for langchain_text_splitters.RecursiveCharacterTextSplitter.
    Splits text by trying each separator in order; merges small pieces back into
    chunks of at most `chunk_size` characters, keeping `chunk_overlap` chars of
    context between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        length_function=len,
        **_,                        # absorb any extra kwargs (e.g. is_separator_regex)
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", ", ", " ", ""]
        self.length_fn = length_function

    # ------------------------------------------------------------------
    # Public API (matches LangChain interface)
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            for chunk_text in self.split_text(doc.page_content):
                result.append(
                    Document(page_content=chunk_text, metadata=dict(doc.metadata))
                )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split `text` using the first matching separator."""
        if not text:
            return []
        if self.length_fn(text) <= self.chunk_size:
            return [text]

        # Find the first separator that is present in the text.
        chosen_sep = separators[-1]     # fallback: empty string → hard split
        remaining_seps = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                chosen_sep = sep
                remaining_seps = separators[i + 1:]
                break

        if chosen_sep == "":
            # Hard split: no separator matched — split by character count.
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [text[i: i + self.chunk_size] for i in range(0, len(text), step)]

        raw_splits = [s for s in text.split(chosen_sep) if s]
        merged = self._merge(raw_splits, chosen_sep)

        # Any chunk that is still too large gets recursively split.
        final: list[str] = []
        for chunk in merged:
            if self.length_fn(chunk) > self.chunk_size and remaining_seps:
                final.extend(self._split(chunk, remaining_seps))
            else:
                final.append(chunk)
        return final

    def _merge(self, splits: list[str], sep: str) -> list[str]:
        """
        Greedily merge `splits` into chunks of at most chunk_size characters.
        After flushing a chunk, retain the tail of it up to chunk_overlap chars
        so the next chunk has context continuity.
        """
        chunks: list[str] = []
        window: list[str] = []          # splits currently accumulated
        window_len: int = 0             # character count of window (excl. future sep)

        def _window_text() -> str:
            return sep.join(window)

        def _sep_overhead(lst: list[str]) -> int:
            return len(sep) * (len(lst) - 1) if len(lst) > 1 else 0

        for split in splits:
            split_len = self.length_fn(split)
            extra = len(sep) + split_len if window else split_len

            if window_len + extra > self.chunk_size and window:
                # Flush
                text = _window_text()
                if text.strip():
                    chunks.append(text)

                # Trim the front of the window until its content fits in chunk_overlap.
                while window and (window_len + _sep_overhead(window)) > self.chunk_overlap:
                    removed = window.pop(0)
                    window_len -= self.length_fn(removed)
                    if window_len < 0:
                        window_len = 0

            window.append(split)
            window_len += split_len

        if window:
            text = _window_text()
            if text.strip():
                chunks.append(text)

        return chunks


# ---------------------------------------------------------------------------
# Public chunker (identical interface to the old DocumentChunker)
# ---------------------------------------------------------------------------

class DocumentChunker:

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()
        self.splitter = _RecursiveCharacterSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
        )

    def chunk(self, pages: list[Document], doc_id: str, filename: str) -> list[Document]:
        if not pages:
            raise ValueError("No pages to chunk.")

        logger.info("Chunking '%s' (%d pages)...", filename, len(pages))
        all_chunks: list[Document] = []

        for page_doc in pages:
            page_number = page_doc.metadata.get("page", 0) + 1
            page_chunks = self.splitter.split_documents([page_doc])

            for idx, chunk in enumerate(page_chunks):
                chunk.metadata.update({
                    "doc_id":      doc_id,
                    "filename":    filename,
                    "page":        page_number,
                    "chunk_index": idx,
                    "chunk_size":  len(chunk.page_content),
                    "char_start":  idx * (self.config.chunk_size - self.config.chunk_overlap),
                })
                all_chunks.append(chunk)

        logger.info("Generated %d chunks from '%s'.", len(all_chunks), filename)
        return all_chunks

"""
vector_store.py
---------------
Manage the FAISS vector database.

Pipeline:
    Chunk Documents -> OpenAI Embeddings -> FAISS -> Similarity Search
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class FAISSVectorStore:

    INDEX_NAME = "index"

    def __init__(self, index_dir: str, embedding_model: Embeddings):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.vector_store = self._load()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> Optional[FAISS]:
        index_file = self.index_dir / f"{self.INDEX_NAME}.faiss"
        if not index_file.exists():
            logger.info("No FAISS index found — will create on first add.")
            return None
        logger.info("Loading existing FAISS index from '%s'...", self.index_dir)
        return FAISS.load_local(
            folder_path=str(self.index_dir),
            embeddings=self.embedding_model,
            index_name=self.INDEX_NAME,
            allow_dangerous_deserialization=True,
        )

    def _save(self):
        if self.vector_store is None:
            return
        self.vector_store.save_local(folder_path=str(self.index_dir), index_name=self.INDEX_NAME)
        logger.info("FAISS index saved to '%s'.", self.index_dir)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]):
        if not documents:
            logger.warning("add_documents called with empty list — skipped.")
            return

        if self.vector_store is None:
            logger.info("Creating new FAISS index with %d chunks...", len(documents))
            self.vector_store = FAISS.from_documents(documents=documents, embedding=self.embedding_model)
        else:
            logger.info("Adding %d chunks to existing FAISS index...", len(documents))
            self.vector_store.add_documents(documents=documents)

        self._save()

    def similarity_search(self, query: str, k: int = 5, doc_id: str | None = None) -> list[Document]:
        if self.vector_store is None:
            return []
        search_filter = {"doc_id": doc_id} if doc_id else None
        return self.vector_store.similarity_search(query=query, k=k, filter=search_filter)

    def similarity_search_with_score(self, query: str, k: int = 5, doc_id: str | None = None):
        if self.vector_store is None:
            return []
        search_filter = {"doc_id": doc_id} if doc_id else None
        return self.vector_store.similarity_search_with_score(query=query, k=k, filter=search_filter)

    def document_count(self) -> int:
        if self.vector_store is None:
            return 0
        return len(self.vector_store.docstore._dict)

    def get_all_doc_ids(self) -> list[str]:
        if self.vector_store is None:
            return []
        return sorted({
            doc.metadata.get("doc_id")
            for doc in self.vector_store.docstore._dict.values()
            if doc.metadata.get("doc_id")
        })

    def delete_document(self, doc_id: str):
        if self.vector_store is None:
            return

        remaining = [
            doc for doc in self.vector_store.docstore._dict.values()
            if doc.metadata.get("doc_id") != doc_id
        ]

        if not remaining:
            self.vector_store = None
            for f in self.index_dir.glob("index.*"):
                f.unlink(missing_ok=True)
            logger.info("FAISS index cleared (last document removed).")
            return

        self.vector_store = FAISS.from_documents(documents=remaining, embedding=self.embedding_model)
        self._save()
        logger.info("Document '%s' deleted from FAISS.", doc_id)

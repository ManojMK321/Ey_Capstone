"""
vector_store_pinecone.py
------------------------

Manage the Pinecone vector database.

Pipeline:
    Chunk Documents -> OpenAI Embeddings -> Pinecone -> Similarity Search

Environment variables required:
    PINECONE_API_KEY        - Pinecone API key
    PINECONE_INDEX_NAME     - Name of the Pinecone index

Vector ID format: "{doc_id}#{uuid4_hex}"
Using doc_id as a prefix allows listing and deleting all chunks
belonging to a document via Pinecone's list API.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_DEFAULT_DIMENSION = 1536


class PineconeVectorStoreManager:

    def __init__(
        self,
        embedding_model: Embeddings,
        index_name: Optional[str] = None,
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.embedding_model = embedding_model
        self.index_name      = index_name or os.environ["PINECONE_INDEX_NAME"]
        self.namespace       = namespace

        self._pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self._ensure_index_exists(cloud=cloud, region=region)
        self._index = self._pc.Index(self.index_name)

        self.vector_store = PineconeVectorStore(
            index=self._index,
            embedding=self.embedding_model,
            namespace=self.namespace,
            text_key="text",
        )

        logger.info(
            "PineconeVectorStore ready  index=%s  namespace=%r",
            self.index_name, self.namespace,
        )

    def _ensure_index_exists(self, cloud: str, region: str) -> None:
        existing = [idx.name for idx in self._pc.list_indexes()]

        if self.index_name in existing:
            logger.info("Pinecone index '%s' already exists.", self.index_name)
            return

        model_name = getattr(self.embedding_model, "model", "")
        dimension  = _EMBEDDING_DIMENSIONS.get(model_name, _DEFAULT_DIMENSION)

        logger.info(
            "Pinecone index '%s' not found — creating (dim=%d, cloud=%s, region=%s).",
            self.index_name, dimension, cloud, region,
        )

        self._pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

        logger.info("Pinecone index '%s' created successfully.", self.index_name)

    def _make_vector_ids(self, documents: list[Document]) -> list[str]:
        return [
            f"{doc.metadata.get('doc_id', 'unknown')}#{uuid.uuid4().hex}"
            for doc in documents
        ]

    def _list_ids_for_doc(self, doc_id: str) -> list[str]:
        ids: list[str] = []
        for batch in self._index.list(prefix=f"{doc_id}#", namespace=self.namespace):
            ids.extend(batch)
        return ids

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            logger.warning("No documents received for indexing.")
            return

        ids = self._make_vector_ids(documents)
        logger.info("Upserting %d chunks to Pinecone.", len(documents))

        self.vector_store.add_documents(
            documents=documents,
            ids=ids,
            namespace=self.namespace,
        )

        logger.info("Upsert complete.")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        doc_id: Optional[str] = None,
    ) -> list[Document]:
        search_filter = {"doc_id": {"$eq": doc_id}} if doc_id else None
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=search_filter,
            namespace=self.namespace,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        doc_id: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        search_filter = {"doc_id": {"$eq": doc_id}} if doc_id else None
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=search_filter,
            namespace=self.namespace,
        )

    def document_count(self) -> int:
        stats = self._index.describe_index_stats()
        ns    = stats.namespaces.get(self.namespace)
        return ns.vector_count if ns else 0

    def get_all_doc_ids(self) -> list[str]:
        doc_ids: set[str] = set()
        for batch in self._index.list(namespace=self.namespace):
            for vec_id in batch:
                if "#" in vec_id:
                    doc_ids.add(vec_id.split("#")[0])
        return sorted(doc_ids)

    def get_all_documents(self, doc_id: Optional[str] = None) -> list[Document]:
        """Fetch all Document objects — used by BM25Retriever to build its term index."""
        prefix  = f"{doc_id}#" if doc_id else ""
        all_ids: list[str] = []

        for batch in self._index.list(prefix=prefix, namespace=self.namespace):
            all_ids.extend(batch)

        if not all_ids:
            return []

        documents: list[Document] = []
        batch_size = 1_000

        for i in range(0, len(all_ids), batch_size):
            response = self._index.fetch(
                ids=all_ids[i : i + batch_size],
                namespace=self.namespace,
            )
            for vector in response.vectors.values():
                metadata     = dict(vector.metadata or {})
                page_content = metadata.pop("text", "")
                documents.append(Document(page_content=page_content, metadata=metadata))

        logger.info("Fetched %d chunks from Pinecone.", len(documents))
        return documents

    def delete_document(self, doc_id: str) -> None:
        ids_to_delete = self._list_ids_for_doc(doc_id)

        if not ids_to_delete:
            logger.warning("No vectors found for doc_id=%s.", doc_id)
            return

        logger.info("Deleting %d chunks for doc_id=%s.", len(ids_to_delete), doc_id)

        batch_size = 1_000
        for i in range(0, len(ids_to_delete), batch_size):
            self._index.delete(
                ids=ids_to_delete[i : i + batch_size],
                namespace=self.namespace,
            )

        logger.info("Document %s deleted.", doc_id)

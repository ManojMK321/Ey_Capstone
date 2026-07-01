"""
embedding.py
------------
Create and manage the OpenAI embedding model.

Pipeline:
    Chunk Documents -> OpenAI Embeddings -> FAISS Vector Store
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    dimensions: int | None = None


class OpenAIEmbedder:
    """
    Thin wrapper around LangChain OpenAIEmbeddings.
    One instance is shared between indexing and querying
    so all vectors live in the same embedding space.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")

        kwargs: dict = {"model": self.config.model, "api_key": api_key}
        if self.config.dimensions:
            kwargs["dimensions"] = self.config.dimensions

        logger.info("Initializing embedding model '%s'...", self.config.model)
        self.embeddings = OpenAIEmbeddings(**kwargs)
        logger.info("Embedding model ready.")

    @property
    def embedding_model(self) -> OpenAIEmbeddings:
        return self.embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)

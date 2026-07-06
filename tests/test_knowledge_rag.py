from unittest.mock import MagicMock

from src.agents.knowledge_rag import KnowledgeRAG


def test_knowledge_rag_accepts_top_k_argument():
    vector_store = MagicMock()
    client = MagicMock()

    rag = KnowledgeRAG(client=client, vector_store=vector_store, top_k=5)

    assert rag.k_final == 5

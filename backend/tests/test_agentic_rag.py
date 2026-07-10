from unittest.mock import MagicMock

from src.agents.agentic_rag import AgenticRAG, TaskRouteResult, TaskType, ValidationResult


def test_agentic_rag_run_routes_and_generates_response():
    client = MagicMock()
    vector_store = MagicMock()

    rag = AgenticRAG(client=client, vector_store=vector_store, top_k=5)

    rag.analyze_query = MagicMock(return_value=["Who assumes more financial risk?"])
    retrieved_doc = {
        "metadata": {"filename": "contract.pdf", "page": 1, "doc_id": "doc1"},
        "content": "The seller assumes the insurance obligations and liability caps.",
    }
    rag.retrieve = MagicMock(return_value=[retrieved_doc])
    rag.route_task = MagicMock(
        return_value=TaskRouteResult(task=TaskType.COMPARISON, reason="risk comparison")
    )
    rag.run_specialist_agent = MagicMock(return_value="Draft comparison answer.")
    rag.validate = MagicMock(
        return_value=ValidationResult(
            is_grounded=True,
            issues=[],
            corrected_answer="Draft comparison answer.",
        )
    )
    rag.generate_response = MagicMock(return_value="Final answer.")

    result = rag.run("Who assumes more risk in this contract?", history="Prior turn")

    assert result["answer"] == "Final answer."
    assert result["task"] == "comparison"
    assert result["task_reason"] == "risk comparison"
    assert result["draft_answer"] == "Draft comparison answer."
    assert result["is_grounded"] is True
    assert result["validation_issues"] == []
    assert result["retrieved"] == [retrieved_doc]
    assert result["sources"] == [
        {
            "subquestion": "",
            "source": "contract.pdf (page 1)",
            "content": "The seller assumes the insurance obligations and liability caps.",
        }
    ]

    rag.analyze_query.assert_called_once_with("Who assumes more risk in this contract?")
    rag.route_task.assert_called_once_with("Who assumes more risk in this contract?")
    rag.run_specialist_agent.assert_called_once()
    rag.validate.assert_called_once()
    rag.generate_response.assert_called_once_with(
        "Who assumes more risk in this contract?", "Draft comparison answer."
    )

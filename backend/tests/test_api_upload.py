"""
Unit tests for upload_api.py.

Uses FastAPI's TestClient (no real server needed) and mocks everything
below the route layer: parser, chunker, session_store, and get_vector_store.
Tests route logic (validation, error handling, response shape) only.

Run with:
    python -m pytest tests/test_api_upload.py -v
"""

import io
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import upload_api


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(upload_api.router, prefix="/upload")
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    mock_parser = MagicMock()
    mock_parser.parse.return_value = ["page 1 text"]

    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [{"text": "chunk 1"}]

    mock_vector_store = MagicMock()
    mock_get_vector_store = MagicMock(return_value=mock_vector_store)

    mock_session_store = MagicMock()
    mock_session_store.make_session.side_effect = (
        lambda requested_id, reset=False: requested_id or "generated-session-id"
    )
    mock_session_store.session_exists.return_value = True
    mock_session_store.list_documents.return_value = []
    mock_session_store.delete_document.return_value = True

    mock_register_document = MagicMock()

    monkeypatch.setattr(
        upload_api,
        "get_pipeline",
        lambda: (mock_parser, mock_chunker, mock_vector_store),
    )
    monkeypatch.setattr(upload_api, "get_vector_store", mock_get_vector_store)
    monkeypatch.setattr(upload_api, "session_store", mock_session_store)
    monkeypatch.setattr(upload_api, "register_document", mock_register_document)

    return {
        "parser": mock_parser,
        "chunker": mock_chunker,
        "vector_store": mock_vector_store,
        "get_vector_store": mock_get_vector_store,
        "session_store": mock_session_store,
        "register_document": mock_register_document,
    }


def _pdf_file(filename="contract.pdf", content=b"%PDF-1.4 fake pdf bytes", content_type="application/pdf"):
    return ("files", (filename, io.BytesIO(content), content_type))


# -----------------------------
# POST /upload/ — validation
# -----------------------------

class TestUploadValidation:
    def test_no_files_returns_400(self, client):
        response = client.post("/upload/", files=[])
        assert response.status_code == 400
        assert "No files provided" in response.json()["detail"]

    def test_too_many_files_returns_400(self, client):
        files = [_pdf_file(filename=f"doc{i}.pdf") for i in range(11)]
        response = client.post("/upload/", files=files)
        assert response.status_code == 400
        assert "Max 10 files per request" in response.json()["detail"]

    def test_non_pdf_extension_is_rejected_as_error_not_exception(self, client):
        files = [_pdf_file(filename="notes.txt", content_type="text/plain")]
        response = client.post("/upload/", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["uploaded_count"] == 0
        assert data["failed_count"] == 1
        assert "not a PDF" in data["errors"][0]["reason"]

    def test_unsupported_content_type_is_rejected(self, client):
        files = [_pdf_file(filename="contract.pdf", content_type="image/png")]
        response = client.post("/upload/", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["failed_count"] == 1
        assert "unsupported content type" in data["errors"][0]["reason"]


# -----------------------------
# POST /upload/ — success path
# -----------------------------

class TestUploadSuccess:
    def test_single_valid_pdf_succeeds(self, client, mock_dependencies):
        response = client.post("/upload/", files=[_pdf_file()])
        assert response.status_code == 200
        data = response.json()

        assert data["uploaded_count"] == 1
        assert data["failed_count"] == 0
        assert len(data["files"]) == 1
        assert data["files"][0]["original_name"] == "contract.pdf"
        assert "file_id" in data["files"][0]
        assert data["session_id"]

        mock_dependencies["parser"].parse.assert_called_once()
        mock_dependencies["chunker"].chunk.assert_called_once()
        mock_dependencies["vector_store"].add_documents.assert_called_once()
        mock_dependencies["session_store"].add_document.assert_called_once()
        mock_dependencies["register_document"].assert_called_once()

    def test_reuses_provided_session_id(self, client, mock_dependencies):
        existing_session = str(uuid.uuid4())
        response = client.post(
            "/upload/",
            files=[_pdf_file()],
            data={"session_id": existing_session},
        )
        assert response.status_code == 200
        assert response.json()["session_id"] == existing_session
        mock_dependencies["session_store"].make_session.assert_called_once_with(
            existing_session, reset=False
        )

    def test_multiple_files_mixed_success_and_failure(self, client, mock_dependencies):
        files = [
            _pdf_file(filename="good.pdf"),
            _pdf_file(filename="bad.txt", content_type="text/plain"),
        ]
        response = client.post("/upload/", files=files)
        data = response.json()
        assert data["uploaded_count"] == 1
        assert data["failed_count"] == 1
        assert data["files"][0]["original_name"] == "good.pdf"
        assert data["errors"][0]["filename"] == "bad.txt"


# -----------------------------
# POST /upload/ — pipeline errors
# -----------------------------

class TestUploadPipelineErrors:
    def test_unexpected_parser_exception_is_caught_and_reported(self, client, mock_dependencies):
        mock_dependencies["parser"].parse.side_effect = RuntimeError("corrupted PDF stream")
        response = client.post("/upload/", files=[_pdf_file()])
        assert response.status_code == 200
        data = response.json()
        assert data["uploaded_count"] == 0
        assert data["failed_count"] == 1
        assert data["errors"][0]["reason"] == "An unexpected error occurred."


# -----------------------------
# GET /upload/list/{session_id}
# -----------------------------

class TestListSessionUploads:
    def test_unknown_session_returns_404(self, client, mock_dependencies):
        mock_dependencies["session_store"].session_exists.return_value = False
        response = client.get("/upload/list/some-session-id")
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]

    def test_known_session_returns_documents(self, client, mock_dependencies):
        mock_dependencies["session_store"].list_documents.return_value = [
            {"file_id": "f1", "original_name": "a.pdf"}
        ]
        response = client.get("/upload/list/some-session-id")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "some-session-id"
        assert data["count"] == 1
        assert data["files"][0]["original_name"] == "a.pdf"

    def test_trailing_period_in_session_id_is_stripped(self, client, mock_dependencies):
        response = client.get("/upload/list/some-session-id.")
        assert response.status_code == 200
        mock_dependencies["session_store"].session_exists.assert_called_with("some-session-id")
        assert response.json()["session_id"] == "some-session-id"


# -----------------------------
# DELETE /upload/{session_id}/{file_id}
# -----------------------------

class TestDeleteSessionDocument:
    def test_delete_missing_document_returns_404(self, client, mock_dependencies):
        mock_dependencies["session_store"].delete_document.return_value = False
        response = client.delete("/upload/session-1/file-1")
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_delete_existing_document_succeeds(self, client, mock_dependencies):
        mock_dependencies["session_store"].delete_document.return_value = True
        response = client.delete("/upload/session-1/file-1")
        assert response.status_code == 200
        assert "removed from session" in response.json()["message"]
        mock_dependencies["session_store"].delete_document.assert_called_once_with(
            "session-1", "file-1"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

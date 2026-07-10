# Contract Intelligence

AI-powered contract analysis: FastAPI backend (Knowledge RAG + Agentic RAG
pipelines) with a Streamlit frontend.

## Layout

```
backend/            FastAPI service
├── main.py          entrypoint (uvicorn main:app)
├── src/             agents, api routes, middleware, observability, orchestrator, retrieval, schema
├── docs/            ingestion pipeline (parse, chunk, embed, vector store)
├── eval/            RAGAS scoring — imported by src/observability/langsmith_eval.py
│                    AND directly by frontend/streamlit_app.py (in-process, no HTTP hop)
├── tests/           pytest suite
├── load_test/       standalone HTTP load-test scripts (hit a running backend)
├── requirements.txt
├── Dockerfile
└── .env             not committed — copy .env.example

frontend/            Streamlit UI
├── streamlit_app.py
├── requirements.txt
├── Dockerfile        build from repo root: docker build -f frontend/Dockerfile .
└── .streamlit/config.toml
```

`backend/eval` is shared: it has no dependency on `src/`, so it's imported
directly by the Streamlit process for live RAGAS scoring instead of round-tripping
through the API. Keep it dependency-free from `src/` if you touch it.

## Running locally

```bash
# Backend (from backend/)
cd backend
python -m venv venv && venv\Scripts\activate   # Python 3.12 required
pip install -r requirements.txt
cp .env.example .env   # fill in real values
python main.py         # http://localhost:8000/docs

# Frontend (from frontend/, separate shell)
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py   # http://localhost:8501
```

The frontend calls the backend over HTTP (`http://localhost:8000`, see
`API` in `streamlit_app.py`) except for RAGAS scoring, which it computes
in-process via `backend/eval/ragas_judge.py`.

## Tests

```bash
cd backend
pytest
```

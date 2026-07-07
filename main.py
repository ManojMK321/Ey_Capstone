from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.upload_api import router as upload_router
from src.api.chat import router as chat_router
from src.observability.telemetry import setup_observability
from src.observability.langsmith import LangSmithRequestTracingMiddleware
from src.observability import metrics
from docs.pipeline import get_pipeline
import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

setup_observability()

app = FastAPI(
    title="Contract Intelligence Chatbot",
    description="AI-powered contract analysis using Knowledge RAG and Agentic RAG pipelines.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
def _warm_up_ingestion_pipeline() -> None:
    # get_pipeline() is a lazy singleton — without this, the heavy
    # langchain/pinecone imports and the first network round trip to
    # OpenAI/Pinecone happen on whichever user's upload request arrives
    # first, making that upload look stuck for up to a minute. Paying that
    # cost once at startup keeps every real upload fast.
    logger.info("Warming up ingestion pipeline...")
    _, _, vector_store = get_pipeline()
    metrics.SYSTEM_HEALTH_STATUS.labels(component="api").set(1)
    metrics.SYSTEM_HEALTH_STATUS.labels(component="vector_store").set(
        1 if getattr(vector_store, "vector_store", None) is not None else 0
    )
    logger.info("Ingestion pipeline warm.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LangSmithRequestTracingMiddleware)

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus metrics exposed at /metrics")
except ImportError:
    logger.warning(
        "prometheus-fastapi-instrumentator not installed; /metrics endpoint disabled. "
        "Install with: pip install prometheus-fastapi-instrumentator"
    )

app.include_router(upload_router, prefix="/upload", tags=["Upload"])
app.include_router(chat_router,   prefix="/chat",   tags=["Chat"])

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        # Scoped on purpose: with no reload_dirs, uvicorn watches the whole
        # project root, including venv/ (70k+ files). watchfiles then has to
        # continuously monitor that entire tree — reload_excludes only
        # filters which changes *trigger* a reload, it doesn't stop the
        # watching itself — and that constant I/O was starving the Pinecone
        # upsert call during uploads, turning a ~2s call into 60-80s.
        reload_dirs=["src", "docs"],
        reload_excludes=["sessions.db", "*.log"],
        log_level="info",
    )

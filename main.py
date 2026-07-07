from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.upload_api import router as upload_router
from src.api.chat import router as chat_router
from docs.pipeline import get_pipeline
import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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
    get_pipeline()
    logger.info("Ingestion pipeline warm.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.upload_api import router as upload_router
from src.api.chat import router as chat_router
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
        # Uploads write to temp_uploads/ and the session store writes to
        # sessions.db mid-request; without these excludes the reload watcher
        # treats its own runtime writes as a source change and restarts the
        # worker, silently killing whatever request was in flight.
        reload_excludes=["temp_uploads/*", "sessions.db", "*.log"],
        log_level="info",
    )

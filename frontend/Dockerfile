# Build from the repo root so backend/eval (RAGAS scoring, shared with the
# backend) can be copied in:
#   docker build -f frontend/Dockerfile -t contract-intelligence-ui .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ .
# score_turn() is imported in-process for live RAGAS scoring — see the
# sys.path setup at the top of streamlit_app.py.
COPY backend/eval ./backend/eval

# OPENAI_API_KEY must be provided via the container environment (docker run
# -e / compose `environment:`) — backend/.env is intentionally not copied
# into this image.
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Metabolic Memory Engine - plug-and-play microservice for Moltbot
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt


COPY src/ /app/src/

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]

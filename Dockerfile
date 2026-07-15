FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/sentence-transformers

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY api ./api
COPY data ./data
COPY database ./database
COPY evaluation ./evaluation
COPY event_extraction ./event_extraction
COPY retrieval ./retrieval
COPY scripts ./scripts
COPY README.md .

RUN mkdir -p /home/app/.cache && chown -R app:app /app /home/app

USER app

EXPOSE 8000

CMD ["python", "-m", "scripts.docker_entrypoint"]

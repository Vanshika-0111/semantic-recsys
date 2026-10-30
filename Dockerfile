# Day 12: the recommendation API in one container.
#
#   1. build the bundle on the laptop:  python scripts/run_serving.py --stages bundle
#   2. docker build -t semantic-recsys-api .
#   3. docker run --rm -p 8000:8000 semantic-recsys-api      -> http://localhost:8000/docs
#
# The image holds code + the ~50 MB bundle (data/serving) + the bge-small encoder (for free-text
# search). No ratings, no training data. Runs as a non-root user.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 \
    RECSYS_BUNDLE=/app/data/serving

# libgomp: OpenMP runtime that LightGBM and FAISS need
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# WITH_TEXT=false -> no torch / sentence-transformers / encoder: a ~4x smaller image; free-text
# search then answers 503, everything else works.  docker build --build-arg WITH_TEXT=false ...
ARG WITH_TEXT=true
WORKDIR /app
COPY requirements-serve.txt .
RUN if [ "$WITH_TEXT" = "true" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-serve.txt; \
    else \
        grep -v sentence-transformers requirements-serve.txt > /tmp/req.txt && pip install -r /tmp/req.txt; \
    fi

# the text encoder, downloaded once at build time (explore mode works offline afterwards)
RUN if [ "$WITH_TEXT" = "true" ]; then \
        python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5', device='cpu')"; \
    fi

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .
COPY data/serving ./data/serving

RUN mkdir -p /opt/hf && useradd --create-home app && chown -R app /app /opt/hf
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "recsys.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Day 13: the public demo image — API (FastAPI, internal :8000) + UI (Streamlit, public :7860).
#
# Built by Hugging Face from the Space repo that scripts/deploy_space.py uploads, and by CI from the
# GitHub repo root:   docker build -f deploy/hf_space/Dockerfile .
# The serving bundle is NOT in the image: deploy/start_space.sh downloads it at startup from the
# model repo named in $BUNDLE_REPO (or uses one mounted at $RECSYS_BUNDLE, as CI does).
FROM python:3.11-slim

# libgomp: OpenMP runtime for LightGBM and FAISS; curl: health checks in the start script
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs Spaces as user id 1000 -> create it and install everything in its home
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 \
    RECSYS_BUNDLE=/home/user/app/data/serving RECSYS_API_URL=http://127.0.0.1:8000
WORKDIR /home/user/app

# CPU-only torch (~200 MB instead of ~2 GB with CUDA), then API + UI requirements
COPY --chown=user requirements-serve.txt requirements-ui.txt ./
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user -r requirements-serve.txt -r requirements-ui.txt

# the text encoder, baked in at build time -> free-text search works without a download at startup
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5', device='cpu')"

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
RUN pip install --user --no-deps .
COPY --chown=user app ./app
COPY --chown=user deploy/start_space.sh ./deploy/start_space.sh

EXPOSE 7860
CMD ["bash", "deploy/start_space.sh"]

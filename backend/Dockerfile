FROM python:3.11-slim

# git is needed to clone the repositories we index.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY reporag ./reporag
RUN pip install --no-cache-dir --no-deps .

RUN useradd --create-home --uid 1000 app && mkdir -p /data /models && chown app:app /data /models
USER app

ENV REPORAG_DATA_DIR=/data \
    HF_HOME=/models \
    REPORAG_DEVICE=cpu \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "reporag.main:app", "--host", "0.0.0.0", "--port", "8000"]

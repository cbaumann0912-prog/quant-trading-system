# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 quant \
    && mkdir -p /out \
    && chown quant:quant /out

# Copy as quant, not root: the container runs as a non-root user, and a
# root-owned /app leaves pytest unable to write .pytest_cache.
COPY --chown=quant:quant . .

USER quant

ENTRYPOINT ["python", "research/run_research.py"]
CMD ["--help"]

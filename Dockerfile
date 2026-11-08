FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LEAFLET_RAW_DIR=/app/data/raw \
    LEAFLET_INDEX_DIR=/app/data/index

WORKDIR /app

# Install dependencies first so code changes do not invalidate this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/raw /app/data/index \
    && chown -R app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if b'\"ok\"' in urllib.request.urlopen('http://localhost:8000/health').read() else 1)"

ENTRYPOINT ["./docker-entrypoint.sh"]

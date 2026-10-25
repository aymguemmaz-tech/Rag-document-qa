# syntax=docker/dockerfile:1.7
# ---- build stage: build a wheel with all dependencies ------------------------------------------
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

# ---- runtime stage: slim image, non-root user ---------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    RAGQA_DATA_DIR=/data RAGQA_PIPELINE_CONFIG=/app/configs/default.yaml
RUN useradd --create-home --uid 10001 ragqa && mkdir -p /data && chown ragqa:ragqa /data
COPY --from=build /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels
WORKDIR /app
COPY configs ./configs
COPY data ./data
USER ragqa
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "ragqa.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

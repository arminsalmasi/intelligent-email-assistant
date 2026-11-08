# One image, two entrypoints (API and dashboard). Multi-stage, non-root.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml README.md ./
COPY email_assistant ./email_assistant
COPY evals ./evals
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache ".[dashboard,tracing]"

FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd -g 10001 app && useradd -u 10001 -g app -m app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app email_assistant ./email_assistant
COPY --chown=app:app evals ./evals
COPY --chown=app:app fixtures ./fixtures
COPY --chown=app:app dashboard ./dashboard
RUN mkdir -p /app/data && chown app:app /app/data
USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "email_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

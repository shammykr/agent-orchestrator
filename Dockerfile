FROM python:3.12-slim AS build
WORKDIR /src
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ORCH_AGENTS_DIR=/app/config/agents \
    ORCH_PIPELINES_DIR=/app/config/pipelines \
    ORCH_USERS_FILE=/app/config/users.yaml \
    ORCH_KNOWLEDGE_BASE_FILE=/app/data/knowledge_base.json \
    ORCH_SKILL_PLUGIN_DIR=/app/plugins/skills
RUN useradd --create-home --uid 1000 orchestrator
COPY --from=build /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels
WORKDIR /app
# Pipelines, agents, skill plugins and data are plain files: mount over them to customise.
COPY config ./config
COPY data ./data
COPY plugins ./plugins
USER orchestrator
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/health').status != 200)"
CMD ["uvicorn", "orchestrator.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

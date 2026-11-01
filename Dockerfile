# One image, three roles (see docker-compose.yml): dashboard (Streamlit), pipeline, results (Dash)
FROM ghcr.io/astral-sh/uv:0.8-python3.12-bookworm-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --all-extras --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --all-extras

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home app
WORKDIR /app
COPY --from=build --chown=app /app /app
COPY --chown=app data ./data
COPY --chown=app streamlit_app.py ./
COPY --chown=app .streamlit ./.streamlit
RUN mkdir -p outputs && chown app outputs
ENV PATH="/app/.venv/bin:$PATH" WH_ROOT=/app PYTHONUNBUFFERED=1
USER app
EXPOSE 8501 8050
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

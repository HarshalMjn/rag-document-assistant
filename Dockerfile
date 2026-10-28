FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /code

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH"

ENV UV_COMPILE_BYTECODE=1

ENV UV_LINK_MODE=copy

# Install dependencies first so this layer is cached until the lock file changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

ENV PYTHONPATH=/code

COPY ./pyproject.toml ./uv.lock ./alembic.ini /code/

COPY ./app /code/app

COPY ./alembic /code/alembic

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

EXPOSE 8000

# Apply pending migrations before serving so the schema always matches the code.
CMD ["sh", "-c", "alembic upgrade head && fastapi run app/main.py --host 0.0.0.0 --port 8000"]

# StrataFi's API, worker and model gateway for compose's "app" profile: one image, three
# commands (stratafi-api, stratafi-worker, stratafi-gateway).
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

LABEL org.opencontainers.image.title="stratafi" \
      org.opencontainers.image.description="StrataFi, a lab for evaluating RAG pipelines: its API, worker and model gateway" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/stratafi/venv

WORKDIR /opt/stratafi/src
# uv at the version that writes uv.lock's format.
RUN pip install --no-cache-dir uv==0.13.5

COPY pyproject.toml uv.lock ./
COPY packages packages
COPY services services
RUN uv sync --locked --no-dev --no-editable \
    --package stratafi-api --package stratafi-worker --package stratafi-gateway

COPY config /opt/stratafi/config
# The app's user, and the directory its local artifacts live in (storage.local_root,
# data/artifacts from the working directory), where compose mounts a volume.
RUN useradd --system --uid 10001 --home-dir /opt/stratafi --shell /usr/sbin/nologin stratafi && \
    mkdir -p /opt/stratafi/data && \
    chown stratafi:stratafi /opt/stratafi/data
USER stratafi
WORKDIR /opt/stratafi
ENV PATH=/opt/stratafi/venv/bin:$PATH \
    STRATAFI_CONFIG_DIR=/opt/stratafi/config \
    STRATAFI_API__HOST=0.0.0.0
EXPOSE 8000 8091
CMD ["stratafi-api"]

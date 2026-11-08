# The sandbox custom checks run in. One check per container, started by the worker with no
# network, a read-only root, every capability dropped, no new privileges, a non-root user, and
# limits on processes, memory, CPU and time (stratafi_worker.sandbox). gVisor (runsc) runs it
# where the host has it. The image holds Python's standard library, the dependency-free
# stratafi.checks module and the entry point: no compiler, no pip, no shell tools a check needs.
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

LABEL org.opencontainers.image.title="stratafi-check-sandbox" \
      org.opencontainers.image.description="StrataFi, a lab for evaluating RAG pipelines: the sandbox custom checks run in" \
      org.opencontainers.image.licenses="MIT"

RUN python -m pip uninstall --yes pip setuptools wheel >/dev/null 2>&1 || true && \
    mkdir -p /opt/sandbox/stratafi /bundle /check /scratch && \
    touch /opt/sandbox/stratafi/__init__.py
COPY packages/sdk/src/stratafi/checks.py /opt/sandbox/stratafi/checks.py
COPY docker/sandbox/run_check.py /opt/sandbox/run_check.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65534:65534
WORKDIR /scratch
ENTRYPOINT ["python", "-I", "-B", "/opt/sandbox/run_check.py"]

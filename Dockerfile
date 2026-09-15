# A downloaded System: the model server plus the invoke path, no control plane.
# The self-host path is a compose file on a GPU box, not a VPC deployment.
#
#   docker build -t mekoy-system .
#   docker run --rm -p 11434:11434 mekoy-system
#   MODEL=qwen2.5:14b
#
# The compiled System is the harness, not the weights. spec.json and report.txt
# come from the compile and describe how to call it.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL=qwen2.5:14b

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir "uvicorn[standard]"

# Ollama serves the open-weight model the System was compiled against.
# zstd is required by the Ollama install script to unpack its archive.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

COPY examples ./examples

EXPOSE 11434

# Pull the model at first start, then serve. Pulling at build time would bake a
# multi-GB layer and pin a model the spec can change.
CMD ["sh", "-c", "ollama serve & sleep 5 && ollama pull \"$MODEL\" && wait"]

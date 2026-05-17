FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip git curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /workspace
COPY pyproject.toml ./
RUN pip install -U pip wheel && pip install -e ".[dev]" hf_transfer
RUN pip install flash-attn==2.6.3 --no-build-isolation

COPY . .
ENV PYTHONPATH=/workspace
CMD ["/bin/bash"]